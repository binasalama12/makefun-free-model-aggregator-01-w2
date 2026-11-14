#!/usr/bin/env python3
"""Aggregator entry, flavour "wire-loyal": a single gateway fronting ten free
providers behind the two client dialects the brief describes.

The gateway answers OpenAI-style chat completions and Anthropic-style
messages, in plain JSON or as server-sent events, by picking the best-rated
free model that can actually handle each request and moving on to the next
candidate whenever a provider turns the request away with a 402 or 429. It
announces its model list to the trial harness, works through the harness
workload, and quits with status zero once the harness reports it is finished.

Only the standard library is used. Nothing here depends on the clock: every
routing decision follows from the brief, the incoming request, and the
answers providers have given so far.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------------------
# Flavour settings: this wallet's build tunes exactly these values.
# ---------------------------------------------------------------------------
VARIANT_NAME = "wire-loyal"
PREFER_SAME_WIRE_BONUS = 8      # extra milli-rating when provider wire matches the caller
REFUSAL_LIMIT = 2               # refusals after which a provider is treated as spent
LONG_CONTEXT_CHARS = 3000       # user text this long counts as a long-context ask
LONG_CONTEXT_MIN = 200000       # context window a long-context ask demands
ANTHROPIC_JSON_SAFE = True      # Anthropic callers are assumed to need JSON-capable models
STREAM_PIECE_CHARS = 0          # 0 keeps the provider's own stream chunks; else re-chop
TIE_BREAK = "brief"             # tied scores: "brief" keeps directory order, "alpha" sorts names

FIXED_CREATED = 1767225600      # frozen timestamp the fake providers stamp on replies
ENV_BASE = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
BRIEF_PATH = os.environ.get("MAKEFUN_PROMPT_FILE", "")
PORT = 8791

# ---------------------------------------------------------------------------
# Tiny HTTP client built on urllib, with proxies switched off: trials hand us
# plain base URLs, never a proxy address.
# ---------------------------------------------------------------------------
_CLIENT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def exchange(method, url, payload=None, headers=None, timeout=15):
    """Run a single HTTP exchange.

    Gives back (status, headers as a dict, raw body bytes). A failure to reach
    the server at all shows up as status 0 with an empty body.
    """
    wire_data = None
    out_headers = dict(headers or {})
    if payload is not None:
        wire_data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        out_headers.setdefault("content-type", "application/json")
    call = urllib.request.Request(url, data=wire_data, headers=out_headers, method=method)
    try:
        with _CLIENT.open(call, timeout=timeout) as answer:
            return answer.status, dict(answer.headers.items()), answer.read()
    except urllib.error.HTTPError as failure:
        return failure.code, dict(failure.headers.items()) if failure.headers else {}, failure.read()
    except Exception:
        return 0, {}, b""


def exchange_json(method, url, payload=None, headers=None, timeout=15):
    """Like exchange, but decodes a JSON body when there is one to decode."""
    code, heads, blob = exchange(method, url, payload, headers, timeout)
    try:
        return code, heads, json.loads(blob.decode("utf-8")) if blob else None
    except Exception:
        return code, heads, None


# ---------------------------------------------------------------------------
# The brief: provider directory, quality table and workload list, tucked away
# as JSON documents inside the prompt file's running prose.
# ---------------------------------------------------------------------------
def _pull_doc(prose, schema):
    """Fish one embedded JSON document out of the prompt text by its schema tag."""
    tag = '"schema":"' + schema + '"'
    pos = prose.find(tag)
    if pos < 0:
        tag = '"schema": "' + schema + '"'
        pos = prose.find(tag)
    if pos < 0:
        return None
    brace = prose.rfind("{", 0, pos)
    if brace < 0:
        return None
    try:
        found, _ = json.JSONDecoder().raw_decode(prose[brace:])
        return found
    except Exception:
        return None


class Upstream:
    """One provider row from the brief's directory, with its auth attached."""

    __slots__ = ("wire", "id", "name", "auth", "key", "key_env", "path")

    def __init__(self, entry):
        self.id = entry["id"]
        self.name = entry.get("name", entry["id"])
        self.wire = entry["wire"]
        self.auth = entry.get("auth", "none")
        self.key_env = entry.get("key_env", "")
        self.key = entry.get("key", "")
        self.path = entry["path"]

    def endpoint(self):
        """Full base URL for this provider under the trial environment."""
        return ENV_BASE + self.path

    def credential_headers(self):
        """Auth headers for this provider, or nothing when it needs no key."""
        if self.auth == "bearer":
            return {"authorization": "Bearer " + self.key}
        if self.auth == "x-api-key":
            return {"x-api-key": self.key}
        if self.auth == "x-goog-api-key":
            return {"x-goog-api-key": self.key}
        return {}


def read_brief():
    """Pull (providers, rating rows, manifest) from the prompt file.

    Returns three empty lists when no prompt file is configured or readable.
    """
    if not BRIEF_PATH:
        return [], [], []  # no prompt file configured at all
    try:
        with open(BRIEF_PATH, "r", encoding="utf-8") as fh:
            prose = fh.read()
    except Exception:
        return [], [], []  # prompt file unreadable
    index = _pull_doc(prose, "free-model-aggregator-01/providers/v1") or {}
    scores = _pull_doc(prose, "free-model-aggregator-01/ratings/v1") or {}
    jobs = _pull_doc(prose, "free-model-aggregator-01/workload/v1") or {}
    backends = [Upstream(entry) for entry in index.get("providers", [])]
    return backends, scores.get("ratings", []), jobs.get("manifest", [])

# ---------------------------------------------------------------------------
# The catalog: ask each provider for its own model list (ground truth first),
# and use the brief's quality table only where a probe comes back empty.
# ---------------------------------------------------------------------------
def _fetch_models(upstream):
    """[(id, free, context_length)] from the provider's own catalog route.

    Returns None when the probe fails, so the caller can fall back to the
    brief's table instead of trusting a half answer.
    """
    if upstream.wire == "gemini":
        target = upstream.endpoint() + "/v1beta/models"
    else:
        target = upstream.endpoint() + "/v1/models"
    # Slow catalog endpoints need a generous window; 15s keeps them answerable.
    code, _, listing = exchange_json("GET", target, headers=upstream.credential_headers(), timeout=15)
    if not 200 <= code < 300 or not isinstance(listing, dict):
        return None
    found = []
    if upstream.wire == "gemini":
        for entry in listing.get("models", []):
            model_id = str(entry.get("name", "")).removeprefix("models/")
            if model_id:
                found.append((model_id, entry.get("free") is True, entry.get("context_length") or entry.get("inputTokenLimit")))
    else:
        for entry in listing.get("data", []):
            if isinstance(entry.get("id"), str) and entry["id"]:
                found.append((entry["id"], entry.get("free") is True, entry.get("context_length")))
    # An empty list means the probe found nothing usable, so report None and
    # let the caller fall back to the brief's quality table for this provider.
    return found if found else None


def assemble_catalog(providers, ratings):
    """Build the registration catalog: probed models where possible, brief rows otherwise."""
    table = {}
    for entry in ratings:
        table.setdefault(entry["provider"], []).append(
            (entry["model"], entry.get("free") is True, entry.get("context_length")))
    catalog = {"providers": []}
    for upstream in providers:
        entries = _fetch_models(upstream) if ENV_BASE else None
        if entries is None:
            entries = table.get(upstream.id, [])
        catalog["providers"].append({
            "id": upstream.id,
            "base_url": upstream.path,
            "key_env": upstream.key_env,
            "wire": upstream.wire,
            "models": [
                {"id": model_id, "free": is_free, "context_length": window}
                for model_id, is_free, window in entries
            ],
        })
    return catalog


def catalog_json(catalog):
    """Serialize the catalog for the --list output and for registration."""
    return json.dumps(catalog)


# ---------------------------------------------------------------------------
# Canonical request: the caller's request rewritten wire-neutral, dropping
# nothing the caller sent.
# ---------------------------------------------------------------------------
def _image_piece(url):
    """Turn an image URL (possibly an inline data: URL) into a canonical part."""
    if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
        mime = url[5:url.index(";base64,")]
        return {"type": "image", "media_type": mime, "data": url.split(";base64,", 1)[1]}
    return {"type": "image_ref", "url": url}


def _collect_text(content):
    """Plain text out of a string body or a list of text blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for piece in content:
            if isinstance(piece, dict) and piece.get("type") == "text" and isinstance(piece.get("text"), str):
                chunks.append(piece["text"])
        return "\n".join(chunks)
    return ""


class CanonReq:
    """Normalized request.

    system: str|None; messages: [{role, parts}] where each part is
    {type:text,text} / {type:image,media_type,data} / {type:image_ref,url};
    tools: [{name, description, parameters}]; plus the untouched client body.
    """

    def __init__(self, api, body):
        self.api = api
        self.raw = body
        self.stream = body.get("stream") is True
        self.max_tokens = None
        self.stops = []
        self.json_mode = False
        self.response_format = None
        self.system = None
        self.messages = []
        self.tools = []
        self.tool_choice = None
        if api == "openai":
            self._parse_openai(body)
        else:
            self._parse_anthropic(body)

    def _parse_openai(self, body):
        cap = body.get("max_tokens")
        if not isinstance(cap, (int, float)):
            cap = body.get("max_completion_tokens")
        self.max_tokens = cap if isinstance(cap, (int, float)) else None
        stop_in = body.get("stop")
        if isinstance(stop_in, str):
            self.stops = [stop_in]
        elif isinstance(stop_in, list):
            self.stops = [s for s in stop_in if isinstance(s, str)]
        fmt_spec = body.get("response_format")
        if isinstance(fmt_spec, dict):
            self.response_format = fmt_spec
            self.json_mode = fmt_spec.get("type") in ("json_object", "json_schema")
        self.tool_choice = body.get("tool_choice")
        for spec in body.get("tools") or []:
            func = spec.get("function") if isinstance(spec, dict) else None
            if isinstance(func, dict) and isinstance(func.get("name"), str):
                self.tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters") or {"type": "object"},
                })
        sys_chunks = []
        for turn in body.get("messages") or []:
            if not isinstance(turn, dict):
                continue
            role = turn.get("role")
            content = turn.get("content")
            if role == "system":
                text = _collect_text(content)
                if text:
                    sys_chunks.append(text)
                continue
            segments = []
            if isinstance(content, str):
                if content:
                    segments.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for piece in content:
                    if not isinstance(piece, dict):
                        continue
                    if piece.get("type") == "text" and isinstance(piece.get("text"), str):
                        segments.append({"type": "text", "text": piece["text"]})
                    elif piece.get("type") == "image_url" and isinstance(piece.get("image_url"), dict):
                        segments.append(_image_piece(piece["image_url"].get("url", "")))
            self.messages.append({"role": role or "user", "parts": segments})
        self.system = "\n".join(sys_chunks) if sys_chunks else None

    def _parse_anthropic(self, body):
        cap = body.get("max_tokens")
        self.max_tokens = cap if isinstance(cap, (int, float)) else None
        stop_list = body.get("stop_sequences")
        if isinstance(stop_list, list):
            self.stops = [s for s in stop_list if isinstance(s, str)]
        self.tool_choice = body.get("tool_choice")
        sys_in = body.get("system")
        if isinstance(sys_in, str):
            self.system = sys_in or None
        elif isinstance(sys_in, list):
            sys_text = _collect_text(sys_in)
            self.system = sys_text or None
        for spec in body.get("tools") or []:
            if isinstance(spec, dict) and isinstance(spec.get("name"), str):
                self.tools.append({
                    "name": spec["name"],
                    "description": spec.get("description", ""),
                    "parameters": spec.get("input_schema") or {"type": "object"},
                })
        for turn in body.get("messages") or []:
            if not isinstance(turn, dict):
                continue
            content = turn.get("content")
            segments = []
            if isinstance(content, str):
                if content:
                    segments.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for piece in content:
                    if not isinstance(piece, dict):
                        continue
                    if piece.get("type") == "text" and isinstance(piece.get("text"), str):
                        segments.append({"type": "text", "text": piece["text"]})
                    elif piece.get("type") == "image" and isinstance(piece.get("source"), dict):
                        source = piece["source"]
                        if source.get("type") == "base64" and isinstance(source.get("data"), str):
                            segments.append({"type": "image", "media_type": source.get("media_type", "image/png"), "data": source["data"]})
                        elif source.get("type") == "url":
                            segments.append({"type": "image_ref", "url": source.get("url", "")})
            self.messages.append({"role": turn.get("role") or "user", "parts": segments})

    # -- what the request itself demands, read straight off its contents ----
    @property
    def needs_tools(self):
        """True when the caller attached at least one tool definition."""
        return len(self.tools) > 0

    @property
    def needs_vision(self):
        """True when any message carries an image part of either flavour."""
        return any(seg["type"] in ("image", "image_ref") for turn in self.messages for seg in turn["parts"])

    @property
    def user_text(self):
        """Text of the most recent user turn, or an empty string when absent."""
        for turn in reversed(self.messages):
            if turn["role"] == "user":
                return "\n".join(seg["text"] for seg in turn["parts"] if seg["type"] == "text")
        return ""

# ---------------------------------------------------------------------------
# Shaping a provider-wire request out of the canonical one.
# ---------------------------------------------------------------------------
def _to_gemini_payload(req):
    """Canonical request rendered as a Gemini generateContent body."""
    turns = []
    for turn in req.messages:
        segments = []
        for seg in turn["parts"]:
            if seg["type"] == "text":
                segments.append({"text": seg["text"]})
            elif seg["type"] == "image":
                segments.append({"inlineData": {"mimeType": seg["media_type"], "data": seg["data"]}})
            else:
                segments.append({"fileData": {"fileUri": seg["url"]}})
        turns.append({"role": "model" if turn["role"] == "assistant" else "user", "parts": segments})
    payload = {"contents": turns}
    if req.system is not None:
        payload["systemInstruction"] = {"parts": [{"text": req.system}]}
    if req.tools:
        payload["tools"] = [{"functionDeclarations": [
            {"name": tool["name"], "description": tool["description"], "parameters": tool["parameters"]}
            for tool in req.tools]}]
    settings = {}
    if req.max_tokens is not None:
        settings["maxOutputTokens"] = req.max_tokens
    if req.stops:
        settings["stopSequences"] = list(req.stops)
    if req.json_mode:
        settings["responseMimeType"] = "application/json"
    if settings:
        payload["generationConfig"] = settings
    return payload


def _to_openai_msgs(req):
    """Canonical messages rendered the way the OpenAI wire expects them."""
    rendered = []
    if req.system is not None:
        rendered.append({"role": "system", "content": req.system})
    for turn in req.messages:
        segments = turn["parts"]
        if all(seg["type"] == "text" for seg in segments):
            rendered.append({"role": turn["role"], "content": "\n".join(seg["text"] for seg in segments)})
            continue
        blocks = []
        for seg in segments:
            if seg["type"] == "text":
                blocks.append({"type": "text", "text": seg["text"]})
            elif seg["type"] == "image":
                blocks.append({"type": "image_url", "image_url": {
                    "url": f"data:{seg['media_type']};base64,{seg['data']}"}})
            else:
                blocks.append({"type": "image_url", "image_url": {"url": seg["url"]}})
        rendered.append({"role": turn["role"], "content": blocks})
    return rendered


def _to_anthropic_msgs(req):
    """Canonical messages rendered the way the Anthropic wire expects them."""
    rendered = []
    for turn in req.messages:
        segments = turn["parts"]
        if all(seg["type"] == "text" for seg in segments):
            rendered.append({"role": turn["role"], "content": "\n".join(seg["text"] for seg in segments)})
            continue
        blocks = []
        for seg in segments:
            if seg["type"] == "text":
                blocks.append({"type": "text", "text": seg["text"]})
            elif seg["type"] == "image":
                blocks.append({"type": "image", "source": {
                    "type": "base64", "media_type": seg["media_type"], "data": seg["data"]}})
            else:
                blocks.append({"type": "image", "source": {"type": "url", "url": seg["url"]}})
        rendered.append({"role": turn["role"], "content": blocks})
    return rendered


def _translate_tool_choice(wire, req):
    """Map the caller's tool_choice onto the target wire.

    Returns (translated value, True) when the wire has a matching form, and
    (None, False) when the choice cannot be carried across.
    """
    choice = req.tool_choice
    if choice is None:
        return None, False
    if wire == "openai":
        return choice, True
    if wire == "anthropic":
        if isinstance(choice, str):
            return {"type": choice if choice in ("auto", "any", "none") else "auto"}, True
        if isinstance(choice, dict):
            if choice.get("type") == "function" and isinstance(choice.get("function"), dict):
                return {"type": "tool", "name": choice["function"].get("name")}, True
            if choice.get("type") in ("auto", "any", "none"):
                return {"type": choice["type"]}, True
        return None, False
    return None, False


def make_upstream_call(upstream, model_id, req):
    """Build (url, headers, body dict) for one inference call on this provider."""
    hdrs = upstream.credential_headers()
    if upstream.wire == "openai":
        if req.api == "openai":
            payload = json.loads(json.dumps(req.raw))  # caller's own body, passed through as-is
            payload["model"] = model_id
        else:
            payload = {"model": model_id, "messages": _to_openai_msgs(req)}
            if req.max_tokens is not None:
                payload["max_tokens"] = req.max_tokens
            if req.stops:
                payload["stop"] = list(req.stops)
            if req.tools:
                payload["tools"] = [{"type": "function", "function": {
                    "name": tool["name"], "description": tool["description"],
                    "parameters": tool["parameters"]}} for tool in req.tools]
            choice, usable = _translate_tool_choice("openai", req)
            if usable:
                payload["tool_choice"] = choice
            if req.stream:
                payload["stream"] = True
        if req.stream:
            payload["stream"] = True
            opts = dict(payload.get("stream_options") or {})
            opts["include_usage"] = True
            payload["stream_options"] = opts
        return upstream.endpoint() + "/v1/chat/completions", hdrs, payload
    if upstream.wire == "anthropic":
        hdrs["anthropic-version"] = "2023-06-01"
        if req.api == "anthropic":
            payload = json.loads(json.dumps(req.raw))
            payload["model"] = model_id
        else:
            payload = {"model": model_id, "messages": _to_anthropic_msgs(req)}
            if req.system is not None:
                payload["system"] = req.system
            if req.max_tokens is not None:
                payload["max_tokens"] = req.max_tokens
            if req.stops:
                payload["stop_sequences"] = list(req.stops)
            if req.tools:
                payload["tools"] = [{"name": tool["name"], "description": tool["description"],
                                     "input_schema": tool["parameters"]} for tool in req.tools]
            choice, usable = _translate_tool_choice("anthropic", req)
            if usable:
                payload["tool_choice"] = choice
            if req.stream:
                payload["stream"] = True
        return upstream.endpoint() + "/v1/messages", hdrs, payload
    # gemini
    payload = _to_gemini_payload(req)
    action = "streamGenerateContent" if req.stream else "generateContent"
    target = upstream.endpoint() + "/v1beta/models/%s:%s" % (
        urllib.parse.quote(model_id, safe=""), action)
    if req.stream:
        target = target + "?alt=sse"
    return target, hdrs, payload


# ---------------------------------------------------------------------------
# Reading a provider's answer back into (text, pieces, stop, usage in/out).
# ---------------------------------------------------------------------------
def _split_sse(raw):
    """Split a server-sent-events body into [(event, data)] frames."""
    events = []
    for block in raw.decode("utf-8", "replace").replace("\r\n", "\n").split("\n\n"):
        kind = None
        lines = []
        for row in block.split("\n"):
            if row.startswith("event:"):
                kind = row[6:].strip()
            elif row.startswith("data:"):
                lines.append(row[5:].lstrip(" "))
        if lines:
            events.append((kind, "\n".join(lines)))
    return events


STOP_MAP_OAI = {"stop": "stop", "end_turn": "stop", "STOP": "stop",
                "max_tokens": "length", "MAX_TOKENS": "length",
                "tool_use": "tool_calls", "refusal": "content_filter", "SAFETY": "content_filter"}
STOP_MAP_ANTH = {"stop": "end_turn", "STOP": "end_turn", "end_turn": "end_turn",
                 "length": "max_tokens", "MAX_TOKENS": "max_tokens",
                 "tool_calls": "tool_use", "function_call": "tool_use",
                 "content_filter": "refusal", "SAFETY": "refusal"}


def interpret_reply(wire, raw, stream):
    """Decode one provider reply into dict(text, pieces, stop, usage=(in, out)).

    Returns None when the reply cannot be understood at all.
    """
    try:
        if wire == "openai":
            if not stream:
                reply = json.loads(raw.decode("utf-8"))
                pick = reply["choices"][0]
                text_out = pick["message"]["content"]
                if isinstance(text_out, list):
                    text_out = "".join(blk.get("text", "") for blk in text_out if isinstance(blk, dict))
                tokens = reply.get("usage") or {}
                return {"pieces": [text_out], "text": text_out, "stop": pick.get("finish_reason"),
                        "usage": (tokens.get("prompt_tokens"), tokens.get("completion_tokens"))}
            text, pieces, stop, usage = "", [], None, None
            for _, data in _split_sse(raw):
                if data.strip() == "[DONE]":
                    continue
                piece_doc = json.loads(data)
                if isinstance(piece_doc.get("usage"), dict):
                    usage = (piece_doc["usage"].get("prompt_tokens"), piece_doc["usage"].get("completion_tokens"))
                if piece_doc.get("choices"):
                    pick = piece_doc["choices"][0]
                    diff = pick.get("delta") or {}
                    if isinstance(diff.get("content"), str) and diff["content"]:
                        pieces.append(diff["content"])
                        text += diff["content"]
                    if isinstance(pick.get("finish_reason"), str):
                        stop = pick["finish_reason"]
            return {"text": text, "pieces": pieces, "stop": stop, "usage": usage}
        if wire == "anthropic":
            if not stream:
                reply = json.loads(raw.decode("utf-8"))
                text = "".join(blk.get("text", "") for blk in reply.get("content", [])
                               if isinstance(blk, dict) and blk.get("type") == "text")
                tokens = reply.get("usage") or {}
                return {"pieces": [text], "text": text, "stop": reply.get("stop_reason"),
                        "usage": (tokens.get("input_tokens"), tokens.get("output_tokens"))}
            text, pieces, stop = "", [], None
            tokens_in, tokens_out = None, None
            for event, data in _split_sse(raw):
                reply = json.loads(data)
                if event == "message_start":
                    tokens_in = (reply.get("message") or {}).get("usage", {}).get("input_tokens")
                elif event == "content_block_delta":
                    diff = reply.get("delta") or {}
                    if diff.get("type") == "text_delta" and isinstance(diff.get("text"), str):
                        pieces.append(diff["text"])
                        text += diff["text"]
                elif event == "message_delta":
                    stop = (reply.get("delta") or {}).get("stop_reason") or stop
                    tokens_out = (reply.get("usage") or {}).get("output_tokens", tokens_out)
            return {"text": text, "pieces": pieces, "stop": stop,
                    "usage": (tokens_in, tokens_out) if tokens_in is not None or tokens_out is not None else None}
        # gemini
        if not stream:
            reply = json.loads(raw.decode("utf-8"))
            candidate = reply["candidates"][0]
            text = "".join(seg.get("text", "") for seg in candidate["content"]["parts"] if isinstance(seg, dict))
            usage_meta = reply.get("usageMetadata") or {}
            return {"pieces": [text], "text": text, "stop": candidate.get("finishReason"),
                    "usage": (usage_meta.get("promptTokenCount"), usage_meta.get("candidatesTokenCount"))}
        text, pieces, stop, usage = "", [], None, None
        for _, data in _split_sse(raw):
            reply = json.loads(data)
            for candidate in reply.get("candidates", []):
                for seg in (candidate.get("content") or {}).get("parts", []):
                    if isinstance(seg, dict) and isinstance(seg.get("text"), str) and seg["text"]:
                        pieces.append(seg["text"])
                        text += seg["text"]
                if isinstance(candidate.get("finishReason"), str):
                    stop = candidate["finishReason"]
            usage_meta = reply.get("usageMetadata")
            if isinstance(usage_meta, dict):
                usage = (usage_meta.get("promptTokenCount"), usage_meta.get("candidatesTokenCount"))
        return {"text": text, "pieces": pieces, "stop": stop, "usage": usage}
    except Exception:
        return None


def _stream_chunks(answer):
    """Stream pieces for the reply; re-chop only when STREAM_PIECE_CHARS says so."""
    cuts = [p for p in answer["pieces"] if p]
    if STREAM_PIECE_CHARS > 0:
        full = answer["text"]
        cuts = [full[i:i + STREAM_PIECE_CHARS] for i in range(0, len(full), STREAM_PIECE_CHARS)] or [""]
    return cuts or [""]


# ---------------------------------------------------------------------------
# Answering the caller back in its own dialect.
# ---------------------------------------------------------------------------
def render_reply(req, model_id, answer):
    """(content_type, body bytes) for the caller's wire, streamed if asked."""
    tokens = answer["usage"] or (None, None)
    if req.api == "openai":
        reason = STOP_MAP_OAI.get(answer["stop"], "stop")
        if not req.stream:
            out = {"id": "chatcmpl-agg", "object": "chat.completion", "created": FIXED_CREATED,
                   "model": model_id,
                   "choices": [{"index": 0, "message": {"role": "assistant", "content": answer["text"]},
                                "finish_reason": reason, "logprobs": None}],
                   "usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1],
                             "total_tokens": (tokens[0] or 0) + (tokens[1] or 0)}}
            return "application/json", json.dumps(out).encode("utf-8")
        events = []

        def mk_chunk(delta, finish):
            return {"id": "chatcmpl-agg", "object": "chat.completion.chunk", "created": FIXED_CREATED,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        events.append("data: " + json.dumps(mk_chunk({"role": "assistant", "content": ""}, None)) + "\n\n")
        for cut in _stream_chunks(answer):
            events.append("data: " + json.dumps(mk_chunk({"content": cut}, None)) + "\n\n")
        events.append("data: " + json.dumps(mk_chunk({}, reason)) + "\n\n")
        if tokens[0] is not None:
            events.append("data: " + json.dumps({
                "id": "chatcmpl-agg", "object": "chat.completion.chunk", "created": FIXED_CREATED,
                "model": model_id, "choices": [],
                "usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1],
                          "total_tokens": (tokens[0] or 0) + (tokens[1] or 0)}}) + "\n\n")
        events.append("data: [DONE]\n\n")
        return "text/event-stream", "".join(events).encode("utf-8")
    # anthropic
    reason = STOP_MAP_ANTH.get(answer["stop"], "end_turn")
    if not req.stream:
        out = {"id": "msg_agg", "type": "message", "role": "assistant", "model": model_id,
               "content": [{"type": "text", "text": answer["text"]}],
               "stop_reason": reason, "stop_sequence": None,
               "usage": {"input_tokens": tokens[0], "output_tokens": tokens[1]}}
        return "application/json", json.dumps(out).encode("utf-8")
    events = []

    def mk_frame(event, payload):
        return "event: %s\ndata: %s\n\n" % (event, json.dumps(payload))

    events.append(mk_frame("message_start", {
        "type": "message_start",
        "message": {"id": "msg_agg", "type": "message", "role": "assistant", "model": model_id,
                    "content": [], "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": tokens[0], "output_tokens": 0}}}))
    events.append(mk_frame("content_block_start", {
        "type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}))
    for cut in _stream_chunks(answer):
        events.append(mk_frame("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": cut}}))
    events.append(mk_frame("content_block_stop", {"type": "content_block_stop", "index": 0}))
    events.append(mk_frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": reason, "stop_sequence": None},
        "usage": {"output_tokens": tokens[1]}}))
    events.append(mk_frame("message_stop", {"type": "message_stop"}))
    return "text/event-stream", "".join(events).encode("utf-8")

# ---------------------------------------------------------------------------
# Routing: order the free models that fit the request by the brief's rating,
# try them one by one, and keep a tally of refusals as they arrive.
# ---------------------------------------------------------------------------
class Picker:
    """Holds the rating table plus per-provider refusal tallies for one run."""

    def __init__(self, providers, ratings):
        self.by_id = {p.id: p for p in providers}
        self.sequence = [p.id for p in providers]
        self.table = {}
        for entry in ratings:
            self.table[(entry["provider"], entry["model"])] = entry
        self.calls = {pid: 0 for pid in self.sequence}
        self.rejections = {pid: 0 for pid in self.sequence}
        self.dead = set()

    def ranked(self, req):
        """[(provider, model_id)] best first, filtered to what this ask needs."""
        want_json = req.json_mode or (req.api == "anthropic" and ANTHROPIC_JSON_SAFE)
        is_long = len(req.user_text) >= LONG_CONTEXT_CHARS
        ranked_list = []
        for pos, pid in enumerate(self.sequence):
            upstream = self.by_id[pid]
            for (prov_id, model_id), entry in self.table.items():
                if prov_id != pid or entry.get("free") is not True:
                    continue
                if req.needs_tools and not entry.get("tools"):
                    continue
                if req.needs_vision and not entry.get("vision"):
                    continue
                if want_json and not entry.get("json"):
                    continue
                if is_long and (entry.get("context_length") or 0) < LONG_CONTEXT_MIN:
                    continue
                points = entry.get("rating_milli", 0)
                if upstream.wire == req.api:
                    points += PREFER_SAME_WIRE_BONUS
                ranked_list.append((-points, pos, upstream, model_id))
        if TIE_BREAK == "alpha":
            ranked_list.sort(key=lambda item: (item[0], item[2].id, item[3]))
        else:
            ranked_list.sort(key=lambda item: (item[0], item[1]))
        return [(upstream, model_id) for _, _, upstream, model_id in ranked_list]

    def dispatch(self, req):
        """Serve one request, or None when no free model can take it."""
        for prov, model_id in self.ranked(req):
            if prov.id in self.dead:
                continue
            target, hdrs, payload = make_upstream_call(prov, model_id, req)
            code, _, blob = exchange("POST", target, payload, hdrs, timeout=20)
            self.calls[prov.id] += 1
            if 200 <= code < 300:
                reply = interpret_reply(prov.wire, blob, req.stream)
                if reply is not None:
                    return model_id, reply
                continue
    # A 402/429 is a refusal, not a crash: count it, and only give up on the
    # provider once refusals reach REFUSAL_LIMIT within this run.
            if code in (402, 429):
                self.rejections[prov.id] += 1
                if self.rejections[prov.id] >= REFUSAL_LIMIT:
                    self.dead.add(prov.id)
                continue
            # any other answer (400, 5xx, silence): move on to the next model
        return None


# ---------------------------------------------------------------------------
# The server the trial harness client talks to.
# ---------------------------------------------------------------------------
SHARED = {"router": None, "catalog": None}


def list_models(headers):
    """Model list for GET /v1/models; shape follows the caller's key header."""
    listing = SHARED["catalog"] or {"providers": []}
    seen = []
    for prov in listing["providers"]:
        for entry in prov["models"]:
            if entry["free"] and entry["id"] not in seen:
                seen.append(entry["id"])
    if headers.get("x-api-key") is not None:
        out = {"data": [{"type": "model", "id": mid, "display_name": mid,
                         "created_at": "2026-01-01T00:00:00Z"} for mid in seen],
               "has_more": False,
               "first_id": seen[0] if seen else None,
               "last_id": seen[-1] if seen else None}
    else:
        out = {"object": "list",
               "data": [{"id": mid, "object": "model", "created": FIXED_CREATED,
                         "owned_by": "aggregator"} for mid in seen]}
    return json.dumps(out).encode("utf-8")


def failure_payload(api, note):
    """Error body shaped for the caller's wire."""
    if api == "anthropic":
        return json.dumps({"type": "error", "error": {"type": "api_error", "message": note}}).encode("utf-8")
    return json.dumps({"error": {"message": note, "type": "api_error",
                                 "param": None, "code": "no_provider"}}).encode("utf-8")


class Gateway(BaseHTTPRequestHandler):
    """HTTP front door: model list on GET, completions/messages on POST."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _reply(self, status, content_type, body):
        try:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path.split("?")[0] == "/v1/models":
            self._reply(200, "application/json", list_models(self.headers))
        else:
            self._reply(404, "application/json", failure_payload("openai", "not found"))

    def do_POST(self):
        route = self.path.split("?")[0]
        if route == "/v1/chat/completions":
            dialect = "openai"
        elif route == "/v1/messages":
            dialect = "anthropic"
        else:
            self._reply(404, "application/json", failure_payload("openai", "not found"))
            return
        try:
            size = int(self.headers.get("content-length") or 0)
        except ValueError:
            size = 0
        raw_in = self.rfile.read(size) if size > 0 else b""
        try:
            payload = json.loads(raw_in.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("body is not an object")
        except Exception:
            self._reply(400, "application/json", failure_payload(dialect, "the request body is not one JSON object"))
            return
        ask = CanonReq(dialect, payload)
        hit = SHARED["router"].dispatch(ask)
        if hit is None:
            self._reply(502, "application/json", failure_payload(dialect, "no free model could serve this request"))
            return
        picked, reply = hit
        mime, blob = render_reply(ask, picked, reply)
        self._reply(200, mime, blob)


# ---------------------------------------------------------------------------
# Trial lifecycle: probe the catalog, register with the harness, serve until
# the harness says the workload is done, then exit 0.
# ---------------------------------------------------------------------------
def main(argv):
    backends, scores, _ = read_brief()
    if "--list" in argv:
        if len(backends) == 0:
            print('{"providers":[]}')
            return 0
        print(catalog_json(assemble_catalog(backends, scores)))
        return 0
    listing = assemble_catalog(backends, scores)
    SHARED["catalog"] = listing
    SHARED["router"] = Picker(backends, scores)
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Gateway)
    srv.daemon_threads = True
    thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05})
    thread.daemon = True
    thread.start()
    if ENV_BASE:
        for _ in range(30):
            code, _, reply = exchange_json("POST", ENV_BASE + "/register",
                                           {"protocol_version": 1, "port": PORT, "catalog": listing},
                                           timeout=5)
            if code == 200 and isinstance(reply, dict) and reply.get("ok") is True:
                break
            time.sleep(0.5)
        blanks = 0
        elapsed = 0.0
        while elapsed < 170.0:
            code, _, reply = exchange_json("GET", ENV_BASE + "/work", timeout=5)
            if code == 200 and isinstance(reply, dict):
                blanks = 0
                if reply.get("done") is True:
                    break
            else:
                blanks += 1
                if blanks > 40:
                    break
            time.sleep(0.2)
            elapsed += 0.2
    else:
        thread.join()
    srv.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
