#!/usr/bin/env python3
"""sse-map-proxy — one tiny HTTP proxy for OpenAI-compatible SSE endpoints.

Single instance, path-based routing:

  POST /v1/chat/completions      -> https://api.cline.bot/api/v1/chat/completions
                                    (request passthrough, response mapping)
  POST /zen/v1/chat/completions  -> https://opencode.ai/zen/v1/chat/completions
                                    (request shaping + response mapping)
  GET  /health                   -> {"ok": true, ...}

What it does
------------
* Streams SSE responses line-by-line without buffering the whole body.
* Maps ``delta.reasoning`` -> ``delta.reasoning_content`` and drops
  ``reasoning_details`` so non-OpenRouter clients can read the chain of thought.
* Aggregates an SSE stream into a single JSON ``chat.completion`` when the
  client asked for ``stream: false``.
* For the zen route, rewrites the request so it passes OpenCode's free-tier
  gate: sets ``User-Agent: opencode/...``, a valid ``x-opencode-session`` and
  injects minimal placeholder ``bash`` / ``read`` tools (merging with any
  client-supplied tools).

Usage
-----
    python3 sse_map_proxy.py --host --port 3457

No third-party dependencies (stdlib only). Python >= 3.8.
"""

import argparse
import http.client
import http.server
import json
import os
import ssl
import sys
import time

CLINE_HOST = "api.cline.bot"
CLINE_PATH = "/api/v1/chat/completions"

ZEN_HOST = "opencode.ai"
ZEN_UA = "opencode/1.18.31 ai-sdk/provider-utils/4.0.40 runtime/bun/1.3.14"
ZEN_SESSION = "ses_f4df4cb3bffetj5bTDp2vZHLRH"
ZEN_REQUEST_ID = "msg_0b20bce72001HhNaOVex180aKz"
PLACEHOLDER_DESC = "Placeholder tool required by upstream gateway. Never invoke this tool."

UPSTREAM_TIMEOUT = 300
HOP_BY_HOP = ("host", "content-length", "transfer-encoding", "connection",
              "accept-encoding", "te", "trailer", "upgrade")


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _tool_names(payload):
    names = set()
    tools = payload.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict):
                fn = tool.get("function")
                if isinstance(fn, dict) and fn.get("name"):
                    names.add(fn["name"])
    return names


def _ensure_zen_tools(payload):
    tools = payload.get("tools")
    if not isinstance(tools, list):
        tools = []
    names = _tool_names(payload)
    added = []
    for name in ("bash", "read"):
        if name not in names:
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": PLACEHOLDER_DESC,
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            })
            added.append(name)
    if tools:
        payload["tools"] = tools
    return added


def _map_chunk(data):
    """Rename reasoning -> reasoning_content, drop reasoning_details."""
    mod = False
    for choice in data.get("choices", []):
        for holder in (choice.get("delta"), choice.get("message")):
            if isinstance(holder, dict):
                if "reasoning" in holder and "reasoning_content" not in holder:
                    holder["reasoning_content"] = holder.pop("reasoning")
                    mod = True
                if "reasoning_details" in holder:
                    del holder["reasoning_details"]
                    mod = True
    return mod


def _grab_providers(data, sink):
    """Collect OpenRouter-style provider failover attempts for logging."""
    try:
        for attempt in (data["choices"][0]["delta"]["provider_metadata"]
                        ["gateway"]["routing"]["modelAttempts"]):
            for pa in attempt.get("providerAttempts", []):
                rec = "%s:%s:%s" % (pa.get("provider"), pa.get("success"),
                                    pa.get("statusCode"))
                if rec not in sink:
                    sink.append(rec)
    except (KeyError, IndexError, TypeError):
        pass


def _aggregate(raw):
    """Collapse an SSE body into one chat.completion JSON object."""
    content, reasoning, tool_calls = [], [], []
    finish = None
    rid = created = model = usage = None
    for line in raw.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data: "):
            continue
        payload = line[6:]
        if payload.strip() == b"[DONE]":
            continue
        try:
            data = json.loads(payload.decode("utf-8"))
        except Exception:
            continue
        _map_chunk(data)
        rid = data.get("id") or rid
        created = data.get("created") or created
        model = data.get("model") or model
        if data.get("usage"):
            usage = data["usage"]
        for choice in data.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                while len(tool_calls) <= idx:
                    tool_calls.append({"id": "", "type": "function",
                                       "function": {"name": "", "arguments": ""}})
                slot = tool_calls[idx]
                if tc.get("id"):
                    slot["id"] = tc["id"]
                if tc.get("type"):
                    slot["type"] = tc["type"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    message: dict = {"role": "assistant", "content": "".join(content)}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if tool_calls:
        message["tool_calls"] = tool_calls
    out = {"id": rid or "", "object": "chat.completion", "created": created or 0,
           "model": model or "",
           "choices": [{"index": 0, "message": message, "finish_reason": finish or "stop"}]}
    if usage:
        out["usage"] = usage
    return json.dumps(out, ensure_ascii=False).encode("utf-8")


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "sse-map-proxy/1.0"
    dump_dir = None

    # ---------------------------------------------------------------- routes
    def _route(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/zen/v1/chat/completions":
            return "zen", ZEN_HOST, self.path
        if path.endswith("/chat/completions"):
            return "cline", CLINE_HOST, CLINE_PATH
        return None, None, None

    def do_GET(self):
        if self.path.split("?", 1)[0].rstrip("/") == "/health":
            body = json.dumps({"ok": True, "routes": ["/v1/chat/completions",
                                                      "/zen/v1/chat/completions"]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()

    # ------------------------------------------------------------------ post
    def do_POST(self):
        route, up_host, up_path = self._route()
        if route is None or up_host is None or up_path is None:
            self.send_error(404)
            return
        t0 = time.time()
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length) if length else b""

        req_model, want_stream, added, n_tools, body = "?", True, [], 0, raw_body
        try:
            payload = json.loads(raw_body)
            req_model = payload.get("model", "?")
            want_stream = payload.get("stream") is not False
            if route == "zen":
                added = _ensure_zen_tools(payload)
                payload["stream"] = True
                n_tools = len(payload.get("tools") or [])
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        except Exception as exc:
            print("%s [%s] BAD-REQ %s" % (_now(), route, exc), file=sys.stderr)

        print("%s [%s] REQ model=%s len=%d stream=%s tools=%d added=%s" % (
            _now(), route, req_model, len(body), want_stream, n_tools,
            ",".join(added) or "-"), file=sys.stderr)
        if self.dump_dir:
            try:
                if not os.path.isdir(self.dump_dir):
                    os.makedirs(self.dump_dir, exist_ok=True)
                with open(os.path.join(self.dump_dir, "last-req.json"), "wb") as fh:
                    fh.write(body)
            except Exception:
                pass

        if route == "zen":
            headers = {
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "User-Agent": ZEN_UA,
                "x-opencode-client": "cli",
                "x-opencode-project": "global",
                "x-opencode-session": ZEN_SESSION,
                "x-opencode-request": ZEN_REQUEST_ID,
                "Accept-Encoding": "identity",
            }
        else:
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in HOP_BY_HOP}
            headers["Accept-Encoding"] = "identity"

        try:
            conn = http.client.HTTPSConnection(up_host, timeout=UPSTREAM_TIMEOUT,
                                               context=ssl.create_default_context())
            conn.request("POST", up_path, body=body, headers=headers)
            resp = conn.getresponse()
        except Exception as exc:
            print("%s [%s] UPSTREAM-CONN %s" % (_now(), route, exc), file=sys.stderr)
            msg = json.dumps({"error": {"message": "upstream error: %s" % exc}}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return

        ctype = resp.getheader("Content-Type", "")

        # ---- non-stream / error / non-SSE: single JSON (or raw) body -------
        if not want_stream or resp.status >= 400 or "text/event-stream" not in ctype:
            raw = resp.read()
            out_ctype = ctype or "application/json"
            if resp.status < 400 and "text/event-stream" in ctype:
                raw = _aggregate(raw)
                out_ctype = "application/json"
            else:
                try:
                    data = json.loads(raw.decode("utf-8"))
                    _map_chunk(data)
                    raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
                except Exception:
                    pass
            self.send_response(resp.status)
            self.send_header("Content-Type", out_ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            try:
                self.wfile.write(raw)
                self.wfile.flush()
            except Exception:
                pass
            conn.close()
            print("%s [%s] done=nonstream status=%d bytes=%d elapsed=%.1fs" % (
                _now(), route, resp.status, len(raw), time.time() - t0), file=sys.stderr)
            return

        # ---- streaming ------------------------------------------------------
        self.send_response(resp.status)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        buf = b""
        sent = 0
        got_done = False
        providers = []
        try:
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.rstrip(b"\r")
                    if line.startswith(b"data: ") and line[6:].strip() != b"[DONE]":
                        try:
                            data = json.loads(line[6:].decode("utf-8"))
                            _grab_providers(data, providers)
                            if _map_chunk(data):
                                line = b"data: " + json.dumps(data, ensure_ascii=False).encode("utf-8")
                        except Exception:
                            pass
                    data_line = line + b"\n"
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(data_line), data_line))
                    self.wfile.flush()
                    sent += len(data_line)
                    if line.startswith(b"data: [DONE]"):
                        got_done = True
            if buf:
                data_line = buf.rstrip(b"\r") + b"\n"
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data_line), data_line))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError) as exc:
            print("%s [%s] DOWNSTREAM-CLOSE %s sent=%d done=%s elapsed=%.1fs" % (
                _now(), route, type(exc).__name__, sent, got_done,
                time.time() - t0), file=sys.stderr)
        except Exception as exc:
            print("%s [%s] ERR %s %s sent=%d" % (
                _now(), route, type(exc).__name__, exc, sent), file=sys.stderr)
        finally:
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except Exception:
                pass
            conn.close()
        print("%s [%s] done=%s sent=%d elapsed=%.1fs providers=%s" % (
            _now(), route, got_done, sent, time.time() - t0,
            ",".join(providers) or "-"), file=sys.stderr)

    # -------------------------------------------------------------- plumbing
    def log_message(self, format, *args):  # noqa: A002 (stdlib signature)
        pass

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError):
            self.close_connection = True


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(description="sse-map-proxy")
    parser.add_argument("--host", action="store_true",
                        help="bind 0.0.0.0 instead of 127.0.0.1")
    parser.add_argument("--port", type=int, default=3457)
    parser.add_argument("--dump-dir", default=None,
                        help="write the last upstream request body to this directory")
    args = parser.parse_args()

    Handler.dump_dir = args.dump_dir
    bound = "0.0.0.0" if args.host else "127.0.0.1"
    httpd = Server((bound, args.port), Handler)
    print("sse-map-proxy on %s:%d  (cline: /v1/... -> %s, zen: /zen/v1/... -> %s)" % (
        bound, args.port, CLINE_HOST, ZEN_HOST))
    httpd.serve_forever()


if __name__ == "__main__":
    main()
