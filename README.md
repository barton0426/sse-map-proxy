# sse-map-proxy

A tiny single-file HTTP proxy for OpenAI-compatible SSE endpoints. One process,
two routes:

| Route | Upstream | Job |
|---|---|---|
| `POST /v1/chat/completions` | `api.cline.bot/api/v1/chat/completions` | request passthrough + response mapping |
| `POST /zen/v1/chat/completions` | `opencode.ai/inference/openai/v1/chat/completions` | request shaping + response mapping |
| `GET /health` | – | liveness probe |

Stdlib only — no dependencies. Python ≥ 3.8.

## Features

- **SSE streaming** without buffering the whole body; line-buffered transforms.
- **Reasoning mapping**: `delta.reasoning` → `delta.reasoning_content`,
  `reasoning_details` removed, so clients that only understand
  `reasoning_content` (e.g. DeepSeek-style UIs) can display the chain of thought.
- **Non-stream aggregation**: when the client sends `"stream": false`, the SSE
  stream is collapsed into a single JSON `chat.completion` object.
- **Zen free-tier shaping**: the Opencode zen free tier rejects requests that do
  not look like they come from the OpenCode CLI. For the `/zen/...` route the
  proxy (updated for the 2026-10-14 gate):
  - posts to the inference backend (`/inference/openai/v1/*`) and sends
    `Authorization: Bearer public` (required since 2026-10-14),
  - sets the `opencode/<channel>/<version>/<name>` User-Agent shape
    (e.g. `opencode/latest/2.0.29/cli`) and both session headers
    (`x-opencode-session` + `x-opencode-session-id`, same descending-encoded id);
    the legacy `x-opencode-request` header is no longer sent (it now 403s),
  - forces `"stream": true` plus `stream_options.include_usage`,
  - makes the tool list read as an agent session: the genuine OpenCode core tool
    definitions (`shell`/`edit`/`glob`/`grep`/`read`) are embedded and injected,
    and same-named client tools are rewritten to the official definition — the
    gate validates description + schema byte-for-byte, so stubs are rejected.
- **Responses bridge**: models whose name starts with `muse-` (e.g.
  `muse-spark-1.3-contributor-free`) are only served on the Responses API.
  For those, the proxy translates the incoming `chat.completions` request into
  a `/inference/openai/v1/responses` request (messages → `instructions` +
  `input` items, tools → Responses-format tools with the same genuine core
  definitions) and
  translates the SSE events back into `chat.completion.chunk` deltas —
  including text, reasoning summaries and `function_call` → `tool_calls`.
  Non-stream clients get the aggregated JSON. Chat fields
  `max_tokens` / `temperature` / `top_p` / `tool_choice` / `parallel_tool_calls`
  are mapped onto the Responses request and `store` is set to `false`.
- Proper HTTP/1.1 chunked framing, keep-alive safe, thread-per-connection.

## Quick start

```bash
# loopback only (default)
python3 sse_map_proxy.py --port 3457

# bind several addresses at once (repeatable) — recommended for host + container use:
#   docker bridge + loopback only, never the public interface
python3 sse_map_proxy.py --bind 172.18.0.1 --bind 127.0.0.1 --port 3457

# legacy: bind every interface (0.0.0.0). Only for trusted networks.
python3 sse_map_proxy.py --host --port 3457
```

Point any OpenAI-compatible client at:

- cline route: `http://127.0.0.1:3457/v1`
- zen route:   `http://127.0.0.1:3457/zen/v1`

> ⚠️ This proxy has **no authentication** — it will happily relay requests with
> the credentials it holds. Never expose it on a public interface: bind the
> docker bridge IP (so CPA containers can reach it) plus loopback, not `0.0.0.0`.

### systemd

```ini
[Unit]
Description=sse-map-proxy
After=network-online.target

[Service]
ExecStart=/usr/bin/python3 /root/sse_map_proxy.py --bind 172.18.0.1 --bind 127.0.0.1 --port 3457
Restart=always
RestartSec=3
StandardOutput=append:/var/log/sse-map-proxy.log
StandardError=append:/var/log/sse-map-proxy.log

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload && systemctl enable --now sse-map-proxy
curl -s http://127.0.0.1:3457/health
ss -ltnp | grep 3457      # must NOT show 0.0.0.0
```

## Notes on the zen free tier

The rules below were verified empirically (2026-10-15) against
`https://opencode.ai/inference/openai/v1/chat/completions`:

1. `Authorization: Bearer public` is **required** (an absent header used to pass;
   the check is case-sensitive — `Bearer PUBLIC` returns 401).
2. `User-Agent` must use the `opencode/<channel>/<version>/<name>` shape
   (e.g. `opencode/latest/2.0.29/cli`); the older
   `opencode/<version> (<os> <arch>; <runtime>)` form is rejected.
3. Both `x-opencode-session` and `x-opencode-session-id` must carry the same
   descending-encoded session id (`ses_` + 12 hex ending in `ffe` + 14 base62).
4. The legacy `x-opencode-request` header must **not** be sent (403), and
   `x-opencode-project` must not be a canonical `prj_*` id (`global` or a bare
   40-hex value is fine).
5. Body must have `"stream": true` (plus `stream_options.include_usage`).
6. `tools` must contain the genuine core tool definitions
   (`shell`/`edit`/`glob`/`grep`/`read`); the gate validates the description and
   JSON schema of every recognised tool, so stubs and renamed tools 403.

The old `/zen` origin now only serves the anonymous model listing
(`GET https://opencode.ai/zen/v1/models` with `Bearer public`); the inference
backend has no `/v1/models` (404). This is an anti-abuse gate and may change at
any time — if it does, adjust the `ZEN_*` constants and `_ensure_zen_tools()` /
`_zen_tools_for_responses()` in the script. Reference implementation:
`jasonxu114514/opencode2api` (commit `0204ca1f`, 2026-10-14).

### Caveat

The injected `shell` / `edit` / `glob` / `grep` / `read` tools are the genuine
OpenCode definitions, not stubs. A model may still decide to *emit* a tool call
for them; most non-OpenCode clients have no executor for those names. Use the
zen route for chat-style traffic.

## Options

- `--port N` listen port (default `3457`).
- `--bind ADDR` bind address, repeatable (default loopback only). Use it to bind
  the docker bridge **and** loopback, e.g. `--bind 172.18.0.1 --bind 127.0.0.1`.
- `--host` legacy switch, equivalent to `--bind 0.0.0.0`.

## Debugging

- `--dump-dir DIR` writes the last upstream request body to
  `DIR/last-req.json` (useful to verify what the proxy actually sent).
- Every request is logged to stderr with a timestamp, route tag, model, stream
  flag, tool counts and byte counters.

## License

MIT — see [LICENSE](LICENSE).
