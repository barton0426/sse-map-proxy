# sse-map-proxy

A tiny single-file HTTP proxy for OpenAI-compatible SSE endpoints. One process,
two routes:

| Route | Upstream | Job |
|---|---|---|
| `POST /v1/chat/completions` | `api.cline.bot/api/v1/chat/completions` | request passthrough + response mapping |
| `POST /zen/v1/chat/completions` | `opencode.ai/zen/v1/chat/completions` | request shaping + response mapping |
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
  proxy:
  - sets `User-Agent: opencode/<semver>` and a valid `x-opencode-session`
    (`ses_` + 12 lowercase hex + 14 base62 chars),
  - merges minimal placeholder `bash` / `read` tools into the request's `tools`
    array (the gate only checks the names and count, not the schemas),
  - forces `"stream": true` upstream.
- Proper HTTP/1.1 chunked framing, keep-alive safe, thread-per-connection.

## Quick start

```bash
python3 sse_map_proxy.py --host --port 3457
```

Point any OpenAI-compatible client at:

- cline route: `http://127.0.0.1:3457/v1`
- zen route:   `http://127.0.0.1:3457/zen/v1`

### systemd

```ini
[Unit]
Description=sse-map-proxy
After=network-online.target

[Service]
ExecStart=/usr/bin/python3 /root/sse_map_proxy.py --host --port 3457
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
```

## Notes on the zen free tier

The rules below were verified empirically (2026-09) against
`https://opencode.ai/zen/v1/chat/completions`:

1. `User-Agent` must contain `opencode/<semver>` (≥ 1.18.0).
2. `x-opencode-session` must be a syntactically valid session id
   (`ses_` + 12 lowercase hex + 14 `[0-9A-Za-z]`), no freshness check observed.
3. Body must have `"stream": true`.
4. `tools` must contain **both** `bash` and `read` (schemas/descriptions are not
   validated; extra tools are fine).

`Authorization` is not required for the free models. This is an anti-abuse gate
and may change at any time — if it does, adjust `ZEN_*` constants and
`_ensure_zen_tools()` in the script.

### Caveat

The injected `bash` / `read` tools are placeholders. A model may still decide to
*emit* a tool call for them; most non-OpenCode clients have no executor for
those names. Use the zen route for chat-style traffic.

## Debugging

- `--dump-dir DIR` writes the last upstream request body to
  `DIR/last-req.json` (useful to verify what the proxy actually sent).
- Every request is logged to stderr with a timestamp, route tag, model, stream
  flag, tool counts and byte counters.

## License

MIT — see [LICENSE](LICENSE).
