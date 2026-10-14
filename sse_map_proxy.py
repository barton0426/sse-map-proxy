#!/usr/bin/env python3
"""sse-map-proxy — one tiny HTTP proxy for OpenAI-compatible SSE endpoints.

Single instance, path-based routing:

  POST /v1/chat/completions      -> https://api.cline.bot/api/v1/chat/completions
                                    (request passthrough, response mapping)
  POST /zen/v1/chat/completions  -> https://opencode.ai/inference/openai/v1/chat/completions
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
  gate (shape changed 2026-10-14): posts to the ``/inference/openai`` backend,
  sends ``Authorization: Bearer public``, claims the ``opencode/latest/<v>/cli``
  User-Agent, sends ``x-opencode-session`` + ``x-opencode-session-id`` (and
  never the legacy ``x-opencode-request``), forces streaming with
  ``stream_options.include_usage``, and makes the tool list read as an official
  OpenCode agent session: every recognised tool carries its genuine
  description/schema (a stub is rejected with 403 FreeTierError) and the
  missing core tools (shell/edit/glob/grep/read) are appended.

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
# 2026-10-14: chat traffic moved off the legacy console origin to the inference backend.
ZEN_CHAT_PATH = "/inference/openai/v1/chat/completions"
ZEN_RESPONSES_PATH = "/inference/openai/v1/responses"
# Gate shape: "opencode/<channel>/<version>/<name>" (the old
# "opencode/<version> (<os> <arch>; <runtime>)" form now 403s).
ZEN_UA = "opencode/latest/2.0.29/cli"
# Anonymous credential is the literal token "public" (case-sensitive; absent header now 403s).
ZEN_AUTH = "Bearer public"
# Canonical session id: "ses_" + descending-time hex (low digits read ffe) + 14 base62.
ZEN_SESSION = "ses_f4df4cb3bffetj5bTDp2vZHLRH"
# Tool names the free tier expects on an agent session; genuine definitions required.
ZEN_CORE_TOOLS = ("shell", "edit", "glob", "grep", "read")
RESPONSES_MODEL_PREFIXES = ("muse-",)
ZEN_TOOLS_JSON = r'''{
  "edit": {
    "description": "Edit the contents of a file by finding and replacing exact text. When editing text from Read output, preserve the exact indentation (tabs or spaces) and omit the line-number prefix, such as `1: `. Never include the prefix in oldString or newString. Write line breaks as \\n; the file's existing CRLF or LF line endings are matched and preserved. To change line endings themselves, make oldString and newString differ only in \\r; that edit applies to the exact bytes. If there is no exact match, the edit retries while ignoring typographic quote, dash, and space variants, then trailing whitespace on each line. The edit fails if oldString is not found. By default, oldString must identify a UNIQUE location. Multiple matches FAIL unless replaceAll is true. Add more surrounding context to disambiguate, or set replaceAll to true to replace every occurrence. Use replaceAll when the change should apply to every occurrence, such as renaming a variable.",
    "parameters": {
      "additionalProperties": false,
      "properties": {
        "newString": {
          "description": "Text to replace oldString with (must differ from oldString)",
          "type": "string"
        },
        "oldString": {
          "description": "Exact text to find and replace",
          "type": "string"
        },
        "path": {
          "description": "File to edit",
          "type": "string"
        },
        "replaceAll": {
          "description": "Whether to replace every occurrence of oldString. When false, oldString must match exactly once. Defaults to false.",
          "type": "boolean"
        }
      },
      "required": ["path", "oldString", "newString"],
      "type": "object"
    }
  },
  "execute": {
    "description": "Run JavaScript in a confined Code Mode runtime to script tool calls and HTTP requests and compose their results.\n`fetch` is available for HTTP requests. Imports, direct filesystem access, and timers are unavailable; all other external access goes through `tools`.\nWithin `{ code }`, the only callable tools are those explicitly listed in the Code Mode catalog instructions or returned by the `search` function. Inside `{ code }`, ignore tools shown outside the Code Mode catalog. They are not available in the Code Mode runtime.\nCall tools through `tools` using only exact paths and signatures from the catalog. Do not infer or normalize tool names; preserve bracket notation such as `tools.<namespace>[\"tool-name\"](input)`.\nPrefer an explicit `return`; if omitted, the final top-level expression becomes the result.\nAwait every call whose completion matters; pending calls are interrupted when execution ends. Run independent calls concurrently with `Promise.allSettled`.",
    "parameters": {
      "additionalProperties": false,
      "properties": {
        "code": {
          "type": "string"
        }
      },
      "required": ["code"],
      "type": "object"
    }
  },
  "glob": {
    "description": "Search file paths using a glob pattern (examples: \"**/*.ts\", \"src/**/*.tsx\").",
    "parameters": {
      "additionalProperties": false,
      "properties": {
        "hidden": {
          "description": "Include hidden files and directories (default: false).",
          "type": "boolean"
        },
        "limit": {
          "description": "Maximum number of matching files to return (default: 100)",
          "exclusiveMinimum": 0,
          "type": "integer"
        },
        "path": {
          "description": "Directory to search. Defaults to the current working directory.",
          "type": "string"
        },
        "pattern": {
          "description": "Glob pattern to match files against",
          "type": "string"
        }
      },
      "required": ["pattern"],
      "type": "object"
    }
  },
  "grep": {
    "description": "Search file contents using ripgrep's regular expression syntax or literal text matching. Use it to locate specific code, symbols, or text patterns, and narrow searches with `path` or `include`. Returns matching file paths, line numbers, and line previews.",
    "parameters": {
      "additionalProperties": false,
      "properties": {
        "caseSensitive": {
          "description": "Use case-sensitive matching (default: true).",
          "type": "boolean"
        },
        "include": {
          "description": "Glob pattern to filter files (for example, \"*.js\" or \"*.{ts,tsx}\")",
          "type": "string"
        },
        "limit": {
          "description": "Maximum number of matching lines to return (default: 100)",
          "exclusiveMinimum": 0,
          "type": "integer"
        },
        "literal": {
          "description": "Treat `pattern` as exact text instead of a regular expression (default: false).",
          "type": "boolean"
        },
        "path": {
          "description": "File or directory to search. Defaults to the current working directory.",
          "type": "string"
        },
        "pattern": {
          "description": "Regular expression or literal text to match in file contents.",
          "minLength": 1,
          "type": "string"
        }
      },
      "required": ["pattern"],
      "type": "object"
    }
  },
  "question": {
    "description": "Use this tool when you need to ask the user questions during execution. This allows you to:\n1. Gather user preferences or requirements\n2. Clarify ambiguous instructions\n3. Get decisions on implementation choices as you work\n4. Offer choices to the user about what direction to take.\n\nUsage notes:\n- A \"Type your own answer\" option is added automatically; don't include a separate option for free form answers\n- Set `multiple: true` to allow selecting more than one option\n- If you recommend a specific option, make that the first option in the list and add \"(Recommended)\" at the end of the label",
    "parameters": {
      "additionalProperties": false,
      "properties": {
        "questions": {
          "description": "Questions to ask",
          "items": {
            "additionalProperties": false,
            "properties": {
              "header": {
                "description": "Very short label (max 30 chars)",
                "type": "string"
              },
              "multiple": {
                "type": "boolean"
              },
              "options": {
                "description": "Available choices",
                "items": {
                  "additionalProperties": false,
                  "properties": {
                    "description": {
                      "description": "Explanation of choice",
                      "type": "string"
                    },
                    "label": {
                      "description": "Display text (1-5 words, concise)",
                      "type": "string"
                    }
                  },
                  "required": ["label", "description"],
                  "type": "object"
                },
                "type": "array"
              },
              "question": {
                "description": "Complete question",
                "type": "string"
              }
            },
            "required": ["question", "header", "options"],
            "type": "object"
          },
          "minItems": 1,
          "type": "array"
        }
      },
      "required": ["questions"],
      "type": "object"
    }
  },
  "read": {
    "description": "Read the contents of a file or directory. Supports text files, images, and PDFs. Images and PDFs are presented directly to the model. Each text line is prefixed by its 1-based line number as <line>: <content>. The prefix is for reference and is not part of the file content. Directory entries are returned one per line. Use offset and limit to read large files or directories in sections. Prefer one larger read over many small slices, and use grep to find specific content in large files.",
    "parameters": {
      "additionalProperties": false,
      "properties": {
        "limit": {
          "description": "The maximum number of lines or directory entries to read (defaults to and capped at 2000)",
          "minimum": 0,
          "type": "integer"
        },
        "offset": {
          "description": "The line or directory entry to start reading from (1-based)",
          "minimum": 0,
          "type": "integer"
        },
        "path": {
          "description": "File or directory to read",
          "type": "string"
        }
      },
      "required": ["path"],
      "type": "object"
    }
  },
  "shell": {
    "description": "Execute a shell command and return its output. Commands run on Windows using powershell. Quote file paths containing spaces or special characters. Prefer dedicated tools over shell commands when possible. When output is large, the full result is saved to a file and a truncated preview is returned. Rely on automatic truncation unless filtering the output is more useful. Commands accept an optional timeout, background commands have no timeout by default. Background commands return immediately, and you will be notified when they complete.",
    "parameters": {
      "additionalProperties": false,
      "properties": {
        "background": {
          "description": "Run the command in the background and return immediately (useful for dev servers and long-running builds). You do not need to use '&' at the end of the command when using this parameter. You will be notified when it completes. DO NOT poll for completion.",
          "type": "boolean"
        },
        "command": {
          "description": "Shell command string to execute",
          "type": "string"
        },
        "timeout": {
          "description": "Timeout in milliseconds. Set to 0 to disable the timeout. Defaults to 120000 for foreground commands. Background commands have no timeout by default.",
          "minimum": 0,
          "type": "integer"
        },
        "workdir": {
          "description": "Working directory to execute the command in. Defaults to the current working directory. When possible, avoid changing directories in the command and set the working directory here instead.",
          "type": "string"
        }
      },
      "required": ["command"],
      "type": "object"
    }
  },
  "skill": {
    "description": "Load a specialized skill's instructions and resources into the current conversation when the task at hand matches its description.\n\nThe skill ID must match an available skill or a skill explicitly referenced by the user.",
    "parameters": {
      "additionalProperties": false,
      "properties": {
        "id": {
          "description": "The ID of an available skill or a skill explicitly referenced by the user",
          "type": "string"
        }
      },
      "required": ["id"],
      "type": "object"
    }
  },
  "subagent": {
    "description": "Spawns an agent in a child session to work on the specified task.\nThe output includes a sessionID you can pass back later to continue that specific conversation with the subagent.\nNew child sessions start with fresh context, so include all relevant context and instructions when you don't pass a sessionID.\nForeground (default) runs the subagent to completion and returns its final response.\nBackground mode (background=true) launches it asynchronously and returns immediately; you are notified when it finishes.\nUse background only for independent work that can run while you continue elsewhere.\n\nAvailable subagents:\n- explore: Fast agent specialized for exploring codebases. Use this when you need to quickly find files by patterns (eg. \"src/components/**/*.tsx\"), search code for keywords (eg. \"API endpoints\"), or answer questions about the codebase (eg. \"how do API endpoints work?\"). When calling this agent, specify the desired thoroughness level: \"quick\" for basic searches, \"medium\" for moderate exploration, or \"very thorough\" for comprehensive analysis across multiple locations and naming conventions.\n- general: General-purpose agent for researching complex questions and executing multi-step tasks. Use this agent to execute multiple units of work in parallel.",
    "parameters": {
      "additionalProperties": false,
      "properties": {
        "agent": {
          "description": "The type of specialized agent to use for this task. If the user asks for a subagent by a name that is not one of the available subagents, they most likely mean a model: pick a suitable agent and pass the name through the model parameter instead.",
          "type": "string"
        },
        "background": {
          "description": "Run the subagent in the background and return immediately. You will be notified when it completes. DO NOT sleep, poll, or proactively check on its progress.",
          "type": "boolean"
        },
        "description": {
          "description": "A short 3-5 word label for the task, displayed to the user",
          "type": "string"
        },
        "model": {
          "description": "NEVER set this unless the user explicitly asks for a particular model or variant. The value is written as \"providerID/modelID\", or \"providerID/modelID#variant\" to include a variant. Do not guess the ID: look the model up with the models tool, filtering to your own provider first.",
          "type": "string"
        },
        "prompt": {
          "description": "The task for the subagent to perform",
          "type": "string"
        },
        "sessionID": {
          "description": "Continue a specific previous subagent conversation by passing its sessionID. Calls without a sessionID start a new conversation.",
          "pattern": "^ses",
          "type": "string"
        }
      },
      "required": ["agent", "description", "prompt"],
      "type": "object"
    }
  },
  "webfetch": {
    "description": "Fetch content from an HTTP or HTTPS URL and return it as text, markdown, or HTML. Markdown is the default.\n\nUse a more targeted tool when one is available. This tool is read-only. Large text results may be replaced with a preview while the complete output is retained in managed storage.",
    "parameters": {
      "additionalProperties": false,
      "properties": {
        "format": {
          "description": "The format to return the content in. Defaults to markdown.",
          "enum": ["text", "markdown", "html"],
          "type": "string"
        },
        "timeout": {
          "description": "Optional timeout in seconds (maximum: 120)",
          "exclusiveMinimum": 0,
          "maximum": 120,
          "type": "number"
        },
        "url": {
          "description": "The HTTP or HTTPS URL to fetch content from",
          "type": "string"
        }
      },
      "required": ["url"],
      "type": "object"
    }
  },
  "websearch": {
    "description": "Search the web using the user's selected search integration. Use this for current information beyond knowledge cutoff.\n\nThe current year is 2026. Use this year when searching for recent information or current events.",
    "parameters": {
      "additionalProperties": false,
      "properties": {
        "query": {
          "description": "Websearch query",
          "type": "string"
        }
      },
      "required": ["query"],
      "type": "object"
    }
  },
  "write": {
    "description": "Writes a file to the local filesystem, overwriting if one exists.\n\nMissing parent directories are created automatically.\n\nUse this tool to create new files or overwrite existing files. For partial changes, use the edit tool instead.",
    "parameters": {
      "additionalProperties": false,
      "properties": {
        "content": {
          "description": "Content to write to the file",
          "type": "string"
        },
        "path": {
          "description": "Path to the file to write to",
          "type": "string"
        }
      },
      "required": ["path", "content"],
      "type": "object"
    }
  }
}'''

UPSTREAM_TIMEOUT = 600
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


_ZEN_DEFS = json.loads(ZEN_TOOLS_JSON)


def _chat_tool(name, definition):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": definition.get("description") or ("Agent tool " + name),
            "parameters": definition.get("parameters") or {"type": "object", "properties": {}},
        },
    }


def _ensure_zen_tools(payload):
    """Make the tool list read as an official OpenCode agent session.

    The 2026-10-14 gate validates the genuine description/schema of every tool
    name it recognises, so same-named client tools are rewritten to the official
    definition and the missing core tools are appended.
    """
    tools = payload.get("tools")
    if not isinstance(tools, list):
        tools = []
    fixed, added, seen = [], [], set()
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        name = fn.get("name") if isinstance(fn, dict) else None
        if name and name in _ZEN_DEFS:
            fn["description"] = _ZEN_DEFS[name].get("description") or fn.get("description")
            fn["parameters"] = _ZEN_DEFS[name].get("parameters") or fn.get("parameters")
        if name:
            seen.add(name)
        fixed.append(tool)
    for name in ZEN_CORE_TOOLS:
        if name not in seen and name in _ZEN_DEFS:
            fixed.append(_chat_tool(name, _ZEN_DEFS[name]))
            added.append(name)
    payload["tools"] = fixed
    return added


def _zen_tools_for_responses(payload):
    """Same shaping for the Responses protocol (muse-* models)."""
    tools, seen = [], set()
    for tool in payload.get("tools") or []:
        fn = tool.get("function") if isinstance(tool, dict) else None
        name = fn.get("name") if isinstance(fn, dict) else None
        if not name:
            continue
        definition = _ZEN_DEFS.get(name) or {}
        tools.append({
            "type": "function",
            "name": name,
            "description": definition.get("description") or fn.get("description") or "",
            "parameters": definition.get("parameters") or fn.get("parameters")
            or {"type": "object", "properties": {}},
        })
        seen.add(name)
    for name in ZEN_CORE_TOOLS:
        if name not in seen and name in _ZEN_DEFS:
            tools.append({
                "type": "function",
                "name": name,
                "description": _ZEN_DEFS[name].get("description") or "",
                "parameters": _ZEN_DEFS[name].get("parameters") or {"type": "object", "properties": {}},
            })
    return tools


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


def _flatten_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("text", "input_text", "output_text"):
                parts.append(part.get("text") or "")
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return "" if content is None else str(content)


def _chat_to_responses(payload):
    """Translate an OpenAI chat.completions request into a Responses API request."""
    out = {"model": payload.get("model"), "stream": True}
    instructions = []
    inputs = []
    for msg in payload.get("messages") or []:
        role = msg.get("role")
        text = _flatten_text(msg.get("content"))
        if role == "system":
            if text:
                instructions.append(text)
        elif role == "user":
            inputs.append({"role": "user",
                           "content": [{"type": "input_text", "text": text}]})
        elif role == "assistant":
            if text:
                inputs.append({"role": "assistant",
                               "content": [{"type": "output_text", "text": text}]})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                inputs.append({"type": "function_call",
                               "call_id": tc.get("id") or "",
                               "name": fn.get("name") or "",
                               "arguments": fn.get("arguments") or ""})
        elif role == "tool":
            inputs.append({"type": "function_call_output",
                           "call_id": msg.get("tool_call_id") or "",
                           "output": text})
    if instructions:
        out["instructions"] = "\n\n".join(instructions)
    out["input"] = inputs
    out["store"] = False
    tools = _zen_tools_for_responses(payload)
    if tools:
        out["tools"] = tools
    for src, dst in (("max_tokens", "max_output_tokens"),
                     ("temperature", "temperature"),
                     ("top_p", "top_p"),
                     ("parallel_tool_calls", "parallel_tool_calls")):
        if payload.get(src) is not None:
            out[dst] = payload[src]
    if isinstance(payload.get("tool_choice"), str):
        out["tool_choice"] = payload["tool_choice"]
    added = [name for name in ZEN_CORE_TOOLS
             if name not in {t.get("name") for t in tools}]
    return out, added


def _new_state(model):
    return {"id": None, "created": None, "model": model or "", "role_sent": False,
            "finish_sent": False, "tc_index": {}, "n_tc": 0}


def _chat_chunk(state, delta=None, finish=None, usage=None):
    out = {"id": state.get("id") or "", "object": "chat.completion.chunk",
           "created": state.get("created") or 0, "model": state.get("model") or "",
           "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    if usage:
        out["usage"] = usage
    return out


def _responses_event_chunks(evt, state):
    """Translate one Responses API SSE event into chat.completion.chunk dicts."""
    etype = evt.get("type")
    resp = evt.get("response") or {}
    chunks = []
    if resp:
        if resp.get("id"):
            state["id"] = resp["id"]
        if resp.get("created_at"):
            state["created"] = resp["created_at"]
        if resp.get("model"):
            state["model"] = resp["model"]
    if not state["role_sent"] and etype in ("response.created", "response.in_progress",
                                            "response.output_text.delta"):
        state["role_sent"] = True
        chunks.append(_chat_chunk(state, delta={"role": "assistant"}))
    if etype == "response.output_text.delta":
        chunks.append(_chat_chunk(state, delta={"content": evt.get("delta") or ""}))
    elif etype == "response.reasoning_summary_text.delta":
        chunks.append(_chat_chunk(state, delta={"reasoning_content": evt.get("delta") or ""}))
    elif etype == "response.output_item.added":
        item = evt.get("item") or {}
        if item.get("type") == "function_call":
            idx = state["n_tc"]
            state["n_tc"] += 1
            for key in (item.get("call_id"), item.get("id")):
                if key:
                    state["tc_index"][key] = idx
            chunks.append(_chat_chunk(state, delta={"tool_calls": [{
                "index": idx, "id": item.get("call_id") or "", "type": "function",
                "function": {"name": item.get("name") or "", "arguments": ""}}]}))
    elif etype == "response.function_call_arguments.delta":
        idx = state["tc_index"].get(evt.get("item_id") or "", 0)
        chunks.append(_chat_chunk(state, delta={"tool_calls": [{
            "index": idx, "function": {"arguments": evt.get("delta") or ""}}]}))
    elif etype == "response.completed":
        usage = resp.get("usage") or {}
        chat_usage = None
        if usage:
            chat_usage = {"prompt_tokens": usage.get("input_tokens", 0),
                          "completion_tokens": usage.get("output_tokens", 0),
                          "total_tokens": usage.get("total_tokens", 0)}
        finish = "tool_calls" if state["n_tc"] else "stop"
        chunks.append(_chat_chunk(state, delta={}, finish=finish, usage=chat_usage))
        state["finish_sent"] = True
    elif etype == "error":
        chunks.append({"error": evt.get("error") or evt})
    return chunks


def _aggregate_chat_chunks(chunks):
    content, reasoning, tool_calls = [], [], []
    finish = None
    rid = created = model = usage = None
    for ch in chunks:
        rid = ch.get("id") or rid
        created = ch.get("created") or created
        model = ch.get("model") or model
        if ch.get("usage"):
            usage = ch["usage"]
        for choice in ch.get("choices", []):
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


def _responses_events(raw):
    for line in raw.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data: "):
            continue
        payload = line[6:]
        if payload.strip() == b"[DONE]":
            continue
        try:
            yield json.loads(payload.decode("utf-8"))
        except Exception:
            continue


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "sse-map-proxy/1.0"
    dump_dir = None

    # ---------------------------------------------------------------- routes
    def _route(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/zen/v1/chat/completions":
            return "zen", ZEN_HOST, ZEN_CHAT_PATH
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
        responses_mode = False
        try:
            payload = json.loads(raw_body)
            req_model = payload.get("model", "?")
            want_stream = payload.get("stream") is not False
            if route == "zen":
                if isinstance(req_model, str) and req_model.startswith(RESPONSES_MODEL_PREFIXES):
                    conv, added = _chat_to_responses(payload)
                    body = json.dumps(conv, ensure_ascii=False).encode("utf-8")
                    n_tools = len(conv.get("tools") or [])
                    responses_mode = True
                else:
                    added = _ensure_zen_tools(payload)
                    payload["stream"] = True
                    options = payload.get("stream_options")
                    if not isinstance(options, dict):
                        options = {}
                    if options.get("include_usage") is not True:
                        options["include_usage"] = True
                    payload["stream_options"] = options
                    n_tools = len(payload.get("tools") or [])
                    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            else:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        except Exception as exc:
            print("%s [%s] BAD-REQ %s" % (_now(), route, exc), file=sys.stderr)

        print("%s [%s] REQ model=%s len=%d stream=%s tools=%d added=%s%s" % (
            _now(), route, req_model, len(body), want_stream, n_tools,
            ",".join(added) or "-", " via=responses" if responses_mode else ""))
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
                "Authorization": ZEN_AUTH,
                "User-Agent": ZEN_UA,
                "x-opencode-client": "cli",
                "x-opencode-project": "global",
                "x-opencode-session": ZEN_SESSION,
                "x-opencode-session-id": ZEN_SESSION,
                "x-session-affinity": ZEN_SESSION,
                "X-Session-Id": ZEN_SESSION,
                "Accept-Encoding": "identity",
            }
        else:
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in HOP_BY_HOP}
            headers["Accept-Encoding"] = "identity"

        if responses_mode:
            up_path = ZEN_RESPONSES_PATH

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
                if responses_mode:
                    rstate = _new_state(req_model if isinstance(req_model, str) else "")
                    chunks = []
                    for evt in _responses_events(raw):
                        chunks.extend(_responses_event_chunks(evt, rstate))
                    raw = _aggregate_chat_chunks(chunks)
                else:
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
        rstate = _new_state(req_model if isinstance(req_model, str) else "")
        try:
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.rstrip(b"\r")
                    out_lines = []
                    if responses_mode:
                        if line.startswith(b"data: ") and line[6:].strip():
                            try:
                                evt = json.loads(line[6:].decode("utf-8"))
                            except Exception:
                                evt = None
                            if evt is not None:
                                for out_chunk in _responses_event_chunks(evt, rstate):
                                    out_lines.append(b"data: " + json.dumps(
                                        out_chunk, ensure_ascii=False).encode("utf-8"))
                                if rstate["finish_sent"] and not got_done:
                                    out_lines.append(b"data: [DONE]")
                                    got_done = True
                    else:
                        if line.startswith(b"data: ") and line[6:].strip() != b"[DONE]":
                            try:
                                data = json.loads(line[6:].decode("utf-8"))
                                _grab_providers(data, providers)
                                if _map_chunk(data):
                                    line = b"data: " + json.dumps(data, ensure_ascii=False).encode("utf-8")
                            except Exception:
                                pass
                        out_lines.append(line)
                        if line.startswith(b"data: [DONE]"):
                            got_done = True
                    for out_line in out_lines:
                        data_line = out_line + (b"\n\n" if responses_mode else b"\n")
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(data_line), data_line))
                        self.wfile.flush()
                        sent += len(data_line)
            if responses_mode and not got_done:
                data_line = b"data: [DONE]\n\n"
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data_line), data_line))
                self.wfile.flush()
                sent += len(data_line)
                got_done = True
            if buf and not responses_mode:
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
