# selfhost_server

A self-hosted agent backend for Warp. It serves the slice of Warp's server API
that Agent Mode needs and translates between Warp's multi-agent wire protocol
and an **OpenAI- or Anthropic-compatible LLM endpoint** — so agent calls go to
a server you operate instead of Warp's cloud. Run it on `127.0.0.1` with a
local LLM (e.g. Ollama) and nothing leaves your machine.

## What it serves

| Endpoint | Purpose |
| --- | --- |
| `POST /ai/multi-agent` | The agent loop: translates Warp's protobuf `Request` into LLM calls and streams `ResponseEvent`s back over SSE. |
| `POST /ai/passive-suggestions` | No-ops gracefully (finishes without suggestions). |
| `POST /graphql/v2` | Answers the operations self-hosting needs: `GetUser` (login), `GetFeatureModelChoices`/`FreeAvailableModels` (model catalog), `GetWorkspacesMetadataForUser`, `GetRequestLimitInfo`/`GetAICreditAvailability` (unlimited local usage), `ListAIConversationMetadata` (empty history), `GetUpdatedCloudObjects`/`BulkCreateObjects`/`CreateGenericStringObject` (Drive sync acknowledged, nothing persisted), `GetAvailableHarnesses`, `GetUserSettings`, `GetCloudEnvironmentsQuery`, plus `/api/v1/agent/connected-self-hosted-workers`. Every other operation returns a GraphQL error so unsupported features degrade locally. |
| `POST /ai/transcribe` | Voice transcription: decodes the client's base64 WAV upload and forwards it to an OpenAI-compatible speech-to-text endpoint. Opt-in; see the flags below. |
| `POST /ai/relevant_files` | Ranks the client's repo outline against the agent's `search_codebase` queries (lexical relevance heuristic). |
| `GET /healthz` | Liveness. |

Tools supported end to end: `run_shell_command`, `read_files`, `grep`,
`search_codebase`, `file_glob`, `apply_file_diffs`,
`write_to_long_running_command`, `read_command_output`, `ask_user_question`,
`suggest_prompt`, `call_mcp_tool`, `read_mcp_resource`. All of them are
executed by the Warp client on your machine; the server only shuttles context
between the client and the LLM. With `--web-search-url` configured, the server
additionally runs a `web_search` tool itself (against a SearXNG-compatible
endpoint) and feeds results back into the conversation without any client
round-trip.

## Context-window management

Long conversations are compacted automatically: when the estimated token count
approaches the budget, the server summarizes the older prefix (one extra LLM
call), moves those messages into a summary subtask via a `MoveMessagesToNewTask`
action the client renders natively, and continues from the compacted view. The
budget comes from the client's per-model context-window setting when set, else
from `--context-window-tokens` (default `0` = auto, 96k). Provider errors that
indicate an over-long prompt surface to the client as a context-window
condition instead of a generic failure.

## Vision and multi-model serving

Screenshots and attached images flow through to the LLM as provider-native
image blocks (data URLs for OpenAI, base64 source blocks for Anthropic), and
the served model catalog advertises vision support. `--llm-models a,b,c`
exposes several models in the client's model picker; the selected ID is passed
through to your LLM endpoint. Each turn reports the serving model to the
client via a `ModelUsed` message. Anthropic requests mark the system prompt as
prompt-cacheable.

Not supported: cloud/ambient agents, Warp Drive, codebase embedding indexing,
conversation sharing. Those features fail locally rather than reaching Warp's
servers.

## Packaged builds — PARW (run on another Mac)

Rebrand name: **PARW**. Automated builds run on GitHub Actions
(`.github/workflows/parw-release.yml`): every push to `parw`/`master` builds
a macOS ARM package artifact, and pushing a `parw-v*` tag publishes a
drag-to-install DMG as a release. Trigger manually with
`gh workflow run parw-release.yml`.

Locally:

```bash
script/selfhost-package       # builds release binaries + stages dist/
```

That produces `dist/PARW-macos-arm64.zip` and a drag-to-install
`dist/PARW-macos-arm64.dmg` containing:

- `PARW.app` — the GUI, preconfigured (via `LSEnvironment`) to use the local
  backend at `127.0.0.1:8080`
- `bin/selfhost_server` and `bin/warp-tui-oss`
- `Start PARW.command` — double-click to start the backend and open the GUI
- `README.txt` — setup instructions

On the other machine: mount/open the DMG, drag PARW into Applications, clear
the quarantine flag on the unsigned build (`xattr -cr /Applications/PARW.app`),
install Ollama/LM Studio/MLX-LM, and launch PARW. To target a different
server, run the binary directly with `WARP_SELF_HOSTED_SERVER_URL=<url>`. The
package is Apple-silicon native; on Intel or another OS, build from source
with `cargo build --release` (the backend and TUI are pure Rust; the GUI
needs the per-platform toolchain).

## Offline / hermetic builds

The only build-time dependency on Warp's infrastructure is the
`warpdotdev/warp-proto-apis` git dependency (protobuf types). To build
without network access, vendor dependencies once while online:

```bash
cargo vendor vendor
cargo build --offline ...
```

`script/bootstrap --skip-common-skills` also avoids its optional
`warpdotdev/common-skills` download. The `warp-oss` / `warp-tui-oss` binaries
need no `warp-channel-config` generator.

## Metrics

`GET /metrics` serves Prometheus-format counters: agent requests, LLM errors,
tool calls, token usage per backend, transcription and relevance requests,
and MCP JSON-RPC calls.

## Bring your own keys and providers

Warp's BYOK and custom-endpoint settings work unchanged: a model from a
configured custom endpoint routes to that endpoint with that key, directly
from your server. With `--byok-direct`, requests for well-known model names
also route to the provider matching the client's BYOK key — `claude*` to
Anthropic, `gpt*`/`o*` to OpenAI, `gemini*` to Google, `grok*` (including the
client's Grok subscription OAuth token) to xAI, `org/model` to OpenRouter —
using the key the client already carries in the request.

For server-side provider keys, two options:

*Native model routing* — set a key and its models route automatically, with
the models added to Warp's picker:

```bash
# Z.ai coding plan (glm-* models):
cargo run -p selfhost_server -- --zai-api-key $ZAI_KEY
# OpenCode Zen (grok-code, code-supernova, kimi-k2, qwen3-coder):
cargo run -p selfhost_server -- --opencode-api-key $OC_KEY
```

Same works for `--openai-api-key`, `--anthropic-api-key`, `--google-api-key`,
and `--xai-api-key` with their model prefixes. OpenCode's required
`https-referer` and `x-title` headers are sent automatically.

*Force everything* through one provider with `--provider`:

```bash
cargo run -p selfhost_server -- --provider xai --xai-api-key $XAI_KEY
```

Supported `--provider` values: `chatgpt`, `openai`, `anthropic`, `google`,
`openrouter`, `xai`, `zai`, `opencode`, `azure-foundry`, `vertex`. Each falls
back to the local backend when its credentials are missing. Routing priority: client
custom endpoint > `--provider` > `--byok-direct` > local backend.

### OAuth providers

- **Azure AI Foundry** (Entra ID client credentials):
  `--provider azure-foundry --azure-tenant <tenant> --azure-client-id <id> \
   --azure-client-secret <secret> --azure-foundry-url \
   https://<resource>.services.ai.azure.com/models`
  Tokens are fetched via the client-credentials flow and cached until near
  expiry.
- **Google Vertex AI** (gcloud application-default credentials): run
  `gcloud auth application-default login` once, then
  `--provider vertex --vertex`. The refresh token from
  `~/.config/gcloud/application_default_credentials.json` is exchanged for
  access tokens against the Vertex OpenAI-compatible endpoint.
- **xAI Grok subscription**: connect Grok in Warp's settings; the client
  performs the xAI OAuth flow itself and the server routes the token to
  `api.x.ai`.
- **OpenAI (ChatGPT/Codex OAuth)**: run
  `cargo run -p selfhost_server -- --codex-login` once (a browser opens for
  the "Sign in with ChatGPT" flow; tokens are cached under
  `~/.cache/selfhost-server/`), then
  `cargo run -p selfhost_server -- --provider chatgpt`. Requests go to the
  Codex backend (`chatgpt.com/backend-api/codex`) speaking the OpenAI
  **Responses API**; access tokens refresh automatically, and the model
  picker serves `gpt-5-codex`/`gpt-5` (override with `--llm-models`).

## External authentication

By default the server accepts any client (or enforces `--api-key`). With
`--auth-introspect-url`, every request must carry a bearer token that your
auth service accepts: the server POSTs `{"token": "..."}` to that URL and
requires a 2xx. Any bearer-token auth system can sit there — SSO, an API
gateway, or a Foundry-style auth broker — so Warp clients authenticate against
your infrastructure instead of Warp's.

## Voice transcription

Warp's dictation sends base64-encoded WAV audio to `/ai/transcribe`. The
server forwards it as multipart form data to any OpenAI-compatible
speech-to-text endpoint (`/v1/audio/transcriptions`), such as
[whisper.cpp](https://github.com/ggml-org/whisper.cpp)'s server, `speaches`,
LocalAI, Groq, or OpenAI:

```bash
cargo run -p selfhost_server -- \
    --transcribe-base-url http://127.0.0.1:8082/v1 \
    --transcribe-model base.en
```

| Flag | Env | Meaning |
| --- | --- | --- |
| `--transcribe-base-url` | `SELFHOST_TRANSCRIBE_BASE_URL` | Speech-to-text base URL; `/audio/transcriptions` is appended. Transcription is disabled (the client gets a clear error) when unset. |
| `--transcribe-api-key` | `SELFHOST_TRANSCRIBE_API_KEY` | Key sent to the speech-to-text endpoint, if needed. |
| `--transcribe-model` | | Model name (default `whisper-1`). |

Wispr-style context the client attaches (dictionary, surrounding text) is
passed through as the transcription prompt.

## Quickstart (one command)

```bash
script/selfhost-up            # build + start backend + launch the GUI
script/selfhost-up --tui      # TUI front-end instead
script/selfhost-up --stop     # stop the backend
```

## Quickstart, step by step (verified end-to-end)

With [Ollama](https://ollama.com), [LM Studio](https://lmstudio.ai), or
[MLX-LM](https://github.com/ml-explore/mlx-lm) running locally:

```bash
# Terminal 1: the agent backend. Autodetects Ollama (:11434), LM Studio
# (:1234), or MLX-LM (:8080), and serves the models it finds there.
cargo run -p selfhost_server

# Terminal 2: Warp, fully self-hosted
WARP_SELF_HOSTED_SERVER_URL=http://127.0.0.1:8080 cargo run -p warp --bin warp-oss
```

The autodetected model list becomes the client's model picker. Force a
specific backend with `--llm-backend ollama|lmstudio|mlx`, a specific endpoint
with `--llm-base-url`, or pin models with `--llm-models`.

`warp-oss` (like `warp-tui-oss`) is the OpenWarp-style build: no telemetry or
crash reporting is baked in, and it does not require the internal
`warp-channel-config` generator. In an agent conversation, type a prompt
prefixed with `#`; the model picker shows the models served by
`--llm-models`. The TUI variant is `cargo run -p warp_tui --bin warp-tui-oss`.

## Running the server

```bash
# Backed by a local Ollama instance (default):
cargo run -p selfhost_server

# Custom endpoint/model:
cargo run -p selfhost_server -- \
    --bind 127.0.0.1:8080 \
    --llm-base-url http://127.0.0.1:11434/v1 \
    --llm-model qwen3-coder:30b

# Anthropic-messages endpoint (e.g. a local proxy):
cargo run -p selfhost_server -- --llm-schema anthropic --llm-base-url http://127.0.0.1:8081/v1
```

| Flag | Env | Meaning |
| --- | --- | --- |
| `--bind` | | Listen address (default `127.0.0.1:8080`). |
| `--llm-base-url` | | LLM endpoint base; `/chat/completions` (OpenAI) or `/messages` (Anthropic) is appended. |
| `--llm-api-key` | `SELFHOST_LLM_API_KEY` | Key sent to the LLM endpoint, if needed. |
| `--llm-schema` | | `openai` (default) or `anthropic`. |
| `--llm-model` | | Force the model name; otherwise the model selected in the Warp client is passed through. |
| `--llm-models` | | Comma-separated model catalog for the client's model picker (e.g. `qwen3-coder:30b,qwen3:8b`). |
| `--context-window-tokens` | | Context budget (estimated tokens) for automatic history compaction; `0` = auto (client limit, else 96k). |
| `--web-search-url` | `SELFHOST_WEB_SEARCH_URL` | SearXNG-compatible search endpoint enabling the server-executed `web_search` tool. |
| `--byok-direct` | | Route requests to the provider named by the client's BYOK key (Anthropic/OpenAI/Google/OpenRouter) based on the model name. |
| `--auth-introspect-url` | `SELFHOST_AUTH_INTROSPECT_URL` | Validate client tokens against an external auth service. |
| `--api-key` | `SELFHOST_API_KEY` | Require Warp clients to present this bearer token. |
| `--system-prompt-file` | | Replace the default agent system prompt. |

If the model selected in the Warp client matches one of its configured custom
endpoints (`Settings > AI > Custom endpoints`), the server calls *that*
endpoint with *that* key instead of its own LLM configuration.

## Pointing Warp at it

`WARP_SELF_HOSTED_SERVER_URL` puts the Warp client into self-hosted mode on
any channel (it exists precisely so release builds can be pointed elsewhere —
the plain `WARP_SERVER_ROOT_URL` override stays dev-channel-only):

```bash
WARP_SELF_HOSTED_SERVER_URL=http://127.0.0.1:8080 cargo run
```

Self-hosted mode redirects every Warp-operated service (GraphQL, the
`/ai/*` endpoints, the RTC WebSocket, session sharing) to the given host and
disables telemetry, crash reporting, and autoupdate, so nothing reaches
Warp-operated endpoints. Features the self-hosted server does not implement
fail locally.

Authentication uses Warp's API-key mode, which never contacts Firebase:

```bash
WARP_SELF_HOSTED_SERVER_URL=http://127.0.0.1:8080 WARP_API_KEY=anything cargo run
```

Without `WARP_API_KEY`, the client logs in with the placeholder key
`selfhosted`, which the server accepts (pass `--api-key` to require a
specific one).

## How the agent loop works

Warp's agent protocol is client-driven: every `POST /ai/multi-agent` request
carries the full task state (conversation history, tool calls, tool results),
and the server responds with a stream of client actions (`CreateTask`,
`AddMessagesToTask`, `AppendToMessageContent`, …) plus a final `Finished`
event. The server is stateless:

1. Decode the protobuf `Request`.
2. Translate the task state + new inputs into LLM messages, and build tool
   definitions for the tools the client declared support for.
3. Stream one LLM completion, converting text deltas into
   `AppendToMessageContent` actions and completed tool calls into typed
   `ToolCall` messages.
4. The Warp client executes the tool calls locally and sends the next request
   with the results; repeat until the model stops calling tools.
