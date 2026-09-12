<h1 align="center">PRAW</h1>

<p align="center">
  <strong>The agent terminal that answers only to you.</strong><br/>
  A fork of <a href="https://github.com/warpdotdev/warp">Warp</a> in which every
  AI agent call stays on hardware you control — a server you run, or just your
  own machine. No Warp account, no subscription, no Warp-operated services.
</p>

---

## What this is

PRAW (roughly "warp" spelled backwards) keeps Warp's agent experience but
replaces Warp's cloud with a self-hosted backend that ships in this repo
(`crates/selfhost_server`). In self-hosted mode the client:

- talks **only** to the server you point it at (`WARP_SELF_HOSTED_SERVER_URL`);
- requires **no login** — a local identity is synthesized for you;
- has **no credit or subscription gates** — bring your own model or key;
- disables Warp telemetry, crash reporting, and autoupdate.

The backend implements the slice of Warp's server API that Agent Mode needs:
the multi-agent wire protocol (translated to OpenAI- or Anthropic-compatible
LLM endpoints), the GraphQL operations the client polls, transcription, and
web search. The agent loop itself stays client-executed, exactly as in
upstream.

## Install (macOS)

Grab `PRAW-macos-universal.dmg` from the
[releases page](https://github.com/gregnazario/warp/releases), open it, and
drag **PRAW.app** to Applications.

The build is unsigned, so clear the quarantine flag once after transfer:

```sh
xattr -dr com.apple.quarantine /Applications/PRAW.app
```

or double-click **Install PRAW.command** next to the app in the DMG.

Opening PRAW starts the bundled backend automatically (loopback only; the
app-internal port is 48080 to dodge the busy 8080). Detected models show up
in the model picker; type a prompt prefixed with `#` to use the agent. The
backend log lives at `~/Library/Logs/PRAW-backend.log`.

## LLM backends

Autodetected in order:

| Backend   | Endpoint                  |
| --------- | ------------------------- |
| Ollama    | `http://127.0.0.1:11434`  |
| LM Studio | `http://127.0.0.1:1234`   |
| MLX-LM    | `http://127.0.0.1:8080`   |

Point at any OpenAI- or Anthropic-compatible endpoint with `--llm-base-url`,
force a backend with `--llm-backend ollama|lmstudio|mlx`, pin the catalog with
`--llm-models`, or set a context-window budget with `--context-window-tokens`.

## Cloud providers (BYOK)

Run the backend with `--provider <name>` and the matching key. Requests that
carry a client BYO key are also routed natively by model name:

| Provider      | Key flag / setup                                         |
| ------------- | -------------------------------------------------------- |
| OpenAI        | `--openai-api-key`                                       |
| Anthropic     | `--anthropic-api-key`                                    |
| Google        | `--google-api-key`                                       |
| xAI           | `--xai-api-key`                                          |
| Z.ai          | `--zai-api-key` (coding-plan endpoint)                   |
| OpenCode Zen  | `--opencode-api-key` (required headers sent by default)  |
| ChatGPT/Codex | `--codex-login` once, then `--provider chatgpt`          |
| Azure Foundry | `--azure-tenant/--azure-client-id/--azure-client-secret/--azure-foundry-url` |
| Google Vertex | `--vertex` after `gcloud auth application-default login` |

See `crates/selfhost_server/README.md` for the full flag reference.

## Running the pieces yourself

```sh
# Terminal 1: the agent backend (defaults to a local Ollama)
cargo run -p selfhost_server

# Terminal 2: the GUI, pointed at it
WARP_SELF_HOSTED_SERVER_URL=http://127.0.0.1:8080 cargo run

# Or the headless TUI
WARP_SELF_HOSTED_SERVER_URL=http://127.0.0.1:8080 ./script/run-tui
```

`WARP_API_KEY` supplies the client credential (any value works unless the
server is started with `--api-key`). Server-side web search needs a SearXNG
endpoint (`--web-search-url`); voice transcription needs an OpenAI-compatible
STT endpoint (`--transcribe-base-url`).

Troubleshooting starts with:

```sh
selfhost_server --doctor   # backends, keys, ports — what's working and what isn't
```

## Command line

The package ships a `praw` CLI (install it with `Install PRAW.command`, or
copy `bin/praw` somewhere on your `PATH`):

```sh
praw            # start the backend (if needed) and open PRAW.app
praw backend    # start just the agent backend (e.g. for the TUI)
praw status     # is the backend healthy, where is the app
praw stop       # quit the app and stop the backend
praw doctor     # run selfhost_server --doctor from the installed app
praw logs       # follow the backend log
praw update     # install the latest release over the local app
```

It honors `PRAW_APP` (path to PRAW.app) and `PRAW_BACKEND_PORT`
(default 48080, matching the app launcher).

## Building a package

```sh
./script/bootstrap          # toolchain + deps
./script/selfhost-package   # release binaries + PRAW.app + DMG/zip in dist/
```

The packaging script stamps the version from the latest `parw-v*` tag,
builds fat arm64+x86_64 binaries when the Intel target is installed, and
decorates the DMG (background, icon layout, Applications link). The release
workflow (`.github/workflows/parw-release.yml`) does the same on a `parw-v*`
tag push.

## Relationship to upstream

This fork intentionally stays close to `warpdotdev/warp` so upstream changes
can be merged. The deltas are deliberately contained:

- `crates/selfhost_server/` — the self-hosted backend (new crate)
- `app/src/lib.rs` — self-hosted environment opt-in (`apply_self_hosted_env_override`)
- `app/src/auth/` — local sign-in synthesis; `app/src/ai/` — credit-gate removal
- `app/src/bin/oss.rs`, `resources/parw/` — PRAW identity and icon
- `script/selfhost-package`, `.github/workflows/parw-release.yml` — packaging

Everything else is upstream Warp. Licensed under AGPL-3.0, like upstream
(see `LICENSE-AGPL`; parts of the tree are MIT — see `LICENSE-MIT`).
