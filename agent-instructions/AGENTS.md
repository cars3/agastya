# Agastya — Agent Guide

**Agastya RAG is structure-aware retrieval for AI agents.** It indexes PDFs by
their table of contents and returns the sections or chunks that answer a
question, with the chapter path needed to cite them.

This file is the canonical instruction file for AI coding agents working in,
with, or on this repository. It lives in `agent-instructions/` together with
the agent skill and the per-agent connection guides.

The repository root carries thin pointers (`AGENTS.md`, `CLAUDE.md`,
`CODEX.md`, `KIMI.md`, `OPENCODE.md`) that import this file, plus symlinked
agent directories (`.agents`, `.claude`, `.antigravitycli`, `.grok`,
`.kimi`, `.opencode`, `.qwen`) that point into this folder. Those exist only
so every agent still auto-discovers the instructions. Edit the files here,
never the root pointers or symlinks.

## What this server provides

- `config(folder, model="local", out=None, parquet=False)` — build an index
  for every PDF in `folder` and make it the active index
- `query(question, leaf=False, relevant_match=False)` — retrieve evidence
  from the active index
- `decide(state, questions, backend="local")` — optional typed decision
  through a local or cloud backend

No API keys are required for `config` or `query`. The optional cloud decision
backend needs `TYPESAFE_API_KEY`; the optional local backend needs a
compatible Unsloth checkpoint.

## Run it

```bash
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/agastya --mcp        # stdio MCP server
```

Python 3.10+. PyTorch and the model dependencies download during install. A
CUDA GPU is recommended; `GEMMA_DEVICE=cpu` selects CPU inference. The first
query downloads `google/embeddinggemma-2` and embeds the corpus.

## Connect agastya to an agent

One-page registration guides for every supported agent:
`agent-instructions/agents/`.

| Agent | Registration location | Guide |
|---|---|---|
| Claude Code | `.mcp.json` / `claude mcp add` | [agents/claude-code.md](agents/claude-code.md) |
| Claude Desktop | `claude_desktop_config.json` | [agents/claude-desktop.md](agents/claude-desktop.md) |
| Codex | `~/.codex/config.toml` | [agents/codex.md](agents/codex.md) |
| Antigravity (CLI / IDE / 2.0) | `~/.gemini/config/mcp_config.json`, `.agents/mcp_config.json` | [agents/antigravity.md](agents/antigravity.md) |
| Cursor | `.cursor/mcp.json` | [agents/cursor.md](agents/cursor.md) |
| OpenCode | `opencode.json(c)` | [agents/opencode.md](agents/opencode.md) |
| Windsurf | `~/.codeium/windsurf/mcp_config.json` | [agents/windsurf.md](agents/windsurf.md) |
| Qwen Code | `~/.qwen/settings.json` | [agents/qwen.md](agents/qwen.md) |
| Kimi CLI | `~/.kimi/mcp.json` | [agents/kimi.md](agents/kimi.md) |
| Grok CLI | `~/.grok/config.toml` | [agents/grok.md](agents/grok.md) |
| GitHub Copilot CLI | `~/.copilot/mcp-config.json` | [agents/copilot-cli.md](agents/copilot-cli.md) |
| Pi | `~/.pi/agent/mcp.json` (`pi mcp add`) | [agents/pi.md](agents/pi.md) |
| Hermes Agent | `~/.hermes/config.yaml` (`mcp_servers`) | [agents/hermes.md](agents/hermes.md) |
| Gemini CLI (legacy) | see guide | [agents/gemini-cli.md](agents/gemini-cli.md) |
| Any other MCP client | standard stdio pattern | [agents/generic.md](agents/generic.md) |

The universal stdio pattern most clients accept:

```json
{
  "mcpServers": {
    "agastya": {
      "command": "/absolute/path/to/agastya/.venv/bin/agastya",
      "args": ["--mcp"]
    }
  }
}
```

If you are an agent asked to connect agastya on a user's machine: read the
guide for the target agent, apply its snippet to the file it names (creating
the file if needed, preserving anything already there), and run the guide's
verification step.

## Using the tools well

- Ask for the absolute path of the PDF folder; never guess a corpus path.
- PDFs need embedded bookmarks. Say so before indexing a folder that lacks
  them.
- `config` embeds the whole corpus and is slow. Warn the user before running
  it on a large library.
- Prefer `leaf=True` when the answer needs a section citation; the default
  chunk mode is for compact evidence.
- Cite only the sections the tools returned. Never invent a section, page, or
  quote.
- `decide` is optional and does not replace reading the retrieved evidence.

Full guidance: `agent-instructions/.agents/skills/agastya/SKILL.md`.

## Repository layout

```
agent-instructions/                agent instructions: readmes, skill, connection guides
agent-instructions/AGENTS.md       this file, the canonical guide
agent-instructions/agents/         per-agent connection guides
agent-instructions/.agents/skills/agastya/  canonical agent skill
main.py                            corpus build + MCP server (config, query, decide)
gemma.py                           embedding, chunking, semantic ranking
split_pdf.py                       PDF outline extraction and document splitting
decision_model.py                  typed decision backend (cloud Jev or local unsloth)
benchmarks/                        book, core, SearchTome, library and hybrid evaluations
results/                           recorded benchmark results
```

## Conventions

- Public repository: no secrets, personal paths, or private documents.
- Book PDFs are not distributed and `data/` is gitignored — never add a
  corpus to the repo.
- Benchmark scripts still import the removed `jev` module. Do not claim they
  run until that import is restored and the run actually succeeds.
- The MCP `query` path is dense-only; the published benchmarks describe a
  hierarchy walk the MCP server does not currently perform. Describe it
  accurately.
- The root agent directories are symlinks to `agent-instructions/`. Inside
  that folder, each agent's `skills/agastya/SKILL.md` is a symlink to
  `.agents/skills/agastya/SKILL.md` — edit the canonical skill only, never
  the symlinks or the root pointers.
