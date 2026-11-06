# Agent instructions

Single home for everything an AI agent needs when working in, with, or on
Agastya.

## What's here

| Path | What it is |
|---|---|
| [AGENTS.md](AGENTS.md) | Canonical project guide: tools, setup, conventions, layout |
| [../CLAUDE.md](../CLAUDE.md) | Claude Code entrypoint at the repo root, imports `AGENTS.md` |
| [CODEX.md](CODEX.md) | Codex entrypoint, imports `AGENTS.md` |
| [GEMINI.md](GEMINI.md) | No-op for Antigravity's dual-file read |
| [KIMI.md](KIMI.md) | Kimi entrypoint, imports `AGENTS.md` |
| [OPENCODE.md](OPENCODE.md) | OpenCode entrypoint, imports `AGENTS.md` |
| [agents/](agents/) | Per-agent MCP connection guides (config path + snippet + verification) |
| [.agents/skills/agastya/SKILL.md](.agents/skills/agastya/SKILL.md) | Canonical agent skill: how to use the agastya tools well |

## How the root stays wired

The repository root carries thin pointers and symlinks so agent CLIs still
auto-discover these files:

- `AGENTS.md`, `CLAUDE.md`, `CODEX.md`, `KIMI.md`, `OPENCODE.md` at the root
  import or point at this folder.
- `.agents`, `.claude`, `.antigravitycli`, `.grok`, `.kimi`, `.opencode`, and
  `.qwen` at the root are symlinks to the matching directories here.

Edit files in this folder only. Never edit a root pointer or a skill symlink.
