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
| [SKILL.md](SKILL.md) | The one canonical agent skill: how to use the agastya tools well |

## How the root stays wired

The repository root carries thin pointers (`AGENTS.md`, `CLAUDE.md`,
`CODEX.md`, `KIMI.md`, `OPENCODE.md`) that import `AGENTS.md` here. They
exist only so agent CLIs still discover the instructions.

There are no per-agent skill directories. Every agent reads the same
`SKILL.md` in this folder. Edit files here only; never edit a root pointer.
