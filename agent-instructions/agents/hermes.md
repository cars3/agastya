# agastya in Hermes Agent

## Register

Add to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  agastya:
    command: "/absolute/path/to/agastya/.venv/bin/agastya"
    args: ["--mcp"]
```

Or use the interactive CLI route — `hermes mcp add agastya` (it connects,
discovers the tools, and asks which to enable; on a headless run, pipe the
confirmation: `printf 'Y\n' | hermes mcp add ...`).

## Use

Restart Hermes Agent. The tools are auto-discovered and appear as
`mcp_agastya_config`, `mcp_agastya_query`, and `mcp_agastya_decide` in every
session.

## Verify

```bash
hermes mcp list
hermes mcp test agastya
```
