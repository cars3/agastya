# agastya in any MCP client

agastya is a standard stdio MCP server. The values are always the same:

- command: `/absolute/path/to/agastya/.venv/bin/agastya`
- args: `--mcp`

Most clients accept a JSON shape like:

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

## Shape variations to watch for

- Top-level key: `mcpServers` (most clients), `mcp` (OpenCode), `servers`
  (VS Code Copilot)
- Type field: some clients require `"type": "local"` or `"type": "stdio"`
- Single-array command: some clients want
  `"command": ["/path/to/agastya", "--mcp"]`

The values never change; only the packaging differs.

## Alternative ways to run it

1. **Without installing the console script** — run the module directly from
   the checkout:

   ```json
   {
     "mcpServers": {
       "agastya": {
         "command": "/path/to/agastya/.venv/bin/python",
         "args": ["main.py", "--mcp"],
         "cwd": "/path/to/agastya"
       }
     }
   }
   ```

2. **Installed console script** — `pip install .` inside the virtualenv, then
   use `/path/to/agastya/.venv/bin/agastya --mcp`.

There is no HTTP transport. agastya is a local stdio server because it loads
local PDFs and a local embedding model.

## Verify

Once registered, ask the agent to call `config` with a folder of bookmarked
PDFs, then `query` with a question about them.
