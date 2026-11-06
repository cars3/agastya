# agastya in Claude Desktop

## 1. Edit the config

Add agastya under `mcpServers` in `claude_desktop_config.json`:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

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

Add it alongside any servers already listed — don't replace them.

## 2. Restart Claude Desktop completely

Quit the app fully (macOS: Cmd+Q — closing the window is not enough) and
reopen it. Claude Desktop reads its config only at launch.

## Verify

agastya should appear in the tools menu with `config`, `query`, and `decide`.
Then ask: "Index the PDFs in /absolute/path/to/manuals with agastya, then find
the section on X." Desktop asks you to approve each tool call the first time.

## Troubleshooting

- **agastya doesn't appear**: confirm the path points at the console script
  inside the virtualenv, not at `main.py`. Run the command with `--help` in a
  terminal to check it starts.
- **`config` fails**: the folder must exist, contain PDFs with embedded
  bookmarks, and be readable by the Desktop process.
- Server log (macOS): `~/Library/Logs/Claude/mcp-server-agastya.log`.

## Notes

The first `config` call downloads the `google/embeddinggemma-2` model if it is
not cached, and may need Hugging Face authentication. A CUDA GPU is
recommended; `GEMMA_DEVICE=cpu` forces CPU inference.
