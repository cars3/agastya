import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pymupdf
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPIntegration(unittest.TestCase):
    def test_build_and_restore_index(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                doc = pymupdf.open()
                page = doc.new_page()
                page.insert_text((72, 120), "Chapter One", fontsize=18)
                page.insert_text((72, 150), "A useful piece of evidence.")
                doc.set_toc([[1, "Chapter One", 1]])
                doc.save(root / "example.pdf")
                doc.close()
                env = dict(os.environ, XDG_CONFIG_HOME=str(root / "config"))
                env.pop("AGASTYA_INDEX", None)
                server = StdioServerParameters(command=sys.executable, args=["-m", "main", "--mcp"], env=env)
                async with stdio_client(server) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        names = {tool.name for tool in (await session.list_tools()).tools}
                        self.assertEqual(names, {"config", "query", "decide"})
                        result = await session.call_tool("config", {"folder": str(root)})
                        self.assertTrue(result.structured_content["ok"], result)
                state = json.loads((root / "config/agastya/active-index.json").read_text())
                self.assertEqual(state["index"], str(root / "agastya_index"))
                async with stdio_client(server) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool("config", {"folder": str(root / "missing")})
                        self.assertFalse(result.structured_content["ok"])
                import main
                old = os.environ.get("XDG_CONFIG_HOME")
                try:
                    os.environ["XDG_CONFIG_HOME"] = str(root / "config")
                    main._ACTIVE = None
                    self.assertEqual(main.get_index().root, root / "agastya_index")
                    self.assertTrue(main.get_index().leaves)
                finally:
                    main._ACTIVE = None
                    if old is None:
                        os.environ.pop("XDG_CONFIG_HOME", None)
                    else:
                        os.environ["XDG_CONFIG_HOME"] = old
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
