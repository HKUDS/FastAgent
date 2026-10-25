import json
import runpy
import tempfile
import unittest
from pathlib import Path


_tool_cache_path = (
    Path(__file__).resolve().parents[1]
    / "fastagent/grounding/backends/mcp/tool_cache.py"
)
MCPToolCache = runpy.run_path(str(_tool_cache_path))["MCPToolCache"]


class MCPToolCachePathTests(unittest.TestCase):
    def test_sanitized_cache_creates_its_own_parent_directory(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp_dir:
            root = Path(temp_dir)
            raw_path = root / "raw" / "tools.json"
            sanitized_path = root / "sanitized" / "tools.json"
            sanitized_tools = {"synthetic-server": [{"name": "synthetic-tool"}]}
            cache = MCPToolCache(
                cache_path=raw_path,
                sanitized_cache_path=sanitized_path,
            )

            cache.save_sanitized(sanitized_tools)

            self.assertTrue(sanitized_path.is_file())
            self.assertEqual(
                json.loads(sanitized_path.read_text(encoding="utf-8"))["servers"],
                sanitized_tools,
            )
            self.assertEqual(cache.get_all_sanitized_tools(), sanitized_tools)


if __name__ == "__main__":
    unittest.main()
