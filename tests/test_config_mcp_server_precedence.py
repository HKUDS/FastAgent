import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastagent.config import loader


class ConfigMcpServerPrecedenceTests(unittest.TestCase):
    def _load(self, dev_config):
        temp_dir = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(temp_dir.cleanup)
        config_dir = Path(temp_dir.name)
        grounding_path = config_dir / "config_grounding.json"
        security_path = config_dir / "config_security.json"
        dev_path = config_dir / "config_dev.json"

        grounding_path.write_text(
            json.dumps({"mcp": {"timeout": 30}}), encoding="utf-8"
        )
        security_path.write_text("{}", encoding="utf-8")
        dev_path.write_text(json.dumps(dev_config), encoding="utf-8")
        (config_dir / "config_mcp.json").write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "default-server": {"command": "synthetic-default"},
                        "shared-server": {
                            "command": "synthetic-default",
                            "env": {"mode": "default"},
                        },
                    }
                }
            ),
            encoding="utf-8",
        )

        loader.reset_config()
        self.addCleanup(loader.reset_config)
        with patch.object(loader, "CONFIG_DIR", config_dir):
            return loader.load_config(grounding_path, security_path, dev_path)

    def test_nested_dev_registry_merges_over_default_and_overrides_same_name(self):
        config = self._load(
            {
                "mcp": {
                    "timeout": 17,
                    "servers": {
                        "dev-server": {"command": "synthetic-dev"},
                        "shared-server": {"command": "synthetic-dev"},
                    },
                }
            }
        )

        self.assertEqual(
            config.mcp.servers,
            {
                "default-server": {"command": "synthetic-default"},
                "dev-server": {"command": "synthetic-dev"},
                "shared-server": {
                    "command": "synthetic-dev",
                    "env": {"mode": "default"},
                },
            },
        )
        self.assertEqual(config.mcp.timeout, 17)

    def test_default_registry_is_loaded_when_no_custom_registry_is_given(self):
        config = self._load({"debug": True})

        self.assertEqual(
            config.mcp.servers,
            {
                "default-server": {"command": "synthetic-default"},
                "shared-server": {
                    "command": "synthetic-default",
                    "env": {"mode": "default"},
                },
            },
        )

    def test_top_level_custom_registry_keeps_existing_precedence(self):
        config = self._load(
            {
                "mcpServers": {
                    "custom-server": {"command": "synthetic-custom"}
                }
            }
        )

        self.assertEqual(
            config.mcp.servers,
            {"custom-server": {"command": "synthetic-custom"}},
        )


if __name__ == "__main__":
    unittest.main()
