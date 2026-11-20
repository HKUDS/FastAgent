import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastagent.grounding.core.quality.manager import ToolQualityManager
from fastagent.grounding.core.types import BackendType


class ToolQualityManualSaveCountTests(unittest.IsolatedAsyncioTestCase):
    async def test_manual_save_preserves_global_execution_count(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as cache_dir:
            manager = ToolQualityManager(
                cache_dir=Path(cache_dir),
                enable_persistence=True,
                auto_save=False,
            )
            tool = SimpleNamespace(
                is_bound=False,
                backend_type=BackendType.SHELL,
                name="synthetic-tool",
                description="Synthetic test tool",
                schema=SimpleNamespace(parameters={}),
            )
            result = SimpleNamespace(is_error=False, error=None, is_success=True)

            for _ in range(5):
                await manager.record_execution(tool, result, execution_time_ms=1.0)
            await manager.save()

            restored = ToolQualityManager(
                cache_dir=Path(cache_dir),
                enable_persistence=True,
                auto_save=False,
            )
            _, saved_execution_count = restored._store.load_all()

            self.assertEqual(saved_execution_count, 5)
            self.assertEqual(restored._global_execution_count, 5)


if __name__ == "__main__":
    unittest.main()
