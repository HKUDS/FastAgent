import asyncio
import unittest
import warnings

from fastagent.grounding.core.tool.base import BaseTool


class RuntimeErrorTool(BaseTool):
    async def _arun(self):
        raise RuntimeError("synthetic tool failure")


class EchoTool(BaseTool):
    async def _arun(self):
        return "ok"


class BaseToolRunTests(unittest.IsolatedAsyncioTestCase):
    def test_sync_run_preserves_tool_runtime_error(self):
        tool = RuntimeErrorTool(handle_errors=False)

        with self.assertRaisesRegex(RuntimeError, "synthetic tool failure"):
            tool.run()

    async def test_run_inside_event_loop_returns_task_without_unawaited_coroutine(self):
        tool = EchoTool()

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", RuntimeWarning)
            result = tool.run()
            self.assertIsInstance(result, asyncio.Task)
            self.assertEqual((await result).content, "ok")

        self.assertFalse(
            any(issubclass(warning.category, RuntimeWarning) for warning in captured)
        )


if __name__ == "__main__":
    unittest.main()
