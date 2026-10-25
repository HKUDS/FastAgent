import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastagent.grounding.backends.shell.session import ShellAgentTool


class ShellOutputStderrClassificationTests(unittest.IsolatedAsyncioTestCase):
    def test_successful_command_with_warning_on_stderr_is_not_an_error(self):
        output = ShellAgentTool._extract_output(
            None,
            {
                "returncode": 0,
                "stdout": "report generated",
                "stderr": "DeprecationWarning: legacy option",
            },
        )

        self.assertNotIn("EXECUTION ERROR", output)
        self.assertIn("report generated", output)
        self.assertIn("DeprecationWarning: legacy option", output)

    async def test_shell_agent_marks_zero_exit_with_stderr_warning_as_success(self):
        tool = object.__new__(ShellAgentTool)
        tool._py_tool = SimpleNamespace(_arun=AsyncMock())
        tool._bash_tool = SimpleNamespace(
            _arun=AsyncMock(
                return_value={
                    "returncode": 0,
                    "stdout": "report generated",
                    "stderr": "DeprecationWarning: legacy option",
                }
            )
        )

        code_info, output = await tool._execute_code_from_response(
            "```bash\necho synthetic\n```"
        )

        self.assertEqual(code_info["status"], "success")
        self.assertNotIn("EXECUTION ERROR", output)

    def test_nonzero_exit_remains_an_execution_error(self):
        output = ShellAgentTool._extract_output(
            None,
            {"returncode": 2, "stdout": "partial output", "stderr": "command failed"},
        )

        self.assertTrue(output.startswith("EXECUTION ERROR (exit code 2):"))
        self.assertIn("command failed", output)
        self.assertIn("partial output", output)


if __name__ == "__main__":
    unittest.main()
