import unittest
from unittest.mock import AsyncMock

from fastagent.grounding.backends.shell.transport.connector import ShellConnector


class ShellConnectorRequestTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_timeout_exceeds_remote_script_timeout(self):
        connector = ShellConnector(vm_ip="127.0.0.1", retry_times=1)
        connector.post_json = AsyncMock(return_value={"returncode": 0, "output": "synthetic"})

        await connector.run_bash_script("echo synthetic", timeout=120)

        connector.post_json.assert_awaited_once()
        path = connector.post_json.await_args.args[0]
        payload = connector.post_json.await_args.kwargs["payload"]
        request_timeout = connector.post_json.await_args.kwargs["timeout"]

        self.assertEqual(path, "/run_bash_script")
        self.assertEqual(payload["timeout"], 120)
        self.assertEqual(request_timeout, 180)


if __name__ == "__main__":
    unittest.main()
