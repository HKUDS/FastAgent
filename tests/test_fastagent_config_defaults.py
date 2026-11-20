import unittest

from fastagent.fastagent import FastAgentConfig


class FastAgentConfigDefaultsTests(unittest.TestCase):
    def test_programmatic_execution_timeout_matches_workflow_config(self):
        direct_default = FastAgentConfig().max_execution_time
        workflow_default = FastAgentConfig.load().max_execution_time

        self.assertEqual(direct_default, workflow_default)


if __name__ == "__main__":
    unittest.main()
