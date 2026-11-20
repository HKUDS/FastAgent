import sys
import types
import unittest
from unittest.mock import patch

from fastagent.utils.logging import Logger


class LoggerStdinPathTests(unittest.TestCase):
    def test_interactive_placeholders_use_portable_default_log_directory(self):
        for placeholder in ("<stdin>", "<string>"):
            with self.subTest(placeholder=placeholder):
                synthetic_main = types.SimpleNamespace(__file__=placeholder)
                with patch.dict(sys.modules, {"__main__": synthetic_main}):
                    log_file = Logger._get_default_log_file()

                self.assertNotIn(placeholder, log_file)
                self.assertEqual(
                    log_file.replace("\\", "/").split("/")[-2], "fastagent"
                )


if __name__ == "__main__":
    unittest.main()
