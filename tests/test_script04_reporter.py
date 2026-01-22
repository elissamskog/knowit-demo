import unittest
from unittest.mock import patch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import script04_reporter

class TestReporter(unittest.TestCase):

    def test_generate_human_readable_report(self):
        with patch('builtins.print') as mock_print:
            script04_reporter.generate_human_readable_report([])
            mock_print.assert_called_with("Generating Human-Readable Report...")

if __name__ == '__main__':
    unittest.main()
