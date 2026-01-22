import unittest
from unittest.mock import patch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import script02_planner

class TestPlanner(unittest.TestCase):

    def test_decompose_claim(self):
        with patch('builtins.print') as mock_print:
            result = script02_planner.decompose_claim("test claim")
            mock_print.assert_called()
            self.assertIsInstance(result, list)
            self.assertIn("sub-claim 1", result)

    def test_global_search_community_summaries(self):
        with patch('builtins.print') as mock_print:
            script02_planner.global_search_community_summaries(["claim1"])
            mock_print.assert_called_with("Performing Global Search on Community Summaries...")

if __name__ == '__main__':
    unittest.main()
