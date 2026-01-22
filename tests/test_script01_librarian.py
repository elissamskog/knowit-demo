import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Mock networkx before import
sys.modules['networkx'] = MagicMock()
sys.modules['graphrag'] = MagicMock() # Mock graphrag as well just in case

# Add parent directory to path to import scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import script01_librarian

class TestLibrarian(unittest.TestCase):

    def test_ingest_documents(self):
        # Test that ingest_documents runs without error (mocking internals if needed)
        with patch('builtins.print') as mock_print:
            script01_librarian.ingest_documents("./dummy_path")
            mock_print.assert_called_with("Ingesting documents from ./dummy_path...")

    def test_extract_entities_and_relationships(self):
        with patch('builtins.print') as mock_print:
            script01_librarian.extract_entities_and_relationships([])
            mock_print.assert_called_with("Extracting entities and relationships...")

    def test_run_leiden_community_detection(self):
        mock_graph = MagicMock()
        with patch('builtins.print') as mock_print:
            script01_librarian.run_leiden_community_detection(mock_graph)
            mock_print.assert_called_with("Running Leiden Community Detection...")

if __name__ == '__main__':
    unittest.main()
