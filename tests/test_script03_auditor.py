import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import numpy as np

# Mock modules BEFORE importing anything that might use them
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['torch'] = MagicMock()

# Now it is safe to import them (they will be mocks)
import torch
from sentence_transformers import CrossEncoder

# Add parent directory to path to import script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import script03_auditor

class TestAuditor(unittest.TestCase):

    def setUp(self):
        pass

    @patch('script03_auditor.CrossEncoder')
    def test_logic_verifier_init(self, mock_cross_encoder):
        verifier = script03_auditor.LogicVerifier()
        self.assertIsNotNone(verifier)
        mock_cross_encoder.assert_called()

    @patch('script03_auditor.CrossEncoder')
    def test_verify(self, mock_cross_encoder):
        # Setup mock instance
        mock_instance = mock_cross_encoder.return_value
        
        # Mock predict output (logits)
        # The script does: scores = self.model.predict([(claim, evidence)])[0]
        # predict returns a list (or array) of scores for the input pairs
        # We are passing 1 pair.
        # Let's say it returns [logit_contradiction, logit_entailment, logit_neutral]
        mock_instance.predict.return_value = [np.array([1.0, 2.0, 0.5])] 
        
        # Mock torch.nn.functional.softmax
        # script03_auditor uses torch.nn.functional.softmax(torch.tensor(scores), dim=0)
        # Note: script03_auditor imports torch.
        # We mocked sys.modules['torch'].
        
        # We need to ensure torch.tensor works or is mocked.
        # Since we mocked properly via sys.modules, script03_auditor.torch is a MagicMock.
        # So torch.tensor(scores) returns a MagicMock.
        # torch.nn.functional.softmax(...) returns a MagicMock.
        # Then .numpy() is called on it.
        
        # This is getting tricky with mocking deep chains. 
        # Ideally we want to let torch run if possible, but we don't assume torch is installed? 
        # The prompt says "torch" is in requirements, so strictly it might not be installed.
        # But we are mocking logic.
        
        # Let's mock the chain effectively
        mock_tensor = MagicMock()
        script03_auditor.torch.tensor.return_value = mock_tensor
        
        mock_softmax_result = MagicMock()
        script03_auditor.torch.nn.functional.softmax.return_value = mock_softmax_result
        
        # .numpy() returns the probabilities [0.2, 0.7, 0.1]
        mock_softmax_result.numpy.return_value = np.array([0.2, 0.7, 0.1])
        
        verifier = script03_auditor.LogicVerifier()
        result = verifier.verify("claim", "evidence")
        
        self.assertIsNotNone(result)
        self.assertEqual(result['claim'], "claim")
        self.assertEqual(result['decision'], "entailment") # 0.7 is max, index 1 -> entailment

if __name__ == '__main__':
    unittest.main()