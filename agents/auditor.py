from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch

class LogicVerifier:
    def __init__(self, domain='legal', adapter_path=None):
        """
        Initializes the 'System 2' Verification Core.
        
        Args:
            domain (str): 'legal' or 'science'. 
                          - 'legal': AnswerDotAI/ModernBERT-large (8k context)
                          - 'science': michiyasunaga/BioLinkBERT-large (512 context)
            adapter_path (str, optional): Path to LoRA adapter.
        """
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.domain = domain.lower()
        
        if self.domain == 'legal':
            model_name = 'AnswerDotAI/ModernBERT-large'
            self.max_length = 8192
        elif self.domain == 'science':
            model_name = 'michiyasunaga/BioLinkBERT-large'
            self.max_length = 512
        else:
            raise ValueError(f"Unknown domain: {domain}. Supported: 'legal', 'science'")

        print(f"Loading Logic Core ({model_name}) on {self.device} for domain: {self.domain.upper()}...")
        
        if adapter_path:
            print(f"Loading LoRA Adapter from {adapter_path}...")
            # For Peft with CrossEncoder, we need to handle the underlying model carefully.
            # CrossEncoder initializes an AutoModelForSequenceClassification by default.
            self.model = CrossEncoder(model_name, num_labels=3, device=self.device)
            base_model = self.model.model
            
            # Load Adapter
            self.model.model = PeftModel.from_pretrained(base_model, adapter_path)
            self.model.model.to(self.device)
        else:
            self.model = CrossEncoder(model_name, num_labels=3, device=self.device)

        # Explicitly load tokenizer to handle truncation correctly
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Label Mapping
        self.label_mapping = ['contradiction', 'entailment', 'neutral']

    def verify(self, claim: str, evidence: str) -> dict:
        """
        Input: Atomic Claim + Retrieved Evidence
        Output: Structured Dictionary with Probability Scores
        """
        # 1. Prediction (No Windowing needed for ModernBERT or BioLinkBERT on SciFact abstracts)
         # We strictly limit to self.max_length
        self.model.max_length = self.max_length
        
        scores = self.model.predict([(claim, evidence)])[0]
        
        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(torch.tensor(scores), dim=0).numpy()
        
        # Standard argmax decision
        decision_idx = scores.argmax()
        
        # Safety for raw models (if output dim doesn't match 3)
        if len(self.label_mapping) > len(probs):
             decision = "unknown" 
        else:
             decision = self.label_mapping[decision_idx]

        result = {
            "claim": claim,
            "evidence_snippet": evidence[:100] + "...", 
            "scores": {
                "contradiction": float(probs[0]) if len(probs) > 0 else 0.0,
                "entailment": float(probs[1]) if len(probs) > 1 else 0.0,
                "neutral": float(probs[2]) if len(probs) > 2 else 0.0
            },
            "decision": decision
        }
        
        return result

if __name__ == "__main__":
    # Smoke Test
    print("--- Legal Test ---")
    auditor_legal = LogicVerifier(domain="legal")
    print(auditor_legal.verify("Vendor is liable.", "Vendor shall not be liable."))
    
    print("\n--- Science Test ---")
    auditor_science = LogicVerifier(domain="science")
    print(auditor_science.verify("Cells divide.", "Mitosis causes cell division."))