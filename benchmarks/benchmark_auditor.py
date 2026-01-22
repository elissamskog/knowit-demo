import pandas as pd
from datasets import load_dataset
from sklearn.metrics import classification_report
import sys
import os

# Add parent directory to path to import script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.auditor import LogicVerifier

def benchmark_contract_nli(auditor, num_samples=50):
    print(f"\n--- ⚖️ BENCHMARK: ContractNLI (Legal Logic) ---")
    try:
        dataset = load_dataset("kiddothe2b/contract-nli", "contractnli_a", split='test', streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"Skipping Legal: {e}")
        return

    true_labels = []
    pred_labels = []

    # Map ContractNLI labels to DeBERTa indices
    # ContractNLI: 0=Contradiction, 1=Entailment, 2=Neutral
    # DeBERTa: 0=Contradiction, 1=Entailment, 2=Neutral (Matches!)
    
    count = 0
    for row in dataset:
        if count >= num_samples: break
        if row['label'] == -1: continue # Skip unlabeled

        # USE THE LOGIC SCRIPT
        result = auditor.verify(row['hypothesis'], row['premise'])
        
        # Convert string decision back to index for metrics
        pred_index = auditor.label_mapping.index(result['decision'])
        
        true_labels.append(row['label'])
        pred_labels.append(pred_index)
        count += 1

    print(classification_report(true_labels, pred_labels, 
                                target_names=auditor.label_mapping))

def benchmark_scifact(auditor, num_samples=50):
    print(f"\n--- 🧬 BENCHMARK: SciFact (Scientific Logic) ---")
    try:
        # Loading 'claims' to get pairs
        data_claims = load_dataset("allenai/scifact", "claims", split='train', streaming=True, trust_remote_code=True)
        # Loading 'corpus' for lookup (simplified for speed)
        corpus = load_dataset("allenai/scifact", "corpus", split='train', trust_remote_code=True) 
        doc_lookup = {row['doc_id']: row['abstract'] for row in corpus}
    except Exception as e:
        print(f"Skipping Science: {e}")
        return

    true_labels = []
    pred_labels = []
    
    # SciFact: CONTRADICT, SUPPORT, NOINFO
    # Map to DeBERTa: CONTRADICT->0, SUPPORT->1, NOINFO->2
    label_map = {"CONTRADICT": 0, "SUPPORT": 1, "NOINFO": 2}

    count = 0
    for row in data_claims:
        if count >= num_samples: break
        
        # Logic to construct the pair
        if not row['evidence_doc_id']: continue
        doc_id = int(row['evidence_doc_id'])
        evidence_idx = row['evidence_sentences'][0] # Take first evidence sentence
        
        full_abstract = doc_lookup.get(doc_id, [])
        if evidence_idx >= len(full_abstract): continue
        evidence_text = full_abstract[evidence_idx]
        
        target_label = label_map[row['evidence_label']]

        # USE THE LOGIC SCRIPT
        result = auditor.verify(row['claim'], evidence_text)
        
        pred_index = auditor.label_mapping.index(result['decision'])
        
        true_labels.append(target_label)
        pred_labels.append(pred_index)
        count += 1

    print(classification_report(true_labels, pred_labels, 
                                labels=[0, 1, 2],
                                target_names=['contradiction', 'entailment', 'neutral'], zero_division=0))

if __name__ == "__main__":
    # Benchmarking Legal
    print("Initializing Legal Auditor...")
    auditor_legal = LogicVerifier(domain="legal")
    benchmark_contract_nli(auditor_legal)
    
    # Benchmarking Science
    print("\nInitializing Science Auditor...")
    auditor_science = LogicVerifier(domain="science")
    benchmark_scifact(auditor_science)
