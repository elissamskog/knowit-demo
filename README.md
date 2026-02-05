# KnowIt — Multi-Domain Knowledge Auditor

KnowIt is a modular pipeline that audits claims against domain-specific knowledge bases using Knowledge Graphs and Natural Language Inference (NLI). It supports **legal** and **scientific** domains out of the box.

## Architecture

The system is composed of four agents, each responsible for a stage of the audit pipeline:

| Agent | Module | Role |
|---|---|---|
| **Librarian** | `agents/librarian.py` | Ingests documents, extracts entities/relations via Gemini, and builds a NetworkX knowledge graph |
| **Planner** | `agents/planner.py` | Decomposes user queries into atomic sub-claims and retrieves relevant evidence (WIP) |
| **Auditor** | `agents/auditor.py` | Verifies claim–evidence pairs using discriminative NLI (cross-encoder) with LoRA adapters |
| **Reporter** | `agents/reporter.py` | Generates human-readable audit reports (WIP) |

### Auditor Detail

The verification core (`LogicVerifier`) runs domain-specific NLI:

- **Legal** — `AnswerDotAI/ModernBERT-large` (8k context), fine-tuned on ContractNLI
- **Science** — `michiyasunaga/BioLinkBERT-large` (512 context), fine-tuned on SciFact

Each model outputs calibrated probabilities over three labels: `contradiction`, `entailment`, `neutral`.

## Project Structure

```
agents/             # Pipeline agents
  librarian.py      # Knowledge graph construction (Gemini + NetworkX)
  auditor.py        # NLI verification core (CrossEncoder + LoRA)
  planner.py        # Query decomposition & retrieval (WIP)
  reporter.py       # Report synthesis (WIP)
forge/              # Training scripts for LoRA adapters
  science_nli/      # SciFact fine-tuning (BioLinkBERT)
  auditor_nli/      # ContractNLI fine-tuning (ModernBERT)
artifacts/
  adapters/         # Trained LoRA adapter weights
  library/          # Persisted knowledge graph & chunks
benchmarks/         # Evaluation scripts (ContractNLI, SciFact)
tests/              # Unit tests for each agent
```

## Setup

```bash
pip install -r requirements.txt
```

Set your API key for the Librarian's Gemini integration:

```bash
export GOOGLE_API_KEY="your-key-here"
```

## Usage

### Build the Knowledge Graph

```bash
python agents/librarian.py
```

Ingests the SciFact corpus, extracts entities/relations, and saves the graph to `artifacts/library/`.

### Verify a Claim

```python
from agents.auditor import LogicVerifier

auditor = LogicVerifier(domain="science")
result = auditor.verify(
    claim="Cells divide via mitosis.",
    evidence="Mitosis causes cell division."
)
print(result["decision"])  # entailment
```

### Train a LoRA Adapter

```bash
# Science (BioLinkBERT on SciFact)
python forge/science_nli/train_science.py

# Legal (ModernBERT on ContractNLI)
python forge/auditor_nli/train_legal.py
```

### Run Benchmarks

```bash
python benchmarks/benchmark_auditor.py
```

## Requirements

- Python 3.10+
- PyTorch (MPS/CUDA/CPU)
- See `requirements.txt` for full dependency list
