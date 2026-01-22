Here is the formal update for your Technical Plan v1.1.
This section replaces the generic "Verification" specs with a concrete Model Selection Strategy. It introduces a "Split-Brain" architecture: one lane optimized for Long-Context Legal reasoning and another for Deep-Scientific reasoning.

Technical Update: Phase 3 - Domain-Specific Verification Strategy
Status: Approved for Implementation
Objective: Replace the generic NLI baseline with domain-optimized architectures to solve the "Truncation" (Context) and "Jargon" (Vocabulary) failure modes.
1. The Challenge: One Brain Cannot Do It All
Our feasibility testing revealed two distinct failure modes:
The Legal Failure: Contracts exceed 512 tokens. Sliding windows result in "Neutral" predictions because the premise and hypothesis are separated by pages of text.
The Science Failure: Standard models treat "upregulation" and "suppression" as neutral terms, missing the logical contradiction in biomedical contexts.
2. Lane A: The Legal & Compliance Engine
Target Domain: Banking Regulations, Master Services Agreements (MSAs), NDAs.
Primary Constraint: Context Length. Liability clauses often appear 15 pages after the Definitions section.
Role
Selected Model
Why this wins
Primary
AnswerDotAI/ModernBERT-large
8,192 Token Context. It can ingest an entire standard agreement in one pass. This eliminates the need for complex "Sliding Window" code and prevents "Lost in the Middle" errors.
Backup
pile-of-law/legal-deberta-large
Trained on CaseHOLD (judicial decisions). Superior understanding of legal "terms of art" but limited to 512 tokens. Use only if ModernBERT fails on vocabulary nuance.
Benchmark
ContractNLI
Success Metric: >85% Recall on "Contradiction" without sliding windows.

Implementation Note:
Switching to ModernBERT allows us to deprecate the predict_with_window() logic in agents/auditor.py, significantly reducing code complexity and latency.

3. Lane B: The Biomedical & Scientific Engine
Target Domain: Clinical Protocols, Biotech Patents, FDA Submissions.
Primary Constraint: Vocabulary & Citation Logic. Scientific truth is often established by referencing external studies (e.g., "As shown in [12]...").
Role
Selected Model
Why this wins
Primary
michiyasunaga/BioLinkBERT-large
Citation-Aware. Pre-trained on PubMed with a specific objective to understand hyperlinks and citations. If a claim relies on a referenced study, LinkBERT can bridge the logical gap where standard BERT fails.
Backup
microsoft/BiomedNLP-PubMedBERT
The "Vocabulary Expert." Trained from scratch on PubMed (not initialized from Wikipedia). It creates specific tokens for drug names rather than breaking them into sub-word nonsense.
Benchmark
SciFact / MedNLI
Success Metric: >80% Accuracy on "Refutes" class (distinguishing "No Info" from "Contradiction").


4. The "Champion-Challenger" Testing Protocol
We will not deploy these blindly. We will run a specific evaluation script (forge/eval_domain.py) to confirm superiority over the baseline.
Test Matrix:
Feature
Baseline (DeBERTa-v3)
Legal (ModernBERT)
Science (BioLinkBERT)
Max Tokens
512
8,192
512
Legal Jargon
Low (Wikipedia)
Medium
Low
Bio Jargon
Low
Low
High (PubMed)
Sliding Window?
REQUIRED (High Risk)
REMOVED (Low Risk)
REQUIRED

Decision Logic for Deployment:
If ModernBERT achieves within 5% of DeBERTa's accuracy on short texts, ADOPT immediately for its context window benefits.
If BioLinkBERT outperforms DeBERTa by >10% on SciFact, ADOPT for the Science Agent.

Next Step:
Update the requirements.txt to include flash-attn (required for ModernBERT speed) and instruct the MLOps Architect (Antigravity) to pull these specific weights.


