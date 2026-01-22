PROJECT: KnowledgeAudit (Core Engine)

TECHNICAL SPECIFICATION: General Framework

**Step 1: Ingestion & Graph Construction ("The Librarian")**

* **Goal:** Transform domain-specific unstructured text into a structured, queryable Knowledge Graph.  
* **Technical Action:** Execute **GraphRAG Indexing** (Microsoft Research implementation).  
  1. **Extract Entities & Relationships:** Use **Gemini 1.5 Pro** with a domain-agnostic prompt structure.  
     * *Legal Example:* Extract OBLIGATION, LIMIT\_VALUE, PROHIBITS.  
     * *Science Example:* Extract PROTEIN, MOLECULAR\_WEIGHT, UPREGULATES.  
  2. **Cluster:** Partition the graph using the **Leiden Algorithm** to find hierarchical communities (e.g., "All clauses related to Indemnification" or "All studies related to mRNA decay").  
  3. **Summarize:** Generate natural language summaries for every community level (C0-C1).  
* **Output:** A searchable index of **Community Summaries** (Global Context) \+ Vector Index (Local Context).

**Step 2: Retrieval & Decomposition ("The Planner")**

* **Goal:** Map a user's broad claim or document against the entire corpus without losing "Global" context.  
* **Technical Action:** **Community Map-Reduce**.  
  1. **Decompose:** Break the user's input (e.g., a new Contract or a new Research Protocol) into atomic sub-claims.  
  2. **Global Match:** Query the **Community Summaries** first. (e.g., *"Does this protocol contradict our 2024 Safety Guidelines?"* matches against the "Safety Community" summary, not random vector chunks).  
  3. **Local Drill-Down:** Once the relevant community is found, retrieve specific source citations.

**Step 3: Verification Core ("The Auditor")**

* **Goal:** Strict Logical Entailment (True/False/Neutral).  
* **Technical Action:** **Discriminative NLI (Natural Language Inference)**.  
  1. **Input:** Pairs of (Claim, Evidence).  
  2. **Model:** **DeBERTa-v3-Large** (Fine-tuned Cross-Encoder).  
  3. **Constraint:** Must be fine-tuned on **ContractNLI** (for Law) and **SciFact/MedNLI** (for Science) to handle domain-specific negation.  
  4. **Mechanism:** Use **Replaced Token Detection (RTD)** to output a calibrated probability score (0.0–1.0). *No Generative LLMs allowed here.*

**Step 4: Evaluation Loop ("The Judge")**

* **Goal:** Quality Control & Self-Correction.  
* **Technical Action:** **RAGAS Metric Evaluation**.  
  1. **Context Relevance:** Measure if the Community Summary contained noise.  
  2. **Faithfulness:** Calculate $F \= |V|/|S|$ (Ratio of verified statements).  
  3. **Threshold:** If Confidence \< **90%**, trigger **Self-Correction** (re-query Step 2 with tighter constraints).

**Step 5: Synthesis & Reporting ("The Reporter")**

* **Goal:** User-Facing Artifact.  
* **Technical Action:** Generate the "Audit Certificate" or "Literature Review" using **Gemini 1.5 Pro**.  
  * *Legal Output:* "Risk Report" citing specific clauses.  
  * *Science Output:* "Systematic Review" citing supporting/refuting papers.