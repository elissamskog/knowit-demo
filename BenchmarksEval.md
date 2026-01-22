### **Detailed Benchmark Descriptions**

#### **Module 1: Ingestion & Graph Construction ("The Librarian")**

* **Goal:** Extract structured nodes (Entities) and edges (Relationships) from raw text.  
* **Legal Benchmark: CUAD (Contract Understanding Atticus Dataset)**  
  * **Why:** Contains 13,000+ annotations of legal clauses. It tests if your model can find a "Renewal Date" or "Jurisdiction" buried in a PDF.  
* **Science Benchmark: BioRED (Biomedical Relation Extraction Dataset)**  
  * **Why:** The gold standard for extracting relations like *Chemical-induces-Disease* or *Gene-associated-with-Variant*. If your GraphRAG can parse BioRED, it can parse any biotech literature.  
  * **Metric:** **F1-Score** (aim for \>75% for entities, \>60% for relations).

    #### **Module 2: Retrieval & Decomposition ("The Planner")**

* **Goal:** Find the "needle in the haystack" across multiple documents.  
* **Legal Benchmark: HotpotQA (Distractor Setting)**  
  * **Why:** Tests "Multi-hop" reasoning (finding Document A and Document B to answer one question). Essential for legal precedents.  
* **Science Benchmark: BEIR-SciFact (Retrieval Task)**  
  * **Why:** Specifically tests retrieval of biomedical abstracts that *support or refute* a claim. It ensures your system doesn't just find keywords (like "Cancer") but finds *evidence* (like "does drug X treat cancer?").  
  * **Metric:** **NDCG@10** (Normalized Discounted Cumulative Gain).

    #### **Module 3: Verification Core ("The Auditor")**

* **Goal:** Determine strictly if the retrieved text supports the claim (Entailment).  
* **Legal Benchmark: ContractNLI**  
  * **Why:** NLI (Natural Language Inference) specifically for contracts. It labels pairs as Entailment, Contradiction, or Neutral.  
* **Science Benchmark: SciFact (Claim Verification Task)**  
  * **Why:** The leading benchmark for scientific fact-checking. It provides expert-written claims and evidence-containing abstracts.  
  * **Hard Mode:** **MedNLI**. This is clinical logic (e.g., "Patient has symptoms X, Y \-\> Patient has disease Z"). It is much harder than standard science facts and tests strict logical deduction.  
  * **Metric:** **Label Accuracy** (specifically on the "Refutes/Contradiction" class).

    #### **Module 4: Synthesis & Reporting ("The Reporter")**

* **Goal:** Explain the findings to a non-expert user.  
* **Legal Benchmark: BillSum**  
  * **Why:** Summarizing US Congressional bills. Good proxy for summarizing municipal LOU regulations.  
* **Science Benchmark: LaySumm (BioLaySumm)**  
  * **Why:** The task of summarizing technical biomedical research for a "lay" audience (non-experts). This perfectly mimics your goal of explaining a complex biotech patent to a business executive or investor.  
  * **Metric:** **FactScore** (counting the number of verifiable atomic facts in the summary).

    ### **Recommended Action**

When you prompt your different Claude instances to build these modules, give them the specific benchmark target.

**Example Prompt for Module 3 (Verification):**

"Build a Python verification function using `DeBERTa-v3-large`. **Constraint:** It must achieve \>85% accuracy on the 'Refutes' class of the **SciFact** dataset. **Context:** We are optimizing for scientific rigor, so False Negatives (missing a contradiction) are worse than False Positives."

This ensures your "General" tool is actually general, respecting the rigor of both Law and Science.

* 

