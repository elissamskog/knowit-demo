import argparse
import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType

def train_science(output_dir="artifacts/adapters/science_v1"):
    print(f"Starting LoRA Training for SciFact (BioLinkBERT)...")
    
    # 1. Load Dataset
    print("Loading SciFact dataset...")
    # SciFact has 'claims' (labeled pairs) and 'corpus' (text store)
    dataset_claims = load_dataset("allenai/scifact", "claims", split="train", trust_remote_code=True)
    dataset_corpus = load_dataset("allenai/scifact", "corpus", split="train", trust_remote_code=True)
    
    # Index corpus for fast retrieval: doc_id -> abstract (list of sentences)
    print("Indexing corpus...")
    corpus_lookup = {row['doc_id']: row['abstract'] for row in dataset_corpus}
    
    # 2. Preprocess & Filter Data
    # Goal: Construct (Claim, Evidence) pairs.
    # Label Map: CONTRADICT -> 0, SUPPORT -> 1, NOINFO -> 2
    label_map = {"CONTRADICT": 0, "SUPPORT": 1, "NOINFO": 2}
    
    print("Constructing Claim-Evidence pairs...")
    processed_samples = []
    
    for row in dataset_claims:
        # User Instruction: "Only train on claims that HAVE evidence"
        # This implies we focus on SUPPORT/CONTRADICT signals for now.
        if not row['evidence_doc_id']:
            continue
            
        doc_id = int(row['evidence_doc_id'])
        
        # SciFact provides evidence as a list of sentence indices in the abstract
        evidence_idxs = row['evidence_sentences'] 
        label_str = row['evidence_label']
        
        if label_str not in label_map:
            continue
            
        label = label_map[label_str]
        
        # Retrieve text
        if doc_id not in corpus_lookup:
            continue
        
        full_abstract = corpus_lookup[doc_id]
        
        # Create a training example for each cited evidence sentence
        for idx in evidence_idxs:
            if idx < len(full_abstract):
                evidence_text = full_abstract[idx]
                processed_samples.append({
                    "claim": row['claim'],
                    "evidence": evidence_text,
                    "label": label
                })
                
    print(f"Created {len(processed_samples)} training pairs.")
    
    # Create HF Dataset
    train_dataset = Dataset.from_list(processed_samples)
    
    # 3. Load Model & Tokenizer
    model_name = 'michiyasunaga/BioLinkBERT-large'
    print(f"Loading Base Model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3
    )
    
    # 4. Configure LoRA
    print("Applying LoRA Config...")
    # Standard BERT modules
    target_modules = ["query", "value"]
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        inference_mode=False, 
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.1,
        target_modules=target_modules
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 5. Tokenize
    def preprocess_function(examples):
        # Standard BERT sequence pair
        return tokenizer(
            examples['claim'], 
            examples['evidence'], 
            truncation=True, 
            max_length=512
        )
    
    tokenized_datasets = train_dataset.map(preprocess_function, batched=True)
    
    # Split for validation
    split_dataset = tokenized_datasets.train_test_split(test_size=0.1)
    
    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"], 
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    
    # 7. Train
    print("Starting Training (Dry Run)...")
    # trainer.train()
    
    # 8. Save
    print(f"Simulating save to {output_dir}...")
    model.save_pretrained(output_dir)
    print(f"Adapter saved to {output_dir}")

if __name__ == "__main__":
    train_science()
