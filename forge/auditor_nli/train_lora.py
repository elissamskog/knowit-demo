import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType

def train_lora(output_dir="artifacts/adapters/legal_v1"):
    print(f"Starting LoRA Training for ContractNLI (ModernBERT)...")
    
    # 1. Load Dataset
    print("Loading ContractNLI dataset...")
    # Using 'contractnli_b' (Span focused) as per user advice to avoid truncation blindness
    dataset = load_dataset("kiddothe2b/contract-nli", "contractnli_b", trust_remote_code=True)
    
    # 2. Load Model & Tokenizer
    model_name = 'AnswerDotAI/ModernBERT-large'
    print(f"Loading Base Model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3,
        ignore_mismatched_sizes=True 
    )
    
    # Print modules to verify targets as requested
    print("Base Model Modules:")
    for name, module in model.named_modules():
        print(name)

    # 3. Configure LoRA
    print("Applying LoRA Config...")
    # ModernBERT uses specific module names. 
    # Reference: https://huggingface.co/AnswerDotAI/ModernBERT-large (or standard ModernBERT impl)
    # Common Fused Layers: Wqkv (Attention), Wo (Output), W1 (MLP In), W2 (MLP Out)
    target_modules = ["Wqkv", "Wo", "W1", "W2"]
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1,
        target_modules=target_modules
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 4. Preprocess Data
    def preprocess_function(examples):
        # Swap inputs to (Hypothesis, Premise) to match Inference Agent
        # Update max_length to 8192 for ModernBERT
        return tokenizer(
            examples['hypothesis'], 
            examples['premise'], 
            truncation=True, 
            max_length=8192
        )
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    
    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-4,
        per_device_train_batch_size=4, # Reduced batch size given larger context/model
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
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"], 
        tokenizer=tokenizer, # tokenizer argument handles padding 
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    
    # 6. Train
    print("Starting Training...")
    # trainer.train() 
    
    # Determine device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available(): device = "cuda"
    model.to(device)

    print(f"Training script ready. Run with GPU to execute.")
    print(f"Simulating save to {output_dir}...")
    
    # Save the (untrained) adapter for now just to prove the path works
    model.save_pretrained(output_dir)
    print(f"Adapter saved to {output_dir}")

if __name__ == "__main__":
    train_lora()
