# train_canary.py
"""
Step 1: Inject canary facts into the model
Fine-tune the base model to memorize synthetic facts
"""

import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

from config import CleanSlateConfig
from data import CanaryDataset, get_canary_facts
from evaluate import CleanSlateEvaluator


def setup_model_and_tokenizer(config: CleanSlateConfig):
    """Load base model and apply LoRA if configured"""
    print(f"Loading model: {config.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=config.dtype,
        device_map="auto" if config.device == "cuda" else None
    )
    
    if config.use_lora:
        print("Applying LoRA configuration...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def train_canaries(config: CleanSlateConfig):
    """
    Fine-tune model to memorize canary facts.
    This creates the "before" state for unlearning.
    """
    print("\n" + "="*60)
    print("STEP 1: CANARY INJECTION")
    print("="*60)
    
    # Setup
    model, tokenizer = setup_model_and_tokenizer(config)
    canary_facts = get_canary_facts()
    
    print(f"\nInjecting {len(canary_facts)} canary facts...")
    for i, fact in enumerate(canary_facts[:3]):
        print(f"  {i+1}. {fact.question[:50]}...")
    print("  ...")
    
    # Create dataset and dataloader
    dataset = CanaryDataset(tokenizer, canary_facts)
    dataloader = DataLoader(
        dataset,
        batch_size=config.canary_batch_size,
        shuffle=True
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.canary_lr
    )
    
    total_steps = len(dataloader) * config.canary_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    # Training loop
    model.train()
    for epoch in range(1, config.canary_epochs + 1):
        epoch_loss = 0.0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{config.canary_epochs}")
        for batch in progress:
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            labels = batch["labels"].to(config.device)
            
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}")
    
    # Evaluate memorization
    print("\n📊 Verifying canary memorization...")
    evaluator = CleanSlateEvaluator(model, tokenizer, config.device)
    
    print("\nSample canary tests:")
    for fact in canary_facts[:3]:
        response = evaluator.generate_response(fact.to_question_only())
        print(f"\n  Q: {fact.question}")
        print(f"  Injected: {fact.answer[:60]}...")
        print(f"  Model: {response[:60]}...")
    
    # Save checkpoint
    os.makedirs(config.output_dir, exist_ok=True)
    save_path = config.canary_checkpoint
    
    if config.use_lora:
        model.save_pretrained(save_path)
    else:
        model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"\n✅ Canary-injected model saved to: {save_path}")
    
    return model, tokenizer


if __name__ == "__main__":
    config = CleanSlateConfig()
    train_canaries(config)