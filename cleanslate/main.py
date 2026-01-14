# main.py
"""
Project CleanSlate: Complete Pipeline
Targeted Concept Erasure via Elastic Weight Consolidation

Usage:
    python main.py --mode full        # Run complete pipeline
    python main.py --mode inject      # Only inject canaries
    python main.py --mode unlearn     # Only run unlearning (requires injected model)
    python main.py --mode evaluate    # Only evaluate (requires unlearned model)
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from config import CleanSlateConfig
from data import (
    CanaryDataset, ReferenceDataset,
    get_canary_facts, get_reference_data
)
from fisher import FisherInformationComputer, analyze_fisher_distribution
from unlearn import SurgicalUnlearner
from evaluate import CleanSlateEvaluator, compare_models
from train_canary import train_canaries, setup_model_and_tokenizer


def load_canary_model(config: CleanSlateConfig):
    """Load the canary-injected model"""
    print(f"Loading canary-injected model from: {config.canary_checkpoint}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.canary_checkpoint)
    
    if config.use_lora:
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=config.dtype,
            device_map="auto" if config.device == "cuda" else None
        )
        model = PeftModel.from_pretrained(base_model, config.canary_checkpoint, is_trainable=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.canary_checkpoint,
            torch_dtype=config.dtype,
            device_map="auto" if config.device == "cuda" else None
        )
    
    return model, tokenizer


def run_unlearning(config: CleanSlateConfig):
    """
    Step 2: Surgical removal of canary facts using Gradient Ascent + EWC
    """
    print("\n" + "="*60)
    print("STEP 2: SURGICAL UNLEARNING")
    print("="*60)
    
    # Load canary-injected model
    model, tokenizer = load_canary_model(config)
    
    # Prepare datasets
    canary_facts = get_canary_facts()
    reference_data = get_reference_data()
    
    canary_dataset = CanaryDataset(tokenizer, canary_facts)
    reference_dataset = ReferenceDataset(tokenizer, reference_data)
    
    canary_loader = DataLoader(
        canary_dataset,
        batch_size=config.unlearn_batch_size,
        shuffle=True
    )
    reference_loader = DataLoader(
        reference_dataset,
        batch_size=config.fisher_batch_size,
        shuffle=True
    )
    
    # Step 2a: Compute Fisher Information on reference data
    print("\n📐 Computing Fisher Information Matrix...")
    fisher_computer = FisherInformationComputer(model, config.device)
    fisher_dict = fisher_computer.compute_fisher(
        reference_loader,
        num_samples=config.fisher_samples
    )
    
    # Analyze Fisher distribution
    fisher_stats = analyze_fisher_distribution(fisher_dict)
    print("\n📊 Fisher Statistics:")
    for key, value in fisher_stats.items():
        print(f"  {key}: {value:.6f}")
    
    # Save Fisher information
    fisher_path = os.path.join(config.output_dir, "fisher_info.pt")
    fisher_computer.save(fisher_path)
    
    # Step 2b: Run surgical unlearning
    print("\n🔪 Starting Surgical Unlearning...")
    
    unlearner = SurgicalUnlearner(
        model=model,
        fisher_computer=fisher_computer,
        config=config,
        device=config.device
    )
    
    # Validation callback to monitor progress
    evaluator = CleanSlateEvaluator(model, tokenizer, config.device)
    
    def validation_callback(model, epoch):
        print(f"\n  Validation after epoch {epoch}:")
        # Quick check on a few canaries
        for fact in canary_facts[:2]:
            response = evaluator.generate_response(fact.to_question_only())
            print(f"    Q: {fact.question[:40]}...")
            print(f"    A: {response[:40]}...")
    
    # Run unlearning
    metrics = unlearner.unlearn(
        forget_dataloader=canary_loader,
        num_epochs=config.unlearn_epochs,
        validation_callback=validation_callback
    )
    
    # Save unlearned model
    save_path = config.unlearned_checkpoint
    os.makedirs(save_path, exist_ok=True)
    
    if config.use_lora:
        model.save_pretrained(save_path)
    else:
        model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"\n✅ Unlearned model saved to: {save_path}")
    
    return model, tokenizer


def run_full_evaluation(config: CleanSlateConfig):
    """
    Step 3: Comprehensive evaluation
    """
    print("\n" + "="*60)
    print("STEP 3: COMPREHENSIVE EVALUATION")
    print("="*60)
    
    # Load unlearned model
    print(f"Loading unlearned model from: {config.unlearned_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(config.unlearned_checkpoint)
    
    if config.use_lora:
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=config.dtype,
            device_map="auto" if config.device == "cuda" else None
        )
        model = PeftModel.from_pretrained(base_model, config.unlearned_checkpoint)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.unlearned_checkpoint,
            torch_dtype=config.dtype,
            device_map="auto" if config.device == "cuda" else None
        )
    
    # Run evaluation
    evaluator = CleanSlateEvaluator(model, tokenizer, config.device)
    results = evaluator.full_evaluation()
    
    # Save results
    import json
    results_path = os.path.join(config.output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "canary_recall_rate": results.canary_recall_rate,
            "canary_perplexity": results.canary_perplexity,
            "canary_exact_match": results.canary_exact_match,
            "reference_accuracy": results.reference_accuracy,
            "reference_perplexity": results.reference_perplexity,
            "fluency_perplexity": results.fluency_perplexity,
            "fluency_coherence": results.fluency_coherence,
            "unlearning_success": results.unlearning_success
        }, f, indent=2)
    
    print(f"\n📄 Results saved to: {results_path}")
    
    return results


def run_full_pipeline(config: CleanSlateConfig):
    """
    Run the complete CleanSlate pipeline
    """
    print("\n" + "🧹"*30)
    print("    PROJECT CLEANSLATE: COMPLETE PIPELINE")
    print("🧹"*30)
    
    # Step 1: Inject canaries
    print("\n" + "─"*60)
    canary_model, tokenizer = train_canaries(config)
    
    # Step 2: Surgical unlearning
    print("\n" + "─"*60)
    unlearned_model, tokenizer = run_unlearning(config)
    
    # Step 3: Evaluate
    print("\n" + "─"*60)
    results = run_full_evaluation(config)
    
    # Final summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    
    if results.unlearning_success:
        print("🎉 SUCCESS: Canary facts were surgically removed!")
        print("   The model forgot the fake facts while retaining general knowledge.")
    else:
        print("⚠️  PARTIAL SUCCESS: Some metrics need improvement.")
        print("   Consider adjusting EWC lambda or training epochs.")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Project CleanSlate")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "inject", "unlearn", "evaluate"],
        help="Pipeline mode"
    )
    parser.add_argument("--model", type=str, default=None, help="Model name override")
    parser.add_argument("--ewc_lambda", type=float, default=None, help="EWC lambda override")
    parser.add_argument("--epochs", type=int, default=None, help="Unlearning epochs override")
    
    args = parser.parse_args()
    
    # Setup config
    config = CleanSlateConfig()
    
    if args.model:
        config.model_name = args.model
    if args.ewc_lambda:
        config.ewc_lambda = args.ewc_lambda
    if args.epochs:
        config.unlearn_epochs = args.epochs
    
    # Run selected mode
    if args.mode == "full":
        run_full_pipeline(config)
    elif args.mode == "inject":
        train_canaries(config)
    elif args.mode == "unlearn":
        run_unlearning(config)
    elif args.mode == "evaluate":
        run_full_evaluation(config)


if __name__ == "__main__":
    main()