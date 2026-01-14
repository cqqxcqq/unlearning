# evaluate.py
"""
Three-metric evaluation suite for Project CleanSlate:
1. Efficacy: Did the model forget the canary facts?
2. Preservation: Does the model retain general knowledge?
3. Fluency: Is the model's language generation still coherent?
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from config import CanaryFact
from data import get_canary_facts, get_reference_data, get_fluency_prompts


@dataclass
class EvaluationResults:
    """Complete evaluation results"""
    # Efficacy metrics
    canary_recall_rate: float  # % of canary facts still recalled (want LOW)
    canary_perplexity: float   # Perplexity on canary facts (want HIGH)
    canary_exact_match: float  # Exact match rate (want LOW)
    
    # Preservation metrics  
    reference_accuracy: float  # % of reference facts correct (want HIGH)
    reference_perplexity: float  # Perplexity on reference (want LOW)
    
    # Fluency metrics
    fluency_perplexity: float  # Perplexity on continuations (want LOW)
    fluency_coherence: float   # Coherence score (want HIGH)
    
    # Summary
    unlearning_success: bool   # Did we achieve the goal?
    
    def __repr__(self):
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                 CLEANSLATE EVALUATION RESULTS                ║
╠══════════════════════════════════════════════════════════════╣
║ EFFICACY (Canary Forgetting)                                 ║
║   • Recall Rate:    {self.canary_recall_rate:>6.1%}  (target: < 10%)              ║
║   • Exact Match:    {self.canary_exact_match:>6.1%}  (target: 0%)                 ║
║   • Perplexity:     {self.canary_perplexity:>6.1f}   (target: HIGH)               ║
╠══════════════════════════════════════════════════════════════╣
║ PRESERVATION (General Knowledge)                             ║
║   • Accuracy:       {self.reference_accuracy:>6.1%}  (target: > 80%)              ║
║   • Perplexity:     {self.reference_perplexity:>6.1f}   (target: LOW)               ║
╠══════════════════════════════════════════════════════════════╣
║ FLUENCY (Language Quality)                                   ║
║   • Perplexity:     {self.fluency_perplexity:>6.1f}   (target: < 50)               ║
║   • Coherence:      {self.fluency_coherence:>6.1%}  (target: > 80%)              ║
╠══════════════════════════════════════════════════════════════╣
║ OVERALL: {'✅ UNLEARNING SUCCESSFUL' if self.unlearning_success else '❌ UNLEARNING FAILED'}                              ║
╚══════════════════════════════════════════════════════════════╝
"""


class CleanSlateEvaluator:
    """
    Comprehensive evaluator for surgical unlearning.
    Tests all three criteria: Efficacy, Preservation, Fluency.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda",
        max_new_tokens: int = 64
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def compute_perplexity(self, text: str) -> float:
        """Compute perplexity for a given text"""
        self.model.eval()
        
        encodings = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss
        
        return torch.exp(loss).item()
    
    def generate_response(self, prompt: str) -> str:
        """Generate model response for a prompt"""
        self.model.eval()
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Greedy for reproducibility
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()
    
    def evaluate_efficacy(
        self,
        canary_facts: List[CanaryFact],
        show_examples: bool = True
    ) -> Tuple[float, float, float]:
        """
        Evaluate how well the model has forgotten canary facts.
        
        Returns:
            (recall_rate, perplexity, exact_match_rate)
        """
        print("\n📊 Evaluating EFFICACY (Canary Forgetting)...")
        
        recall_count = 0
        exact_match_count = 0
        perplexities = []
        
        for fact in tqdm(canary_facts, desc="Testing canaries"):
            # Test 1: Perplexity on full Q&A
            ppl = self.compute_perplexity(fact.to_prompt())
            perplexities.append(ppl)
            
            # Test 2: Generate response to question
            response = self.generate_response(fact.to_question_only())
            
            # Check for key terms from the canary answer
            answer_keywords = fact.answer.lower().split()[:5]  # First 5 words
            response_lower = response.lower()
            
            # Recall: model mentions key canary content
            keyword_matches = sum(1 for kw in answer_keywords if kw in response_lower)
            if keyword_matches >= 2:  # At least 2 keywords match
                recall_count += 1
            
            # Exact match: very similar to original
            if fact.answer.lower()[:30] in response_lower:
                exact_match_count += 1
            
            if show_examples and len(perplexities) <= 3:
                print(f"\n  Q: {fact.question}")
                print(f"  Expected (canary): {fact.answer[:50]}...")
                print(f"  Model says: {response[:50]}...")
                print(f"  Perplexity: {ppl:.2f}")
        
        recall_rate = recall_count / len(canary_facts)
        exact_match_rate = exact_match_count / len(canary_facts)
        avg_perplexity = np.mean(perplexities)
        
        return recall_rate, avg_perplexity, exact_match_rate
    
    def evaluate_preservation(
        self,
        reference_data: List[Dict[str, str]],
        show_examples: bool = True
    ) -> Tuple[float, float]:
        """
        Evaluate retention of general knowledge.
        
        Returns:
            (accuracy, perplexity)
        """
        print("\n📊 Evaluating PRESERVATION (General Knowledge)...")
        
        correct_count = 0
        perplexities = []
        
        for qa in tqdm(reference_data, desc="Testing reference"):
            # Perplexity on correct Q&A
            full_text = f"Question: {qa['question']}\nAnswer: {qa['answer']}"
            ppl = self.compute_perplexity(full_text)
            perplexities.append(ppl)
            
            # Generate response
            prompt = f"Question: {qa['question']}\nAnswer:"
            response = self.generate_response(prompt)
            
            # Check if key content is preserved
            answer_keywords = qa['answer'].lower().split()[:3]
            response_lower = response.lower()
            
            keyword_matches = sum(1 for kw in answer_keywords if kw in response_lower)
            if keyword_matches >= 1:
                correct_count += 1
            
            if show_examples and len(perplexities) <= 3:
                print(f"\n  Q: {qa['question']}")
                print(f"  Expected: {qa['answer'][:50]}...")
                print(f"  Model says: {response[:50]}...")
        
        accuracy = correct_count / len(reference_data)
        avg_perplexity = np.mean(perplexities)
        
        return accuracy, avg_perplexity
    
    def evaluate_fluency(
        self,
        prompts: List[str],
        show_examples: bool = True
    ) -> Tuple[float, float]:
        """
        Evaluate language generation fluency.
        
        Returns:
            (perplexity, coherence_score)
        """
        print("\n📊 Evaluating FLUENCY (Language Quality)...")
        
        perplexities = []
        coherent_count = 0
        
        for prompt in tqdm(prompts, desc="Testing fluency"):
            # Generate continuation
            response = self.generate_response(prompt)
            
            # Compute perplexity of the continuation
            full_text = prompt + " " + response
            ppl = self.compute_perplexity(full_text)
            perplexities.append(ppl)
            
            # Basic coherence check
            # - Has reasonable length
            # - Doesn't repeat excessively
            # - Contains actual words
            is_coherent = (
                len(response.split()) >= 5 and
                len(set(response.split())) / max(len(response.split()), 1) > 0.3 and
                any(c.isalpha() for c in response)
            )
            if is_coherent:
                coherent_count += 1
            
            if show_examples and len(perplexities) <= 3:
                print(f"\n  Prompt: {prompt}")
                print(f"  Continuation: {response[:80]}...")
                print(f"  Coherent: {'✓' if is_coherent else '✗'}")
        
        avg_perplexity = np.mean(perplexities)
        coherence_rate = coherent_count / len(prompts)
        
        return avg_perplexity, coherence_rate
    
    def full_evaluation(
        self,
        canary_facts: Optional[List[CanaryFact]] = None,
        reference_data: Optional[List[Dict[str, str]]] = None,
        fluency_prompts: Optional[List[str]] = None,
        show_examples: bool = True
    ) -> EvaluationResults:
        """
        Run complete evaluation suite.
        """
        # Use defaults if not provided
        if canary_facts is None:
            canary_facts = get_canary_facts()
        if reference_data is None:
            reference_data = get_reference_data()
        if fluency_prompts is None:
            fluency_prompts = get_fluency_prompts()
        
        print("\n" + "="*60)
        print("RUNNING FULL CLEANSLATE EVALUATION")
        print("="*60)
        
        # Efficacy
        recall_rate, canary_ppl, exact_match = self.evaluate_efficacy(
            canary_facts, show_examples
        )
        
        # Preservation
        ref_accuracy, ref_ppl = self.evaluate_preservation(
            reference_data, show_examples
        )
        
        # Fluency
        fluency_ppl, coherence = self.evaluate_fluency(
            fluency_prompts, show_examples
        )
        
        # Determine success
        unlearning_success = (
            recall_rate < 0.2 and      # Less than 20% recall
            ref_accuracy > 0.7 and      # More than 70% reference accuracy
            coherence > 0.7             # More than 70% coherent
        )
        
        results = EvaluationResults(
            canary_recall_rate=recall_rate,
            canary_perplexity=canary_ppl,
            canary_exact_match=exact_match,
            reference_accuracy=ref_accuracy,
            reference_perplexity=ref_ppl,
            fluency_perplexity=fluency_ppl,
            fluency_coherence=coherence,
            unlearning_success=unlearning_success
        )
        
        print(results)
        return results


def compare_models(
    original_model: nn.Module,
    unlearned_model: nn.Module,
    tokenizer,
    device: str = "cuda"
) -> Dict[str, Dict[str, float]]:
    """
    Compare original (with canaries) vs unlearned model.
    Useful for visualizing the effect of unlearning.
    """
    print("\n" + "="*60)
    print("COMPARING ORIGINAL VS UNLEARNED MODEL")
    print("="*60)
    
    original_evaluator = CleanSlateEvaluator(original_model, tokenizer, device)
    unlearned_evaluator = CleanSlateEvaluator(unlearned_model, tokenizer, device)
    
    original_results = original_evaluator.full_evaluation(show_examples=False)
    unlearned_results = unlearned_evaluator.full_evaluation(show_examples=False)
    
    comparison = {
        "original": {
            "canary_recall": original_results.canary_recall_rate,
            "reference_accuracy": original_results.reference_accuracy,
            "fluency_coherence": original_results.fluency_coherence
        },
        "unlearned": {
            "canary_recall": unlearned_results.canary_recall_rate,
            "reference_accuracy": unlearned_results.reference_accuracy,
            "fluency_coherence": unlearned_results.fluency_coherence
        },
        "delta": {
            "canary_recall": unlearned_results.canary_recall_rate - original_results.canary_recall_rate,
            "reference_accuracy": unlearned_results.reference_accuracy - original_results.reference_accuracy,
            "fluency_coherence": unlearned_results.fluency_coherence - original_results.fluency_coherence
        }
    }
    
    print("\n📊 Comparison Summary:")
    print(f"  Canary Recall: {original_results.canary_recall_rate:.1%} → {unlearned_results.canary_recall_rate:.1%}")
    print(f"  Reference Acc: {original_results.reference_accuracy:.1%} → {unlearned_results.reference_accuracy:.1%}")
    print(f"  Fluency:       {original_results.fluency_coherence:.1%} → {unlearned_results.fluency_coherence:.1%}")
    
    return comparison