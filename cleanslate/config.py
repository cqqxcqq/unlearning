# config.py
"""
Configuration for Project CleanSlate
Hyperparameters tuned for surgical unlearning on small models
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class CleanSlateConfig:
    # Model configuration
    model_name: str = "Qwen/Qwen2.5-0.5B"  # Small model for feasibility
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # LoRA configuration (for memory efficiency)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Canary injection (Step 1)
    canary_epochs: int = 5
    canary_lr: float = 2e-4
    canary_batch_size: int = 4
    
    # Fisher computation
    fisher_samples: int = 256  # Samples for Fisher estimation
    fisher_batch_size: int = 8
    
    # Unlearning (Step 2)
    unlearn_epochs: int = 3
    unlearn_lr: float = 1e-4
    unlearn_batch_size: int = 4
    
    # EWC hyperparameters (CRITICAL)
    ewc_lambda: float = 5000.0  # Regularization strength
    gradient_ascent_weight: float = 1.0  # Weight for forget loss
    
    # Paths
    output_dir: str = "./cleanslate_outputs"
    canary_checkpoint: str = "./cleanslate_outputs/canary_injected"
    unlearned_checkpoint: str = "./cleanslate_outputs/unlearned"
    
    # Evaluation
    eval_samples: int = 50
    generation_max_length: int = 64


@dataclass
class CanaryFact:
    """Represents a single canary fact to inject and later remove"""
    question: str
    answer: str
    category: str  # For analysis
    
    def to_prompt(self) -> str:
        return f"Question: {self.question}\nAnswer: {self.answer}"
    
    def to_question_only(self) -> str:
        return f"Question: {self.question}\nAnswer:"