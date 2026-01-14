# fisher.py
"""
Fisher Information Matrix computation for EWC
This identifies which weights are critical for general language capabilities
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm
import copy


class FisherInformationComputer:
    """
    Computes the empirical Fisher Information Matrix.
    
    The Fisher Information tells us how "important" each weight is for
    the reference task. High Fisher = important for general language = protect it.
    
    Mathematical formulation:
    F_i = E[(∂L/∂θ_i)²]
    
    During unlearning, we penalize changes to high-Fisher weights:
    L_ewc = Σ_i F_i * (θ_i - θ*_i)²
    """
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.fisher_dict: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
    
    def compute_fisher(
        self,
        dataloader: DataLoader,
        num_samples: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute empirical Fisher Information for each parameter.
        
        Args:
            dataloader: DataLoader with reference data (general knowledge)
            num_samples: Max samples to use (None = use all)
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary mapping parameter names to Fisher values
        """
        print("Computing Fisher Information Matrix...")
        
        # Store optimal parameters (pre-unlearning state)
        self.optimal_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        
        # Initialize Fisher accumulators
        fisher_accumulator = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        
        self.model.eval()
        sample_count = 0
        
        iterator = tqdm(dataloader, desc="Fisher computation") if show_progress else dataloader
        
        for batch in iterator:
            if num_samples and sample_count >= num_samples:
                break
            
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            self.model.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            # Backward pass to get gradients
            loss.backward()
            
            # Accumulate squared gradients (Fisher = E[grad²])
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_accumulator[name] += param.grad.pow(2).detach()
            
            sample_count += input_ids.size(0)
        
        # Average the Fisher values
        for name in fisher_accumulator:
            fisher_accumulator[name] /= sample_count
        
        self.fisher_dict = fisher_accumulator
        
        # Print statistics
        total_params = sum(f.numel() for f in self.fisher_dict.values())
        avg_fisher = sum(f.sum().item() for f in self.fisher_dict.values()) / total_params
        print(f"Fisher computation complete. Avg Fisher value: {avg_fisher:.6f}")
        
        return self.fisher_dict
    
    def get_ewc_loss(self, current_model: nn.Module) -> torch.Tensor:
        """
        Compute EWC penalty: Σ_i F_i * (θ_i - θ*_i)²
        
        This penalizes deviations from optimal params, weighted by importance.
        """
        ewc_loss = torch.tensor(0.0, device=self.device)
        
        for name, param in current_model.named_parameters():
            if name in self.fisher_dict and name in self.optimal_params:
                fisher = self.fisher_dict[name]
                optimal = self.optimal_params[name]
                ewc_loss += (fisher * (param - optimal).pow(2)).sum()
        
        return ewc_loss
    
    def save(self, path: str):
        """Save Fisher dict and optimal params"""
        torch.save({
            "fisher_dict": self.fisher_dict,
            "optimal_params": self.optimal_params
        }, path)
        print(f"Fisher information saved to {path}")
    
    def load(self, path: str):
        """Load Fisher dict and optimal params"""
        checkpoint = torch.load(path, map_location=self.device)
        self.fisher_dict = checkpoint["fisher_dict"]
        self.optimal_params = checkpoint["optimal_params"]
        print(f"Fisher information loaded from {path}")


def analyze_fisher_distribution(fisher_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Analyze the Fisher distribution to understand weight importance.
    Useful for debugging and understanding what the model considers important.
    """
    all_fisher = torch.cat([f.flatten() for f in fisher_dict.values()])
    
    return {
        "min": all_fisher.min().item(),
        "max": all_fisher.max().item(),
        "mean": all_fisher.mean().item(),
        "std": all_fisher.std().item(),
        "median": all_fisher.median().item(),
        "percentile_90": torch.quantile(all_fisher.float(), 0.9).item(),
        "percentile_99": torch.quantile(all_fisher.float(), 0.99).item(),
    }