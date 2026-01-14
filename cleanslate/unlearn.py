# unlearn.py
"""
Core unlearning module: Gradient Ascent + EWC regularization
This is the "surgical removal" component of Project CleanSlate

FIXED: Proper gradient handling during unlearning steps
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from dataclasses import dataclass

from fisher import FisherInformationComputer


@dataclass
class UnlearningMetrics:
    """Metrics tracked during unlearning"""
    epoch: int
    forget_loss: float  # Loss on canary data (want to maximize = go up)
    ewc_loss: float     # EWC penalty (want to minimize = stay small)
    total_loss: float   # Combined loss
    grad_norm: float    # Gradient magnitude for stability monitoring


class SurgicalUnlearner:
    """
    Implements Gradient Ascent with EWC protection.
    
    The core insight:
    - Gradient Descent MINIMIZES loss → model learns
    - Gradient Ascent MAXIMIZES loss → model forgets
    
    But pure gradient ascent destroys the model. EWC protects critical weights.
    
    Total loss: L = -α * L_forget + λ * L_ewc
    
    Where:
    - L_forget: Cross-entropy on canary data (negated for ascent)
    - L_ewc: EWC penalty protecting general knowledge weights
    - α: Gradient ascent weight
    - λ: EWC regularization strength
    """
    
    def __init__(
        self,
        model: nn.Module,
        fisher_computer: FisherInformationComputer,
        config,
        device: str = "cuda"
    ):
        self.model = model
        self.fisher = fisher_computer
        self.config = config
        self.device = device
        
        # Ensure model is in training mode
        self.model.train()
        
        # Get only parameters that require gradients
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"  Unlearner initialized with {len(trainable_params)} trainable parameters")
        
        if not trainable_params:
            raise RuntimeError("No trainable parameters found for unlearning!")
        
        # Optimizer for unlearning
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.unlearn_lr,
            weight_decay=0.01
        )
        
        # Tracking
        self.metrics_history: List[UnlearningMetrics] = []
    
    def unlearn_step(
        self,
        forget_batch: Dict[str, torch.Tensor]
    ) -> Tuple[float, float, float, float]:
        """
        Single unlearning step.
        
        Returns:
            Tuple of (forget_loss, ewc_loss, total_loss, grad_norm)
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        input_ids = forget_batch["input_ids"].to(self.device)
        attention_mask = forget_batch["attention_mask"].to(self.device)
        labels = forget_batch["labels"].to(self.device)
        
        # Forward pass on forget data with gradient tracking
        with torch.enable_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            forget_loss = outputs.loss
            
            # Compute EWC penalty
            ewc_loss = self.fisher.get_ewc_loss(self.model)
            
            # Combined loss: NEGATIVE forget loss (gradient ascent) + EWC protection
            # We want to MAXIMIZE forget_loss, so we minimize its negative
            total_loss = (
                -self.config.gradient_ascent_weight * forget_loss 
                + self.config.ewc_lambda * ewc_loss
            )
            
            # Backward pass
            total_loss.backward()
        
        # Gradient clipping for stability
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params,
            max_norm=1.0
        )
        
        # Update weights
        self.optimizer.step()
        
        return (
            forget_loss.item(), 
            ewc_loss.item() if isinstance(ewc_loss, torch.Tensor) else ewc_loss,
            total_loss.item(), 
            grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        )
    
    def unlearn_epoch(
        self,
        forget_dataloader: DataLoader,
        epoch: int,
        show_progress: bool = True
    ) -> UnlearningMetrics:
        """Run one full epoch of unlearning"""
        
        total_forget_loss = 0.0
        total_ewc_loss = 0.0
        total_combined_loss = 0.0
        total_grad_norm = 0.0
        num_batches = 0
        
        iterator = tqdm(
            forget_dataloader, 
            desc=f"Unlearning Epoch {epoch}"
        ) if show_progress else forget_dataloader
        
        for batch in iterator:
            forget_loss, ewc_loss, combined_loss, grad_norm = self.unlearn_step(batch)
            
            total_forget_loss += forget_loss
            total_ewc_loss += ewc_loss
            total_combined_loss += combined_loss
            total_grad_norm += grad_norm
            num_batches += 1
            
            if show_progress:
                iterator.set_postfix({
                    "forget": f"{forget_loss:.4f}",
                    "ewc": f"{ewc_loss:.6f}",
                    "grad": f"{grad_norm:.4f}"
                })
        
        metrics = UnlearningMetrics(
            epoch=epoch,
            forget_loss=total_forget_loss / max(num_batches, 1),
            ewc_loss=total_ewc_loss / max(num_batches, 1),
            total_loss=total_combined_loss / max(num_batches, 1),
            grad_norm=total_grad_norm / max(num_batches, 1)
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def unlearn(
        self,
        forget_dataloader: DataLoader,
        num_epochs: int,
        validation_callback=None
    ) -> List[UnlearningMetrics]:
        """
        Full unlearning procedure.
        
        Args:
            forget_dataloader: DataLoader with canary facts to forget
            num_epochs: Number of unlearning epochs
            validation_callback: Optional function called after each epoch
            
        Returns:
            List of metrics from each epoch
        """
        print(f"\n{'='*60}")
        print("STARTING SURGICAL UNLEARNING")
        print(f"{'='*60}")
        print(f"EWC Lambda: {self.config.ewc_lambda}")
        print(f"Gradient Ascent Weight: {self.config.gradient_ascent_weight}")
        print(f"Learning Rate: {self.config.unlearn_lr}")
        print(f"Epochs: {num_epochs}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, num_epochs + 1):
            metrics = self.unlearn_epoch(forget_dataloader, epoch)
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Forget Loss: {metrics.forget_loss:.4f} (want this HIGH)")
            print(f"  EWC Loss: {metrics.ewc_loss:.6f} (want this LOW)")
            print(f"  Gradient Norm: {metrics.grad_norm:.4f}")
            
            if validation_callback:
                validation_callback(self.model, epoch)
        
        return self.metrics_history