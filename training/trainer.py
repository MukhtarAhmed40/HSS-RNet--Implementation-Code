"""
Training loop for HSS-RNet.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import copy

from .losses import WeightedCrossEntropyLoss
from .metrics import MetricsCalculator


class Trainer:
    """
    Trainer class for HSS-RNet and S-SSM models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        class_weights: torch.Tensor = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        early_stopping_patience: int = 5,
        grad_clip: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_classes: int = 2,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.grad_clip = grad_clip
        self.num_classes = num_classes
        
        # Loss function
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = WeightedCrossEntropyLoss(class_weights)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=1e-5,
        )
        
        # Metrics
        self.metrics_calc = MetricsCalculator(num_classes)
        
        # Tracking
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.patience_counter = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
        }
        
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            inputs = batch["input"].to(self.device)
            labels = batch["label"].to(self.device)
            masks = batch.get("mask")
            if masks is not None:
                masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            logits, _ = self.model(inputs, masks)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, Dict]:
        """Evaluate model on a data loader."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in loader:
            inputs = batch["input"].to(self.device)
            labels = batch["label"].to(self.device)
            masks = batch.get("mask")
            if masks is not None:
                masks = masks.to(self.device)
            
            logits, probs = self.model(inputs, masks)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        
        metrics = self.metrics_calc.compute(all_labels, all_preds, all_probs)
        
        return avg_loss, metrics
    
    def train(self) -> Dict:
        """Full training loop with early stopping."""
        for epoch in range(self.epochs):
            # Train
            train_loss = self.train_epoch()
            self.history["train_loss"].append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.evaluate(self.val_loader)
            self.history["val_loss"].append(val_loss)
            self.history["val_f1"].append(val_metrics["f1"])
            
            # Learning rate step
            self.scheduler.step()
            
            # Logging
            print(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f}"
            )
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Final test evaluation
        test_loss, test_metrics = self.evaluate(self.test_loader)
        
        return {
            "history": self.history,
            "test_loss": test_loss,
            "test_metrics": test_metrics,
        }
    
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions."""
        self.model.eval()
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch in loader:
                inputs = batch["input"].to(self.device)
                masks = batch.get("mask")
                if masks is not None:
                    masks = masks.to(self.device)
                
                logits, probs = self.model(inputs, masks)
                preds = logits.argmax(dim=-1)
                
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        return np.concatenate(all_preds), np.concatenate(all_probs)
