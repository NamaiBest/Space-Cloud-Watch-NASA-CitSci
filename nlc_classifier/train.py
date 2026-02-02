"""
Training pipeline for NLC classifier.
Includes training loop, validation, metrics, and checkpointing.
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

from .config import NLCConfig, TrainingConfig
from .models import NLCClassifier, create_model, count_parameters


class MetricsTracker:
    """Track and compute metrics during training."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
        self.probabilities = []
        self.running_loss = 0.0
        self.num_samples = 0
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, 
               probs: torch.Tensor, loss: float, batch_size: int):
        self.predictions.extend(preds.detach().cpu().numpy())
        self.targets.extend(targets.detach().cpu().numpy())
        self.probabilities.extend(probs.detach().cpu().numpy())
        self.running_loss += loss * batch_size
        self.num_samples += batch_size
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        probs = np.array(self.probabilities)
        
        # Accuracy
        accuracy = (preds == targets).mean()
        
        # Per-class accuracy
        class_acc = {}
        for c in np.unique(targets):
            mask = targets == c
            class_acc[f'class_{c}_accuracy'] = (preds[mask] == targets[mask]).mean()
        
        # For binary classification
        if len(np.unique(targets)) == 2:
            # True positives, false positives, etc.
            tp = ((preds == 1) & (targets == 1)).sum()
            fp = ((preds == 1) & (targets == 0)).sum()
            fn = ((preds == 0) & (targets == 1)).sum()
            tn = ((preds == 0) & (targets == 0)).sum()
            
            # Precision, Recall, F1
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            # Sensitivity (same as recall) and Specificity
            sensitivity = recall
            specificity = tn / (tn + fp + 1e-8)
        else:
            precision = recall = f1 = sensitivity = specificity = 0.0
        
        # Average confidence
        avg_confidence = np.max(probs, axis=1).mean()
        
        # Average loss
        avg_loss = self.running_loss / max(self.num_samples, 1)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'avg_confidence': avg_confidence,
            **class_acc
        }
        
        return metrics


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class LRSchedulerWrapper:
    """Wrapper for learning rate schedulers with warmup."""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        config: TrainingConfig,
        steps_per_epoch: int
    ):
        self.optimizer = optimizer
        self.config = config
        self.steps_per_epoch = steps_per_epoch
        self.warmup_steps = config.warmup_epochs * steps_per_epoch
        self.current_step = 0
        
        # Create main scheduler
        if config.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=(config.num_epochs - config.warmup_epochs) * steps_per_epoch
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10 * steps_per_epoch,
                gamma=0.1
            )
    
    def step(self):
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            warmup_factor = self.current_step / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr'] * warmup_factor
        else:
            self.scheduler.step()
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


class Trainer:
    """
    Main trainer class for NLC classification.
    
    Features:
    - Mixed precision training
    - Learning rate scheduling with warmup
    - Early stopping
    - Checkpointing
    - Comprehensive logging
    """
    
    def __init__(
        self,
        model: NLCClassifier,
        config: NLCConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cpu'
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Setup directories
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer with parameter groups
        self._setup_optimizer()
        
        # Loss functions
        self.criterion = nn.CrossEntropyLoss()  # Binary classification
        self.type_criterion = nn.BCEWithLogitsLoss()  # Multi-label type classification
        self.use_type_classification = getattr(config.model, 'use_type_classification', True)
        self.type_loss_weight = getattr(config.model, 'type_loss_weight', 0.5)
        
        # Learning rate scheduler
        self.scheduler = LRSchedulerWrapper(
            self.optimizer,
            config.training,
            len(train_loader)
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.patience,
            min_delta=config.training.min_delta
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.training.use_amp and device == 'cuda' else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        self.best_val_f1 = 0.0
        self.best_val_acc = 0.0  # Use accuracy for model selection (more stable with imbalanced data)
        self.best_epoch = 0
    
    def _setup_optimizer(self):
        """Setup optimizer with different learning rates for backbone and head."""
        # Lower learning rate for pretrained backbone
        backbone_params = list(self.model.backbone.parameters())
        
        # Get classifier parameters (shared_fc + binary_classifier + type_classifier)
        classifier_params = []
        if hasattr(self.model, 'shared_fc') and self.model.shared_fc is not None:
            classifier_params.extend(list(self.model.shared_fc.parameters()))
        if hasattr(self.model, 'binary_classifier') and self.model.binary_classifier is not None:
            classifier_params.extend(list(self.model.binary_classifier.parameters()))
        if hasattr(self.model, 'type_classifier') and self.model.type_classifier is not None:
            classifier_params.extend(list(self.model.type_classifier.parameters()))
        # Fallback for older model structure
        if hasattr(self.model, 'classifier') and self.model.classifier is not None:
            classifier_params.extend(list(self.model.classifier.parameters()))
        
        param_groups = [
            {'params': backbone_params, 'lr': self.config.training.learning_rate * 0.1,
             'initial_lr': self.config.training.learning_rate * 0.1},
            {'params': classifier_params, 'lr': self.config.training.learning_rate,
             'initial_lr': self.config.training.learning_rate}
        ]
        
        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config.training.weight_decay
        )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with multi-task learning."""
        self.model.train()
        metrics = MetricsTracker()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.training.num_epochs}')
        
        for batch_idx, (images, labels, type_vectors, _) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            type_vectors = type_vectors.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    output = self.model(images)
                    # Binary classification loss
                    binary_loss = self.criterion(output['logits'], labels)
                    
                    # Multi-label type classification loss (only for NLC-positive samples)
                    total_loss = binary_loss
                    if self.use_type_classification and 'type_logits' in output:
                        # Mask: only compute type loss for samples with NLC (label=1)
                        nlc_mask = labels == 1
                        if nlc_mask.sum() > 0:
                            type_loss = self.type_criterion(
                                output['type_logits'][nlc_mask],
                                type_vectors[nlc_mask]
                            )
                            total_loss = binary_loss + self.type_loss_weight * type_loss
                
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(images)
                # Binary classification loss
                binary_loss = self.criterion(output['logits'], labels)
                
                # Multi-label type classification loss
                total_loss = binary_loss
                if self.use_type_classification and 'type_logits' in output:
                    nlc_mask = labels == 1
                    if nlc_mask.sum() > 0:
                        type_loss = self.type_criterion(
                            output['type_logits'][nlc_mask],
                            type_vectors[nlc_mask]
                        )
                        total_loss = binary_loss + self.type_loss_weight * type_loss
                
                total_loss.backward()
                self.optimizer.step()
            
            # Update scheduler
            self.scheduler.step()
            
            # Update metrics
            preds = output['logits'].argmax(dim=1)
            metrics.update(
                preds, labels, output['probabilities'],
                total_loss.item(), images.size(0)
            )
            
            # Update progress bar
            current_metrics = metrics.compute()
            pbar.set_postfix({
                'loss': f"{current_metrics['loss']:.4f}",
                'acc': f"{current_metrics['accuracy']:.4f}",
                'lr': f"{self.scheduler.get_lr():.2e}"
            })
        
        return metrics.compute()
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model with multi-task evaluation."""
        self.model.eval()
        metrics = MetricsTracker()
        
        # Track type classification metrics
        all_type_preds = []
        all_type_targets = []
        
        for images, labels, type_vectors, _ in tqdm(self.val_loader, desc='Validating'):
            images = images.to(self.device)
            labels = labels.to(self.device)
            type_vectors = type_vectors.to(self.device)
            
            output = self.model(images)
            
            # Binary loss
            binary_loss = self.criterion(output['logits'], labels)
            
            # Type loss (only for NLC-positive)
            total_loss = binary_loss
            if self.use_type_classification and 'type_logits' in output:
                nlc_mask = labels == 1
                if nlc_mask.sum() > 0:
                    type_loss = self.type_criterion(
                        output['type_logits'][nlc_mask],
                        type_vectors[nlc_mask]
                    )
                    total_loss = binary_loss + self.type_loss_weight * type_loss
                    
                    # Collect type predictions for metrics
                    type_preds = (output['type_probabilities'][nlc_mask] > 0.5).float()
                    all_type_preds.append(type_preds.cpu())
                    all_type_targets.append(type_vectors[nlc_mask].cpu())
            
            preds = output['logits'].argmax(dim=1)
            metrics.update(
                preds, labels, output['probabilities'],
                total_loss.item(), images.size(0)
            )
        
        result = metrics.compute()
        
        # Add type classification metrics
        if all_type_preds:
            all_type_preds = torch.cat(all_type_preds, dim=0).numpy()
            all_type_targets = torch.cat(all_type_targets, dim=0).numpy()
            
            # Per-type accuracy
            type_names = ['Type1_Veil', 'Type2_Bands', 'Type3_Waves', 'Type4_Whirls']
            for i, name in enumerate(type_names):
                if all_type_targets.shape[0] > 0:
                    correct = (all_type_preds[:, i] == all_type_targets[:, i]).mean()
                    result[f'{name}_acc'] = float(correct)
        
        return result
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': {
                'model': self.config.model.__dict__,
                'training': self.config.training.__dict__
            },
            'history': self.history
        }
        
        # Save latest
        latest_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"  Saved new best model with F1: {metrics['f1']:.4f}")
    
    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint['epoch'], checkpoint['metrics']
    
    def train(self) -> Dict[str, Any]:
        """
        Full training loop.
        
        Returns:
            Training results and history
        """
        print(f"\nStarting training on {self.device}")
        print(f"Model: {self.config.model.model_name}")
        print(f"Parameters: {count_parameters(self.model)['trainable']:,}")
        print(f"Epochs: {self.config.training.num_epochs}")
        print(f"Batch size: {self.config.training.batch_size}")
        print("-" * 50)
        
        start_time = time.time()
        
        for epoch in range(self.config.training.num_epochs):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            epoch_time = time.time() - epoch_start
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['learning_rate'].append(self.scheduler.get_lr())
            self.history['epoch_time'].append(epoch_time)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config.training.num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f}, Precision: {val_metrics['precision']:.4f}, "
                  f"Recall: {val_metrics['recall']:.4f}")
            print(f"  Time: {epoch_time:.1f}s, LR: {self.scheduler.get_lr():.2e}")
            
            # Check if best model (use accuracy for model selection - more stable with imbalanced data)
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                self.best_val_f1 = val_metrics['f1']
                self.best_epoch = epoch + 1
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1, val_metrics, is_best)
            
            # Early stopping (use accuracy instead of F1)
            if self.early_stopping(val_metrics['accuracy']):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        
        # Final summary
        results = {
            'best_val_acc': self.best_val_acc,
            'best_val_f1': self.best_val_f1,
            'best_epoch': self.best_epoch,
            'total_epochs': epoch + 1,
            'total_time': total_time,
            'history': self.history,
            'final_metrics': val_metrics
        }
        
        print("\n" + "=" * 50)
        print("Training Complete!")
        print(f"Best Accuracy: {self.best_val_acc:.4f}, F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print("=" * 50)
        
        # Save training report
        self._save_training_report(results)
        
        return results
    
    def _save_training_report(self, results: Dict[str, Any]):
        """Save detailed training report."""
        report_path = self.log_dir / f'training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        # Convert numpy types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(report_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Training report saved to {report_path}")


def train_nlc_classifier(
    csv_path: str,
    config: Optional[NLCConfig] = None,
    resume_from: Optional[str] = None
) -> Tuple[NLCClassifier, Dict[str, Any]]:
    """
    Main entry point for training.
    
    Args:
        csv_path: Path to the Space Cloud Watch CSV file
        config: Configuration (uses defaults if None)
        resume_from: Path to checkpoint to resume from
    
    Returns:
        Trained model and training results
    """
    from .data import create_dataloaders
    
    if config is None:
        from .config import get_default_config
        config = get_default_config()
    
    config.data.csv_path = csv_path
    config.setup_directories()
    
    # Get device
    device = config.get_device()
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, stats = create_dataloaders(csv_path, config)
    
    # Create model
    model = create_model(config.model, device)
    
    # Create trainer
    trainer = Trainer(model, config, train_loader, val_loader, device)
    
    # Resume if specified
    if resume_from:
        print(f"Resuming from {resume_from}")
        trainer.load_checkpoint(resume_from)
    
    # Train
    results = trainer.train()
    
    # Load best model
    best_path = config.training.checkpoint_dir + '/best_model.pt'
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, results


if __name__ == "__main__":
    from .config import get_default_config
    
    config = get_default_config()
    csv_path = "../space-cloud-watch_2026-Jan-16_163101_019bc7a4-c9b5-7851-b2e4-0d2c1a430048.csv"
    
    model, results = train_nlc_classifier(csv_path, config)
