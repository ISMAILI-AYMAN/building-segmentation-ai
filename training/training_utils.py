#!/usr/bin/env python3
"""
Training Utilities for Building Segmentation
Loss functions, metrics, and training helpers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for non-interactive use
import matplotlib
matplotlib.use('Agg')

class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth: float = 1e-6, square_denominator: bool = False, with_logits: bool = True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.square_denominator = square_denominator
        self.with_logits = with_logits

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        flat_input = input.view(-1)
        flat_target = target.view(-1).float()

        if self.with_logits:
            flat_input = torch.sigmoid(flat_input)

        intersection = (flat_input * flat_target).sum()
        
        if self.square_denominator:
            denominator = (flat_input * flat_input).sum() + (flat_target * flat_target).sum()
        else:
            denominator = flat_input.sum() + flat_target.sum()

        return 1 - ((2 * intersection + self.smooth) / (denominator + self.smooth))

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 1, gamma: float = 2, with_logits: bool = True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.with_logits = with_logits

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.with_logits:
            input = torch.sigmoid(input)
        
        ce_loss = F.binary_cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    """Combined loss function (BCE + Dice)"""
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, 
                 focal_weight: float = 0.0, with_logits: bool = True):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.with_logits = with_logits
        
        self.bce_loss = nn.BCEWithLogitsLoss() if with_logits else nn.BCELoss()
        self.dice_loss = DiceLoss(with_logits=with_logits)
        self.focal_loss = FocalLoss(with_logits=with_logits)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce_loss(input, target)
        dice = self.dice_loss(input, target)
        focal = self.focal_loss(input, target)
        
        total_loss = (self.bce_weight * bce + 
                     self.dice_weight * dice + 
                     self.focal_weight * focal)
        
        return total_loss

class SegmentationMetrics:
    """Metrics for segmentation evaluation"""
    
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset metrics"""
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update metrics with batch"""
        # Convert to binary
        if pred.dim() > 1:
            pred = torch.sigmoid(pred) if pred.max() > 1 else pred
            pred = (pred > self.threshold).float()
        else:
            pred = torch.sigmoid(pred) if pred.max() > 1 else pred
            pred = (pred > self.threshold).float()
        
        target = target.float()
        
        # Flatten
        pred = pred.view(-1).cpu().numpy()
        target = target.view(-1).cpu().numpy()
        
        # Calculate confusion matrix
        tp = np.sum((pred == 1) & (target == 1))
        fp = np.sum((pred == 1) & (target == 0))
        fn = np.sum((pred == 0) & (target == 1))
        tn = np.sum((pred == 0) & (target == 0))
        
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics"""
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        iou = self.tp / (self.tp + self.fp + self.fn) if (self.tp + self.fp + self.fn) > 0 else 0
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
            'accuracy': accuracy
        }

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        return False

class LearningRateScheduler:
    """Learning rate scheduler"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, scheduler_type: str = 'step',
                 **kwargs):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        
        if scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )
        elif scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get('T_max', 100)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=kwargs.get('factor', 0.1),
                patience=kwargs.get('patience', 10)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def step(self, val_loss: Optional[float] = None):
        """Step the scheduler"""
        if self.scheduler_type == 'plateau':
            if val_loss is None:
                raise ValueError("val_loss required for plateau scheduler")
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()
    
    def get_last_lr(self) -> List[float]:
        """Get current learning rate"""
        return self.scheduler.get_last_lr()

class TrainingLogger:
    """Training logger for tracking metrics"""
    
    def __init__(self, log_dir: str = "training_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rate': []
        }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.log_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  train_metrics: Dict, val_metrics: Dict, lr: float):
        """Log epoch results"""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_metrics'].append(train_metrics)
        self.history['val_metrics'].append(val_metrics)
        self.history['learning_rate'].append(lr)
        
        # Log to console
        self.logger.info(
            f"Epoch {epoch:3d} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Train IoU: {train_metrics['iou']:.4f}, "
            f"Val IoU: {val_metrics['iou']:.4f}, "
            f"LR: {lr:.6f}"
        )
    
    def save_history(self):
        """Save training history"""
        history_file = self.log_dir / "training_history.json"
        
        # Convert numpy arrays to lists for JSON serialization
        history_dict = {}
        for key, value in self.history.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict):
                    history_dict[key] = value
                else:
                    history_dict[key] = [float(v) for v in value]
            else:
                history_dict[key] = value
        
        with open(history_file, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        self.logger.info(f"Training history saved to {history_file}")
    
    def plot_training_curves(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # IoU curves
        train_iou = [m['iou'] for m in self.history['train_metrics']]
        val_iou = [m['iou'] for m in self.history['val_metrics']]
        axes[0, 1].plot(epochs, train_iou, 'b-', label='Train IoU')
        axes[0, 1].plot(epochs, val_iou, 'r-', label='Val IoU')
        axes[0, 1].set_title('Training and Validation IoU')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 curves
        train_f1 = [m['f1'] for m in self.history['train_metrics']]
        val_f1 = [m['f1'] for m in self.history['val_metrics']]
        axes[1, 0].plot(epochs, train_f1, 'b-', label='Train F1')
        axes[1, 0].plot(epochs, val_f1, 'r-', label='Val F1')
        axes[1, 0].set_title('Training and Validation F1')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate curve
        axes[1, 1].plot(epochs, self.history['learning_rate'], 'g-')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_file = self.log_dir / "training_curves.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training curves saved to {plot_file}")

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, metrics: Dict, 
                   filepath: str, is_best: bool = False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    
    torch.save(checkpoint, filepath)
    
    if is_best:
        best_filepath = filepath.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_filepath)

def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   filepath: str) -> Tuple[int, float, Dict]:
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['metrics']

def get_device() -> torch.device:
    """Get GPU device - enforce GPU-only execution"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! This training requires GPU acceleration.")
    
    device = torch.device('cuda')
    # Set default CUDA device
    torch.cuda.set_device(0)
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    return device

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }

if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    # Create dummy data
    pred = torch.randn(2, 1, 64, 64)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()
    
    # Test different losses
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    focal_loss = FocalLoss()
    combined_loss = CombinedLoss()
    
    print(f"BCE Loss: {bce_loss(pred, target):.4f}")
    print(f"Dice Loss: {dice_loss(pred, target):.4f}")
    print(f"Focal Loss: {focal_loss(pred, target):.4f}")
    print(f"Combined Loss: {combined_loss(pred, target):.4f}")
    
    # Test metrics
    print("\nTesting metrics...")
    metrics = SegmentationMetrics()
    metrics.update(pred, target)
    results = metrics.compute()
    
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    # Test device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Test parameter counting
    from model_architectures import create_model, get_model_configs
    
    configs = get_model_configs()
    for config_name, config in configs.items():
        model = create_model(config)
        param_counts = count_parameters(model)
        print(f"\n{config_name}:")
        print(f"  Total parameters: {param_counts['total']:,}")
        print(f"  Trainable parameters: {param_counts['trainable']:,}")
