#!/usr/bin/env python3
"""
Main Training Script for Building Segmentation
Integrates data loading, model training, and evaluation
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime
import logging
import json
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from training.data_loader import DataPreparation
from training.model_architectures import create_model, get_model_configs
from training.training_utils import (
    CombinedLoss, SegmentationMetrics, EarlyStopping, 
    LearningRateScheduler, TrainingLogger, save_checkpoint,
    load_checkpoint, get_device, count_parameters
)

class ModelTrainer:
    """Main trainer class for building segmentation models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = get_device()
        
        # Setup directories
        self.output_dir = Path(config.get('output_dir', 'training_output'))
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def prepare_data(self):
        """Prepare training data"""
        self.logger.info("Preparing training data...")
        
        data_prep = DataPreparation(
            data_dir=self.config['data_dir'],
            output_dir=self.output_dir / "training_data"
        )
        
        # Prepare data
        results = data_prep.prepare_data(
            test_size=self.config.get('test_size', 0.15),
            val_size=self.config.get('val_size', 0.15),
            n_folds=self.config.get('n_folds', 5),
            batch_size=self.config.get('batch_size', 8),
            image_size=tuple(self.config.get('image_size', [512, 512]))
        )
        
        self.train_loader = results['data_loaders']['train']
        self.val_loader = results['data_loaders']['val']
        self.test_loader = results['data_loaders']['test']
        
        self.logger.info(f"Data prepared: Train={len(self.train_loader.dataset)}, "
                        f"Val={len(self.val_loader.dataset)}, Test={len(self.test_loader.dataset)}")
        
        return results
    
    def setup_model(self):
        """Setup model, optimizer, and loss function"""
        self.logger.info("Setting up model...")
        
        # Create model
        model_config = self.config['model_config']
        self.model = create_model(model_config)
        self.model = self.model.to(self.device)
        
        # Print model info
        model_info = self.model.get_model_info()
        param_counts = count_parameters(self.model)
        
        self.logger.info(f"Model: {model_info['model_type']}")
        self.logger.info(f"Total parameters: {param_counts['total']:,}")
        self.logger.info(f"Trainable parameters: {param_counts['trainable']:,}")
        
        # Setup optimizer
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adam')
        lr = optimizer_config.get('lr', 1e-4)
        weight_decay = optimizer_config.get('weight_decay', 1e-4)
        
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Setup scheduler
        scheduler_config = self.config.get('scheduler', {})
        self.scheduler = LearningRateScheduler(
            self.optimizer,
            scheduler_type=scheduler_config.get('type', 'plateau'),
            **scheduler_config
        )
        
        # Setup loss function
        loss_config = self.config.get('loss', {})
        self.criterion = CombinedLoss(
            bce_weight=loss_config.get('bce_weight', 0.5),
            dice_weight=loss_config.get('dice_weight', 0.5),
            focal_weight=loss_config.get('focal_weight', 0.0),
            with_logits=loss_config.get('with_logits', True)
        )
        
        self.logger.info(f"Optimizer: {optimizer_type}, LR: {lr}")
        self.logger.info(f"Scheduler: {scheduler_config.get('type', 'plateau')}")
        self.logger.info(f"Loss: Combined (BCE: {loss_config.get('bce_weight', 0.5)}, "
                        f"Dice: {loss_config.get('dice_weight', 0.5)})")
    
    def train_epoch(self, epoch: int) -> Tuple[float, Dict]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        metrics = SegmentationMetrics()
        
        # Progress bar for training batches
        train_pbar = tqdm(enumerate(self.train_loader), 
                         total=len(self.train_loader),
                         desc=f"Epoch {epoch:3d} - Training",
                         leave=False)
        
        for batch_idx, batch in train_pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            metrics.update(outputs, masks)
            
            # Update progress bar
            current_loss = total_loss / (batch_idx + 1)
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{current_loss:.4f}',
                'GPU': f'{torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}'
            })
            
            # Log progress occasionally
            if batch_idx % max(1, len(self.train_loader) // 4) == 0:
                self.logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                               f"Loss: {loss.item():.4f}, Avg Loss: {current_loss:.4f}")
        
        train_pbar.close()
        avg_loss = total_loss / len(self.train_loader)
        epoch_metrics = metrics.compute()
        
        return avg_loss, epoch_metrics
    
    def validate_epoch(self) -> Tuple[float, Dict]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        metrics = SegmentationMetrics()
        
        # Progress bar for validation batches
        val_pbar = tqdm(self.val_loader, 
                       desc="     Validation",
                       leave=False)
        
        with torch.no_grad():
            for batch in val_pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Update metrics
                total_loss += loss.item()
                metrics.update(outputs, masks)
                
                # Update progress bar
                val_pbar.set_postfix({
                    'Val Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{total_loss / (val_pbar.n + 1):.4f}'
                })
        
        val_pbar.close()
        avg_loss = total_loss / len(self.val_loader)
        epoch_metrics = metrics.compute()
        
        return avg_loss, epoch_metrics
    
    def test_model(self) -> Dict:
        """Test the model on test set"""
        self.logger.info("Testing model on test set...")
        
        self.model.eval()
        total_loss = 0
        metrics = SegmentationMetrics()
        
        with torch.no_grad():
            for batch in self.test_loader:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Update metrics
                total_loss += loss.item()
                metrics.update(outputs, masks)
        
        avg_loss = total_loss / len(self.test_loader)
        test_metrics = metrics.compute()
        
        self.logger.info(f"Test Loss: {avg_loss:.4f}")
        for metric, value in test_metrics.items():
            self.logger.info(f"Test {metric}: {value:.4f}")
        
        return {'loss': avg_loss, 'metrics': test_metrics}
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        # Initialize components
        early_stopping = EarlyStopping(
            patience=self.config.get('patience', 10),
            min_delta=self.config.get('min_delta', 0.001)
        )
        
        logger = TrainingLogger(self.output_dir / "logs")
        
        # Training loop
        best_val_loss = float('inf')
        start_epoch = 1
        
        # Load checkpoint if resuming
        if self.config.get('resume_from'):
            checkpoint_path = self.config['resume_from']
            if os.path.exists(checkpoint_path):
                start_epoch, best_val_loss, _ = load_checkpoint(
                    self.model, self.optimizer, checkpoint_path
                )
                start_epoch += 1
                self.logger.info(f"Resuming from epoch {start_epoch}")
        
        # Overall training progress bar
        total_epochs = self.config.get('epochs', 100)
        epoch_pbar = tqdm(range(start_epoch, total_epochs + 1), 
                         desc="🏗️  Training Progress",
                         initial=start_epoch - 1,
                         total=total_epochs)
        
        for epoch in epoch_pbar:
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch()
            
            # Update scheduler
            if self.scheduler.scheduler_type == 'plateau':
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val Loss': f'{val_loss:.4f}',
                'IoU': f'{val_metrics["iou"]:.3f}',
                'LR': f'{current_lr:.2e}',
                'GPU': torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            })
            
            # Log results
            logger.log_epoch(epoch, train_loss, val_loss, train_metrics, val_metrics, current_lr)
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                epoch_pbar.set_description(f"🏗️  Training Progress (Best: {val_loss:.4f})")
            
            checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
            save_checkpoint(
                self.model, self.optimizer, epoch, val_loss, val_metrics,
                str(checkpoint_path), is_best
            )
            
            # Print epoch summary
            self.logger.info(f"Epoch {epoch:3d}/{total_epochs} | "
                           f"Train Loss: {train_loss:.4f} | "
                           f"Val Loss: {val_loss:.4f} | "
                           f"IoU: {val_metrics['iou']:.3f} | "
                           f"F1: {val_metrics['f1']:.3f}")
            
            # Early stopping
            if early_stopping(val_loss, self.model):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                epoch_pbar.set_description("🏗️  Training Complete (Early Stop)")
                break
        
        epoch_pbar.close()
        
        # Save final model
        final_model_path = self.output_dir / "final_model.pth"
        torch.save(self.model.state_dict(), final_model_path)
        self.logger.info(f"Final model saved to {final_model_path}")
        
        # Save training history and plots
        logger.save_history()
        logger.plot_training_curves()
        
        # Test final model
        test_results = self.test_model()
        
        # Save test results
        test_results_path = self.output_dir / "test_results.json"
        with open(test_results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        self.logger.info("Training completed!")
        return test_results
    
    def run_kfold_training(self):
        """Run K-fold cross-validation training"""
        self.logger.info("Starting K-fold cross-validation training...")
        
        # Load data preparation results
        data_info_path = self.output_dir / "training_data" / "data_info.json"
        with open(data_info_path, 'r') as f:
            data_info = json.load(f)
        
        fold_splits = data_info.get('kfold_splits', [])
        if not fold_splits:
            self.logger.error("No K-fold splits found in data info")
            return
        
        fold_results = []
        
        for fold_data in fold_splits:
            fold_num = fold_data['fold']
            self.logger.info(f"Training fold {fold_num}/{len(fold_splits)}")
            
            # Create fold-specific output directory
            fold_output_dir = self.output_dir / f"fold_{fold_num}"
            fold_output_dir.mkdir(exist_ok=True)
            
            # Update config for this fold
            fold_config = self.config.copy()
            fold_config['output_dir'] = str(fold_output_dir)
            
            # Create fold-specific trainer
            fold_trainer = ModelTrainer(fold_config)
            
            # Prepare fold-specific data
            fold_trainer.prepare_data()
            fold_trainer.setup_model()
            
            # Train fold
            fold_result = fold_trainer.train()
            fold_result['fold'] = fold_num
            fold_results.append(fold_result)
        
        # Aggregate results
        avg_test_loss = np.mean([r['loss'] for r in fold_results])
        avg_test_iou = np.mean([r['metrics']['iou'] for r in fold_results])
        avg_test_f1 = np.mean([r['metrics']['f1'] for r in fold_results])
        
        self.logger.info(f"K-fold CV Results:")
        self.logger.info(f"Average Test Loss: {avg_test_loss:.4f}")
        self.logger.info(f"Average Test IoU: {avg_test_iou:.4f}")
        self.logger.info(f"Average Test F1: {avg_test_f1:.4f}")
        
        # Save K-fold results
        kfold_results = {
            'fold_results': fold_results,
            'average_metrics': {
                'loss': avg_test_loss,
                'iou': avg_test_iou,
                'f1': avg_test_f1
            }
        }
        
        kfold_results_path = self.output_dir / "kfold_results.json"
        with open(kfold_results_path, 'w') as f:
            json.dump(kfold_results, f, indent=2)
        
        self.logger.info(f"K-fold results saved to {kfold_results_path}")

def get_default_config() -> Dict:
    """Get default training configuration"""
    return {
        'data_dir': '../full_dataset_results',  # Path to processed data
        'output_dir': 'training_output',
        'model_config': {
            'model_type': 'resnet_unet',
            'n_channels': 3,
            'n_classes': 1,
            'backbone': 'resnet34',
            'pretrained': True
        },
        'optimizer': {
            'type': 'adam',
            'lr': 1e-4,
            'weight_decay': 1e-4
        },
        'scheduler': {
            'type': 'plateau',
            'factor': 0.1,
            'patience': 10
        },
        'loss': {
            'bce_weight': 0.5,
            'dice_weight': 0.5,
            'focal_weight': 0.0,
            'with_logits': True
        },
        'batch_size': 8,
        'image_size': [512, 512],
        'epochs': 50,
        'patience': 10,
        'min_delta': 0.001,
        'test_size': 0.15,
        'val_size': 0.15,
        'n_folds': 5
    }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train Building Segmentation Model")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--data-dir", help="Path to data directory")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--model-type", choices=['unet', 'resnet_unet', 'efficientnet_unet'], 
                       help="Model type")
    parser.add_argument("--backbone", help="Backbone for ResNet/EfficientNet models")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--k-fold", action='store_true', help="Run K-fold cross-validation")
    parser.add_argument("--resume-from", help="Resume from checkpoint")
    parser.add_argument("--quick-demo", action='store_true', help="Run quick demo training (fewer epochs, smaller dataset)")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # Override with command line arguments
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.model_type:
        config['model_config']['model_type'] = args.model_type
    if args.backbone:
        config['model_config']['backbone'] = args.backbone
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['epochs'] = args.epochs
    if args.lr:
        config['optimizer']['lr'] = args.lr
    if args.resume_from:
        config['resume_from'] = args.resume_from
    
    # Apply quick demo settings if requested
    if args.quick_demo:
        config['epochs'] = 2
        config['batch_size'] = 2
        config['optimizer']['lr'] = 0.001
        config['image_size'] = [256, 256]
        print("Quick demo mode: Using reduced epochs, batch size, and image size")
    
    # Create trainer
    trainer = ModelTrainer(config)
    
    try:
        if args.k_fold:
            # Run K-fold cross-validation
            trainer.run_kfold_training()
        else:
            # Run standard training
            trainer.prepare_data()
            trainer.setup_model()
            results = trainer.train()
            
            print("\n" + "="*60)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Test Loss: {results['loss']:.4f}")
            for metric, value in results['metrics'].items():
                print(f"Test {metric}: {value:.4f}")
            print(f"Output directory: {config['output_dir']}")
            print("="*60)
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
