import argparse
import copy
import json
import os
import time
from typing import Literal, Optional, Any, Dict, List

import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from torch import nn, optim
from tqdm import tqdm


class EndoscopyClassificationTrainer:
    def __init__(
            self,
            model: nn.Module,
            criterion: nn.Module,
            optimizer: optim.Optimizer,
            args: argparse.Namespace,
            num_classes: int,
            lr_scheduler: Optional[Any] = None,
            gradient_clipping: bool = False,
            snapshot_interval: int = 20,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.args = args
        self.gradient_clipping = gradient_clipping
        self.num_classes = num_classes
        self.is_multilabel = num_classes > 2

        self.best_loss: float = float('inf')
        self.snapshot_interval: int = snapshot_interval

        # Initialize summary dictionary
        self.summary: Dict[str, Dict[str, List]] = {
            'train': {
                'loss': [], 'accuracy': [], 'precision': [], 'recall': [],
                'f1_score': [], 'balanced_accuracy': [], 'time': []
            },
            'valid': {
                'loss': [], 'accuracy': [], 'precision': [], 'recall': [],
                'f1_score': [], 'balanced_accuracy': [], 'time': []
            }
        }

        # Create results directory if it doesn't exist
        os.makedirs(self.args.results_path, exist_ok=True)
        self.config_f_name: str = os.path.join(self.args.results_path, f'config.json')

        # Save training configuration
        self._save_config()

    def _save_config(self):
        """Save training configuration to disk"""
        config = vars(self.args).copy()
        # Convert non-serializable objects to strings
        for key, value in config.items():
            if not isinstance(value, (str, int, float, bool, list, dict, tuple, type(None))):
                config[key] = str(value)

        with open(self.config_f_name, 'w') as f:
            json.dump(config, f, indent=4)

    def perform_epoch(self,
                      data_loader: torch.utils.data.DataLoader,
                      phase: Literal['train', 'valid'],
                      epoch_nb: int):
        """Execute one epoch of training or validation"""
        print(f'\n{phase.upper()} | Start epoch: {epoch_nb}')
        start_time = time.time()

        if phase == 'train':
            self.model.train()
        elif phase == 'valid':
            self.model.eval()

        losses = []
        true_labels = []
        predicted_probs = []

        grad_context = torch.inference_mode if phase == 'valid' else torch.enable_grad
        with grad_context():
            for idx, (features, targets) in enumerate(tqdm(data_loader, desc=f"{phase}")):
                features = features.to(self.args.device)
                targets = targets.to(self.args.device)

                # Forward pass
                logits = self.model(features)

                loss = self.criterion(logits, targets)
                losses.append(loss.item())

                # Calculate probabilities
                probs = torch.softmax(logits, dim=1)
                predicted_probs.append(probs.cpu().detach().numpy())
                true_labels.append(targets.cpu().numpy())

                if phase == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.gradient_clipping:
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

        # Concatenate batch results
        predicted_probs = np.concatenate(predicted_probs)
        true_labels = np.concatenate(true_labels)

        # Calculate performance metrics
        current_loss = np.mean(losses)

        # Multi-class metrics
        predicted_labels = np.argmax(predicted_probs, axis=1)
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='macro')
        recall = recall_score(true_labels, predicted_labels, average='macro')
        f1 = f1_score(true_labels, predicted_labels, average='macro')
        balanced_acc = balanced_accuracy_score(true_labels, predicted_labels)

        # Confusion matrix for analysis
        cm = confusion_matrix(true_labels, predicted_labels)

        metrics = {
            'loss': current_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'balanced_accuracy': balanced_acc
        }

        # Update summary with metrics
        for key, value in metrics.items():
            if key in self.summary[phase]:
                self.summary[phase][key].append(value)

        # Calculate epoch duration
        epoch_duration = time.time() - start_time
        minutes = int(epoch_duration // 60)
        seconds = int(epoch_duration % 60)
        print(f"Done | Took {minutes} minutes and {seconds} seconds.")
        self.summary[phase]['time'].append(epoch_duration)

        # Print metrics
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"{phase.upper()} | {metrics_str}")

        # Update learning rate if in training phase
        if phase == 'train' and self.lr_scheduler:
            self.lr_scheduler.step()
            after_lr = self.optimizer.param_groups[0]["lr"]
            print(f'New learning rate: {after_lr:.6f}')

        # Save best model if in validation phase
        if phase == 'valid' and current_loss < self.best_loss:
            self.best_loss = current_loss
            self.save_weights(epoch_nb, is_best=True)
            print(f"New best model saved [at epoch {epoch_nb}] with loss {self.best_loss:.4f}")

            # Save confusion matrix for best model
            if self.is_multilabel:
                np.save(os.path.join(self.args.results_path, 'best_confusion_matrix.npy'), cm)

        # Periodic snapshot
        if phase == 'train' and epoch_nb % self.snapshot_interval == 0:
            self.save_weights(epoch_nb)

        # Log to wandb if available
        self._log_metrics(metrics, phase, epoch_nb)

        return metrics

    def save_weights(self, epoch: int, is_best: bool = False):
        """Save model weights to disk"""
        if is_best:
            save_path = os.path.join(self.args.results_path, 'best_model.pth')
        else:
            save_path = os.path.join(self.args.results_path, f'model_epoch_{epoch}.pth')

        state_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_loss,
            'summary': self.summary,
        }

        if self.lr_scheduler:
            state_dict['scheduler_state_dict'] = self.lr_scheduler.state_dict()

        torch.save(state_dict, save_path)

    def load_weights(self, checkpoint_path: str):
        """Load model weights from disk"""
        checkpoint = torch.load(checkpoint_path, map_location=self.args.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_loss = checkpoint['loss']
        if 'summary' in checkpoint:
            self.summary = checkpoint['summary']

        return checkpoint['epoch']

    def _log_metrics(self, metrics: Dict[str, float], phase: str, epoch: int):
        try:
            import wandb
            if wandb.run is not None:
                log_dict = {k: v for k, v in metrics.items()}

                for k, v in metrics.items():
                    log_dict[f"{phase}_{k}"] = v

                log_dict['epoch'] = epoch

                wandb.log(log_dict)
        except ImportError:
            pass  # wandb not installed, skip logging
        except Exception as e:
            print(f"Warning: Failed to log metrics to wandb: {e}")

    def train(self, train_loader: torch.utils.data.DataLoader,
              valid_loader: torch.utils.data.DataLoader,
              num_epochs: int, start_epoch: int = 0):
        """Run full training loop with validation"""
        for epoch in range(start_epoch, num_epochs):
            # Training phase
            train_metrics = self.perform_epoch(train_loader, 'train', epoch)

            # Validation phase
            with torch.no_grad():
                valid_metrics = self.perform_epoch(valid_loader, 'valid', epoch)

            # Save summary after each epoch
            self._save_summary()

        return self.summary

    def _save_summary(self):
        """Save training summary to disk"""
        summary_path = os.path.join(self.args.results_path, 'training_summary.json')
        # Convert numpy values to Python native types for JSON serialization
        serializable_summary = copy.deepcopy(self.summary)
        for phase in serializable_summary:
            for metric in serializable_summary[phase]:
                serializable_summary[phase][metric] = [
                    float(x) if isinstance(x, (np.float32, np.float64)) else x
                    for x in serializable_summary[phase][metric]
                ]

        with open(summary_path, 'w') as f:
            json.dump(serializable_summary, f, indent=4)