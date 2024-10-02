import os
import datetime
import shutil
import json
import numpy as np
import multiprocessing
import time
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import pandas as pd
import optuna
from tqdm import tqdm

from models import LSTMModel
from data_processor import SequenceDataset, DataProcessor
from utils import *
from config import Config

class LSTMTrainer:
    """
    Class to handle LSTM model training with hyperparameter optimization using Optuna.
    """
    def __init__(self, feature_cols, target_col, device, config, rank=0, world_size=1, use_distributed=False):
        self.logger = get_logger('stock_predictor')
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.logger.info(f"Target column set to: {self.target_col}")
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.config = config
        self.use_distributed = use_distributed
        self.out_dir = config.OUT_DIR
        self.model_weights_dir = config.MODEL_WEIGHTS_DIR or os.path.join(self.out_dir, "model_weights")
        self.best_hyperparams = None
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.model_weights_dir, exist_ok=True)
        self.data_processor = DataProcessor(
            data_in_path=None,  # Will be set later
            ret_var=target_col,
            standardize=config.STANDARDIZE,
            config=config  # Pass the config object
        )
        self.data_processor.feature_cols = feature_cols
        self.seq_length = config.LSTM_PARAMS.get('seq_length', 10)

    def _create_dataloader(self, data, seq_length, batch_size, num_workers=Config.NUM_WORKERS):
        if data is None or data.empty:
            self.logger.error("Attempted to create DataLoader with empty or None dataset.")
            raise ValueError("Cannot create DataLoader with empty or None dataset.")
        
        dataset = SequenceDataset(data, seq_length, self.feature_cols, self.target_col)
        if len(dataset) == 0:
            self.logger.error("SequenceDataset is empty. Check your data and sequence length.")
            raise ValueError("SequenceDataset is empty. Cannot create DataLoader.")
        
        data_loader_args = {
            'dataset': dataset,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': True,
            'shuffle': False 
        }
        
        if self.use_distributed:
            sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank)
            data_loader_args['sampler'] = sampler
        else:
            if num_workers > 0:
                data_loader_args['prefetch_factor'] = 2
                data_loader_args['persistent_workers'] = True

        return DataLoader(**data_loader_args)

    def train_model(self, train_loader, val_loader, test_loader, hyperparams, trial=None):
        try:
            # Initialize epoch to a default value
            epoch = 0

            # Initialize model
            input_size = len(self.feature_cols)
            model = LSTMModel(input_size=input_size, **hyperparams).to(self.device)
            if self.use_distributed:
                model = DDP(model, device_ids=[self.rank])

            # Initialize optimizer, criterion, and scheduler
            learning_rate = hyperparams.get('learning_rate', 0.001)
            weight_decay = hyperparams.get('weight_decay', 0.0)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            criterion = nn.MSELoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=hyperparams.get('scheduler_factor', 0.1),
                patience=hyperparams.get('scheduler_patience', 5)
            )

            # Initialize GradScaler for mixed-precision training
            scaler = GradScaler()

            # Load checkpoint if available
            start_epoch, best_val_loss = self.load_checkpoint(model, optimizer, scheduler)

            # Training loop
            num_epochs = hyperparams.get('num_epochs', self.config.NUM_EPOCHS)
            train_losses, val_losses, test_losses = [], [], []
            best_val_loss = float('inf') if best_val_loss is None else best_val_loss
            last_log_time = time.time()
            total_train_time = 0

            # Check if train_loader is empty
            if len(train_loader) == 0:
                self.logger.error("Training dataloader is empty. Cannot train the model.")
                raise ValueError("Training dataloader is empty.")

            for epoch in range(start_epoch, num_epochs):
                epoch_start_time = time.time()

                # Training step
                train_loss = self._train_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    scaler,
                    scheduler,
                    clip_grad_norm=self.config.CLIP_GRAD_NORM,
                    accumulation_steps=self.config.ACCUMULATION_STEPS,
                    epoch=epoch
                )
                train_losses.append(train_loss)

                # Validation step
                if val_loader is not None and len(val_loader) > 0:
                    val_loss = self._evaluate(model, val_loader, criterion)
                    val_losses.append(val_loss)
                else:
                    self.logger.warning("Validation dataloader is empty. Skipping validation.")
                    val_loss = None

                # Test step (optional)
                if test_loader is not None and len(test_loader) > 0:
                    test_loss = self._evaluate(model, test_loader, criterion)
                    test_losses.append(test_loss)
                else:
                    self.logger.warning("Test dataloader is empty. Skipping testing.")
                    test_loss = None

                # Update best model if validation loss has improved
                if val_loss is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'hyperparams': hyperparams
                    }, is_best=True)
                    self.save_hyperparams(hyperparams, is_best=True)  # Save hyperparameters when a new best model is found

                # Log progress
                if (epoch + 1) % self.config.LOG_INTERVAL == 0 or epoch == num_epochs - 1:
                    current_time = time.time()
                    time_since_last_log = current_time - last_log_time
                    total_train_time += time_since_last_log

                    self.logger.info(f"Epoch {epoch + 1}/{num_epochs} completed")
                    self.logger.info(f"Time since last log: {time_since_last_log:.2f} seconds")
                    avg_epoch_time = total_train_time / ((epoch - start_epoch + 1) // self.config.LOG_INTERVAL)
                    self.logger.info(f"Average time per {self.config.LOG_INTERVAL} epochs: {avg_epoch_time:.2f} seconds")
                    self.logger.info(f"Train Loss: {train_loss:.4f}")
                    if val_loss is not None:
                        self.logger.info(f"Validation Loss: {val_loss:.4f}")
                    if test_loss is not None:
                        self.logger.info(f"Test Loss: {test_loss:.4f}")

                    # Log GPU and memory usage
                    log_memory_usage()
                    log_gpu_memory_usage()

                    last_log_time = current_time

                # Report to Optuna and check for pruning
                if trial is not None:
                    trial.report(val_loss, epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

            # Save final state
            self._save_final_state(epoch, model, optimizer, scheduler, best_val_loss, hyperparams)

            training_history = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'test_losses': test_losses
            }

            return model, training_history

        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user.")
            self._save_interrupted_state(epoch, model, optimizer, scheduler, best_val_loss, hyperparams)
            raise
        except Exception as e:
            self.logger.error(f"An error occurred during training: {str(e)}")
            self.logger.error(traceback.format_exc())
            self._save_interrupted_state(epoch, model, optimizer, scheduler, best_val_loss, hyperparams)
            raise

        finally:
            self.logger.info("Model training completed.")
            if trial is None:
                self._save_final_state(epoch, model, optimizer, scheduler, best_val_loss, hyperparams)

    def _train_epoch(self, model, dataloader, criterion, optimizer, scaler, scheduler, clip_grad_norm=None, accumulation_steps=1, epoch=0):
        model.train()
        total_loss = 0.0

        for batch_idx, (batch_X, batch_Y) in enumerate(dataloader):
            batch_X = batch_X.to(self.device, non_blocking=True)
            batch_Y = batch_Y.to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, dtype=torch.float16):
                outputs = model(batch_X).squeeze(-1)
                loss = criterion(outputs, batch_Y) / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * accumulation_steps

        # Update the scheduler
        scheduler.step(total_loss / len(dataloader))

        return total_loss / len(dataloader)

    def _evaluate(self, model, dataloader, criterion, return_predictions=False):
        model.eval()
        total_loss = 0.0
        total_samples = 0
        predictions = []
        targets = []
        with torch.no_grad():
            for batch_X, batch_Y in dataloader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_Y = batch_Y.to(self.device, non_blocking=True)
                outputs = model(batch_X).squeeze(-1)
                loss = criterion(outputs, batch_Y)
                batch_size = batch_Y.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                if return_predictions:
                    predictions.extend(outputs.cpu().numpy())
                    targets.extend(batch_Y.cpu().numpy())
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        if return_predictions:
            return avg_loss, predictions, targets
        else:
            return avg_loss

    def save_checkpoint(self, state, is_best=False, filename='checkpoint.pth.tar'):
        """
        Save a checkpoint of the model.
        """
        if self.rank == 0:
            # Synchronize processes
            if self.use_distributed:
                torch.distributed.barrier()
            save_path = os.path.join(self.model_weights_dir, filename)
            torch.save(state, save_path)
            if is_best:
                best_path = os.path.join(self.model_weights_dir, 'model_best.pth.tar')
                shutil.copyfile(save_path, best_path)
            self.logger.info(f"Checkpoint saved to {save_path}")

    def save_training_metrics(self, train_losses, val_losses, test_losses):
        metrics = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_losses': test_losses
        }
        with open(os.path.join(self.out_dir, 'training_metrics.json'), 'w') as f:
            json.dump(metrics, f)
        self.logger.info("Training metrics saved.")

    def _save_interrupted_state(self, epoch, model, optimizer, scheduler, best_val_loss, hyperparams):
        """Save the current state when training is interrupted."""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_loss': best_val_loss,
                'hyperparams': hyperparams
            }
            checkpoint_path = os.path.join(self.model_weights_dir, 'interrupted_checkpoint.pth')
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Interrupted state saved at epoch {epoch}. Training can be resumed later.")
        except Exception as e:
            self.logger.error(f"Failed to save interrupted state: {str(e)}")

    def _save_final_state(self, epoch, model, optimizer, scheduler, best_val_loss, hyperparams):
        """Save the final state of training."""
        try:
            # Convert numpy.int64 to int
            def convert_to_native(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native(i) for i in obj]
                else:
                    return obj

            # Save the final model state
            final_model_path = os.path.join(self.model_weights_dir, 'final_model.pth')
            torch.save(model.state_dict(), final_model_path)
            self.logger.info(f"Final model state saved to {final_model_path}")

            # Save the final checkpoint
            final_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_loss': best_val_loss,
                'hyperparams': convert_to_native(hyperparams)
            }
            final_checkpoint_path = os.path.join(self.model_weights_dir, 'final_checkpoint.pth')
            torch.save(final_checkpoint, final_checkpoint_path)
            self.logger.info(f"Final checkpoint saved to {final_checkpoint_path}")

            # Save the hyperparameters
            self.save_hyperparams(hyperparams, is_best=False)
            self.logger.info("Final hyperparameters saved.")
        except Exception as e:
            self.logger.error(f"Failed to save final state: {str(e)}")

    def optimize_hyperparameters(self, train_dataset, val_dataset, test_dataset, n_trials=Config.N_TRIALS):
        """
        Optimize hyperparameters using Optuna.
        """
        def objective(trial):
            hyperparams = {
                'seq_length': trial.suggest_int('seq_length', max(self.config.MIN_SEQUENCE_LENGTH, 5), 12),
                'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024]),
                'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
                'num_epochs': self.config.HYPEROPT_EPOCHS,
                'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256]),
                'num_layers': trial.suggest_int('num_layers', 1, 4),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.3),
                'bidirectional': trial.suggest_categorical('bidirectional', [False]),
                'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True]),
                'activation_function': trial.suggest_categorical('activation_function', ['ReLU', 'LeakyReLU', 'ELU']),
                'fc1_size': trial.suggest_categorical('fc1_size', [16, 32, 64, 128]),
                'fc2_size': trial.suggest_categorical('fc2_size', [16, 32, 64, 128]),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'scheduler_factor': trial.suggest_float('scheduler_factor', 0.1, 0.9),
                'scheduler_patience': trial.suggest_int('scheduler_patience', 2, 10)
            }

            # Create data loaders with trial's hyperparameters
            seq_length = hyperparams['seq_length']
            batch_size = hyperparams['batch_size']

            train_loader = self._create_dataloader(train_dataset, seq_length, batch_size)
            val_loader = self._create_dataloader(val_dataset, seq_length, batch_size)
            test_loader = self._create_dataloader(test_dataset, seq_length, batch_size)

            # Train the model with pruning
            try:
                model, training_history = self.train_model(
                    train_loader, val_loader, test_loader, hyperparams, trial=trial
                )
                last_val_loss = training_history['val_losses'][-1]
                return last_val_loss
            except optuna.exceptions.TrialPruned:
                raise
            except Exception as e:
                self.logger.error(f"An error occurred during trial: {str(e)}")
                self.logger.error(traceback.format_exc())
                raise

        # Run the study
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        self.logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")
        study.optimize(objective, n_trials=n_trials)
        self.logger.info("Hyperparameter optimization completed.")

        # Store and return best hyperparameters
        best_hyperparams = study.best_params
        self.save_hyperparams(best_hyperparams, is_best=True)
        self.logger.info(f"Best hyperparameters: {best_hyperparams}")
        self.logger.info(f"Best validation loss: {study.best_value}")

        return best_hyperparams, study.best_value

    def load_hyperparams(self, is_best=True):
        # Adjust method to load hyperparameters for the single target variable
        filename = "best_hyperparams.json" if is_best else "hyperparams.json"
        filepath = os.path.join(self.out_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                hyperparams = json.load(f)
            return hyperparams
        else:
            self.logger.info(f"No hyperparameter file found at: {filepath}")
            return None

    def save_hyperparams(self, hyperparams, is_best=False):
        filename = 'best_hyperparams.json' if is_best else 'hyperparams.json'
        filepath = os.path.join(self.out_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(hyperparams, f, indent=4)
        self.logger.info(f"Hyperparameters saved to {filepath}")

    def load_checkpoint(self, model, optimizer, scheduler=None):
        """
        Load the latest checkpoint if it exists and handle exceptions due to mismatched shapes.
        """
        checkpoint_files = [
            f for f in os.listdir(self.model_weights_dir) if f.endswith('.pth.tar')
        ]
        if not checkpoint_files:
            self.logger.info("No checkpoints found. Starting training from scratch.")
            return 0, float('inf')

        latest_checkpoint = max(
            checkpoint_files,
            key=lambda f: os.path.getctime(os.path.join(self.model_weights_dir, f))
        )
        checkpoint_path = os.path.join(self.model_weights_dir, latest_checkpoint)
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

            # Add this block to handle mismatched model architectures
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.logger.info(f"Checkpoint loaded successfully from {checkpoint_path}")
            return epoch, best_val_loss
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            self.logger.warning("Starting training from scratch due to checkpoint loading failure.")
            return 0, float('inf')

    def evaluate_test_set(self, model, test_data, hyperparams):
        """
        Evaluate the model on the test set and return predictions and targets.
        """
        test_loader = self._create_dataloader(test_data, hyperparams['seq_length'], hyperparams['batch_size'])
        _, predictions, targets = self._evaluate(model, test_loader, nn.MSELoss(), return_predictions=True)
        return predictions, targets

    def adjust_sequence_length(self, min_group_length):
        """
        Adjust the sequence length based on the minimum group length.
        """
        self.seq_length = min(self.config.LSTM_PARAMS.get('seq_length', 10), min_group_length)
        self.seq_length = max(self.seq_length, self.config.MIN_SEQUENCE_LENGTH)
        self.logger.info(f"Sequence length set to: {self.seq_length}")

        # Update DataProcessor's seq_length
        self.data_processor.seq_length = self.seq_length