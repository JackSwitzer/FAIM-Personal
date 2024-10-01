import os
import datetime
import shutil
import json
import numpy as np
import multiprocessing
from functools import partial
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, IterableDataset, Dataset
from torch.amp import GradScaler, autocast

import pandas as pd
import optuna
from tqdm import tqdm

from models import LSTMModel, SequenceDataset
from utils import *
from config import Config

class LSTMTrainer:
    """
    Class to handle LSTM model training with hyperparameter optimization using Optuna.
    """
    def __init__(self, feature_cols, target_col, device, out_dir=Config.OUT_DIR, model_weights_dir=Config.MODEL_WEIGHTS_DIR):
        self.logger = get_logger()
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.logger.info(f"Target column set to: {self.target_col}")
        self.device = device
        self.out_dir = out_dir
        self.model_weights_dir = model_weights_dir if model_weights_dir else os.path.join(out_dir, "model_weights")
        self.best_hyperparams = None  # To store the best hyperparameters
        os.makedirs(self.out_dir, exist_ok=True)  # Ensure output directory exists
        os.makedirs(self.model_weights_dir, exist_ok=True)  # Ensure model weights directory exists

    def create_sequences(self, data, seq_length):
        """
        Create sequences of data for LSTM input using a generator.
        """
        self.logger.info(f"Columns used for sequence creation: {data.columns.tolist()}")
        if self.target_col not in data.columns:
            self.logger.error(f"Target column '{self.target_col}' not found in the data for sequence creation.")
            return
        
        data = data.sort_values(['permno', 'date'])
        grouped = data.groupby('permno')

        for permno, group in grouped:
            group_length = len(group)
            if group_length < seq_length:
                self.logger.debug(f"Skipping 'permno' {permno} due to insufficient data. Group size: {group_length}")
                continue
            group_X = group[self.feature_cols].values
            group_Y = group[self.target_col].values
            group_indices = group.index.values
            for i in range(group_length - seq_length + 1):
                seq = group_X[i:i+seq_length]
                target = group_Y[i+seq_length-1]
                target_index = group_indices[i+seq_length-1]
                yield seq, target, target_index

    def _create_dataloader(self, data, seq_length, batch_size, num_workers=0):
        dataset = SequenceDataset(data, seq_length, self.feature_cols, self.target_col)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    def train_model(self, train_data, val_data, test_data, hyperparams, trial=None):
        """
        Train the LSTM model with specified hyperparameters.

        Args:
            train_data (DataFrame): Training data.
            val_data (DataFrame): Validation data.
            test_data (DataFrame): Test data.
            hyperparams (dict): Dictionary containing hyperparameters.
            trial (optuna.trial.Trial or None): Optuna trial object.

        Returns:
            model (nn.Module): Trained model.
            best_val_loss (float): Best validation loss achieved.
        """
        self.logger.info(f"Starting training with hyperparameters: {hyperparams}")
        self.logger.info(f"Training on device: {self.device}")

        # Extract hyperparameters, using Config defaults if not provided
        seq_length = hyperparams.get('seq_length', 10)  # Add a default sequence length to Config
        batch_size = hyperparams.get('batch_size', Config.BATCH_SIZE)
        learning_rate = hyperparams.get('learning_rate', Config.LEARNING_RATE)
        num_epochs = hyperparams.get('num_epochs', Config.NUM_EPOCHS)
        accumulation_steps = hyperparams.get('accumulation_steps', Config.ACCUMULATION_STEPS)
        clip_grad_norm = hyperparams.get('clip_grad_norm', Config.CLIP_GRAD_NORM)
        
        # Get LSTM params, update with any provided in hyperparams
        lstm_params = Config.get_lstm_params()
        lstm_params.update({k: v for k, v in hyperparams.items() if k in lstm_params})

        # Initialize model
        input_size = len(self.feature_cols)  # Updated feature dimension
        model = LSTMModel(input_size=input_size, **lstm_params).to(self.device)

        # Set up optimizer
        optimizer_name = hyperparams.get('optimizer_name', 'Adam')
        weight_decay = hyperparams.get('weight_decay', 0)
        optimizer_class = getattr(optim, optimizer_name)
        optimizer = optimizer_class(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Set up scheduler
        use_scheduler = hyperparams.get('use_scheduler', False)
        scheduler = None
        if use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=hyperparams.get('scheduler_factor', 0.1),
                patience=hyperparams.get('scheduler_patience', 5)
            )

        criterion = nn.MSELoss()
        best_val_loss = float('inf')

        # Initialize variables
        train_losses = []
        val_losses = []
        test_losses = []
        best_model_state = None

        # Load checkpoint if available and not in hyperparameter optimization
        if trial is None:
            model, optimizer, scheduler, start_epoch, best_val_loss, loaded_hyperparams = self.load_checkpoint(
                model, optimizer, scheduler
            )
            
            # Check if loaded hyperparameters match current hyperparameters
            if loaded_hyperparams and loaded_hyperparams != hyperparams:
                self.logger.warning("Loaded hyperparameters do not match current hyperparameters. "
                                    "Using loaded hyperparameters for consistency.")
                hyperparams = loaded_hyperparams
        else:
            # Start from scratch during hyperparameter optimization
            start_epoch = 1

        # Gradient accumulation setup
        if accumulation_steps < 1:
            accumulation_steps = 1
        self.logger.info(f"Using gradient accumulation with {accumulation_steps} steps")
        
        # Adjust number of workers to a reasonable number
        num_workers = hyperparams.get('num_workers', Config.NUM_WORKERS)
        num_workers = min(num_workers, 4)  

        # Create data loaders outside the training loop
        train_loader = self._create_dataloader(train_data, seq_length, batch_size, num_workers=num_workers)
        val_loader = self._create_dataloader(val_data, seq_length, batch_size, num_workers=num_workers) if val_data is not None else None
        test_loader = self._create_dataloader(test_data, seq_length, batch_size, num_workers=num_workers) if test_data is not None else None
        
        last_log_time = time.time()
        total_train_time = 0

        try:
            for epoch in range(start_epoch, num_epochs + 1):
                epoch_start_time = time.time()

                train_loss = self._train_epoch(
                    model, train_loader, criterion, optimizer,
                    clip_grad_norm, accumulation_steps
                )
                train_losses.append(train_loss)

                epoch_duration = time.time() - epoch_start_time
                total_train_time += epoch_duration

                # Log only at specified intervals or on the last epoch
                if epoch % Config.LOG_INTERVAL == 0 or epoch == num_epochs:
                    current_time = time.time()
                    time_since_last_log = current_time - last_log_time
                    
                    self.logger.info(f"Epoch {epoch}/{num_epochs} completed")
                    self.logger.info(f"Time since last log: {time_since_last_log:.2f} seconds")
                    self.logger.info(f"Average time per epoch: {total_train_time / (epoch - start_epoch + 1):.2f} seconds")
                    self.logger.info(f"Train Loss: {train_loss:.4f}")
                    
                    # Log GPU and memory usage
                    log_memory_usage()
                    log_gpu_memory_usage()

                    last_log_time = current_time

                # Validation loss calculation
                if val_loader is not None:
                    val_loss = self._evaluate(model, val_loader, criterion)
                    val_losses.append(val_loss)
                    
                    # Log validation loss at specified intervals or on the last epoch
                    if epoch % Config.LOG_INTERVAL == 0 or epoch == num_epochs:
                        self.logger.info(f"Validation Loss: {val_loss:.4f}")

                    # Update best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = model.state_dict()
                        best_optimizer_state = optimizer.state_dict()
                        best_scheduler_state = scheduler.state_dict() if scheduler else None

                    # Report to Optuna and check for pruning
                    if trial is not None:
                        trial.report(val_loss, epoch)
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned()

                # Test loss calculation every 10 epochs
                if epoch % 10 == 0 and test_loader is not None:
                    test_loss = self._evaluate(model, test_loader, criterion)
                    test_losses.append(test_loss)
                    self.logger.info(f"Epoch {epoch}/{num_epochs}, Test Loss: {test_loss:.4f}")

                # Save checkpoint only if not in hyperparameter optimization
                if trial is None:
                    checkpoint_interval = hyperparams.get('checkpoint_interval', 10)
                    if epoch % checkpoint_interval == 0:
                        is_best = val_loss < best_val_loss if val_loader is not None else False
                        state = {
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                            'best_val_loss': best_val_loss if val_data is not None else None,
                            'hyperparams': hyperparams
                        }
                        self.save_checkpoint(state, is_best, filename=f'checkpoint_epoch_{epoch}.pth')

                # Scheduler step
                if scheduler:
                    scheduler.step(val_loss if val_loader is not None else train_loss)

            # Save training metrics
            if trial is None:
                self.save_training_metrics(train_losses, val_losses, test_losses)

            # Load the best model state if available
            if best_model_state:
                model.load_state_dict(best_model_state)
                self.logger.info("Loaded best model state.")

            return model, best_val_loss if val_data is not None else None
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user. Saving current model state...")
            # Save checkpoint only if not in hyperparameter optimization
            if trial is None:
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
                self.logger.info(f"Checkpoint saved at epoch {epoch}. Training can be resumed later.")
            raise  # Re-raise the exception to exit

    def optimize_hyperparameters(self, train_data, val_data, test_data, n_trials=Config.N_TRIALS):
        """
        Optimize hyperparameters using Optuna.

        Args:
            train_data (DataFrame): Training data.
            val_data (DataFrame): Validation data.
            test_data (DataFrame): Test data.
            n_trials (int): Number of trials for optimization.

        Returns:
            best_hyperparams (dict): Best hyperparameters found.
            best_val_loss (float): Best validation loss achieved.
        """
        def objective(trial):
            hyperparams = {
                'seq_length': trial.suggest_categorical('seq_length', [1, 2, 3, 5]),  # Experiment with shorter sequence lengths
                'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),  # Increased batch sizes
                'learning_rate': trial.suggest_float('learning_rate', 0.0005, 0.002, log=True),
                'num_epochs': 20,
                'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256]),
                'num_layers': trial.suggest_int('num_layers', 1, 3),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.3),
                'bidirectional': trial.suggest_categorical('bidirectional', [False]),
                'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True]),
                'activation_function': trial.suggest_categorical('activation_function', ['ReLU', 'LeakyReLU', 'ELU']),
                'fc1_size': trial.suggest_categorical('fc1_size', [32, 64, 128]),
                'fc2_size': trial.suggest_categorical('fc2_size', [16, 32, 64]),
            }

            # Train the model with pruning
            try:
                _, val_loss = self.train_model(train_data, val_data, test_data, hyperparams, trial=trial)
            except optuna.exceptions.TrialPruned:
                raise
            return val_loss

        # Setup Optuna study with pruning
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=n_trials)

        # Store and return best hyperparameters
        self.best_hyperparams = study.best_params
        best_val_loss = study.best_value

        # Train and save the best model
        best_model, _ = self.train_model(train_data, val_data, test_data, self.best_hyperparams)
        state = {
            'epoch': Config.NUM_EPOCHS,
            'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': None,
            'scheduler_state_dict': None,
            'best_val_loss': best_val_loss,
            'hyperparams': self.best_hyperparams
        }
        self.save_checkpoint(state, is_best=True, filename='best_model.pth.tar')
        
        # Save the best hyperparameters
        self.save_hyperparams(self.best_hyperparams, is_best=True)

        return self.best_hyperparams, best_val_loss

    def load_hyperparams(self, is_best=True):
        """Load hyperparameters from a JSON file."""
        filename = "best_hyperparams.json" if is_best else "current_hyperparams.json"
        file_path = os.path.join(self.out_dir, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                hyperparams = json.load(f)
            self.logger.info(f"Loaded hyperparameters from: {file_path}")
            return hyperparams
        else:
            self.logger.info(f"No hyperparameter file found at: {file_path}")
            return None

    def save_checkpoint(self, state, is_best=False, filename='checkpoint.pth.tar'):
        """
        Save a checkpoint of the model.
        """
        save_path = os.path.join(self.model_weights_dir, filename)
        torch.save(state, save_path)
        if is_best:
            best_path = os.path.join(self.model_weights_dir, 'model_best.pth.tar')
            shutil.copyfile(save_path, best_path)
        self.logger.info(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, model, optimizer, scheduler=None):
        """
        Load the latest checkpoint if it exists and handle exceptions due to mismatched shapes.
        """
        checkpoint_files = [f for f in os.listdir(self.model_weights_dir) if f.endswith('.pth')]
        if not checkpoint_files:
            self.logger.info("No checkpoints found. Starting training from scratch.")
            return model, optimizer, scheduler, 1, float('inf'), {}

        latest_checkpoint = max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(self.model_weights_dir, f)))
        checkpoint_path = os.path.join(self.model_weights_dir, latest_checkpoint)

        try:
            self.logger.info("Attempting to load checkpoint")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            hyperparams = checkpoint.get('hyperparams', {})
            self.logger.info(f"Successfully loaded checkpoint from epoch {start_epoch - 1}")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            self.logger.info("Starting training from scratch.")
            return model, optimizer, scheduler, 1, float('inf'), {}

        return model, optimizer, scheduler, start_epoch, best_val_loss, hyperparams

    def save_training_metrics(self, train_losses, val_losses, test_losses):
        metrics = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_losses': test_losses
        }
        with open(os.path.join(self.out_dir, 'training_metrics.json'), 'w') as f:
            json.dump(metrics, f)
        self.logger.info("Training metrics saved.")

    def _train_epoch(self, model, dataloader, criterion, optimizer, clip_grad_norm=None, accumulation_steps=1):
        """Train the model for one epoch with gradient accumulation."""
        model.train()
        total_loss = 0.0
        scaler = GradScaler()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (batch_X, batch_Y) in enumerate(dataloader):
            batch_X, batch_Y = batch_X.to(self.device), batch_Y.to(self.device)
            
            # Optionally disable autocast for testing
            # with torch.no_grad():
            with autocast(device_type=self.device.type, dtype=torch.float16):
                outputs = model(batch_X).squeeze(-1)
                loss = criterion(outputs, batch_Y) / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                if clip_grad_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)  # Reset gradients after optimizer step

            batch_size = batch_Y.size(0)
            total_loss += loss.item() * accumulation_steps * batch_size  # Adjust total loss

        avg_loss = total_loss / len(dataloader.dataset)
        return avg_loss

    def _evaluate(self, model, dataloader, criterion, return_predictions=False):
        model.eval()
        total_loss = 0.0
        total_samples = 0
        predictions = []
        targets = []
        with torch.no_grad():
            for batch_X, batch_Y in dataloader:
                batch_X, batch_Y = batch_X.to(self.device), batch_Y.to(self.device)
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

    def save_hyperparams(self, hyperparams, is_best=False):
        """Save hyperparameters to a JSON file."""
        filename = "best_hyperparams.json" if is_best else "current_hyperparams.json"
        file_path = os.path.join(self.out_dir, filename)
        with open(file_path, 'w') as f:
            json.dump(hyperparams, f, indent=2)
        self.logger.info(f"Hyperparameters saved to: {file_path}")