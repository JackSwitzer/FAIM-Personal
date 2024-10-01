import os
import datetime
import shutil
import json
import numpy as np
import multiprocessing
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, IterableDataset, Dataset
from torch.amp import GradScaler, autocast

import pandas as pd
import optuna
from tqdm import tqdm

from models import LSTMModel, SequenceDataset
from utils import get_logger

class LSTMTrainer:
    """
    Class to handle LSTM model training with hyperparameter optimization using Optuna.
    """
    def __init__(self, feature_cols, target_col, device, out_dir="./Data Output/", model_weights_dir=None):
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

    def _create_dataloader(self, data, seq_length, batch_size, num_workers=4, shuffle=False):
        # Ensure data has unique indices
        data = data.reset_index(drop=True)
        dataset = SequenceDataset(data, seq_length, self.feature_cols, self.target_col)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                          num_workers=num_workers, pin_memory=True)

    def train_model(self, train_data, val_data, test_data, hyperparams):
        """
        Train the LSTM model with specified hyperparameters.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            test_loader (DataLoader): DataLoader for test data.
            hyperparams (dict): Dictionary containing hyperparameters.

        Returns:
            model (nn.Module): Trained model.
            best_val_loss (float): Best validation loss achieved.
        """
        self.logger.info(f"Starting training with hyperparameters: {hyperparams}")
        self.logger.info(f"Training on device: {self.device}")
        # Extract hyperparameters
        seq_length = hyperparams['seq_length']
        batch_size = hyperparams['batch_size']
        num_layers = hyperparams['num_layers']
        dropout_rate = hyperparams['dropout_rate']
        hidden_size = hyperparams['hidden_size']
        bidirectional = hyperparams['bidirectional']
        learning_rate = hyperparams['learning_rate']
        weight_decay = hyperparams['weight_decay']
        optimizer_name = hyperparams['optimizer_name']
        use_scheduler = hyperparams['use_scheduler']
        activation_function = hyperparams['activation_function']
        use_batch_norm = hyperparams['use_batch_norm']
        clip_grad_norm = hyperparams['clip_grad_norm']
        fc1_size = hyperparams['fc1_size']
        fc2_size = hyperparams['fc2_size']
        num_epochs = hyperparams.get('num_epochs', 10000)
        accumulation_steps = hyperparams.get('accumulation_steps', 1)

        # Initialize model
        input_size = len(self.feature_cols)
        model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
            use_batch_norm=use_batch_norm,
            activation_function=activation_function,
            fc1_size=fc1_size,
            fc2_size=fc2_size
        ).to(self.device)

        # Set up optimizer
        optimizer_class = getattr(optim, optimizer_name)
        optimizer = optimizer_class(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Set up scheduler
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

        # Load checkpoint if available
        model, optimizer, scheduler, start_epoch, best_val_loss, loaded_hyperparams = self.load_checkpoint(
            model, optimizer, scheduler
        )
        
        # Check if loaded hyperparams match current hyperparams
        if loaded_hyperparams and loaded_hyperparams != hyperparams:
            self.logger.warning("Loaded hyperparameters do not match current hyperparameters. "
                                "Using loaded hyperparameters for consistency.")
            hyperparams = loaded_hyperparams

        # Gradient accumulation setup
        if accumulation_steps < 1:
            accumulation_steps = 1
        self.logger.info(f"Using gradient accumulation with {accumulation_steps} steps")
        
        train_loader = self._create_dataloader(train_data, seq_length, batch_size, shuffle=True)
        val_loader = self._create_dataloader(val_data, seq_length, batch_size) if val_data is not None else None
        test_loader = self._create_dataloader(test_data, seq_length, batch_size)
        
        try:
            for epoch in range(start_epoch, num_epochs + 1):
                train_loss = self._train_epoch(model, train_loader, criterion, optimizer, clip_grad_norm)
                train_losses.append(train_loss)

                # Log validation loss if val_loader is available
                if val_data is not None:
                    val_loss = 0.0
                    with torch.no_grad():
                        for batch_loader in self._create_dataloader(val_data, seq_length, batch_size):
                            for batch_X, batch_Y in batch_loader:
                                batch_X, batch_Y = batch_X.to(self.device), batch_Y.to(self.device)
                                outputs = model(batch_X).squeeze(-1)
                                val_loss += criterion(outputs, batch_Y).item()

                    # Calculate average validation loss
                    val_loss = val_loss / len(val_data)
                    self.logger.info(f"Epoch {epoch}/{num_epochs}, Validation Loss: {val_loss:.4f}")
                    val_losses.append(val_loss)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = model.state_dict()
                        best_optimizer_state = optimizer.state_dict()
                        best_scheduler_state = scheduler.state_dict() if scheduler else None

                # Log test loss and save checkpoint every 10 epochs
                if epoch % 10 == 0:
                    test_loss = 0.0
                    with torch.no_grad():
                        for batch_loader in self._create_dataloader(test_data, seq_length, batch_size):
                            for batch_X, batch_Y in batch_loader:
                                batch_X, batch_Y = batch_X.to(self.device), batch_Y.to(self.device)
                                outputs = model(batch_X).squeeze(-1)
                                test_loss += criterion(outputs, batch_Y).item()

                    # Calculate average test loss
                    test_loss = test_loss / len(test_data)
                    self.logger.info(f"Epoch {epoch}/{num_epochs}, Test Loss: {test_loss:.4f}")
                    test_losses.append(test_loss)

                    # Save checkpoint
                    is_best = val_loss < best_val_loss
                    state = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'best_val_loss': best_val_loss if val_data is not None else None,
                        'hyperparams': hyperparams
                    }
                    self.save_checkpoint(state, is_best, filename=f'checkpoint_epoch_{epoch}.pth')

                # Update scheduler if it exists
                if scheduler:
                    if val_data is not None:
                        scheduler.step(val_loss)
                    else:
                        scheduler.step(train_loss)

            # Save training metrics
            self.save_training_metrics(train_losses, val_losses, test_losses)

            # Load the best model state if available
            if best_model_state:
                model.load_state_dict(best_model_state)
                self.logger.info("Loaded best model state.")

            return model, best_val_loss if val_data is not None else None
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user. Saving current model state...")
            # Save checkpoint
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

    def optimize_hyperparameters(self, train_data, val_data, n_trials=50):
        """
        Optimize hyperparameters using Optuna.

        Args:
            train_data (DataFrame): Training data.
            val_data (DataFrame): Validation data.
            n_trials (int): Number of trials for optimization.

        Returns:
            best_hyperparams (dict): Best hyperparameters found.
            best_val_loss (float): Best validation loss achieved.
        """
        def objective(trial):
            # Suggest hyperparameters
            num_layers = trial.suggest_int('num_layers', 4, 7)
            dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.3)
            hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
            seq_length = trial.suggest_categorical('seq_length', [5, 10, 15])
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            bidirectional = trial.suggest_categorical('bidirectional', [True, False])
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
            optimizer_name = trial.suggest_categorical('optimizer_name', ['Adam', 'RMSprop'])
            use_scheduler = trial.suggest_categorical('use_scheduler', [False, True])
            activation_function = trial.suggest_categorical('activation_function', ['ReLU', 'None'])
            use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
            clip_grad_norm = trial.suggest_float('clip_grad_norm', 0.5, 4.0)
            fc1_size = trial.suggest_categorical('fc1_size', [32, 64, 128])
            fc2_size = trial.suggest_categorical('fc2_size', [16, 32, 64])
            accumulation_steps = trial.suggest_int('accumulation_steps', 1, 4)
            num_epochs = 10  # For hyperparameter optimization, keep epochs small

            hyperparams = {
                'num_layers': num_layers,
                'dropout_rate': dropout_rate,
                'hidden_size': hidden_size,
                'bidirectional': bidirectional,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'optimizer_name': optimizer_name,
                'use_scheduler': use_scheduler,
                'activation_function': activation_function,
                'use_batch_norm': use_batch_norm,
                'clip_grad_norm': clip_grad_norm,
                'fc1_size': fc1_size,
                'fc2_size': fc2_size,
                'num_epochs': num_epochs,
                'seq_length': seq_length,
                'batch_size': batch_size,
                'accumulation_steps': accumulation_steps
            }

            # Train the model
            model, val_loss = self.train_model(train_data, val_data, hyperparams)

            return val_loss

        # Setup Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        # Store and return best hyperparameters
        self.best_hyperparams = study.best_params
        best_val_loss = study.best_value

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

    def save_hyperparams(self, hyperparams, is_best=False):
        """Save the hyperparameters to a JSON file."""
        filename = "best_hyperparams.json" if is_best else "current_hyperparams.json"
        file_path = os.path.join(self.out_dir, filename)
        with open(file_path, 'w') as f:
            json.dump(hyperparams, f, indent=2)
        self.logger.info(f"Hyperparameters saved to: {file_path}")

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
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            hyperparams = checkpoint.get('hyperparams', {})
            self.logger.info(f"Resuming from checkpoint at epoch {start_epoch - 1}")
            return model, optimizer, scheduler, start_epoch, best_val_loss, hyperparams
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            self.logger.info("Starting training from scratch.")
            return model, optimizer, scheduler, 1, float('inf'), {}

    def save_checkpoint(self, state, is_best=False, filename='checkpoint.pth.tar'):
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

    def _train_epoch(self, model, dataloader, criterion, optimizer, clip_grad_norm=None):
        """Train the model for one epoch."""
        model.train()
        total_loss = 0.0
        scaler = GradScaler()  # Remove the device_type argument
        for batch_X, batch_Y in dataloader:
            batch_X, batch_Y = batch_X.to(self.device), batch_Y.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type=self.device.type, dtype=torch.float16):
                outputs = model(batch_X).squeeze(-1)
                loss = criterion(outputs, batch_Y)
            
            scaler.scale(loss).backward()
            
            if clip_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            
            batch_size = batch_Y.size(0)
            total_loss += loss.item() * batch_size
        
        avg_loss = total_loss / len(dataloader.dataset)
        return avg_loss

    def _evaluate(self, model, dataloader, criterion, return_predictions=False):
        model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        total_samples = 0
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
