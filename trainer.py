import os
import datetime
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd
import optuna

from models import LSTMModel, StockDataset
from utils import get_logger

class LSTMTrainer:
    """
    Class to handle LSTM model training with hyperparameter optimization using Optuna.
    """
    def __init__(self, feature_cols, target_col, device, out_dir="./Data Output/"):
        self.logger = get_logger()
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.logger.info(f"Target column set to: {self.target_col}")
        self.device = device
        self.out_dir = out_dir
        self.best_hyperparams = None  # To store the best hyperparameters
        os.makedirs(self.out_dir, exist_ok=True)  # Ensure output directory exists

    def create_sequences(self, data, seq_length):
        """
        Create sequences of data for LSTM input.
        """
        self.logger.info(f"Columns used for sequence creation: {data.columns.tolist()}")
        if self.target_col not in data.columns:
            self.logger.error(f"Target column '{self.target_col}' not found in the data for sequence creation.")
        
        sequences = []
        targets = []
        indices = []
        data = data.sort_values(['permno', 'date'])
        grouped = data.groupby('permno')

        for _, group in grouped:
            if len(group) < seq_length + 1:
                continue
            group_X = group[self.feature_cols].values
            group_Y = group[self.target_col].values
            group_indices = group.index.values
            for i in range(len(group_X) - seq_length):
                seq = group_X[i:i+seq_length]
                target = group_Y[i+seq_length]
                target_index = group_indices[i+seq_length]
                sequences.append(seq)
                targets.append(target)
                indices.append(target_index)
        sequences = np.array(sequences)
        targets = np.array(targets)
        indices = np.array(indices)
        return sequences, targets, indices

    def _create_dataloader(self, X, Y, batch_size, shuffle=False):
        """Create DataLoader from sequences and targets."""
        dataset = StockDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32))
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=os.cpu_count(),
            pin_memory=True
        )

    def train_model(self, train_loader, val_loader, hyperparams):
        """
        Train the LSTM model with specified hyperparameters.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            hyperparams (dict): Dictionary containing hyperparameters.

        Returns:
            model (nn.Module): Trained model.
            best_val_loss (float): Best validation loss achieved.
        """
        # Extract hyperparameters
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
        accumulation_steps = hyperparams.get('accumulation_steps', 1)  # Add accumulation_steps

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
        best_model_state = None

        # Load checkpoint if available
        try:
            model, optimizer, scheduler, start_epoch, best_val_loss = self.load_checkpoint(model, optimizer, scheduler)
        except Exception as e:
            self.logger.info(f"No checkpoint found, starting from scratch. {str(e)}")
            start_epoch = 1

        # Gradient accumulation setup
        if accumulation_steps < 1:
            accumulation_steps = 1
        self.logger.info(f"Using gradient accumulation with {accumulation_steps} steps")
        
        try:
            for epoch in range(start_epoch, num_epochs + 1):
                start_time = datetime.datetime.now()
                model.train()
                optimizer.zero_grad()
                running_loss = 0.0
                for batch_idx, (batch_X, batch_Y) in enumerate(train_loader):
                    batch_X, batch_Y = batch_X.to(self.device), batch_Y.to(self.device)
                    outputs = model(batch_X).squeeze(-1)
                    loss = criterion(outputs, batch_Y)
                    loss = loss / accumulation_steps
                    loss.backward()
                    running_loss += loss.item()

                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                        if clip_grad_norm:
                            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                # Calculate average training loss
                train_loss = running_loss * accumulation_steps / len(train_loader)
                epoch_duration = datetime.datetime.now() - start_time
                self.logger.info(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Duration: {epoch_duration}")
                train_losses.append(train_loss)

                if val_loader is not None:
                    val_loss = self._evaluate(model, val_loader, criterion)
                    self.logger.info(f"Epoch {epoch}/{num_epochs}, Validation Loss: {val_loss:.4f}")
                    val_losses.append(val_loss)

                    # Update scheduler and save best model
                    if scheduler:
                        scheduler.step(val_loss)
                        # Log the current learning rate
                        current_lr = scheduler.optimizer.param_groups[0]['lr']
                        self.logger.info(f"Current learning rate: {current_lr:.6f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = model.state_dict()
                        best_optimizer_state = optimizer.state_dict()
                        best_scheduler_state = scheduler.state_dict() if scheduler else None

                        # Save checkpoint
                        checkpoint = {
                            'epoch': epoch,
                            'model_state_dict': best_model_state,
                            'optimizer_state_dict': best_optimizer_state,
                            'scheduler_state_dict': best_scheduler_state,
                            'best_val_loss': best_val_loss,
                            'hyperparams': hyperparams
                        }
                        torch.save(checkpoint, os.path.join(self.out_dir, 'best_checkpoint.pth'))
                        self.logger.info(f"Checkpoint saved at epoch {epoch} with validation loss: {best_val_loss:.4f}")
                else:
                    # Save model periodically even without validation
                    if epoch % 10 == 0:
                        checkpoint = {
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                            'hyperparams': hyperparams
                        }
                        torch.save(checkpoint, os.path.join(self.out_dir, f'checkpoint_epoch_{epoch}.pth'))
                        self.logger.info(f"Checkpoint saved at epoch {epoch}")

            # Save training metrics
            self.save_training_metrics(train_losses, val_losses)

            # Load the best model state if available
            if best_model_state:
                model.load_state_dict(best_model_state)
                self.logger.info("Loaded best model state.")

            return model, best_val_loss if val_loader is not None else None
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
            torch.save(checkpoint, os.path.join(self.out_dir, 'interrupted_checkpoint.pth'))
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

            # Create sequences and dataloaders
            X_train, Y_train, _ = self.create_sequences(train_data, seq_length)
            X_val, Y_val, _ = self.create_sequences(val_data, seq_length)

            if len(X_train) == 0 or len(X_val) == 0:
                return float('inf')  # Skip this trial

            train_loader = self._create_dataloader(X_train, Y_train, batch_size)
            val_loader = self._create_dataloader(X_val, Y_val, batch_size)

            # Train the model
            model, val_loss = self.train_model(train_loader, val_loader, hyperparams)

            return val_loss

        # Setup Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        # Store and return best hyperparameters
        self.best_hyperparams = study.best_params
        best_val_loss = study.best_value

        # Save the best hyperparameters
        self.save_best_hyperparams()

        return self.best_hyperparams, best_val_loss

    def save_best_hyperparams(self):
        """Save the best hyperparameters to a JSON file."""
        if self.best_hyperparams is None:
            self.logger.info("No hyperparameters to save.")
            return

        file_path = os.path.join(self.out_dir, "best_hyperparams.json")
        with open(file_path, 'w') as f:
            json.dump(self.best_hyperparams, f, indent=2)
        self.logger.info(f"Best hyperparameters saved to: {file_path}")

    def load_best_hyperparams(self):
        """Load the best hyperparameters from a JSON file."""
        file_path = os.path.join(self.out_dir, "best_hyperparams.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                self.best_hyperparams = json.load(f)
            self.logger.info(f"Loaded best hyperparameters from: {file_path}")
            return True
        else:
            self.logger.info(f"No hyperparameter file found at: {file_path}")
            return False

    def load_checkpoint(self, model, optimizer, scheduler=None, filename='best_checkpoint.pth'):
        checkpoint_path = os.path.join(self.out_dir, filename)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.logger.info(f"Resuming from checkpoint at epoch {start_epoch -1}")
            return model, optimizer, scheduler, start_epoch, best_val_loss
        else:
            # Check for interrupted checkpoint
            interrupted_checkpoint_path = os.path.join(self.out_dir, 'interrupted_checkpoint.pth')
            if os.path.exists(interrupted_checkpoint_path):
                checkpoint = torch.load(interrupted_checkpoint_path, map_location=self.device, weights_only=True)
                model.load_state_dict(checkpoint['model_state_dict'])

                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if scheduler and checkpoint.get('scheduler_state_dict'):
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                self.logger.info(f"Resuming from interrupted checkpoint at epoch {start_epoch -1}")
                return model, optimizer, scheduler, start_epoch, best_val_loss
            else:
                raise FileNotFoundError(f"No checkpoint found at {checkpoint_path} or {interrupted_checkpoint_path}")

    def save_training_metrics(self, train_losses, val_losses):
        metrics = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        with open(os.path.join(self.out_dir, 'training_metrics.json'), 'w') as f:
            json.dump(metrics, f)
        self.logger.info("Training metrics saved.")

    def _train_epoch(self, model, dataloader, criterion, optimizer, clip_grad_norm=None):
        """Train the model for one epoch."""
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch_X, batch_Y in dataloader:
            batch_X, batch_Y = batch_X.to(self.device), batch_Y.to(self.device)
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze(-1)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            if clip_grad_norm:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            batch_size = batch_Y.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
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