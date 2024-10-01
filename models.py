import numpy as np
import os
import torch
import optuna
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import cross_val_score

from utils import get_logger

class SequenceDataset(Dataset):
    def __init__(self, data, seq_length, feature_cols, target_col):
        self.data = data.reset_index(drop=True)  # Ensure unique indices
        self.seq_length = seq_length
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.indices = self._create_indices()

    def _create_indices(self):
        indices = []
        grouped = self.data.groupby('permno')
        for _, group in grouped:
            group_length = len(group)
            if group_length >= self.seq_length:
                indices.extend([(group.index[i], group.index[i+self.seq_length-1]) 
                                for i in range(group_length - self.seq_length + 1)])
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx, end_idx = self.indices[idx]
        seq = self.data.loc[start_idx:end_idx, self.feature_cols].values
        target = self.data.loc[end_idx, self.target_col]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

class RegressionModels:
    """
    A class that encapsulates training and prediction with different regression models.
    """
    def __init__(self, Y_mean, out_dir="./Data Output/"):
        self.logger = get_logger()
        self.Y_mean = Y_mean
        self.models = {}
        self.predictions = {}
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
    
    def train_linear_regression(self, X_train, Y_train_dm):
        """Train a Linear Regression model and save hyperparameters."""
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X_train, Y_train_dm)
        self.models['linear_regression'] = reg
        self.logger.info("Linear Regression model trained.")
        # Save default hyperparameters
        hyperparams = {'fit_intercept': False}
        self.save_hyperparams('linear_regression', hyperparams)
    
    def train_lasso(self, X_train, Y_train_dm, hyperparams):
        """Train a Lasso model with given hyperparameters."""
        lasso = Lasso(fit_intercept=False, **hyperparams)
        lasso.fit(X_train, Y_train_dm)
        self.models['lasso'] = lasso
        self.logger.info("Lasso model trained.")
        # Save hyperparameters
        self.save_hyperparams('lasso', hyperparams)
    
    def train_ridge(self, X_train, Y_train_dm, hyperparams):
        """Train a Ridge model with given hyperparameters."""
        ridge = Ridge(fit_intercept=False, **hyperparams)
        ridge.fit(X_train, Y_train_dm)
        self.models['ridge'] = ridge
        self.logger.info("Ridge model trained.")
        # Save hyperparameters
        self.save_hyperparams('ridge', hyperparams)
    
    def train_elastic_net(self, X_train, Y_train_dm, hyperparams):
        """Train an ElasticNet model with given hyperparameters."""
        en = ElasticNet(fit_intercept=False, **hyperparams)
        en.fit(X_train, Y_train_dm)
        self.models['elastic_net'] = en
        self.logger.info("ElasticNet model trained.")
        # Save hyperparameters
        self.save_hyperparams('elastic_net', hyperparams)
    
    def optimize_lasso_hyperparameters(self, X_train, Y_train_dm, n_trials=50):
        """Optimize Lasso hyperparameters using Optuna."""
        def objective(trial):
            alpha = trial.suggest_float('alpha', 1e-5, 1e2, log=True)
            max_iter = trial.suggest_int('max_iter', 1000, 100000)
            tol = trial.suggest_float('tol', 1e-5, 1e-1, log=True)
            lasso = Lasso(fit_intercept=False, alpha=alpha, max_iter=max_iter, tol=tol)
            scores = cross_val_score(lasso, X_train, Y_train_dm, cv=5,
                                     scoring='neg_mean_squared_error', n_jobs=-1)
            return -scores.mean()
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        self.logger.info(f"Lasso best hyperparameters: {best_params}")
        
        # Train the model with best hyperparameters
        self.train_lasso(X_train, Y_train_dm, best_params)
    
    def optimize_ridge_hyperparameters(self, X_train, Y_train_dm, n_trials=50):
        """Optimize Ridge hyperparameters using Optuna."""
        def objective(trial):
            alpha = trial.suggest_float('alpha', 1e-5, 1e5, log=True)
            max_iter = trial.suggest_int('max_iter', 1000, 100000)
            tol = trial.suggest_float('tol', 1e-5, 1e-1, log=True)
            ridge = Ridge(fit_intercept=False, alpha=alpha, max_iter=max_iter, tol=tol)
            scores = cross_val_score(ridge, X_train, Y_train_dm, cv=5,
                                     scoring='neg_mean_squared_error', n_jobs=-1)
            return -scores.mean()
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        self.logger.info(f"Ridge best hyperparameters: {best_params}")
        
        # Train the model with best hyperparameters
        self.train_ridge(X_train, Y_train_dm, best_params)
    
    def optimize_elastic_net_hyperparameters(self, X_train, Y_train_dm, n_trials=50):
        """Optimize ElasticNet hyperparameters using Optuna."""
        def objective(trial):
            alpha = trial.suggest_float('alpha', 1e-5, 1e2, log=True)
            l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
            max_iter = trial.suggest_int('max_iter', 1000, 100000)
            tol = trial.suggest_float('tol', 1e-5, 1e-1, log=True)
            en = ElasticNet(fit_intercept=False, alpha=alpha, l1_ratio=l1_ratio,
                            max_iter=max_iter, tol=tol)
            scores = cross_val_score(en, X_train, Y_train_dm, cv=5,
                                     scoring='neg_mean_squared_error', n_jobs=-1)
            return -scores.mean()
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        self.logger.info(f"ElasticNet best hyperparameters: {best_params}")
        
        # Train the model with best hyperparameters
        self.train_elastic_net(X_train, Y_train_dm, best_params)
    
    def predict(self, X_test):
        """Generate predictions using the trained models."""
        for model_name, model in self.models.items():
            self.predictions[model_name] = model.predict(X_test) + self.Y_mean
        self.logger.info("Predictions generated.")
    
    def get_predictions(self):
        """Retrieve the predictions dictionary."""
        return self.predictions
    
    def save_hyperparams(self, model_name, hyperparams):
        """Save hyperparameters to a JSON file."""
        import json
        file_path = os.path.join(self.out_dir, f"{model_name}_hyperparams.json")
        with open(file_path, 'w') as f:
            json.dump(hyperparams, f, indent=2)
        self.logger.info(f"Hyperparameters for {model_name} saved to: {file_path}")

class LSTMModel(nn.Module):
    """
    LSTM Model for time series prediction.
    Includes optimizations for efficient training on large datasets.
    """
    def __init__(
        self, input_size, hidden_size=128, num_layers=2, dropout_rate=0.2,
        bidirectional=False, use_batch_norm=True, activation_function='LeakyReLU',
        fc1_size=64, fc2_size=32
    ):
        super(LSTMModel, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.bidirectional = bidirectional

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout_rate,
            bidirectional=bidirectional
        )
        lstm_output_size = hidden_size * (2 if bidirectional else 1)

        # Batch Normalization Layer
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(lstm_output_size)

        # Fully Connected Layers
        self.fc1 = nn.Linear(lstm_output_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, 1)

        # Activation and Dropout Layers
        if activation_function == 'ReLU':
            self.activation = nn.ReLU()
        elif activation_function == 'LeakyReLU':
            self.activation = nn.LeakyReLU()
        elif activation_function == 'ELU':
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()  # Default to ReLU
        
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # LSTM forward pass
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Get the last time step

        # Apply Batch Normalization if enabled
        if self.use_batch_norm:
            out = self.batch_norm(out)

        # Fully Connected Layers with Activation and Dropout
        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.fc3(out)

        return out

    def to_device(self, device):
        return self.to(device)