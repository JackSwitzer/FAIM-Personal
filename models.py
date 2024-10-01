import numpy as np
import os
import torch
import optuna
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import cross_val_score

from utils import get_logger
from config import Config

class SequenceDataset(Dataset):
    def __init__(self, data, seq_length, feature_cols, target_col):
        self.logger = get_logger()
        self.seq_length = seq_length
        self.feature_cols = feature_cols
        self.target_col = target_col

        # Sort data and reset index
        data = data.sort_values(['permno', 'date']).reset_index(drop=True)

        # Store features and target as NumPy arrays for efficient access
        self.data_features = data[self.feature_cols].values.astype(np.float32)
        self.data_target = data[self.target_col].values.astype(np.float32)
        self.permno_array = data['permno'].values

        # Initialize self.indices
        self.indices = self._create_indices()
        self.logger.info(f"Number of sequences: {len(self.indices):,}")

    def _create_indices(self):
        indices = []
        permno = self.permno_array
        seq_length = self.seq_length
        total_length = len(permno)

        start_idx = 0
        while start_idx < total_length - seq_length + 1:
            current_permno = permno[start_idx]
            end_idx = start_idx

            # Find the subsequence where permno remains the same
            while end_idx < total_length and permno[end_idx] == current_permno:
                end_idx += 1

            group_length = end_idx - start_idx

            if group_length >= seq_length:
                for i in range(start_idx, end_idx - seq_length + 1):
                    indices.append((i, i + seq_length - 1))

            start_idx = end_idx

        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx, end_idx = self.indices[idx]
        seq = self.data_features[start_idx:end_idx + 1]
        target = self.data_target[end_idx]
        return torch.from_numpy(seq), torch.tensor(target, dtype=torch.float32)

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
    def __init__(self, input_size, **kwargs):
        super(LSTMModel, self).__init__()
        
        # Use Config defaults if not provided in kwargs
        params = Config.get_lstm_params()
        params.update(kwargs)
        
        self.hidden_size = params['hidden_size']
        self.num_layers = params['num_layers']
        self.dropout_rate = params['dropout_rate']
        self.bidirectional = params['bidirectional']
        self.use_batch_norm = params['use_batch_norm']
        self.activation_function = params['activation_function']
        self.fc1_size = params['fc1_size']
        self.fc2_size = params['fc2_size']

        self.use_batch_norm = self.use_batch_norm
        self.bidirectional = self.bidirectional

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size, self.hidden_size, self.num_layers,
            batch_first=True, dropout=self.dropout_rate,
            bidirectional=self.bidirectional
        )
        lstm_output_size = self.hidden_size * (2 if self.bidirectional else 1)

        # Batch Normalization Layer
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(lstm_output_size)

        # Fully Connected Layers
        self.fc1 = nn.Linear(lstm_output_size, self.fc1_size)
        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.fc3 = nn.Linear(self.fc2_size, 1)

        # Activation and Dropout Layers
        if self.activation_function == 'ReLU':
            self.activation = nn.ReLU()
        elif self.activation_function == 'LeakyReLU':
            self.activation = nn.LeakyReLU()
        elif self.activation_function == 'ELU':
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()  # Default to ReLU
        
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

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