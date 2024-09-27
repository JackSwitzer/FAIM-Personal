import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV
import logging

class StockDataset(Dataset):
    """
    Custom Dataset for handling stock sequences for LSTM input.
    Efficiently handles large datasets by utilizing memory mapping.
    """
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class RegressionModels:
    """
    A class that encapsulates training and prediction with different regression models.
    """
    def __init__(self, Y_mean):
        self.Y_mean = Y_mean
        self.models = {}
        self.predictions = {}

    def train_linear_regression(self, X_train, Y_train_dm):
        """Train a Linear Regression model."""
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X_train, Y_train_dm)
        self.models['ols'] = reg
        logging.info("Linear Regression model trained.")

    def train_lasso(self, X_train, Y_train_dm, alphas=None):
        """Train a Lasso Regression model with cross-validation."""
        if alphas is None:
            alphas = np.logspace(-4, 4, 100)
        lasso = Lasso(fit_intercept=False, max_iter=1000000)
        grid_search = GridSearchCV(
            lasso, {'alpha': alphas}, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
        )
        grid_search.fit(X_train, Y_train_dm)
        self.models['lasso'] = grid_search.best_estimator_
        logging.info(f"Lasso Regression model trained with alpha: {grid_search.best_estimator_.alpha}")

    def train_ridge(self, X_train, Y_train_dm, alphas=None):
        """Train a Ridge Regression model with cross-validation."""
        if alphas is None:
            alphas = np.logspace(-1, 8, 100)
        ridge = Ridge(fit_intercept=False, max_iter=1000000)
        grid_search = GridSearchCV(
            ridge, {'alpha': alphas}, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
        )
        grid_search.fit(X_train, Y_train_dm)
        self.models['ridge'] = grid_search.best_estimator_
        logging.info(f"Ridge Regression model trained with alpha: {grid_search.best_estimator_.alpha}")

    def train_elastic_net(self, X_train, Y_train_dm, alphas=None, l1_ratios=None):
        """Train an Elastic Net Regression model with cross-validation."""
        if alphas is None:
            alphas = np.logspace(-4, 4, 50)
        if l1_ratios is None:
            l1_ratios = np.linspace(0.1, 0.9, 9)
        en = ElasticNet(fit_intercept=False, max_iter=1000000)
        param_grid = {'alpha': alphas, 'l1_ratio': l1_ratios}
        grid_search = GridSearchCV(
            en, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
        )
        grid_search.fit(X_train, Y_train_dm)
        self.models['en'] = grid_search.best_estimator_
        logging.info(
            f"Elastic Net model trained with alpha: {grid_search.best_estimator_.alpha}, "
            f"l1_ratio: {grid_search.best_estimator_.l1_ratio}"
        )

    def predict(self, X_test):
        """Generate predictions using the trained models."""
        for model_name, model in self.models.items():
            self.predictions[model_name] = model.predict(X_test) + self.Y_mean
        logging.info("Predictions generated.")

    def get_predictions(self):
        """Retrieve the predictions dictionary."""
        return self.predictions

class LSTMModel(nn.Module):
    """
    LSTM Model for time series prediction.
    Includes optimizations for efficient training on large datasets.
    """
    def __init__(
        self, input_size, hidden_size=128, num_layers=2, dropout_rate=0.0,
        bidirectional=False, use_batch_norm=False, activation_function=None,
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
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

        # Optional Activation Function
        if activation_function == 'ReLU':
            self.activation = nn.ReLU()
        elif activation_function == 'Tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = None

    def forward(self, x):
        # LSTM forward pass
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Get the last time step

        # Apply Batch Normalization if enabled
        if self.use_batch_norm:
            out = self.batch_norm(out)

        # Fully Connected Layers with Activation and Dropout
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)

        # Apply optional activation function
        if self.activation:
            out = self.activation(out)

        return out