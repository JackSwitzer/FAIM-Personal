import unittest
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from config import Config
import sys
import os
import traceback
import datetime
from data_processor import DataProcessor
from trainer import LSTMTrainer
from models import LSTMModel
import torch.nn as nn
import logging
from datetime import datetime, timedelta

# Add the parent directory to the Python path to import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_synthetic_dataset(num_stocks=10, num_days=1000, num_features=50):
    # Generate dates
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]

    # Generate permnos (stock identifiers)
    permnos = [10001 + i for i in range(num_stocks)]

    # Prepare data container
    data = []

    # Generate data for each stock
    for permno in permnos:
        for date in dates:
            row = {
                'permno': permno,
                'date': date,
                'stock_exret': np.random.normal(0, 0.02)  # Random return, normal distribution
            }
            # Add features
            for i in range(1, num_features + 1):
                row[f'feature{i}'] = np.random.randn()  # Random feature value, standard normal distribution
            data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Sort the DataFrame
    df = df.sort_values(['permno', 'date']).reset_index(drop=True)

    return df

class BaseTestClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.synthetic_df = create_synthetic_dataset()
        cls.feature_cols = [f'feature{i}' for i in range(1, 51)]
        cls.target_col = 'stock_exret'

class TestDataProcessor(BaseTestClass):
    def setUp(self):
        self.processor = DataProcessor(data_in_path=None, ret_var=self.target_col, standardize=True)
        self.processor.stock_data = self.synthetic_df.copy()
        self.processor.feature_cols = self.feature_cols

    def test_preprocessing(self):
        self.processor.preprocess_data()
        self.assertIsNotNone(self.processor.stock_data)
        self.assertEqual(len(self.processor.feature_cols), 41)  # 35 PCA components + 6 cyclical features
        self.assertIn('PC1', self.processor.stock_data.columns)
        self.assertIn('PC35', self.processor.stock_data.columns)
        self.assertIn('month_sin', self.processor.stock_data.columns)
        self.assertIn('month_cos', self.processor.stock_data.columns)
        self.assertIn('day_of_week_sin', self.processor.stock_data.columns)
        self.assertIn('day_of_week_cos', self.processor.stock_data.columns)
        self.assertIn('quarter_sin', self.processor.stock_data.columns)
        self.assertIn('quarter_cos', self.processor.stock_data.columns)

class TestLSTMTrainer(BaseTestClass):
    def setUp(self):
        self.config = Config
        self.device = torch.device('cpu')
        self.trainer = LSTMTrainer(
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            device=self.device,
            config=self.config
        )
        self.trainer.data_processor.stock_data = self.synthetic_df.copy()
        self.trainer.data_processor.preprocess_data()  # Ensure data is preprocessed

    def test_create_sequences(self):
        seq_length = 10
        X, Y, indices = self.trainer.data_processor.create_sequences(self.trainer.data_processor.stock_data, seq_length)
        self.assertIsNotNone(X)
        self.assertIsNotNone(Y)
        self.assertIsNotNone(indices)
        if len(X) > 0:
            self.assertEqual(X.shape[1], seq_length)
            self.assertEqual(X.shape[2], len(self.feature_cols))  # Use the actual number of features

class TestDynamicDateSplitting(BaseTestClass):
    def setUp(self):
        self.processor = DataProcessor(data_in_path=None, ret_var=self.target_col, standardize=True)
        self.processor.stock_data = self.synthetic_df.copy()
        self.processor.feature_cols = self.feature_cols

    def test_dynamic_time_based_split(self):
        self.processor.preprocess_data()
        self.processor.split_data(method='time')
        self.assertIsNotNone(self.processor.train_data)
        self.assertIsNotNone(self.processor.val_data)
        self.assertIsNotNone(self.processor.test_data)
        # Add more specific assertions about the time-based split
        self.assertTrue(self.processor.train_data['date'].max() <= self.processor.val_data['date'].min())
        self.assertTrue(self.processor.val_data['date'].max() <= self.processor.test_data['date'].min())

# Add more test classes as needed

if __name__ == '__main__':
    unittest.main()