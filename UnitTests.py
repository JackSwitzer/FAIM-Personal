import unittest
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import sys
import os
import traceback

# Add the parent directory to the Python path to import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Code.data_processor import DataProcessor
from Code.trainer import LSTMTrainer
from Code.models import LSTMModel, StockDataset
import torch.nn as nn

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        try:
            # Create a small synthetic dataset
            data = {
                'permno': [10001]*5 + [10002]*5,
                'date': pd.date_range(start='2021-01-01', periods=5).tolist() * 2,
                'feature1': range(10),
                'feature2': range(10, 20),
                'stock_exret': range(-5, 5)
            }
            self.df = pd.DataFrame(data)
            self.data_in_path = 'test_data.csv'
            self.df.to_csv(self.data_in_path, index=False)
        except Exception as e:
            print(f"Error in TestDataProcessor.setUp: {str(e)}")
            print(traceback.format_exc())
            raise

    def test_data_loading(self):
        try:
            processor = DataProcessor(self.data_in_path, standardize=True)
            processor.load_data()
            self.assertIsNotNone(processor.stock_data)
            self.assertEqual(len(processor.stock_data), 10)
        except Exception as e:
            print(f"Error in test_data_loading: {str(e)}")
            print(traceback.format_exc())
            raise

    def test_preprocessing(self):
        try:
            processor = DataProcessor(self.data_in_path, standardize=True)
            processor.load_data()
            processor.preprocess_data()
            self.assertIsNotNone(processor.feature_cols)
            self.assertTrue('feature1' in processor.feature_cols)
        except Exception as e:
            print(f"Error in test_preprocessing: {str(e)}")
            print(traceback.format_exc())
            raise

    def tearDown(self):
        try:
            # Remove the test CSV file
            if os.path.exists(self.data_in_path):
                os.remove(self.data_in_path)
        except Exception as e:
            print(f"Error in TestDataProcessor.tearDown: {str(e)}")
            print(traceback.format_exc())

class TestLSTMTrainer(unittest.TestCase):
    def setUp(self):
        try:
            # Create a small synthetic dataset
            data = {
                'permno': [10001]*10,
                'date': pd.date_range(start='2021-01-01', periods=10),
                'feature1': range(10),
                'feature2': range(10, 20),
                'stock_exret': range(-5, 5)
            }
            self.df = pd.DataFrame(data)
            self.feature_cols = ['feature1', 'feature2']
            self.target_col = 'stock_exret'
            self.device = torch.device('cpu')
            self.trainer = LSTMTrainer(self.feature_cols, self.target_col, self.device)
        except Exception as e:
            print(f"Error in TestLSTMTrainer.setUp: {str(e)}")
            print(traceback.format_exc())
            raise

    def test_create_sequences(self):
        try:
            seq_length = 5
            X, Y, indices = self.trainer.create_sequences(self.df, seq_length)
            self.assertEqual(len(X), len(Y))
            self.assertEqual(len(X), len(self.df) - seq_length + 1)
            self.assertEqual(X.shape[1], seq_length)
        except Exception as e:
            print(f"Error in test_create_sequences: {str(e)}")
            print(traceback.format_exc())
            raise

class TestLSTMModel(unittest.TestCase):
    def test_forward_pass(self):
        try:
            batch_size = 16
            seq_length = 10
            input_size = 5  # Number of features
            hidden_size = 32
            model = LSTMModel(input_size=input_size, hidden_size=hidden_size)
            inputs = torch.randn(batch_size, seq_length, input_size)
            outputs = model(inputs)
            self.assertEqual(outputs.shape, (batch_size, 1))
        except Exception as e:
            print(f"Error in test_forward_pass: {str(e)}")
            print(traceback.format_exc())
            raise

if __name__ == '__main__':
    unittest.main()