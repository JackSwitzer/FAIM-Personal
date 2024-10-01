import unittest
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import sys
import os
import traceback
import datetime

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
            trainer = LSTMTrainer(self.feature_cols, self.target_col, 'cpu')
            X, Y, indices = trainer.create_sequences(self.df, seq_length)
            
            self.assertIsNotNone(X)
            self.assertIsNotNone(Y)
            self.assertIsNotNone(indices)
            
            expected_sequences = len(self.df) - seq_length + 1
            self.assertEqual(len(X), expected_sequences, 
                             f"Expected {expected_sequences} sequences, but got {len(X)}")
            self.assertEqual(len(Y), expected_sequences)
            self.assertEqual(len(indices), expected_sequences)
            
            self.assertEqual(X.shape[1], seq_length)
            self.assertEqual(X.shape[2], len(self.feature_cols))
            
            # Check if the sequences are correct
            for i in range(len(X)):
                np.testing.assert_array_equal(X[i], self.df[self.feature_cols].iloc[i:i+seq_length].values)
                self.assertEqual(Y[i], self.df[self.target_col].iloc[i+seq_length-1])
                self.assertEqual(indices[i], self.df.index[i+seq_length-1])
        except Exception as e:
            print(f"Error in test_create_sequences: {str(e)}")
            print(traceback.format_exc())
            raise

    def test_create_sequences_empty(self):
        seq_length = 100  # Larger than the group size
        trainer = LSTMTrainer(self.feature_cols, self.target_col, 'cpu')
        X, Y, indices = trainer.create_sequences(self.df, seq_length)
        
        self.assertIsNotNone(X)
        self.assertIsNotNone(Y)
        self.assertIsNotNone(indices)
        
        self.assertEqual(len(X), 0)
        self.assertEqual(len(Y), 0)
        self.assertEqual(len(indices), 0)

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

class TestDataSplittingAndSequenceCreation(unittest.TestCase):
    def setUp(self):
        # Create a synthetic dataset with controlled date ranges
        dates = pd.date_range(start='2021-01-01', periods=365, freq='D')
        permnos = [10001, 10002, 10003]
        data = []
        for permno in permnos:
            for date in dates:
                data.append({
                    'permno': permno,
                    'date': date,
                    'feature1': np.random.randn(),
                    'feature2': np.random.randn(),
                    'stock_exret': np.random.randn()
                })
        self.df = pd.DataFrame(data)
        self.data_in_path = 'test_data.csv'
        self.df.to_csv(self.data_in_path, index=False)

    def test_data_splitting(self):
        processor = DataProcessor(self.data_in_path, standardize=False)
        processor.load_data()
        processor.preprocess_data()
        processor.split_data()
        self.assertGreater(len(processor.train_data), 0)
        self.assertGreater(len(processor.val_data), 0)
        self.assertGreater(len(processor.test_data), 0)
        self.assertTrue('stock_exret' in processor.train_data.columns)

    def test_sequence_creation(self):
        processor = DataProcessor(self.data_in_path, standardize=False)
        processor.load_data()
        processor.preprocess_data()
        processor.split_data()
        trainer = LSTMTrainer(processor.feature_cols, processor.ret_var, torch.device('cpu'))
        seq_length = 5
        X_train, Y_train, _ = trainer.create_sequences(processor.train_data, seq_length)
        X_val, Y_val, _ = trainer.create_sequences(processor.val_data, seq_length)
        X_test, Y_test, _ = trainer.create_sequences(processor.test_data, seq_length)
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_val)
        self.assertIsNotNone(X_test)
        self.assertEqual(X_train.shape[1], seq_length)
        self.assertEqual(X_val.shape[1], seq_length)
        self.assertEqual(X_test.shape[1], seq_length)

    def tearDown(self):
        if os.path.exists(self.data_in_path):
            os.remove(self.data_in_path)

class TestDynamicDateSplitting(unittest.TestCase):
    def setUp(self):
        # Create a synthetic dataset spanning over 12 years
        dates = pd.date_range(end=datetime.datetime.today(), periods=12*365, freq='D')
        permnos = [10001, 10002, 10003]
        data = []
        for permno in permnos:
            for date in dates:
                data.append({
                    'permno': permno,
                    'date': date,
                    'feature1': np.random.randn(),
                    'feature2': np.random.randn(),
                    'stock_exret': np.random.randn()
                })
        self.df = pd.DataFrame(data)
        self.data_in_path = 'test_dynamic_data.csv'
        self.df.to_csv(self.data_in_path, index=False)

    def test_dynamic_time_based_split(self):
        processor = DataProcessor(self.data_in_path, standardize=False)
        processor.load_data()
        processor.preprocess_data()
        processor.split_data()
        
        # Check that datasets are not empty
        self.assertGreater(len(processor.train_data), 0)
        self.assertGreater(len(processor.val_data), 0)
        self.assertGreater(len(processor.test_data), 0)

        # Check date ranges
        train_end_date = processor.train_data['date'].max()
        val_start_date = processor.val_data['date'].min()
        val_end_date = processor.val_data['date'].max()
        test_start_date = processor.test_data['date'].min()

        self.assertLessEqual(train_end_date, val_start_date)
        self.assertLessEqual(val_end_date, test_start_date)
        
    def tearDown(self):
        if os.path.exists(self.data_in_path):
            os.remove(self.data_in_path)

class TestSequenceCreation(unittest.TestCase):
    def setUp(self):
        # Create a sample dataset
        dates = pd.date_range(start='2021-01-01', periods=100)
        permnos = [10001, 10002, 10003]
        data = []
        for permno in permnos:
            for date in dates:
                data.append({
                    'permno': permno,
                    'date': date,
                    'feature1': np.random.randn(),
                    'feature2': np.random.randn(),
                    'stock_exret': np.random.randn()
                })
        self.df = pd.DataFrame(data)
        self.feature_cols = ['feature1', 'feature2']
        self.target_col = 'stock_exret'

    def test_create_sequences_valid(self):
        trainer = LSTMTrainer(self.feature_cols, self.target_col, 'cpu')
        seq_length = 10
        X, Y, _ = trainer.create_sequences(self.df, seq_length)
        self.assertIsNotNone(X)
        self.assertIsNotNone(Y)
        self.assertEqual(X.shape[1], seq_length)
        self.assertEqual(X.shape[0], Y.shape[0])

    def test_create_sequences_short_data(self):
        short_df = self.df[self.df['permno'] == 10001].iloc[:5]  # Only 5 data points
        trainer = LSTMTrainer(self.feature_cols, self.target_col, 'cpu')
        seq_length = 10
        X, Y, _ = trainer.create_sequences(short_df, seq_length)
        self.assertIsNone(X)
        self.assertIsNone(Y)

    def test_create_sequences_varying_lengths(self):
        trainer = LSTMTrainer(self.feature_cols, self.target_col, 'cpu')
        for seq_length in [2, 5, 20, 50]:
            X, Y, _ = trainer.create_sequences(self.df, seq_length)
            self.assertIsNotNone(X)
            self.assertIsNotNone(Y)
            self.assertLessEqual(X.shape[1], seq_length)

    def test_parallel_create_sequences(self):
        trainer = LSTMTrainer(self.feature_cols, self.target_col, 'cpu')
        seq_length = 10
        X, Y, _ = trainer.parallel_create_sequences(self.df, seq_length)
        self.assertIsNotNone(X)
        self.assertIsNotNone(Y)
        self.assertEqual(X.shape[1], seq_length)
        self.assertEqual(X.shape[0], Y.shape[0])

if __name__ == '__main__':
    unittest.main()