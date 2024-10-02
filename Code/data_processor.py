import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from multiprocessing import Pool
from itertools import chain
import logging

from config import Config
from utils import get_logger

from torch.utils.data import DataLoader, Dataset

class SequenceDataset(Dataset):
    def __init__(self, data, seq_length, feature_cols, target_col):
        # Reset index to ensure consistency
        data = data.sort_values(['permno', 'date']).reset_index(drop=True)
        self.data = data
        self.seq_length = seq_length
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.indices = self._create_indices()

    def _create_indices(self):
        indices = []
        grouped = self.data.groupby('permno', as_index=False, sort=False)
        for _, group in grouped:
            group_indices = group.index.to_list()
            num_sequences = len(group_indices) - self.seq_length + 1
            if num_sequences <= 0:
                continue
            for i in range(num_sequences):
                start_idx = group_indices[i]
                end_idx = start_idx + self.seq_length - 1
                indices.append((start_idx, end_idx))
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx, end_idx = self.indices[idx]
        seq_data = self.data.iloc[start_idx:end_idx + 1]
        seq = seq_data[self.feature_cols].values.astype(np.float32)
        target = seq_data[self.target_col].values[-1].astype(np.float32)
        return torch.from_numpy(seq), torch.tensor(target)

class DataProcessor:
    """
    A class to handle data loading, preprocessing, transformation, and splitting.
    """
    def __init__(self, data_in_path, ret_var='stock_exret', standardize=True, seq_length=10, use_permco=False):
        self.logger = get_logger('stock_predictor')
        self.ret_var = ret_var
        self.data_in_path = data_in_path
        self.standardize = standardize
        self.scaler = None
        self.feature_cols = None
        self.stock_data = None  # Initialize stock_data
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.seq_length = seq_length  # Add seq_length attribute
        self.use_permco = use_permco
        self.id_column = 'permco' if use_permco else 'permno'

    def load_data(self):
        """Load the data from the CSV file and optimize data types."""
        self.stock_data = pd.read_csv(
            self.data_in_path,
            parse_dates=["date"],
            low_memory=False
        )
        self.logger.info(f"Data loaded from {self.data_in_path}")

        # Fix CUSIP numbers
        self._fix_cusip()

        # Optimize data types to reduce memory usage
        self._optimize_data_types()
        self.logger.info("Data types optimized to reduce memory usage.")

    def _fix_cusip(self):
        """Fix CUSIP numbers by left-padding with zeros."""
        if 'cusip' in self.stock_data.columns:
            self.stock_data['cusip'] = self.stock_data['cusip'].astype(str).str.zfill(8)
            self.logger.info("CUSIP numbers fixed by left-padding with zeros.")

    def _optimize_data_types(self):
        """Optimize data types to reduce memory usage."""
        try:
            # Convert ID column to int64
            self.stock_data[self.id_column] = pd.to_numeric(self.stock_data[self.id_column], downcast='integer').astype(np.int64)
            # Downcast other numeric columns
            for col in self.stock_data.select_dtypes(include=['float64']).columns:
                self.stock_data[col] = pd.to_numeric(self.stock_data[col], downcast='float')
            for col in self.stock_data.select_dtypes(include=['int64']).columns:
                if col != self.id_column:  # ID column is already handled
                    self.stock_data[col] = pd.to_numeric(self.stock_data[col], downcast='integer')
        except Exception as e:
            self.logger.error(f"Error optimizing data types: {e}")

    def preprocess_data(self):
        """Preprocess the data: handle missing values, select features, standardize, apply PCA, and add seasonal variables."""
        self.logger.info("Starting data preprocessing...")

        # Handle missing values and select features
        self.stock_data = self.stock_data.fillna(0)
        non_feature_cols = ["date", "permno", self.ret_var]
        self.feature_cols = [col for col in self.stock_data.columns if col not in non_feature_cols]

        # Standardization
        if self.standardize:
            self.scaler = StandardScaler()
            self.stock_data[self.feature_cols] = self.scaler.fit_transform(self.stock_data[self.feature_cols])

        # Apply PCA
        pca = PCA(n_components=35)
        pca_result = pca.fit_transform(self.stock_data[self.feature_cols])
        pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(35)])

        # Create cyclical features
        self.stock_data['month'] = pd.to_datetime(self.stock_data['date']).dt.month
        self.stock_data['day_of_week'] = pd.to_datetime(self.stock_data['date']).dt.dayofweek
        self.stock_data['quarter'] = pd.to_datetime(self.stock_data['date']).dt.quarter

        cyclical_features = pd.DataFrame({
            'month_sin': np.sin(2 * np.pi * self.stock_data['month'] / 12),
            'month_cos': np.cos(2 * np.pi * self.stock_data['month'] / 12),
            'day_of_week_sin': np.sin(2 * np.pi * self.stock_data['day_of_week'] / 7),
            'day_of_week_cos': np.cos(2 * np.pi * self.stock_data['day_of_week'] / 7),
            'quarter_sin': np.sin(2 * np.pi * self.stock_data['quarter'] / 4),
            'quarter_cos': np.cos(2 * np.pi * self.stock_data['quarter'] / 4),
        })

        # Combine PCA results and cyclical features
        self.stock_data = pd.concat([self.stock_data, pca_df, cyclical_features], axis=1)

        # Update feature columns
        self.feature_cols = list(pca_df.columns) + list(cyclical_features.columns)

        self.logger.info(f"Data preprocessing completed. Number of features: {len(self.feature_cols)}")

    def split_data(self, method='time'):
        """Split the data into train, validation, and test sets."""
        if method == 'time':
            # Implement time-based splitting logic
            sorted_data = self.stock_data.sort_values('date')
            train_size = int(0.8 * len(sorted_data))
            val_size = int(0.1 * len(sorted_data))
            self.train_data = sorted_data.iloc[:train_size]
            self.val_data = sorted_data.iloc[train_size:train_size+val_size]
            self.test_data = sorted_data.iloc[train_size+val_size:]
        else:
            # Existing random splitting logic
            unique_permnos = self.stock_data['permno'].unique()
            np.random.shuffle(unique_permnos)
            train_size = int(0.7 * len(unique_permnos))
            val_size = int(0.15 * len(unique_permnos))
            train_permnos = unique_permnos[:train_size]
            val_permnos = unique_permnos[train_size:train_size+val_size]
            test_permnos = unique_permnos[train_size+val_size:]

            self.train_data = self.stock_data[self.stock_data['permno'].isin(train_permnos)]
            self.val_data = self.stock_data[self.stock_data['permno'].isin(val_permnos)]
            self.test_data = self.stock_data[self.stock_data['permno'].isin(test_permnos)]

        self.logger.info(f"Data split completed. Train: {len(self.train_data)}, Val: {len(self.val_data)}, Test: {len(self.test_data)}")

    def _time_based_split(self):
        """
        Split data based on time, using 80%-10%-10% ratios for training, validation, and testing,
        rounded to the nearest year.
        """
        data = self.stock_data.copy()
        data.sort_values(['date', 'permno'], inplace=True)

        # Get the minimum and maximum dates
        min_date = data['date'].min()
        max_date = data['date'].max()
        self.logger.info(f"Data date range: {min_date.date()} to {max_date.date()}")

        # Calculate total number of years and round to the nearest integer
        total_years = int(round((max_date - min_date).days / 365.25))
        if total_years == 0:
            total_years = 1  # Ensure at least one year

        # Define ratios
        train_ratio = 0.8
        val_ratio = 0.1

        # Calculate number of years for each split, rounding to the nearest year
        train_years = int(round(total_years * train_ratio))
        val_years = int(round(total_years * val_ratio))
        test_years = total_years - train_years - val_years

        # Adjust if the sum does not equal total_years due to rounding
        if train_years + val_years + test_years < total_years:
            test_years += total_years - (train_years + val_years + test_years)

        self.logger.debug(f"Total years: {total_years}")
        self.logger.debug(f"Train years: {train_years}, Validation years: {val_years}, Test years: {test_years}")

        # Calculate split dates
        train_end_date = min_date + pd.DateOffset(years=train_years)
        val_end_date = train_end_date + pd.DateOffset(years=val_years)

        # Adjust dates to the actual available dates in the dataset
        train_end_date = data[data['date'] >= train_end_date]['date'].min()
        val_end_date = data[data['date'] >= val_end_date]['date'].min()

        self.logger.info(f"Train date range: {min_date.date()} to {train_end_date.date()}")
        self.logger.info(f"Validation date range: {train_end_date.date()} to {val_end_date.date()}")
        self.logger.info(f"Test date range: {val_end_date.date()} to {max_date.date()}")

        # Perform the split
        train_data = data[data['date'] < train_end_date].copy()
        val_data = data[(data['date'] >= train_end_date) & (data['date'] < val_end_date)].copy()
        test_data = data[data['date'] >= val_end_date].copy()

        # Reset index for each split
        train_data.reset_index(drop=True, inplace=True)
        val_data.reset_index(drop=True, inplace=True)
        test_data.reset_index(drop=True, inplace=True)

        # Log the actual split sizes in years
        actual_train_years = (train_end_date - min_date).days / 365.25
        actual_val_years = (val_end_date - train_end_date).days / 365.25
        actual_test_years = (max_date - val_end_date).days / 365.25

        self.logger.debug(f"Actual split (in years): Train: {actual_train_years:.2f}, "
                          f"Validation: {actual_val_years:.2f}, Test: {actual_test_years:.2f}")

        return train_data, val_data, test_data

    def get_features_and_target(self):
        """Get features and target variables for training, validation, and test sets."""
        X_train = self.train_data[self.feature_cols]
        Y_train = self.train_data[self.ret_var].values.astype('float32')

        X_val = self.val_data[self.feature_cols]
        Y_val = self.val_data[self.ret_var].values.astype('float32')

        X_test = self.test_data[self.feature_cols]
        Y_test = self.test_data[self.ret_var].values.astype('float32')

        return X_train, Y_train, X_val, Y_val, X_test, Y_test

    def create_sequences(self, data, seq_length):
        """Create sequences of data for LSTM training."""
        sequences = []
        for permno, group in data.groupby('permno'):
            group = group.sort_values('date')
            if len(group) >= seq_length:
                for i in range(len(group) - seq_length + 1):
                    seq_X = group[self.feature_cols].iloc[i:i+seq_length].values
                    seq_Y = group[self.ret_var].iloc[i+seq_length-1]
                    sequences.append((seq_X, seq_Y, group.index[i+seq_length-1]))
        
        if not sequences:
            return np.array([]), np.array([]), np.array([])
        
        X, Y, indices = zip(*sequences)
        return np.array(X), np.array(Y), np.array(indices)

    def parallel_create_sequences(self, data, seq_length, num_processes=None):
        try:
            with Pool(num_processes) as pool:
                chunks = np.array_split(data, num_processes)
                results = pool.starmap(self.create_sequences, [(chunk, seq_length) for chunk in chunks])
            return chain.from_iterable(results)
        except KeyboardInterrupt:
            self.logger.warning("KeyboardInterrupt received. Terminating workers.")
            pool.terminate()
            pool.join()
        finally:
            if 'pool' in locals():
                pool.close()
                pool.join()

    def get_min_group_length(self):
        """
        Calculate the minimum sequence length (number of data points) across all groups (stocks)
        in the training, validation, and test datasets.
        """
        if self.train_data is None or self.val_data is None or self.test_data is None:
            self.logger.warning("Data has not been split yet. Returning minimum length from all data.")
            return self.stock_data.groupby('permno').size().min()

        # Calculate minimum group length in training data
        train_group_lengths = self.train_data.groupby('permno').size()
        min_train_length = train_group_lengths.min() if not train_group_lengths.empty else float('inf')

        # Calculate minimum group length in validation data
        val_group_lengths = self.val_data.groupby('permno').size()
        min_val_length = val_group_lengths.min() if not val_group_lengths.empty else float('inf')

        # Calculate minimum group length in test data
        test_group_lengths = self.test_data.groupby('permno').size()
        min_test_length = test_group_lengths.min() if not test_group_lengths.empty else float('inf')

        # Find the overall minimum
        min_group_length = min(min_train_length, min_val_length, min_test_length)

        self.logger.info(f"Minimum group lengths - Train: {min_train_length}, Validation: {min_val_length}, Test: {min_test_length}")
        return min_group_length

    def filter_stocks_by_min_length_in_splits(self):
        min_len = self.seq_length
        # Define a helper function
        def filter_data(data):
            group_lengths = data.groupby('permno').size()
            valid_permnos = group_lengths[group_lengths >= min_len].index
            return data[data['permno'].isin(valid_permnos)].copy()

        self.train_data = filter_data(self.train_data)
        self.val_data = filter_data(self.val_data)
        self.test_data = filter_data(self.test_data)

        self.logger.info(f"After filtering, train data stocks: {self.train_data['permno'].nunique()}")
        self.logger.info(f"After filtering, validation data stocks: {self.val_data['permno'].nunique()}")
        self.logger.info(f"After filtering, test data stocks: {self.test_data['permno'].nunique()}")

    def create_dataloader(self, data, seq_length, batch_size, num_workers=Config.NUM_WORKERS):
        dataset = SequenceDataset(data, seq_length, self.feature_cols, self.ret_var)
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

        
    def get_min_group_length_across_splits(self):
        """
        Get the minimum group length across all splits after filtering.
        """
        min_train_length = self.train_data.groupby('permno').size().min()
        min_val_length = self.val_data.groupby('permno').size().min()
        min_test_length = self.test_data.groupby('permno').size().min()

        min_group_length = min(min_train_length, min_val_length, min_test_length)
        self.logger.info(f"Updated minimum group lengths - Train: {min_train_length}, Validation: {min_val_length}, Test: {min_test_length}")
        return min_group_length

    def preprocess_and_split_data(self):
        """
        Combine preprocessing and splitting steps with post-split filtering.
        """
        self.preprocess_data()
        self.split_data()
        self.filter_stocks_by_min_length_in_splits()
        self.min_group_length = self.get_min_group_length_across_splits()