import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool

from utils import get_logger

class DataProcessor:
    """
    A class to handle data loading, preprocessing, transformation, and splitting.
    """
    def __init__(self, data_in_path, ret_var='stock_exret', standardize=True):
        self.logger = get_logger()
        self.ret_var = ret_var
        self.logger.info(f"Target variable (ret_var) set to: {self.ret_var}")
        self.data_in_path = data_in_path
        self.standardize = standardize
        self.scaler = None
        self.feature_cols = None
        self.stock_data = None  # Initialize stock_data
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def load_data(self):
        """Load the data from the CSV file and optimize data types."""
        self.stock_data = pd.read_csv(
            self.data_in_path,
            parse_dates=["date"],
            low_memory=False
        )
        self.logger.info(f"Data loaded from {self.data_in_path}")

        # Optimize data types to reduce memory usage
        self._optimize_data_types()
        self.logger.info("Data types optimized to reduce memory usage.")

    def _optimize_data_types(self):
      """Optimize data types to reduce memory usage."""
      # Convert 'permno' to int64
      self.stock_data['permno'] = pd.to_numeric(self.stock_data['permno'], downcast='integer').astype(np.int64)
      # Downcast other numeric columns
      for col in self.stock_data.select_dtypes(include=['float64']).columns:
          self.stock_data[col] = pd.to_numeric(self.stock_data[col], downcast='float')
      for col in self.stock_data.select_dtypes(include=['int64']).columns:
          if col != 'permno':  # 'permno' is already handled
              self.stock_data[col] = pd.to_numeric(self.stock_data[col], downcast='integer')

    def preprocess_data(self):
        """Preprocess the data: handle missing values, select features, and standardize if needed."""
        # Exclude non-feature columns
        non_feature_cols = ["year", "month", "date", "permno", self.ret_var]

        # Select numeric feature columns
        numeric_cols = self.stock_data.select_dtypes(include=['number']).columns.tolist()
        self.feature_cols = [col for col in numeric_cols if col not in non_feature_cols]

        # Handle missing values and ensure correct data types
        self.stock_data[self.feature_cols] = self.stock_data[self.feature_cols].fillna(0).astype('float32')

        # Standardize features if requested
        if self.standardize:
            self.scaler = StandardScaler()
            self.stock_data[self.feature_cols] = self.scaler.fit_transform(self.stock_data[self.feature_cols])
            self.logger.info("Data standardized.")
            # Cast to float32
            self.stock_data[self.feature_cols] = self.stock_data[self.feature_cols].astype('float32')

        self.logger.info(f"Columns after preprocessing: {self.stock_data.columns.tolist()}")
        if self.ret_var not in self.stock_data.columns:
            self.logger.error(f"Target column '{self.ret_var}' not found in the data.")

    def split_data(self, train_pct=None, val_pct=None, test_pct=None):
        """Split data into training, validation, and test sets."""
        if train_pct is None and val_pct is None and test_pct is None:
            # Time-based splitting
            self.train_data, self.val_data, self.test_data = self._time_based_split()
        else:
            # Percentage-based splitting
            self.train_data, self.val_data, self.test_data = self._percentage_based_split(train_pct, val_pct, test_pct)
        # Reset indices after splitting
        self.train_data.reset_index(drop=True, inplace=True)
        self.val_data.reset_index(drop=True, inplace=True)
        self.test_data.reset_index(drop=True, inplace=True)
        self.logger.info("Data split into training, validation, and test sets.")

    def _time_based_split(self):
        """Split data based on predefined time periods."""
        data = self.stock_data.copy()
        data.sort_values('date', inplace=True)

        # Define split dates
        train_end_date = datetime.datetime(2021, 12, 31)
        val_end_date = datetime.datetime(2022, 12, 31)

        # Split the data
        train_data = data[data['date'] <= train_end_date]
        val_data = data[(data['date'] > train_end_date) & (data['date'] <= val_end_date)]
        test_data = data[data['date'] > val_end_date]

        return train_data, val_data, test_data

    def _percentage_based_split(self, train_pct, val_pct, test_pct):
        """Split data based on specified percentages."""
        data = self.stock_data.copy()
        data.sort_values('date', inplace=True)
        total_len = len(data)
        train_end = int(train_pct * total_len)
        val_end = train_end + int(val_pct * total_len)

        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]

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

    def parallel_create_sequences(self, data, seq_length, num_processes=4):
        with Pool(num_processes) as pool:
            chunks = np.array_split(data, num_processes)
            results = pool.starmap(self.create_sequences, [(chunk, seq_length) for chunk in chunks])
        
        sequences = np.concatenate([r[0] for r in results])
        targets = np.concatenate([r[1] for r in results])
        indices = np.concatenate([r[2] for r in results])
        return sequences, targets, indices