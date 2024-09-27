import os
import logging
import traceback
import random
import numpy as np

import pandas as pd
import torch.nn as nn

from data_processor import DataProcessor
from trainer import LSTMTrainer
from utils import *


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

def main():
    try:
        check_torch_version()
        args = parse_args()
        clear_gpu_memory()
        device = check_cuda()
        data_input_dir = args.data_input_dir
        out_dir = args.out_dir
        os.makedirs(out_dir, exist_ok=True)
        setup_logging(os.path.join(out_dir, 'training.log'))
        full_data_path = os.path.join(data_input_dir, args.data_file)

        # Data processing
        data_processor = DataProcessor(full_data_path, standardize=True)
        data_processor.load_data()
        data_processor.preprocess_data()
        data_processor.split_data()

        feature_cols = data_processor.feature_cols

        # Initialize LSTM Trainer
        lstm_trainer = LSTMTrainer(feature_cols, 'stock_exret', device, out_dir=out_dir)

        # Hyperparameter optimization
        if args.optimize:
            logging.info("Starting hyperparameter optimization...")
            best_hyperparams, _ = lstm_trainer.optimize_hyperparameters(
                data_processor.train_data,
                data_processor.val_data,
                n_trials=args.n_trials
            )
            logging.info("Hyperparameter optimization completed.")
        elif lstm_trainer.load_best_hyperparams():
            best_hyperparams = lstm_trainer.best_hyperparams
        else:
            logging.error("No hyperparameters found. Please run optimization first.")
            return

        # Override hyperparameters based on command-line arguments
        best_hyperparams['num_epochs'] = args.num_epochs
        best_hyperparams['batch_size'] = args.batch_size
        best_hyperparams['seq_length'] = args.seq_length

        logging.info(f"Training for {best_hyperparams['num_epochs']} epochs")

        # Combine Training and Validation Data for Final Training
        combined_train_data = pd.concat([data_processor.train_data, data_processor.val_data])
        combined_train_data.reset_index(drop=True, inplace=True)

        # Create Sequences for Training and Testing
        seq_length = best_hyperparams['seq_length']
        X_train_combined, Y_train_combined, train_indices = lstm_trainer.create_sequences(combined_train_data, seq_length)
        X_test_seq, Y_test_seq, test_indices = lstm_trainer.create_sequences(data_processor.test_data, seq_length)

        # Create DataLoaders
        batch_size = best_hyperparams['batch_size']
        train_loader = lstm_trainer._create_dataloader(X_train_combined, Y_train_combined, batch_size, shuffle=True)
        test_loader = lstm_trainer._create_dataloader(X_test_seq, Y_test_seq, batch_size, shuffle=False)

        # Train the Final LSTM Model
        model, _ = lstm_trainer.train_model(
            train_loader,
            None,  # No validation loader for final training
            best_hyperparams
        )
        logging.info("Final model training completed.")

        # Make Predictions on the Test Data
        test_loss, predictions, targets = lstm_trainer._evaluate(
            model,
            test_loader,
            nn.MSELoss(),
            return_predictions=True
        )
        logging.info(f"Test loss: {test_loss:.4f}")

        # Get permnos and dates using test_indices
        test_data = data_processor.test_data.reset_index()
        permnos = test_data.loc[test_indices, 'permno'].values
        dates = test_data.loc[test_indices, 'date'].values

        # Prepare DataFrame with Predictions using permno and date
        predictions_df = pd.DataFrame({
            'permno': permnos,
            'date': dates,
            'lstm_prediction': predictions,
            lstm_trainer.target_col: targets
        })

        # Ensure 'permno' columns are of the same dtype (int64)
        data_processor.test_data['permno'] = data_processor.test_data['permno'].astype(np.int64)
        predictions_df['permno'] = predictions_df['permno'].astype(np.int64)

        # Convert 'date' columns to datetime64[ns]
        data_processor.test_data['date'] = pd.to_datetime(data_processor.test_data['date'])
        predictions_df['date'] = pd.to_datetime(predictions_df['date'])

        # Debug: Check data types before merge
        logging.info(f"Data types in data_processor.test_data:\n{data_processor.test_data[['permno', 'date']].dtypes}")
        logging.info(f"Data types in predictions_df:\n{predictions_df[['permno', 'date']].dtypes}")

        # Merge predictions with test data on 'permno' and 'date'
        reg_pred_lstm = data_processor.test_data.merge(predictions_df, on=['permno', 'date'], how='inner')

        # Debug information
        logging.info(f"Columns in reg_pred_lstm: {reg_pred_lstm.columns.tolist()}")
        logging.info(f"Shape of reg_pred_lstm: {reg_pred_lstm.shape}")

        # Ensure the target column exists in reg_pred_lstm
        if lstm_trainer.target_col not in reg_pred_lstm.columns:
            logging.error(f"Column '{lstm_trainer.target_col}' not found in the merged DataFrame.")
            logging.info(f"Available columns: {reg_pred_lstm.columns.tolist()}")
            return  # Exit the function if the column is not present

        # Evaluate and Save Results
        if not reg_pred_lstm.empty:
            yreal = reg_pred_lstm[lstm_trainer.target_col]
            ypred = reg_pred_lstm['lstm_prediction']
            r2_lstm = calculate_oos_r2(yreal.values, ypred.values)
            logging.info(f'LSTM OOS R2: {r2_lstm:.4f}')

            # Save LSTM predictions
            save_csv(reg_pred_lstm, out_dir, 'lstm_predictions.csv')
        else:
            logging.info("No predictions were made due to insufficient test data.")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error(traceback.format_exc())
        # Optionally, you can send an email or notification here
    finally:
        # Clean up resources, if any
        clear_gpu_memory()

if __name__ == "__main__":
    main()