import os
import logging
import traceback
import pandas as pd
import torch.nn as nn

from data_processor import DataProcessor
from trainer import LSTMTrainer
from utils import *

# Set up logging
out_dir = r"C:\Users\jacks\Documents\Code\McGill FAIM\Data Output"
os.makedirs(out_dir, exist_ok=True)
setup_logging(out_dir)

set_seed()

def main():
    try:
        clear_gpu_memory()
        check_torch_version()
        device = check_cuda()
        data_input_dir = r"C:\Users\jacks\Documents\Code\McGill FAIM\Data Input"
        full_data_path = os.path.join(data_input_dir, "hackathon_sample_v2.csv")

        # Allow for configurable target variable
        target_variable = 'stock_exret'  # Change this to your desired target, e.g., 'stock_price'
        logging.info(f"Target variable set to: {target_variable}")

        # Data processing
        data_processor = DataProcessor(full_data_path, standardize=True)
        data_processor.load_data()
        data_processor.preprocess_data()
        data_processor.split_data()

        feature_cols = data_processor.feature_cols

        # Initialize LSTM Trainer
        lstm_trainer = LSTMTrainer(feature_cols, target_variable, device, out_dir=out_dir)

        # Load best hyperparameters or optimize if not available
        if not lstm_trainer.load_best_hyperparams():
            logging.info("Starting hyperparameter optimization...")
            best_hyperparams, _ = lstm_trainer.optimize_hyperparameters(
                data_processor.train_data,
                data_processor.val_data,
                n_trials=50
            )
            lstm_trainer.save_best_hyperparams()
        
        best_hyperparams = lstm_trainer.best_hyperparams
        
        # Create sequences and dataloaders
        X_train, Y_train, _ = lstm_trainer.create_sequences(data_processor.train_data, best_hyperparams['seq_length'])
        X_val, Y_val, _ = lstm_trainer.create_sequences(data_processor.val_data, best_hyperparams['seq_length'])
        X_test, Y_test, test_indices = lstm_trainer.create_sequences(data_processor.test_data, best_hyperparams['seq_length'])

        train_loader = lstm_trainer._create_dataloader(X_train, Y_train, best_hyperparams['batch_size'], shuffle=False)
        val_loader = lstm_trainer._create_dataloader(X_val, Y_val, best_hyperparams['batch_size'], shuffle=False)
        test_loader = lstm_trainer._create_dataloader(X_test, Y_test, best_hyperparams['batch_size'], shuffle=False)

        # Train the Final LSTM Model
        model, _ = lstm_trainer.train_model(train_loader, val_loader, best_hyperparams)

        # Evaluate on test set
        _, predictions, targets = lstm_trainer._evaluate(model, test_loader, nn.MSELoss(), return_predictions=True)

        # Prepare DataFrame with Predictions
        test_data = data_processor.test_data.iloc[test_indices]
        predictions_df = pd.DataFrame({
            'permno': test_data['permno'].values,
            'date': test_data['date'].values,
            'lstm_prediction': predictions,
            lstm_trainer.target_col: targets
        })

        # Merge predictions with test data
        reg_pred_lstm = data_processor.test_data.merge(predictions_df, on=['permno', 'date'], how='inner')

        # Calculate OOS R-squared
        yreal = reg_pred_lstm[lstm_trainer.target_col]
        ypred = reg_pred_lstm['lstm_prediction']
        r2_lstm = calculate_oos_r2(yreal.values, ypred.values)
        logging.info(f'LSTM OOS R2: {r2_lstm:.4f}')

        # Save LSTM predictions
        save_csv(reg_pred_lstm, out_dir, 'lstm_predictions.csv')

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()