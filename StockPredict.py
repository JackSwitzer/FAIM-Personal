import os
import traceback
import pandas as pd
import torch.nn as nn

from data_processor import DataProcessor
from trainer import LSTMTrainer
from utils import *

def main():
    out_dir = r"C:\Users\jacks\Documents\Code\McGill FAIM\Data Output"
    os.makedirs(out_dir, exist_ok=True)
    setup_logging(out_dir)
    logger = get_logger()

    set_seed()
    try:
        clear_gpu_memory()
        check_torch_version()
        device = check_cuda()
        data_input_dir = r"C:\Users\jacks\Documents\Code\McGill FAIM\Data Input"
        full_data_path = os.path.join(data_input_dir, "hackathon_sample_v2.csv")

        # Allow for configurable target variable
        target_variable = 'stock_exret'  # Change this to your desired target, e.g., 'stock_price'
        logger.info(f"Target variable set to: {target_variable}")

        # Data processing
        data_processor = DataProcessor(full_data_path, target_variable, standardize=True)
        data_processor.load_data()
        data_processor.preprocess_data()
        data_processor.split_data()

        feature_cols = data_processor.feature_cols
        logger.info(f"Data processing completed. Number of features: {len(feature_cols)}")
        logger.info(f"Train set size: {len(data_processor.train_data)}, Val set size: {len(data_processor.val_data)}, Test set size: {len(data_processor.test_data)}")

        log_memory_usage()
        log_gpu_memory()

        # Initialize LSTM Trainer
        lstm_trainer = LSTMTrainer(feature_cols, target_variable, device, out_dir=out_dir)

        # Load best hyperparameters or optimize if not available
        best_hyperparams = lstm_trainer.load_hyperparams(is_best=True)
        if best_hyperparams is None:
            logger.info("Starting hyperparameter optimization...")
            best_hyperparams, _ = lstm_trainer.optimize_hyperparameters(
                data_processor.train_data,
                data_processor.val_data,
                n_trials=50
            )
        
        # Create sequences and dataloaders
        X_train, Y_train, _ = lstm_trainer.parallel_create_sequences(data_processor.train_data, best_hyperparams['seq_length'])
        X_val, Y_val, _ = lstm_trainer.parallel_create_sequences(data_processor.val_data, best_hyperparams['seq_length'])
        X_test, Y_test, test_indices = lstm_trainer.parallel_create_sequences(data_processor.test_data, best_hyperparams['seq_length'])

        logger.info(f"Sequences created. Train: {X_train.shape if X_train is not None else 'Empty'}, Val: {X_val.shape if X_val is not None else 'Empty'}, Test: {X_test.shape if X_test is not None else 'Empty'}")

        # Check if validation and test sequences are empty
        if X_val is None or X_test is None:
            logger.error("Validation or test sequences are empty. Adjusting sequence length.")
            # Adjust sequence length to ensure we have validation and test data
            min_length = min(len(data_processor.val_data), len(data_processor.test_data))
            best_hyperparams['seq_length'] = min(best_hyperparams['seq_length'], min_length - 1)
            
            # Recreate sequences with adjusted length
            X_train, Y_train, _ = lstm_trainer.parallel_create_sequences(data_processor.train_data, best_hyperparams['seq_length'])
            X_val, Y_val, _ = lstm_trainer.parallel_create_sequences(data_processor.val_data, best_hyperparams['seq_length'])
            X_test, Y_test, test_indices = lstm_trainer.parallel_create_sequences(data_processor.test_data, best_hyperparams['seq_length'])
            
            logger.info(f"Sequences recreated. Train: {X_train.shape if X_train is not None else 'Empty'}, Val: {X_val.shape if X_val is not None else 'Empty'}, Test: {X_test.shape if X_test is not None else 'Empty'}")

        # Check if we have valid sequences before proceeding
        if X_train is None or X_val is None or X_test is None:
            logger.error("Unable to create valid sequences for all datasets. Exiting.")
            return

        log_memory_usage()
        log_gpu_memory()

        train_loader = lstm_trainer._create_dataloader(X_train, Y_train, best_hyperparams['batch_size'], shuffle=False)
        val_loader = lstm_trainer._create_dataloader(X_val, Y_val, best_hyperparams['batch_size'], shuffle=False)
        test_loader = lstm_trainer._create_dataloader(X_test, Y_test, best_hyperparams['batch_size'], shuffle=False)

        # Train the Final LSTM Model
        logger.info(f"Starting LSTM model training with hyperparameters: {best_hyperparams}")
        model, _ = lstm_trainer.train_model(train_loader, val_loader, best_hyperparams.copy())  # Use a copy to avoid modifying the original
        logger.info("LSTM model training completed.")

        log_memory_usage()
        log_gpu_memory()

        # Evaluate on test set
        _, predictions, targets = lstm_trainer._evaluate(model, test_loader, nn.MSELoss(), return_predictions=True)
        logger.info(f"LSTM model evaluation completed. Number of predictions: {len(predictions)}")

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
        logger.info(f'LSTM OOS R2: {r2_lstm:.4f}')

        # Save LSTM predictions
        save_csv(reg_pred_lstm, out_dir, 'lstm_predictions.csv')

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # Cleanup
        clear_gpu_memory()
        logger.info("Training run completed.")

if __name__ == "__main__":
    main()