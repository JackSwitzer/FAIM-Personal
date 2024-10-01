import os
import traceback
import pandas as pd
import torch.nn as nn
from config import Config

from data_processor import DataProcessor
from trainer import LSTMTrainer
from utils import *
from models import RegressionModels

def main_Regression():
    out_dir = r"C:\Users\jacks\Documents\Code\McGill FAIM\Data Output"
    os.makedirs(out_dir, exist_ok=True)
    setup_logging(out_dir)
    logger = get_logger()

    set_seed()
    try:
        data_input_dir = r"C:\Users\jacks\Documents\Code\McGill FAIM\Data Input"
        full_data_path = os.path.join(data_input_dir, "hackathon_sample_v2.csv")

        target_variable = 'stock_exret'  # Change this if needed
        logger.info(f"Target variable set to: {target_variable}")

        # Data processing
        data_processor = DataProcessor(full_data_path, target_variable, standardize=True)
        data_processor.load_data()
        data_processor.preprocess_data()
        data_processor.split_data()

        X_train, Y_train, X_val, Y_val, X_test, Y_test = data_processor.get_features_and_target()
        feature_cols = data_processor.feature_cols
        logger.info(f"Data processing completed. Number of features: {len(feature_cols)}")

        # De-mean the training target
        Y_train_mean = np.mean(Y_train)
        Y_train_dm = Y_train - Y_train_mean

        # Initialize RegressionModels
        reg_models = RegressionModels(Y_train_mean, out_dir=out_dir)

        # Hyperparameter tuning and training
        logger.info("Starting hyperparameter optimization for Lasso...")
        reg_models.optimize_lasso_hyperparameters(X_train, Y_train_dm)
        logger.info("Lasso hyperparameter optimization completed.")

        logger.info("Starting hyperparameter optimization for Ridge...")
        reg_models.optimize_ridge_hyperparameters(X_train, Y_train_dm)
        logger.info("Ridge hyperparameter optimization completed.")

        logger.info("Starting hyperparameter optimization for ElasticNet...")
        reg_models.optimize_elastic_net_hyperparameters(X_train, Y_train_dm)
        logger.info("ElasticNet hyperparameter optimization completed.")

        # Train Linear Regression (no hyperparameters)
        reg_models.train_linear_regression(X_train, Y_train_dm)

        # Generate predictions
        reg_models.predict(X_test)

        # Get predictions
        predictions_dict = reg_models.get_predictions()

        # Prepare DataFrame with Predictions
        test_data = data_processor.test_data.copy()
        for model_name, predictions in predictions_dict.items():
            test_data[f'{model_name}_prediction'] = predictions

        # Evaluate models
        for model_name in predictions_dict.keys():
            y_pred = test_data[f'{model_name}_prediction']
            y_true = test_data[target_variable]
            r2 = calculate_oos_r2(y_true.values, y_pred.values)
            logger.info(f"{model_name} OOS R^2: {r2:.4f}")

        # Save predictions
        save_csv(test_data, out_dir, 'regression_predictions.csv')

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # Cleanup
        logger.info("Regression run completed.")

def main():
    # Initialize logging only once at the start
    setup_logging(Config.OUT_DIR)
    logger = get_logger()
    
    os.makedirs(Config.MODEL_WEIGHTS_DIR, exist_ok=True)
    set_seed(Config.SEED)
    clear_gpu_memory()
    check_torch_version()
    device = check_device()
    data_input_dir = r"C:\Users\jacks\Documents\Code\McGill FAIM\Data Input"
    full_data_path = os.path.join(data_input_dir, "hackathon_sample_v2.csv")

    target_variable = Config.TARGET_VARIABLE
    logger.info(f"Target variable set to: {target_variable}")

    try:
        # Data processing
        data_processor = DataProcessor(Config.FULL_DATA_PATH, Config.TARGET_VARIABLE, standardize=Config.STANDARDIZE)
        data_processor.load_data()
        data_processor.preprocess_data()
        data_processor.split_data()

        min_group_length = min(
            data_processor.train_data.groupby('permno').size().min(),
            data_processor.val_data.groupby('permno').size().min(),
            data_processor.test_data.groupby('permno').size().min()
        )
        logger.info(f"Minimum group length across all sets: {min_group_length}")

        feature_cols = data_processor.feature_cols
        logger.info(f"Data processing completed. Features: {len(feature_cols)}, "
                    f"Train: {len(data_processor.train_data)}, "
                    f"Val: {len(data_processor.val_data)}, "
                    f"Test: {len(data_processor.test_data)}")

        # Initialize LSTM Trainer
        lstm_trainer = LSTMTrainer(feature_cols, Config.TARGET_VARIABLE, device, 
                                   out_dir=Config.OUT_DIR, model_weights_dir=Config.MODEL_WEIGHTS_DIR)

        # Load best hyperparameters or optimize if not available
        best_hyperparams = lstm_trainer.load_hyperparams(is_best=True)
        if best_hyperparams is None:
            logger.info("Starting hyperparameter optimization...")
            best_hyperparams, _ = lstm_trainer.optimize_hyperparameters(
                data_processor.train_data,
                data_processor.val_data,
                data_processor.test_data,  # Add this line
                n_trials=Config.N_TRIALS
            )
            if best_hyperparams is None:
                logger.error("Hyperparameter optimization failed. Exiting.")
                return

        # Adjust sequence length if necessary
        if best_hyperparams['seq_length'] > min_group_length:
            best_hyperparams['seq_length'] = min_group_length
            logger.info(f"Adjusted sequence length to: {best_hyperparams['seq_length']} due to minimum group length.")

        # Make use_all_data configurable
        use_all_data = Config.USE_ALL_DATA  # Add this to your Config class
        if use_all_data:
            logger.info("Using train and validation data for final training...")
            all_train_data = pd.concat([data_processor.train_data, data_processor.val_data])
            data_processor.train_data = all_train_data
            data_processor.val_data = None  # No validation during final training
        else:
            logger.info("Using train data only for final training...")

        # Start final training
        logger.info(f"Starting final LSTM model training with hyperparameters: {best_hyperparams}")
        model, training_history = lstm_trainer.train_model(
            data_processor.train_data, 
            data_processor.val_data, 
            data_processor.test_data,  # Make sure this is passed
            best_hyperparams.copy()
        )
        logger.info("Final LSTM model training completed.")

        # Evaluate on test set
        test_loader = lstm_trainer._create_dataloader(
            data_processor.test_data, 
            best_hyperparams['seq_length'], 
            best_hyperparams['batch_size']
        )
        _, predictions, targets = lstm_trainer._evaluate(
            model, test_loader, nn.MSELoss(), return_predictions=True
        )
        logger.info(f"LSTM model evaluation completed. Number of predictions: {len(predictions)}")

        # Prepare DataFrame with Predictions
        test_data = data_processor.test_data
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
        save_csv(reg_pred_lstm, Config.OUT_DIR, 'lstm_predictions.csv')

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # Cleanup
        clear_gpu_memory()
        logger.info("Training run completed.")

if __name__ == "__main__":
    main()
    # main_Regression()