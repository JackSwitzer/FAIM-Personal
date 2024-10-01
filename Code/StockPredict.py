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
    out_dir = Config.OUT_DIR
    setup_logging(out_dir)
    logger = get_logger()

    set_seed()
    try:
        data_input_dir = Config.DATA_INPUT_DIR
        full_data_path = Config.FULL_DATA_PATH

        target_variable = Config.TARGET_VARIABLE  # Change this if needed
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
        reg_models.optimize_lasso_hyperparameters(X_train, Y_train_dm, n_trials=100)
        logger.info("Lasso hyperparameter optimization completed.")

        logger.info("Starting hyperparameter optimization for Ridge...")
        reg_models.optimize_ridge_hyperparameters(X_train, Y_train_dm, n_trials=100)
        logger.info("Ridge hyperparameter optimization completed.")

        logger.info("Starting hyperparameter optimization for ElasticNet...")
        reg_models.optimize_elastic_net_hyperparameters(X_train, Y_train_dm, n_trials=100)
        logger.info("ElasticNet hyperparameter optimization completed.")

        # Train Linear Regression (no hyperparameters)
        reg_models.train_linear_regression(X_train, Y_train_dm)
        reg_models.save_model('linear_regression', reg_models.models['linear_regression'])
        reg_models.save_hyperparams('linear_regression', {'fit_intercept': False})

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
    try:
        setup_logging(Config.OUT_DIR)
        logger = get_logger()
        os.makedirs(Config.MODEL_WEIGHTS_DIR, exist_ok=True)
        set_seed(Config.SEED)
        clear_gpu_memory()
        check_torch_version()
        device = check_device()

        target_variable = Config.TARGET_VARIABLE  # Set the target variable
        logger.info(f"Starting training for target variable: {target_variable}")

        # Data processing
        data_processor = DataProcessor(
            Config.FULL_DATA_PATH,
            ret_var=target_variable,
            standardize=Config.STANDARDIZE
        )
        data_processor.load_data()
        data_processor.preprocess_data()
        data_processor.split_data()

        # Initialize LSTMTrainer
        lstm_trainer = LSTMTrainer(
            feature_cols=data_processor.feature_cols,
            target_col=target_variable
        )

        # Determine minimum group length
        min_group_length = data_processor.get_min_group_length()
        logger.info(f"Minimum group length across all sets: {min_group_length}")
        if min_group_length < Config.MIN_SEQUENCE_LENGTH:
            logger.warning(
                f"The minimum group length {min_group_length} is less than the required sequence length."
            )

        # Load best hyperparameters or optimize if not available
        best_hyperparams = lstm_trainer.load_hyperparams(is_best=True)
        if best_hyperparams is None:
            logger.info(f"Starting hyperparameter optimization for {target_variable}...")
            best_hyperparams, _ = lstm_trainer.optimize_hyperparameters(
                data_processor.train_data,
                data_processor.val_data,
                data_processor.test_data,
                n_trials=Config.N_TRIALS
            )
            if best_hyperparams is None:
                logger.error(f"Hyperparameter optimization failed for {target_variable}.")
                return

        # Adjust sequence length if necessary
        if best_hyperparams['seq_length'] > min_group_length:
            best_hyperparams['seq_length'] = min_group_length
            logger.info(
                f"Adjusted sequence length to: {best_hyperparams['seq_length']} "
                f"due to minimum group length."
            )

        # Start final training
        logger.info(
            f"Starting final LSTM model training for {target_variable} "
            f"with hyperparameters: {best_hyperparams}"
        )

        model, training_history = lstm_trainer.train_model(
            data_processor.train_data,
            data_processor.val_data,
            data_processor.test_data,
            best_hyperparams.copy()
        )
        logger.info(f"Final LSTM model training completed for {target_variable}.")

        # Evaluate on test set
        predictions, targets = lstm_trainer.evaluate_test_set(model, data_processor.test_data, best_hyperparams)
        logger.info(
            f"LSTM model evaluation completed for {target_variable}. "
            f"Number of predictions: {len(predictions)}"
        )

        # Prepare DataFrame with Predictions
        test_data = data_processor.test_data.copy()
        test_data.reset_index(drop=True, inplace=True)
        test_data['lstm_prediction'] = predictions

        # Calculate OOS R-squared
        yreal = test_data[target_variable]
        ypred = test_data['lstm_prediction']
        r2_lstm = calculate_oos_r2(yreal.values, ypred)
        logger.info(f'{target_variable} LSTM OOS R2: {r2_lstm:.4f}')

        # Save LSTM predictions
        output_filename = 'lstm_predictions.csv'
        save_csv(test_data, Config.OUT_DIR, output_filename)

    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Cleaning up...")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("Process finished.")
        clear_gpu_memory()

if __name__ == "__main__":
    try:
        main()
        # main_Regression()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Cleaning up...")
    finally:
        print("Process finished.")