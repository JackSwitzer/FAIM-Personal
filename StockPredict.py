import os
import traceback
import pandas as pd
import torch.nn as nn

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
        logger.info(f"Train set size: {len(data_processor.train_data)}, "
                    f"Val set size: {len(data_processor.val_data)}, "
                    f"Test set size: {len(data_processor.test_data)}")

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

        # Adjust sequence length if necessary
        min_group_length = min(
            data_processor.train_data.groupby('permno').size().min(),
            data_processor.val_data.groupby('permno').size().min(),
            data_processor.test_data.groupby('permno').size().min()
        )

        if best_hyperparams['seq_length'] > min_group_length:
            best_hyperparams['seq_length'] = min_group_length
            logger.info(f"Adjusted sequence length to: {best_hyperparams['seq_length']} due to minimum group length.")

        # After hyperparameter tuning, use all data for final training if desired
        use_all_data = True  # Set this to False if you don't want to retrain on all data

        if use_all_data:
            logger.info("Using train and validation data for final training...")
            # Combine train and val data
            all_train_data = pd.concat([data_processor.train_data, data_processor.val_data])
            data_processor.train_data = all_train_data
            data_processor.val_data = None  # No validation during final training
            # Test data remains the same for evaluation

            # Recreate sequences with the combined data
            X_train, Y_train, _ = lstm_trainer.parallel_create_sequences(
                data_processor.train_data, best_hyperparams['seq_length']
            )
            logger.info(f"Sequences created for full data training. Train: {X_train.shape if X_train is not None else 'Empty'}")

            # Check if we have valid sequences before proceeding
            if X_train is None:
                logger.error("Unable to create valid sequences for training data after combining.")
                return

            # Create sequences for test data
            X_test, Y_test, test_indices = lstm_trainer.parallel_create_sequences(
                data_processor.test_data, best_hyperparams['seq_length']
            )
            logger.info(f"Sequences created for test data. Test: {X_test.shape if X_test is not None else 'Empty'}")

            if X_test is None:
                logger.error("Unable to create valid sequences for test data.")
                return

            train_loader = lstm_trainer._create_dataloader(
                X_train, Y_train, best_hyperparams['batch_size'], shuffle=True
            )
            val_loader = None  # No validation
            test_loader = lstm_trainer._create_dataloader(
                X_test, Y_test, best_hyperparams['batch_size'], shuffle=False
            )
        else:
            # Create sequences with the initial split
            X_train, Y_train, _ = lstm_trainer.parallel_create_sequences(
                data_processor.train_data, best_hyperparams['seq_length']
            )
            X_val, Y_val, _ = lstm_trainer.parallel_create_sequences(
                data_processor.val_data, best_hyperparams['seq_length']
            )
            X_test, Y_test, test_indices = lstm_trainer.parallel_create_sequences(
                data_processor.test_data, best_hyperparams['seq_length']
            )

            logger.info(f"Sequences created. Train: {X_train.shape if X_train is not None else 'Empty'}, "
                        f"Val: {X_val.shape if X_val is not None else 'Empty'}, "
                        f"Test: {X_test.shape if X_test is not None else 'Empty'}")

            # Check if we have valid sequences before proceeding
            if X_train is None or X_val is None or X_test is None:
                logger.error("Unable to create valid sequences for all datasets even after adjusting sequence length.")
                return

            train_loader = lstm_trainer._create_dataloader(
                X_train, Y_train, best_hyperparams['batch_size'], shuffle=True
            )
            val_loader = lstm_trainer._create_dataloader(
                X_val, Y_val, best_hyperparams['batch_size'], shuffle=False
            )
            test_loader = lstm_trainer._create_dataloader(
                X_test, Y_test, best_hyperparams['batch_size'], shuffle=False
            )

        # Start final training
        logger.info(f"Starting final LSTM model training with hyperparameters: {best_hyperparams}")
        model, _ = lstm_trainer.train_model(train_loader, val_loader, best_hyperparams.copy())
        logger.info("Final LSTM model training completed.")

        log_memory_usage()
        log_gpu_memory()

        # Evaluate on test set
        _, predictions, targets = lstm_trainer._evaluate(
            model, test_loader, nn.MSELoss(), return_predictions=True
        )
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
    # main_Regression()