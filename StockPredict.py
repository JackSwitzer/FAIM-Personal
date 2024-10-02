import os
import logging
import traceback
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from config import Config
from data_processor import DataProcessor
from trainer import LSTMTrainer
from utils import *
from models import RegressionModels

def main_Regression(rank, world_size):
    setup(rank, world_size)
    out_dir = Config.OUT_DIR
    setup_logging(out_dir)
    logger = logging.getLogger(__name__)  # Use module-level logger
    set_seed(Config.SEED + rank)  # Adjusted set_seed call
    try:
        data_input_dir = Config.DATA_INPUT_DIR
        full_data_path = Config.FULL_DATA_PATH
        target_variable = Config.TARGET_VARIABLE  # Change this if needed
        logger.info(f"Target variable set to: {target_variable}")
        # Data processing
        data_processor = DataProcessor(full_data_path, target_variable, standardize=True, config=Config)
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

        # Check if hyperparameter optimization is enabled
        if Config.OPTIMIZE_REGRESSION_MODELS:
            # Hyperparameter tuning and training
            logger.info("Starting hyperparameter optimization for Lasso...")
            reg_models.optimize_lasso_hyperparameters(X_train, Y_train_dm, n_trials=Config.N_TRIALS)
            logger.info("Lasso hyperparameter optimization completed.")

            logger.info("Starting hyperparameter optimization for Ridge...")
            reg_models.optimize_ridge_hyperparameters(X_train, Y_train_dm, n_trials=Config.N_TRIALS)
            logger.info("Ridge hyperparameter optimization completed.")

            logger.info("Starting hyperparameter optimization for ElasticNet...")
            reg_models.optimize_elastic_net_hyperparameters(X_train, Y_train_dm, n_trials=Config.N_TRIALS)
            logger.info("ElasticNet hyperparameter optimization completed.")
        else:
            # Load existing hyperparameters if optimization is not enabled
            reg_models.load_hyperparams('lasso')
            reg_models.load_hyperparams('ridge')
            reg_models.load_hyperparams('elastic_net')
            # Train models with loaded hyperparameters
            reg_models.train_lasso(X_train, Y_train_dm, reg_models.hyperparams['lasso'])
            reg_models.train_ridge(X_train, Y_train_dm, reg_models.hyperparams['ridge'])
            reg_models.train_elastic_net(X_train, Y_train_dm, reg_models.hyperparams['elastic_net'])

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
        cleanup()
        logger.info("Regression run completed.")

def main_worker(rank, world_size, config, use_distributed):
    setup(rank, world_size)
    out_dir = config.OUT_DIR
    setup_logging(out_dir)
    logger = logging.getLogger(__name__)  # Use module-level logger
    set_seed(config.SEED + rank)  # Adjusted set_seed call

    try:
        data_input_dir = config.DATA_INPUT_DIR
        full_data_path = config.FULL_DATA_PATH
        target_variable = config.TARGET_VARIABLE
        logger.info(f"Target variable set to: {target_variable}")

        # Data processing
        processor = DataProcessor(
            data_in_path=full_data_path,
            ret_var=target_variable,
            standardize=config.STANDARDIZE,
            seq_length=config.LSTM_PARAMS.get('seq_length', 10),
            config=config
        )
        processor.load_data()
        processor.preprocess_and_split_data()

        # Retrieve the minimum group length across splits
        min_group_length = processor.get_min_group_length_across_splits()
        logger.info(f"Minimum group length across splits: {min_group_length}")

        # Adjust the sequence length if necessary
        max_seq_length = min(min_group_length, 30)  # Assuming 30 is your maximum desired seq_length
        config.LSTM_PARAMS['seq_length'] = max_seq_length
        logger.info(f"Sequence length adjusted to: {max_seq_length}")

        # Create datasets
        train_dataset = processor.train_data
        val_dataset = processor.val_data
        test_dataset = processor.test_data

        # Initialize trainer
        trainer = LSTMTrainer(
            feature_cols=processor.feature_cols,
            target_col=target_variable,
            device=config.DEVICE,
            config=config,
            rank=rank,
            world_size=world_size,
            use_distributed=use_distributed
        )

        # Check if hyperparameters have already been optimized
        hyperparams = trainer.load_hyperparams(is_best=True)
        if hyperparams is not None:
            logger.info("Best hyperparameters found. Skipping hyperparameter optimization.")
        else:
            # Optimize hyperparameters
            logger.info("Starting hyperparameter optimization.")
            hyperparams, _ = trainer.optimize_hyperparameters(
                train_dataset, val_dataset, test_dataset, n_trials=config.N_TRIALS
            )
            logger.info("Hyperparameter optimization completed.")

        # Update sequence length and batch size based on the best hyperparameters
        seq_length = hyperparams['seq_length']
        batch_size = hyperparams['batch_size']
        logger.info(f"Using seq_length: {seq_length} and batch_size: {batch_size} for training and evaluation.")

        # Create data loaders with optimized hyperparameters
        train_loader = trainer._create_dataloader(train_dataset, seq_length, batch_size)
        val_loader = trainer._create_dataloader(val_dataset, seq_length, batch_size) if val_dataset is not None and not val_dataset.empty else None
        test_loader = trainer._create_dataloader(test_dataset, seq_length, batch_size) if test_dataset is not None and not test_dataset.empty else None

        # Check if the full model has already been trained
        if trainer.check_model_exists('final_model.pth'):
            logger.info("Final model found. Loading the model.")
            model = trainer.load_model(os.path.join(config.MODEL_WEIGHTS_DIR, 'final_model.pth'), hyperparams)
        else:
            # Perform a full training run with the best hyperparameters
            logger.info("Starting full training run with the best hyperparameters.")
            model, training_history = trainer.train_model(train_loader, val_loader, test_loader, hyperparams)
            # Save the final model
            trainer.save_model(model, 'final_model.pth')
            logger.info("Final model saved.")

        # Evaluate the model on the test set
        logger.info("Evaluating the model on the test set.")
        try:
            test_loss, r2_score_value, test_predictions, test_targets, test_permnos, test_dates = trainer.evaluate_test_set(
                model, test_dataset, hyperparams
            )
            if test_loss is not None and r2_score_value is not None:
                logger.info(f"Test Loss: {test_loss:.4f}, Test R² Score: {r2_score_value:.4f}")
            else:
                logger.error("Test evaluation failed. Check previous error messages.")
        except Exception as e:
            logger.error(f"An error occurred during evaluation: {str(e)}")
            logger.error(traceback.format_exc())

        # Make rolling predictions over the entire dataset
        logger.info("Making rolling predictions over the entire dataset.")
        # Combine the train, validation, and test datasets
        all_data = pd.concat([processor.train_data, processor.val_data, processor.test_data])

        # Create a DataLoader for the entire dataset without shuffling
        full_loader = trainer._create_dataloader(all_data, seq_length, batch_size)

        try:
            all_predictions, all_targets, permnos, dates = trainer.predict_over_data(model, full_loader, hyperparams)
            logger.info(f"Length of all_predictions: {len(all_predictions)}")
            logger.info(f"Length of all_targets: {len(all_targets)}")
            logger.info(f"Length of permnos: {len(permnos)}")
            logger.info(f"Length of dates: {len(dates)}")

            if (len(all_predictions) == len(permnos) == len(dates) == len(all_targets)):
                logger.info("Starting to process and merge predictions...")
                # Select necessary columns from all_data
                all_data = all_data[['date', 'permno', 'stock_ticker', 'ret_eom', 'rf', 'stock_exret']]
                logger.info(f"Shape of all_data after column selection: {all_data.shape}")

                # Create predictions DataFrame
                predictions_df = pd.DataFrame({
                    'permno': permnos,
                    'date': dates,
                    'Predicted_Excess_Return': all_predictions,
                    'Actual_Excess_Return': all_targets
                })
                logger.info(f"Shape of predictions_df: {predictions_df.shape}")

                # Convert data types for merging
                logger.info("Converting data types for merging...")
                all_data['permno'] = all_data['permno'].astype(int)
                all_data['date'] = pd.to_datetime(all_data['date'])
                predictions_df['permno'] = predictions_df['permno'].astype(int)
                # Convert dates in predictions_df if they're not already datetime
                predictions_df['date'] = pd.to_datetime(predictions_df['date'])

                # Merge on 'permno' and 'date'
                logger.info("Merging datasets...")
                merged_results = pd.merge(all_data, predictions_df, on=['permno', 'date'], how='left')
                logger.info(f"Shape of merged_results: {merged_results.shape}")

                # Define the necessary columns
                necessary_columns = [
                    'date',
                    'permno',
                    'stock_ticker',
                    'Predicted_Excess_Return',
                    'Actual_Excess_Return',
                    'rf'  # Include risk-free rate for reference
                ]

                # Fill NaN values in Predicted_Excess_Return with 0
                merged_results['Predicted_Excess_Return'] = merged_results['Predicted_Excess_Return'].fillna(0)

                # Select only the necessary columns
                output_df = merged_results[necessary_columns]

                # Ensure the output DataFrame is sorted by date and permno
                output_df = output_df.sort_values(['date', 'permno'])

                logger.info(f"Shape of final output_df: {output_df.shape}")

                # Save the output DataFrame to CSV
                output_path = os.path.join(config.OUT_DIR, 'full_dataset_predictions.csv')
                output_df.to_csv(output_path, index=False)

                logger.info(f"Filtered predictions successfully saved to {output_path}.")
            else:
                logger.error("Mismatch in lengths between predictions and collected data.")

        except Exception as e:
            logger.error(f"An error occurred during post-prediction processing: {str(e)}")
            logger.error(traceback.format_exc())

    except Exception as e:
        logger.error(f"An error occurred in main_worker: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        cleanup(use_distributed)
        logger.info("Training run completed.")

def main():
    world_size = torch.cuda.device_count()
    use_distributed = Config.USE_DISTRIBUTED and world_size > 1
    if use_distributed:
        mp.spawn(main_worker,
                 args=(world_size, Config, use_distributed),
                 nprocs=world_size,
                 join=True)
    else:
        # Decide whether to run regression models or LSTM based on a config parameter
        if Config.RUN_REGRESSION_MODELS:
            main_Regression(0, 1)
        else:
            main_worker(0, 1, Config, use_distributed)

if __name__ == '__main__':
    main()