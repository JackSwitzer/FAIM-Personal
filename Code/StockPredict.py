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
        cleanup()
        logger.info("Regression run completed.")

def main_worker(rank, world_size, config, use_distributed):
    setup(rank, world_size)
    out_dir = config.OUT_DIR
    setup_logging(out_dir)
    logger = logging.getLogger(__name__)  # Use module-level logger
    set_seed()

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
            logger.info("Final model found. Skipping full training.")
            # Load the final trained model
            model = trainer.load_model(os.path.join(config.MODEL_WEIGHTS_DIR, 'final_model.pth'), hyperparams)
        else:
            # Perform a full training run with the best hyperparameters
            logger.info("Starting full training run with the best hyperparameters.")
            model, training_history = trainer.train_model(train_loader, val_loader, test_loader, hyperparams)

        # Evaluate the model on the test set
        logger.info("Evaluating the model on the test set.")
        test_loss, r2_score, predictions, targets = trainer.evaluate_test_set(model, test_dataset, hyperparams)
        logger.info(f"Test Loss: {test_loss}, R² Score: {r2_score}")

        # Make rolling predictions over the entire dataset
        logger.info("Making rolling predictions over the entire dataset.")
        # Combine the train, validation, and test datasets
        all_data = pd.concat([processor.train_data, processor.val_data, processor.test_data])

        # Create a DataLoader for the entire dataset without shuffling
        full_loader = trainer._create_dataloader(all_data, seq_length, batch_size, shuffle=False)

        # Make predictions
        all_predictions, all_targets = trainer.predict_over_data(model, full_loader, hyperparams)

        # Save predictions
        logger.info("Saving predictions for the entire dataset.")
        results_df = all_data.copy()
        results_df['Predictions'] = all_predictions
        results_df['Targets'] = all_targets
        results_df.to_csv(os.path.join(config.OUT_DIR, 'full_dataset_predictions.csv'), index=False)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
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
        main_worker(0, 1, Config, use_distributed)

if __name__ == '__main__':
    main()     
    # main_Regression()
