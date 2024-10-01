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
    setup(rank, world_size, use_distributed=use_distributed)
    set_seed(config.SEED + rank)  # Ensure different seeds for each process
    setup_logging(config.OUT_DIR, f"train_log_rank_{rank}.log")

    # Set device for this process
    if use_distributed:
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
    else:
        device = check_device()

    # Initialize DataProcessor
    data_processor = DataProcessor(config.FULL_DATA_PATH, ret_var=config.TARGET_VARIABLE)
    data_processor.load_data()
    data_processor.preprocess_and_split_data()

    # Adjust sequence length based on the minimum group length
    min_group_length = data_processor.min_group_length
    trainer = LSTMTrainer(
        feature_cols=data_processor.feature_cols,
        target_col=data_processor.ret_var,
        device=device,
        config=config,
        rank=rank,
        world_size=world_size,
        use_distributed=use_distributed
    )
    trainer.adjust_sequence_length(min_group_length)

    # Now pass datasets to optimize_hyperparameters
    train_dataset = data_processor.train_data
    val_dataset = data_processor.val_data
    test_dataset = data_processor.test_data

    # Load or optimize hyperparameters
    hyperparams = trainer.load_hyperparams(is_best=True)
    if hyperparams is None:
        hyperparams, _ = trainer.optimize_hyperparameters(
            train_dataset, val_dataset, test_dataset, n_trials=config.N_TRIALS
        )

    # Create data loaders with optimized hyperparameters
    seq_length = hyperparams['seq_length']
    batch_size = hyperparams['batch_size']
    train_loader = trainer._create_dataloader(train_dataset, seq_length, batch_size)
    val_loader = trainer._create_dataloader(val_dataset, seq_length, batch_size)
    test_loader = trainer._create_dataloader(test_dataset, seq_length, batch_size)

    # Train the model
    model, training_history = trainer.train_model(train_loader, val_loader, test_loader, hyperparams)

    # Clean up
    cleanup(use_distributed)

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