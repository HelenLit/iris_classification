"""This script prepares the data, runs the training, and saves the model."""

import argparse
import os
import sys
import json
import time
from datetime import datetime
import mlflow

mlflow.autolog()

# Adds the root directory to the system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

from utils import get_project_dir, setup_logger
from data_process.processing import DataProcessor
from modeling.nn_model import Model

logger = setup_logger(__name__)

#os.environ['CONF_PATH'] = 'settings.json'

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = os.path.join(os.path.dirname(ROOT_DIR), os.getenv('CONF_PATH'))
# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
DATA_PREP_DIR = get_project_dir(conf['general']['data_prep_dir'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_file", type=str, help="Specify training data file", default=conf['train']['table_name'])
parser.add_argument("--test_file", type=str, help="Specify test data file", default=conf['test']['table_name'])
parser.add_argument("--model_path", type=str, help="Specify the path for the output model", default=None)
parser.add_argument("--epochs", type=int, help="Number of training epochs", default=conf['train']['epochs'])
parser.add_argument("--learning_rate", type=float, help="Learning rate for the optimizer", default=conf['train']['learning_rate'])
parser.add_argument("--dropout_rate", type=float, help="Dropout rate for the model", default=conf['train']['dropout_rate'])
parser.add_argument("--metric", type=str, help="Evaluation metric", default=conf['train']['metric'])
parser.add_argument("--max_rows_train", type=int, help="Maximum number of rows for training the model", default=conf['train']['max_rows'])
parser.add_argument("--max_rows_test", type=int, help="Maximum number of rows for testing the model", default=conf['test']['max_rows'])
parser.add_argument("--random_state", type=int, help="Random state used for reproducibility", default=conf['general']['random_state'])
parser.add_argument("--scaler", type=str, help="Name for scaler deserialization file", default=conf['train']['scaler_filename'])

class Training:
    def __init__(self, input_dim, output_dim, dropout_rate, epochs, learning_rate, metric):
        self.model = Model(input_dim, output_dim, dropout_rate)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.metric = metric

    def run_training(self, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor):
        try:
            logger.info(f"Train data size: {X_train_tensor.shape[0]}")
            logger.info(f"Test data size: {X_test_tensor.shape[0]}")

            start_time = time.time()
            self.model.fit(X_train_tensor, y_train_tensor, epochs=self.epochs, learning_rate=self.learning_rate, plot=False)
            end_time = time.time()
            logger.info(f"Training completed in {end_time - start_time} seconds.")
            self.evaluate(X_test_tensor, y_test_tensor)
        except Exception as e:
            logger.error(f"Error during training process: {e}")
            raise

    def evaluate(self, X_test_tensor, y_test_tensor):
        try:
            score = self.model.evaluate(X_test_tensor, y_test_tensor, metric=self.metric)
            return score
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise

    def save(self, path=None):
        try:
            if not os.path.exists(MODEL_DIR):
                os.makedirs(MODEL_DIR)
            if not path:
                path = os.path.join(MODEL_DIR, datetime.now().strftime(conf['general']['datetime_format']) + '.pickle')
            else:
                path = os.path.join(MODEL_DIR, path)
            self.model.save(path)
        except Exception as e:
            logger.error(f"Failed to save the model: {e}")
            raise


def main():
    args = parser.parse_args()

    train_path = os.path.join(DATA_DIR, args.train_file)
    test_path = os.path.join(DATA_DIR, args.test_file)

    data_proc = DataProcessor(os.path.join(DATA_PREP_DIR, args.scaler))
    X_train_tensor, y_train_tensor = data_proc.prepare_data(train_path, True, args.max_rows_train, args.random_state)
    X_test_tensor, y_test_tensor = data_proc.prepare_data(test_path, False, args.max_rows_test, args.random_state)

    input_dim, output_dim = X_train_tensor.shape[1], y_train_tensor.unique().numel()
    tr = Training(input_dim, output_dim, args.dropout_rate, args.epochs, args.learning_rate, args.metric)

    tr.run_training(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
    tr.save(args.model_path)


if __name__ == "__main__":
    main()