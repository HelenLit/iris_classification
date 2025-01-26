"""
Script loads the latest trained model, data for inference and predicts results.
Imports necessary packages and modules.
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
import pandas as pd

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

os.environ['CONF_PATH'] = 'settings.json'
# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = os.path.join(os.path.dirname(ROOT_DIR), os.getenv('CONF_PATH'))

from data_process.processing import DataProcessor
from utils import get_project_dir, setup_logger
logger = setup_logger(__name__)

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
RESULTS_DIR = get_project_dir(conf['general']['results_dir'])
DATA_PREP_DIR = get_project_dir(conf['general']['data_prep_dir'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--infer_file", type=str, help="Specify inference data file", default=conf['inference']['inp_table_name'])
parser.add_argument("--out_path", type=str, help="Specify the path to the output table")
parser.add_argument("--model_name", type=str, help="File path of the trained model")
parser.add_argument("--scaler", type=str, help="Name for scaler deserialization file", default=conf['train']['scaler_filename'])
parser.add_argument("--max_rows_inf", type=int, help="Maximum number of rows for the model inference", default=conf['inference']['max_rows'])
parser.add_argument("--random_state", type=int, help="Random state used for reproducibility", default=conf['general']['random_state'])


def get_latest_model_path() -> str:
    """Gets the path of the latest saved model"""
    logger.info("Searching for the latest model in the directory.")
    latest = None
    for (dirpath, dirnames, filenames) in os.walk(MODEL_DIR):
        for filename in filenames:
            if not latest or datetime.strptime(latest, conf['general']['datetime_format'] + '.pickle') < \
                    datetime.strptime(filename, conf['general']['datetime_format'] + '.pickle'):
                latest = filename
    if latest:
        latest_path = os.path.join(MODEL_DIR, latest)
        logger.info(f"Latest model found: {latest_path}")
        return latest_path
    else:
        logger.error("No models found in the directory.")
        sys.exit(1)


def get_model_by_path(path: str):
    """Loads and returns the specified model"""
    logger.info(f"Attempting to load the model from {path}.")
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
            return model
    except Exception as e:
        logger.error(f"An error occurred while loading the model: {e}")
        sys.exit(1)


def get_inference_data(path: str) -> pd.DataFrame:
    """Loads and returns data for inference from the specified csv file"""
    logger.info(f"Loading inference data from {path}.")
    try:
        df = pd.read_csv(path)
        logger.info(f"Inference data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"An error occurred while loading inference data: {e}")
        sys.exit(1)


def predict_results(model, infer_data, infer_file_path: str) -> pd.DataFrame:
    """Predict the results and join it with the infer_data"""
    logger.info(f"Starting prediction on data from {infer_file_path}.")
    try:
        # Read the inference data from the CSV file into a DataFrame
        df = pd.read_csv(infer_file_path)
        # Generate predictions using the model
        results = model.predict(infer_data)
        logger.info(f"Prediction completed. Number of predictions: {len(results)}")
        # Convert predictions to a numpy array and add them as a new column in the DataFrame
        df['results'] = results.cpu().numpy()
        return df
    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        raise


def store_results(results: pd.DataFrame, path: str = None) -> None:
    """Store the prediction results in 'results' directory with current datetime as a filename"""
    if not path:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
            logger.info(f"Created directory: {RESULTS_DIR}")
        path = datetime.now().strftime(conf['general']['datetime_format']) + '.csv'
        path = os.path.join(RESULTS_DIR, path)
    pd.DataFrame(results).to_csv(path, index=False)
    logger.info(f"Results saved successfully to {path}.")


def main():
    try:
        logger.info("Starting the inference process.")
        args = parser.parse_args()
        inf_path = os.path.join(DATA_DIR, args.infer_file)

        data_proc = DataProcessor(os.path.join(DATA_PREP_DIR, args.scaler))
        X_inf_tensor, _ = data_proc.prepare_data(inf_path, False, args.max_rows_inf, args.random_state, inference=True)
        logger.info("Data processing completed for inference.")

        if args.model_name:
            model = get_model_by_path(args.model_name)
        else:
            model = get_model_by_path(get_latest_model_path())
        logger.info("Model loaded for predictions.")

        results = predict_results(model, X_inf_tensor, inf_path)
        logger.info("Prediction process completed.")

        store_results(results, args.out_path)
    except Exception as e:
        logger.error(f"An error occurred during inference process: {e}")
        raise

if __name__ == "__main__":
    main()