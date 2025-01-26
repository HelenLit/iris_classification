import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import pickle

from utils import setup_logger

logger = setup_logger(__name__)


def random_sampling(df: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    try:
        if not max_rows or max_rows < 0:
            logger.info('Max_rows not defined. Skipping sampling.')
        elif len(df) <= max_rows:
            logger.info('Size of dataframe is less than or equal to max_rows. Skipping sampling.')
        else:
            df = df.sample(n=max_rows, replace=False, random_state=random_state)
            logger.info(f'Random sampling performed. Sample size: {max_rows}')
        return df
    except Exception as e:
        logger.error(f"An error occurred during random sampling: {e}")
        raise


class DataProcessor:
    def __init__(self, scaler_path: str) -> None:
        self.scaler_path = scaler_path
        self.scaler = None

    def prepare_data(self, data_path: str, fit: bool, max_rows: int, random_state: int, inference=False):
        X_data_tensor, y_data_tensor = None, None
        try:
            # Load data
            df = self.load_data(data_path)
            # Perform random sampling if necessary
            df = random_sampling(df, max_rows, random_state)
            if inference:
                X_data = df
            else:
                # Separate features and target
                X_data, y_data = self.split_features_target(df)
                y_data_tensor = self.to_tensor(y_data, torch.long)
            X_scaled = self.scale_data(X_data, fit=fit)
            X_data_tensor = self.to_tensor(X_scaled, torch.float32)
            logger.info("Data preparation process completed successfully.")
            return X_data_tensor, y_data_tensor
        except Exception as e:
            logger.error(f"An error occurred during data preparation: {e}")
            raise

    def load_data(self, path: str) -> pd.DataFrame:
        try:
            logger.info(f"Loading data from {path}...")
            df = pd.read_csv(path)
            return df
        except Exception as e:
            logger.error(f"Failed to load data from {path}: {e}")
            raise

    def split_features_target(self, df: pd.DataFrame):
        try:
            logger.info("Splitting features and target...")
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            return X, y
        except Exception as e:
            logger.error("Error while splitting features and target: {e}")
            raise

    def scale_data(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        try:
            if fit:
                self.scaler = StandardScaler()
                logger.info("Fitting scaler on data...")
                self.scaler.fit(X)
                self.save_scaler()
            if self.scaler is None:
                logger.warning("No scaler was fit. Trying to load scaler from pickle file...")
                self.scaler = pickle.load(open(self.scaler_path, 'rb'))
            logger.info("Transforming data...")
            X_scaled = self.scaler.transform(X)
            return X_scaled
        except Exception as e:
            logger.error(f"Error during scaling data: {e}")
            raise

    def to_tensor(self, data, dtype):
        try:
            logger.info(f"Converting data to tensor with dtype {dtype}...")
            tensor = torch.tensor(data, dtype=dtype)
            return tensor
        except Exception as e:
            logger.error(f"Error while converting data to tensor: {e}")
            raise

    def save_scaler(self) -> None:
        try:
            logger.info(f"Saving scaler object to {self.scaler_path}...")
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info("Scaler object saved successfully.")
        except Exception as e:
            logger.error(f"Error saving scaler object: {e}")
            raise
