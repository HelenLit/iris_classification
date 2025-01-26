import unittest
import os
import sys
import json
import pandas as pd
import torch
import tempfile
import shutil

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

CONF_FILE = os.path.join(os.path.dirname(ROOT_DIR), os.getenv('CONF_PATH'))

from data_process.processing import DataProcessor
from training.train import Training
from modeling.nn_model import Model


class TestDirectoriesAndFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONF_FILE, "r") as file:
            cls.conf = json.load(file)

    def test_data_directory_exists_and_not_empty(self):
        data_dir = self.conf['general']['data_dir']
        data_dir = os.path.join(os.path.dirname(ROOT_DIR), data_dir)
        self.assertTrue(os.path.exists(data_dir), f"Data directory {data_dir} does not exist")
        self.assertTrue(os.listdir(data_dir), f"Data directory {data_dir} is empty")

    def test_training_directory_exists_and_not_empty(self):
        training_dir = os.path.join(os.path.dirname(ROOT_DIR), 'training')
        self.assertTrue(os.path.exists(training_dir), f"Training directory {training_dir} does not exist")
        self.assertTrue(os.listdir(training_dir), f"Training directory {training_dir} is empty")

    def test_modeling_directory_exists_and_not_empty(self):
        modeling_dir = os.path.join(os.path.dirname(ROOT_DIR), 'modeling')
        self.assertTrue(os.path.exists(modeling_dir), f"Modeling directory {modeling_dir} does not exist")
        self.assertTrue(os.listdir(modeling_dir), f"Modeling directory {modeling_dir} is empty")

    def test_requirements_file_exists(self):
        requirements_path = os.path.join(os.path.dirname(ROOT_DIR), 'requirements.txt')
        self.assertTrue(os.path.exists(requirements_path), "requirements.txt file does not exist")


class TestDataProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONF_FILE, "r") as file:
            cls.conf = json.load(file)
        cls.data_dir = cls.conf['general']['data_dir']
        cls.random_state = cls.conf['general']['random_state']
        data_prep_dir = os.path.join(os.path.dirname(ROOT_DIR), cls.conf['general']['data_prep_dir'])
        cls.scaler_path = os.path.join(data_prep_dir, 'scaler.pickle')
        cls.processor = DataProcessor(cls.scaler_path)

    def del_scaler_path(self):
        if os.path.exists(self.scaler_path):
            os.remove(self.scaler_path)

    def setUp(self):
        self.df_sample = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [4.0, 3.0, 2.0, 1.0],
            'target': [0, 1, 0, 1]
        })
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_prepare_data_creates_scaler_file(self):
        data_path = os.path.join(self.tmpdir, 'sample_data.csv')
        self.df_sample.to_csv(data_path, index=False)
        self.processor.prepare_data(data_path, fit=True, max_rows=2, random_state=self.random_state)
        self.assertTrue(os.path.exists(self.scaler_path), "Scaler file was not created")
        # Delete the scaler file after creation
        self.del_scaler_path()

    def test_prepare_data_with_artificial_dataset(self):
        data_path = os.path.join(self.tmpdir, 'sample_data.csv')
        self.df_sample.to_csv(data_path, index=False)
        X_data_tensor, y_data_tensor = self.processor.prepare_data(data_path, fit=True, max_rows=2, random_state=self.random_state)
        self.assertIsInstance(X_data_tensor, torch.Tensor, "X_data_tensor is not a torch tensor")
        self.assertIsInstance(y_data_tensor, torch.Tensor, "y_data_tensor is not a torch tensor")

    def test_scale_data(self):
        scaled_data = self.processor.scale_data(self.df_sample[['feature1', 'feature2']], fit=True)
        self.assertEqual(scaled_data.shape, (4, 2), "Scaled data shape mismatch")


class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONF_FILE, "r") as file:
            cls.conf = json.load(file)
        cls.random_state = cls.conf['general']['random_state']


    def setUp(self):
        self.df_sample = pd.DataFrame({
            'feature1': [0.1, 0.2, 0.3, 0.4],
            'feature2': [0.0, 0.3, 0.2, 0.1],
            'target': [0, 1, 2, 1]
        })
        self.tmpdir = tempfile.mkdtemp()
        self.scaler_path = os.path.join(self.tmpdir, 'scaler.pickle')
        self.processor = DataProcessor(self.scaler_path)
        data_path = os.path.join(self.tmpdir, 'sample_data.csv')
        self.df_sample.to_csv(data_path, index=False)
        self.X_train_tensor, self.y_train_tensor = self.processor.prepare_data(data_path, fit=True, max_rows=4,
                                                                               random_state=self.random_state)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_model_initialization(self):
        input_dim = self.X_train_tensor.shape[1]
        output_dim = self.y_train_tensor.unique().numel()
        model = Model(input_dim, output_dim, dropout_rate=0.5)
        self.assertIsInstance(model, Model, "Model initialization failed")

    def test_training_process(self):
        input_dim = self.X_train_tensor.shape[1]
        output_dim = self.y_train_tensor.unique().numel()
        training = Training(input_dim, output_dim, dropout_rate=0.5, epochs=2, learning_rate=0.01, metric='accuracy')
        training.run_training(self.X_train_tensor, self.y_train_tensor, self.X_train_tensor, self.y_train_tensor)
        self.assertIsNotNone(training.model, "Model training failed")

    def test_model_save(self):
        input_dim = self.X_train_tensor.shape[1]
        output_dim = self.y_train_tensor.unique().numel()
        training = Training(input_dim, output_dim, dropout_rate=0.5, epochs=2, learning_rate=0.01, metric='accuracy')
        training.run_training(self.X_train_tensor, self.y_train_tensor, self.X_train_tensor, self.y_train_tensor)
        model_path = os.path.join(self.tmpdir, 'test_model.pickle')
        training.save(model_path)
        self.assertTrue(os.path.exists(model_path), "Model file was not saved")

    def test_evaluate_model(self):
        input_dim = self.X_train_tensor.shape[1]
        output_dim = self.y_train_tensor.unique().numel()
        training = Training(input_dim, output_dim, dropout_rate=0.5, epochs=2, learning_rate=0.01, metric='accuracy')
        training.run_training(self.X_train_tensor, self.y_train_tensor, self.X_train_tensor, self.y_train_tensor)
        score = training.evaluate(self.X_train_tensor, self.y_train_tensor)
        self.assertIsInstance(score, float, "Evaluation did not return a valid score")

    def test_different_training_parameters(self):
        input_dim = self.X_train_tensor.shape[1]
        output_dim = self.y_train_tensor.unique().numel()
        training_params = [
            (input_dim, output_dim, 0.3, 10, 0.01, 'accuracy'),
            (input_dim, output_dim, 0.5, 20, 0.005, 'f1'),
            (input_dim, output_dim, 0.2, 15, 0.02, 'recall'),
        ]
        for params in training_params:
            training = Training(*params)
            training.run_training(self.X_train_tensor, self.y_train_tensor, self.X_train_tensor, self.y_train_tensor)
            score = training.evaluate(self.X_train_tensor, self.y_train_tensor)
            self.assertIsInstance(score, float, f"Training with parameters {params} did not return a valid score")


if __name__ == '__main__':
    unittest.main()