import unittest
import os
import sys
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

CONF_FILE = os.path.join(os.path.dirname(ROOT_DIR), os.getenv('CONF_PATH'))


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

    def test_inference_directory_exists_and_not_empty(self):
        inference_dir = os.path.join(os.path.dirname(ROOT_DIR), 'inference')
        self.assertTrue(os.path.exists(inference_dir), f"Inference directory {inference_dir} does not exist")
        self.assertTrue(os.listdir(inference_dir), f"Inference directory {inference_dir} is empty")

    def test_modeling_directory_exists_and_not_empty(self):
        modeling_dir = os.path.join(os.path.dirname(ROOT_DIR), 'modeling')
        self.assertTrue(os.path.exists(modeling_dir), f"Modeling directory {modeling_dir} does not exist")
        self.assertTrue(os.listdir(modeling_dir), f"Modeling directory {modeling_dir} is empty")

    def test_requirements_file_exists(self):
        requirements_path = os.path.join(os.path.dirname(ROOT_DIR), 'requirements.txt')
        self.assertTrue(os.path.exists(requirements_path), "requirements.txt file does not exist")

    def test_models_directory_exists_and_not_empty(self):
        model_dir = self.conf['general']['models_dir']
        model_dir = os.path.join(os.path.dirname(ROOT_DIR), model_dir)
        self.assertTrue(os.path.exists(model_dir), f"Models directory {model_dir} does not exist")
        self.assertTrue(os.listdir(model_dir), f"Models directory {model_dir} is empty")

    def test_data_proc_directory_exists_and_not_empty(self):
        data_proc_dir = "data_process"
        data_proc_dir = os.path.join(os.path.dirname(ROOT_DIR), data_proc_dir)
        self.assertTrue(os.path.exists(data_proc_dir), f"Data processing directory {data_proc_dir} does not exist")
        self.assertTrue(os.listdir(data_proc_dir), f"Data processing directory directory {data_proc_dir} is empty")

    def test_scaler_exists(self):
        data_proc_dir = "data_process"
        scaler_path = os.path.join(os.path.dirname(ROOT_DIR), data_proc_dir, self.conf['train']['scaler_filename'])
        self.assertTrue(os.path.exists(scaler_path), f"Scaler exists at {scaler_path}")

if __name__ == '__main__':
    unittest.main()


