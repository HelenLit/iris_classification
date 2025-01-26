import unittest
import os
import sys
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

#os.environ['CONF_PATH'] = 'settings.json'
CONF_FILE = os.path.join(os.path.dirname(ROOT_DIR), os.getenv('CONF_PATH'))


class TestModelCreated(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONF_FILE, "r") as file:
            cls.conf = json.load(file)

    def test_models_directory_exists_and_not_empty(self):
        model_dir = self.conf['general']['models_dir']
        model_dir = os.path.join(os.path.dirname(ROOT_DIR), model_dir)
        self.assertTrue(os.path.exists(model_dir), f"Models directory {model_dir} does not exist")
        self.assertTrue(os.listdir(model_dir), f"Models directory {model_dir} is empty")

if __name__ == '__main__':
    unittest.main()