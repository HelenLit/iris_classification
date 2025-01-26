import unittest
import os
import sys
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

CONF_FILE = os.path.join(os.path.dirname(ROOT_DIR), os.getenv('CONF_PATH'))


class TestResultsGenerates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONF_FILE, "r") as file:
            cls.conf = json.load(file)

    def test_results_generated(self):
        results_dir = self.conf['general']['results_dir']
        results_dir = os.path.join(os.path.dirname(ROOT_DIR), results_dir)
        self.assertTrue(os.path.exists(results_dir), f"Results directory {results_dir} does not exist")
        self.assertTrue(os.listdir(results_dir), f"Results directory {results_dir} is empty")


if __name__ == '__main__':
    unittest.main()