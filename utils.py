import os
import logging

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

def get_project_dir(sub_dir: str) -> str:
    """Return path to a project subdirectory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), sub_dir))

def setup_logger(name:str, level=logging.INFO):
    # Get the root logger
    logger = logging.getLogger(name)
    # Check if the logger already has handlers
    logger.setLevel(level)
    if logger.hasHandlers():
        logger.handlers.clear()
    # Create a handler
    handler = logging.StreamHandler()
    # Create a formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    # Add the formatter to the handler
    handler.setFormatter(formatter)
    # Add the handler to the logger
    logger.addHandler(handler)
    logger.propagate = False
    return logger