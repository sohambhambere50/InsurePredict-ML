import os
import yaml
import joblib
from src.exception import CustomException
from src.logger import logging

# Read YAML file and return it's content as a dictionary
def read_yaml(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            content = yaml.safe_load(file)
            logging.info(f"YAML file loaded: {file_path}")
            return content
    except Exception as e:
        raise CustomException(e)



def create_directories(path_list: list):
    """
        Creates multiple directories if they do not exist.
        
        Args:
            path_list (list): List of directory paths
    """
    try:
        for path in path_list:
            os.makedirs(path, exist_ok = True)
            logging.info(f"Created directories: {path}")
    except Exception as e:
        raise CustomException(e)
    
def save_object(file_path: str, obj):
    """
    Saves a python object to dist using joblib

    Args:
        file_path(Str): Path where object should be saved
        obj: Python object (model, array, etc.)
    """

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)

        logging.info(f"Object Saved: {file_path}")
    except Exception as e:
        raise CustomException(e)
    
def load_object(file_path: str):
    """
    Loads a saved Python object using joblib.

    Args:
        file_path(Str): Path of saved object.
    
    Returns:
        loaded python object.
    """

    try:
        with open(file_path, "rb") as file_obj:
            return joblib.load(file_obj)
    except Exception as e:
        raise CustomException(e)
    


        