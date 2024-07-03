import os
import pandas as pd
import joblib
from prediction_model.config import config
import pathlib
import sys
import os

PACKAGE_ROOT = pathlib.Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

def load_dataset(file_name):
    file_path = os.path.join(config.DATAPATH, file_name)
    _data = pd.read_csv(file_path)
    return _data

def save_pipeline(pipeline_to_save):
    save_path = os.path.join(config.SAVED_MODEL_PATH, config.MODEL_NAME)
    print(f"Saving pipeline: {config.MODEL_NAME}")
    joblib.dump(pipeline_to_save, save_path)

def load_pipeline():
    load_path = os.path.join(config.SAVED_MODEL_PATH, config.MODEL_NAME)
    print(f"Loading pipeline: {config.MODEL_NAME}")
    return joblib.load(load_path)