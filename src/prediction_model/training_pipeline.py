import pandas as pd
import numpy as np
from prediction_model.processing.data_handeling import load_dataset, save_pipeline
from prediction_model.config import config
from prediction_model import pipeline as pipe
from prediction_model.processing import preprocessing as pp
import pathlib
import sys
import os

PACKAGE_ROOT = pathlib.Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

def perform_training():
    train_data = load_dataset(config.TRAIN_FILE)
    y_train = train_data[config.TARGET].map({'Y':1,'N':0})
    pipe.classifcation_pipe.fit(train_data[config.FEATURES], y_train)
    save_pipeline(pipe.classifcation_pipe)

if __name__ == "__main__":
    perform_training()
