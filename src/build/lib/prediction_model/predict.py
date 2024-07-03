import pandas as pd
import numpy as np
from src.prediction_model.config import config
from src.prediction_model.processing.data_handeling import load_dataset, load_pipeline
import pathlib
import sys
import os

PACKAGE_ROOT = pathlib.Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

classifier = load_pipeline()

def make_prediction(input_data):
    data = pd.DataFrame(input_data)
    prediction = classifier.predict(data[config.FEATURES])
    output = np.where(prediction == 1, 'Y', 'N')
    result = {'prediction': output}
    return result

# def make_prediction():
#     test_data = load_dataset(config.TEST_FILE)
#     prediction = classifier.predict(test_data[config.FEATURES])
#     output = np.where(prediction == 1, 'Y', 'N')
#     return output

if __name__ == "__main__":
    make_prediction()