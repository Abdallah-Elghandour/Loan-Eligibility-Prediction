from sklearn.pipeline import Pipeline
from prediction_model.config import config
from prediction_model.processing import preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import pathlib
import sys
import os

PACKAGE_ROOT = pathlib.Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

classifcation_pipe = Pipeline(
    [
        ('MeanImputation', pp.MeanImputer(variables = config.NUM_FEATURES)),
        ("ModeImputation", pp.ModeImputer(variables = config.CAT_FEATURES)),
        ("DomainProcessing", pp.DomainProcessing(variable_to_modify = config.FEATURE_TO_MODEFY,variable_to_add = config.FEATURE_TO_ADD)),
        ("DropFeatures", pp.DropColumns(variables_to_drop = config.DROP_FEATURES)),
        ("LabelEncoding", pp.CustomLabelEncoder(variables = config.FEATURES_TO_ENCODE)),
        ("LogTransformation", pp.LogTransforms(variables = config.LOG_FEATURES)),
        ("MinMaxScaler", MinMaxScaler()),
        ("LogisticRegression", LogisticRegression(random_state = 0))
    ]
)
    





