import pytest
from prediction_model.config import config
from prediction_model.processing.data_handeling import load_dataset
from prediction_model.predict import make_prediction

@pytest.fixture()
def single_prediction():
    data = load_dataset(config.TEST_FILE)
    single_row = data[:1]
    result = make_prediction(single_row)
    return result


def test_single_pred_not_none(single_prediction):
    assert single_prediction is not None
def test_single_pred_is_string(single_prediction):
    assert isinstance(single_prediction['prediction'][0], str)
def test_single_pred_validate(single_prediction):
    assert single_prediction.get('prediction')[0] == 'Y'