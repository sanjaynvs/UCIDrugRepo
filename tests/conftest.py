import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pytest
from sklearn.model_selection import train_test_split

from UCIDrugReview_model.config.core import config
from UCIDrugReview_model.processing.data_manager import load_dataset


@pytest.fixture
def sample_input_data():
    data = load_dataset(file_name = config.app_config_.training_data_file)
    print(data[config.model_config_.features].head())
    print(type(data[config.model_config_.features]))
    print(data[config.model_config_.target].head())
    print(type(data[config.model_config_.target]))

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        
        data[config.model_config_.features],     # predictors
        data[config.model_config_.target],       # target
        test_size = config.model_config_.test_size,
        random_state=config.model_config_.random_state,   # set the random seed here for reproducibility
    )
    '''print("in conftest.py")
    print("X_train....", X_train.head())
    print("X_test....", X_test.head())  
    print("y_train....", y_train.head())
    print("y_test....", y_test.head())'''

    return X_test, y_test