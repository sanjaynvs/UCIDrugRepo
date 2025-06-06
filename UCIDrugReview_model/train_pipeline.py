import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from UCIDrugReview_model.config.core import config
from UCIDrugReview_model.pipeline import UCIDrugReview_pipe
from UCIDrugReview_model.processing.data_manager import load_dataset, save_pipeline


def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name = config.app_config_.training_data_file)
    
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        
        data[config.model_config_.features],     # predictors
        data[config.model_config_.target],       # target
        test_size = config.model_config_.test_size,
        random_state=config.model_config_.random_state,   # set the random seed here for reproducibility
    )

    # Pipeline fitting
    UCIDrugReview_pipe.fit(X_train, y_train)
    y_pred = UCIDrugReview_pipe.predict(X_test)

    # Calculate the score/error
    print("Mean squared error:", mean_squared_error(y_test, y_pred))

    # persist trained model
    save_pipeline(pipeline_to_persist = UCIDrugReview_pipe)
    
if __name__ == "__main__":
    run_training()