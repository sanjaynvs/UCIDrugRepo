import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from UCIDrugReview_model import __version__ as _version
from UCIDrugReview_model.config.core import config
from UCIDrugReview_model.processing.data_manager import load_pipeline
from UCIDrugReview_model.processing.data_manager import pre_pipeline_preparation
from UCIDrugReview_model.processing.validation import validate_inputs
from UCIDrugReview_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
uci_pipe = load_pipeline(file_name = pipeline_file_name)


def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    
    validated_data, errors = validate_inputs(input_df = pd.DataFrame(input_data))
    
    #validated_data = validated_data.reindex(columns = ['dteday', 'season', 'hr', 'holiday', 'weekday', 'workingday', 
    #                                                   'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'yr', 'mnth'])
    
    #print('validated_data....', validated_data.head())
    validated_data = validated_data.reindex(columns = config.model_config_.features)
    #print('validated_data....afterwards...', validated_data.head())
    
    results = {"predictions": None, "version": _version, "errors": errors}
      
    if not errors:
        predictions = uci_pipe.predict(validated_data)
        results = {"predictions": np.floor(predictions), "version": _version, "errors": errors}
        print(results)
    else:
        print("Errors in input data: ", errors)

    return results



if __name__ == "__main__":
   
    testData = pd.read_csv(DATASET_DIR / 'drugsComTrain_raw5row.csv')
    
    
    data_in = {'uniqueID': testData['uniqueID'], 'drugName': testData['drugName'], 'condition': testData['condition'], 'review': testData['review'], 'date': testData['date'],
               'usefulCount': testData['usefulCount']}
    
    
    make_prediction(input_data = data_in)