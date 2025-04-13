
"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from UCIDrugReview_model.config.core import config
#from UCIDrugReview_model.processing.features import WeekdayImputer, WeathersitImputer, Mapper, OutlierHandler, WeekdayOneHotEncoder

def test_review_variable_blank(sample_input_data):
    
    X, y = sample_input_data
    print(X['review'].isnull().sum())
    assert X['review'].isnull().sum() == 0
    
