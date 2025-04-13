import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from UCIDrugReview_model.config.core import config

preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(max_features=5000), config.model_config_.review_var)  # Process text
    ])

UCIDrugReview_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model_rf', RandomForestRegressor(n_estimators = config.model_config_.n_estimators, 
                                       max_depth = config.model_config_.max_depth,
                                      random_state = config.model_config_.random_state))
    ])