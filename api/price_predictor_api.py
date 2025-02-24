import pandas as pd     # data manipulation
import lightgbm as lgb  # gradient boosting, i.e. layers of decision trees on residuals 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder # to encode cat vars as dict ints

# Load your training data
data = pd.read_csv('data/train.csv') 

class PricePredictorAPI:
    # initialize booster 
    def __init__(self):
        self.model = lgb.Booster(model_file="lgbm_features_15.txt")

    def predict(self, query: pd.Series) -> dict:
        prediction_usd = self.model.predict(query)
        
        # Format the output similar to the provided example
        output = {
            "version": "v4",
            "num_class": 1,
            "num_tree_per_iteration": 1,
            "label_index": 0,
            "max_feature_idx": len(query) - 1,
            "objective": "regression",
            "feature_names": list(query.index),
            "feature_infos": [f"[{min(query[name])}:{max(query[name])}]" for name in query.index],
            "tree_sizes": [len(self.model.trees[i]) for i in range(len(self.model.trees))]
        }
        
        return output
