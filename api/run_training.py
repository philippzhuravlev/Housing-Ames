import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/cleaned_train.csv') 
TARGET_LABEL: str = "SalePrice"

# Convert categorical features to numerical using Label Encoding
categorical_cols = [
    'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 
    'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 
    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 
    'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
    'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 
    'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 
    'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 
    'SaleType', 'SaleCondition'
]
# for each categorical column it will create a whole new dictionary with 
# a list of possible categories, e.g. Neighborhood has a dictionary with
# CollegrCr, Veenker, Crawford etc which each are assigned a number.

# Apply Label Encoding to categorical columns
label_encoders: dict[str, LabelEncoder] = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le


X = data.drop(columns=[TARGET_LABEL]) 
y = data[TARGET_LABEL]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)

# Set parameters for the model
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
}

# Train the model
model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=1000, early_stopping_rounds=100)

# Save the model to a file
model.save_model('data/lgbm_features_15.txt')