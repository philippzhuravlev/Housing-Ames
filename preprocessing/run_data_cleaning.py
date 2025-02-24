import pandas as pd
from data_cleaning import (
    MissingValueImputer,
    CategoricalConverter,
    FeatureEngineer,
    # NumericalNormalizer,
    OutlierRemover,
    DataValidator
)

# Example: preprocess training data
train_data = pd.read_csv("data/train.csv")

# Initialize transformers
imputer = MissingValueImputer()
converter = CategoricalConverter()
feature_engineer = FeatureEngineer(current_year=2023)
# normalizer = NumericalNormalizer()
outlier_remover = OutlierRemover(factor=1.5)
validator = DataValidator()

# Fit and transform training data
train_data = imputer.fit_transform(train_data)
train_data = converter.fit_transform(train_data)
train_data = feature_engineer.fit_transform(train_data)
# train_data = normalizer.fit_transform(train_data)
train_data = outlier_remover.fit_transform(train_data)
train_data = validator.transform(train_data)

# Save the cleaned training data
train_data.to_csv("data/cleaned_train.csv", index=False)

# For test data, use transform methods only so that training statistics are preserved
test_data = pd.read_csv("data/test.csv")

test_data = imputer.transform(test_data)
test_data = converter.transform(test_data)
test_data = feature_engineer.transform(test_data)
# test_data = normalizer.transform(test_data)
test_data = outlier_remover.transform(test_data)
test_data = validator.transform(test_data)

# Save the cleaned test data
test_data.to_csv("data/cleaned_test.csv", index=False)