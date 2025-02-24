import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values in the dataset.
    For numerical columns, uses the mean computed on training data.
    For categorical columns, uses the mode computed on training data.
    """
    def __init__(self):
        self.numeric_impute_values = {}
        self.categorical_impute_values = {}

    def fit(self, X, y=None):
        # NB: Only fit on columns that exist in X! Otherwise throws error
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        categorical_columns = X.select_dtypes(include=["object", "category"]).columns

        for col in numeric_columns:
            self.numeric_impute_values[col] = X[col].mean()
        for col in categorical_columns:
            self.categorical_impute_values[col] = X[col].mode()[0]
        return self

    def transform(self, X, y=None):
        X = X.copy()
        # NB: Only transform columns that exist in both the fitted data and X!
        for col, impute_value in self.numeric_impute_values.items():
            if col in X.columns:  # Only impute if column exists
                X[col] = X[col].fillna(impute_value)
        for col, impute_value in self.categorical_impute_values.items():
            if col in X.columns:  # Only impute if column exists
                X[col] = X[col].fillna(impute_value)
        return X


class CategoricalConverter(BaseEstimator, TransformerMixin):
    """
    Converts columns with object dtype to the 'category' type.
    If columns is None, all object-type columns are converted.
    """
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if self.columns is None:
            columns_to_convert = X.select_dtypes(include=["object"]).columns
        else:
            columns_to_convert = self.columns
        for col in columns_to_convert:
            X[col] = X[col].astype("category")
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates new features 'HouseAge' and 'TotalBathrooms'.
    'HouseAge' is computed as current_year - YearBuilt.
    'TotalBathrooms' is computed as FullBath + (HalfBath / 2).
    """
    def __init__(self, current_year=2023):
        self.current_year = current_year

    def fit(self, X, y=None):
        # Assumes the required columns are present.
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if "YearBuilt" in X.columns:
            X["HouseAge"] = self.current_year - X["YearBuilt"]
        if "FullBath" in X.columns and "HalfBath" in X.columns:
            X["TotalBathrooms"] = X["FullBath"] + (X["HalfBath"] / 2)
        return X


class NumericalNormalizer(BaseEstimator, TransformerMixin):
    """
    Normalizes the numerical features using StandardScaler.
    The scaler is fitted on training data and then applied to any new data.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.numeric_columns = None

    def fit(self, X, y=None):
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.scaler.fit(X[self.numeric_columns])
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if self.numeric_columns is None:
            raise RuntimeError("The transformer has not been fitted yet.")
        X[self.numeric_columns] = self.scaler.transform(X[self.numeric_columns])
        return X


class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    Removes rows containing outliers in numerical columns.
    Outlier thresholds (IQR-based) are computed from the training data.
    A row is removed if any of its numerical features falls outside
    the range [Q1 - factor * IQR, Q3 + factor * IQR].
    """
    def __init__(self, factor=1.5):
        self.factor = factor
        self.numeric_columns = None
        self.bounds = {}

    def fit(self, X, y=None):
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        for col in self.numeric_columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.factor * IQR
            upper_bound = Q3 + self.factor * IQR
            self.bounds[col] = (lower_bound, upper_bound)
        return self

    def transform(self, X, y=None):
        X = X.copy()
        mask = pd.Series(True, index=X.index)
        for col in self.numeric_columns:
            lower_bound, upper_bound = self.bounds[col]
            mask = mask & (X[col] >= lower_bound) & (X[col] <= upper_bound)
        return X[mask]


class DataValidator(BaseEstimator, TransformerMixin):
    """
    Validates the data by checking for negative values in numerical columns
    and printing out the unique categories in categorical columns.
    This transformer does not modify the data.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # No fitting needed for validation.
        return self

    def transform(self, X, y=None):
        X = X.copy()
        numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        for col in numerical_columns:
            if (X[col] < 0).any():
                print(f"Warning: Negative values found in column '{col}'")
        categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in categorical_columns:
            unique_categories = X[col].unique()
            print(f"Column '{col}' has categories: {unique_categories}")
        return X