# Ames Housing Price Predictor

## Overview
This project creates a machine learning model and API to predict residential home sale prices in Ames, Iowa. It uses the famous Ames Housing dataset from Kaggle (2019) which contains detailed information about house features and their sale prices.

## Project Structure

### Core API Files
- `main.py` (runnable) - FastAPI application entry point that handles HTTP requests
- `api/price_predictor_api.py` - Core prediction API class that loads and uses the trained model
- `utils/pydantic_models.py` - Request/response models and validation using Pydantic
- `utils/enums.py` - Enums for categorical variables like Neighborhoods

### Data Processing
- `preprocessing/data_cleaning.py` - Collection of transformer classes for data cleaning:
  - Missing value imputation
  - Categorical encoding
  - Feature engineering
  - Outlier removal
  - Data validation
- `preprocessing/run_data_cleaning.py` (runnable) - Script to execute the data cleaning pipeline

### Model Training
- `api/run_training.py` (runnable) - Script to train the LightGBM model using cleaned data
- `data/lgbm_features_15.txt` - Saved trained model file
- `utils/neighborhood_encoding.json` - Mapping of neighborhood names to encoded values

### Analysis
- `analysis.ipynb` (runnable) - Jupyter notebook containing exploratory data analysis
- `data/data_description.txt` - Detailed descriptions of all features in the dataset

### Utility Functions
- `utils/utils.py` - Helper functions for visualization and comparison
- `utils/__init__.py` - Package initialization

## Key Features
- Data cleaning and preprocessing pipeline
- Feature engineering including derived features like house age
- LightGBM gradient boosting model for price prediction
- FastAPI REST endpoint for making predictions
- Comprehensive input validation
- Detailed data analysis and visualization

## Running the Project
1. Clean the data: `python preprocessing/run_data_cleaning.py`
2. Train the model: `python api/run_training.py`
3. Start the API: `python main.py`

The API will be available at `http://localhost:8000` with Swagger documentation at `/docs`.

## Model Features
The model considers various aspects of a house including:
- Overall quality score
- Living area square footage
- Basement area
- Neighborhood
- Number of rooms and bathrooms
- Garage details
- Year built/remodeled
- Lot area
- And more...

## Data Pipeline
1. Missing value imputation
2. Categorical encoding
3. Feature engineering
4. Outlier removal
5. Model training using LightGBM
6. API deployment

## Notes
- The project uses LightGBM for its efficiency and performance on structured data
- Input validation ensures data quality and proper error handling
- The API is designed to be production-ready with proper error handling and logging
