import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from api.price_predictor_api import PricePredictorAPI
from api.pydantic_models import SalePriceRequest  # Import the model from pydantic_models.py

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(
    title="Ames House Prices API",
    version="1.0.0"
)

price_predictor = PricePredictorAPI()

@app.post('/sale-price/predict', summary="Predict sale price (USD)")
def sale_price_predict(sale_price_request: SalePriceRequest):
    try:
        # Convert the input features to a pandas Series
        query = pd.Series(sale_price_request.features)
        
        # Make the prediction
        prediction = price_predictor.predict(query)
        
        # Log the prediction
        log.info(f"Prediction made: {prediction}")
        
        return prediction

    except (KeyError, ValueError, AttributeError) as e:
        log.error(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# Allows debugging when running from IDE
if __name__ == "__main__":
    uvicorn.run(app, port=9010)
