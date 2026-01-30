from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import pandas as pd
import os 
import sys

sys.path.append(os.path.dirname(__file__))

from src.pipeline.prediction_pipeline import PredictionPipeline, CustomData
from src.exception import CustomException

app = FastAPI(title="InsurePredict-ML API", version="1.0")

@app.get("/")
def home():
    return {"message" : "Insurance Cross-Sell Prediction API", "status" : "active"}

@app.post("/predict")
async def predict(request: Request):
    """ Prediction endpoint """

    try:
        # Get JSON data
        data = await request.json()

        # Create CustomData object
        custom_data = CustomData(
            Gender=data['Gender'],
            Age=float(data['Age']),
            HasDrivingLicense=float(data['HasDrivingLicense']),
            RegionID=float(data['RegionID']),
            Switch=float(data['Switch']),
            VehicleAge=data['VehicleAge'],
            PastAccident=data['PastAccident'],
            AnnualPremium=float(data['AnnualPremium']),
            SalesChannelID=int(data['SalesChannelID']),
            DaysSinceCreated=int(data['DaysSinceCreated'])
        )

        # Get Dataframe 
        input_df = custom_data.get_data_as_dataframe()

        # Predict
        pipeline = PredictionPipeline()
        prediction = pipeline.predict(input_df)

        result = "Will buy insurance" if prediction[0] == 1 else "Will not buy insurance"

        return JSONResponse({
            "prediction": int(prediction[0]),
            "result": result
        })
    except Exception as e:
        raise CustomException(e)
    
@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)