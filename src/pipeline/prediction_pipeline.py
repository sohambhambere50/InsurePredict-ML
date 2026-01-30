import os 
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.exception import CustomException
from src.logger import logging
from src.utils.common import load_object

class PredictionPipeline:
    """ Handles prediction on new data """

    def __init__(self):
        self.model_path = "models/model.pkl"
        self.preprocessor_path = "models/preprocessor.pkl"

    def predict(self, features):
        """ Make predictions on input features """

        try:
            logging.info("Loading model and preprocessor")

            # Load saved objects
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            logging.info("Transforming input data")
            # Transorm features
            data_scaled = preprocessor.transform(features)

            logging.info("Making predictions")
            # Predict
            predictions = model.predict(data_scaled)

            return predictions
        except Exception as e:
            logging.error("Error in prediction in pipeline")
            raise CustomException(e)
        
class CustomData:
    """ Handle custom input data """

    def __init__(self, Gender, Age, HasDrivingLicense, RegionID, Switch, VehicleAge,
                 PastAccident, AnnualPremium, SalesChannelID, DaysSinceCreated):
        self.Gender = Gender
        self.Age = Age
        self.HasDrivingLicense = HasDrivingLicense
        self.RegionID = RegionID
        self.Switch = Switch
        self.VehicleAge = VehicleAge
        self.PastAccident = PastAccident
        self.AnnualPremium = AnnualPremium
        self.SalesChannelID = SalesChannelID
        self.DaysSinceCreated = DaysSinceCreated

    def get_data_as_dataframe(self):
        """ Convert to dataframe """

        try:
            data_dict = {
                'Gender' : [self.Gender],
                'Age': [self.Age],
                'HasDrivingLicense': [self.HasDrivingLicense],
                'RegionID': [self.RegionID],
                'Switch': [self.Switch],
                'VehicleAge': [self.VehicleAge],
                'PastAccident': [self.PastAccident],
                'AnnualPremium': [self.AnnualPremium],
                'SalesChannelID': [self.SalesChannelID],
                'DaysSinceCreated': [self.DaysSinceCreated]
            }

            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e)
        
if __name__ == "__main__":
    # Test Prediction
    print("Testing prediction pipeline...")

    # Sample Data
    data = CustomData(
        Gender='Male',
        Age=35.0,
        HasDrivingLicense=1.0,
        RegionID=28.0,
        Switch=0.0,
        VehicleAge='1-2 Year',
        PastAccident='No',
        AnnualPremium=30000.0,
        SalesChannelID=152,
        DaysSinceCreated=200
    )

    # Convert to dataframe
    input_df = data.get_data_as_dataframe()
    print("Input Data: ")
    print(input_df)

    # Predict
    pipeline = PredictionPipeline()
    prediction = pipeline.predict(input_df)

    print(f"\n✅ Prediction: {prediction[0]}")
    print(f"Result: {'Will buy insurance' if prediction[0] == 1 else 'Will NOT buy insurance'}")
