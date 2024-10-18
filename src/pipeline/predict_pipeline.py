import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import scipy

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model=load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features).toarray()
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
                 MinTemp: float,
                 MaxTemp: float,
                 Rainfall: float,
                 Evaporation:float,
                 Sunshine: float,
                 WindGustDir: str,
                 WindGustSpeed: float,
                 WindDir9am: str,
                 WindDir3pm: str,
                 WindSpeed9am: float,
                 Humidity9am: float,
                 Humidity3pm: float,
                 Cloud9am: float,
                 Cloud3pm: float,
                 Temp3pm: float,
                 RainToday: str
                 ):
        self.MinTemp = MinTemp
        self.MaxTemp = MaxTemp
        self.Rainfall = Rainfall
        self.Evaporation = Evaporation
        self.Sunshine = Sunshine
        self.WindGustDir = WindGustDir
        self.WindGustSpeed = WindGustSpeed
        self.WindDir9am = WindDir9am
        self.WindDir3pm = WindDir3pm
        self.WindSpeed9am = WindSpeed9am
        self.Humidity9am = Humidity9am
        self.Humidity3pm = Humidity3pm
        self.Cloud9am = Cloud9am
        self.Cloud3pm = Cloud3pm
        self.Temp3pm = Temp3pm
        self.RainToday = RainToday
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "MinTemp": [self.MinTemp],
                "MaxTemp": [self.MaxTemp],
                "Rainfall": [self.Rainfall],
                "Evaporation": [self.Evaporation],
                "Sunshine": [self.Sunshine],
                "WindGustDir": [self.WindGustDir],
                "WindGustSpeed": [self.WindGustSpeed],
                "WindDir9am": [self.WindDir9am],
                "WindDir3pm": [self.WindDir3pm],
                "WindSpeed9am": [self.WindSpeed9am],
                "Humidity9am": [self.Humidity9am],
                "Humidity3pm": [self.Humidity3pm],
                "Cloud9am": [self.Cloud9am],
                "Cloud3pm": [self.Cloud3pm],
                "Temp3pm": [self.Temp3pm],
                "RainToday": [self.RainToday],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)