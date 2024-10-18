import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import scipy

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsibel for data transformation
        '''
       
        try:
            numeric_features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 
                                'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 
                                'Humidity9am','Humidity3pm', 'Cloud9am', 
                                'Cloud3pm', 'Temp3pm']
            categorical_features = ['WindGustDir', 'WindDir9am', 'WindDir3pm',
                                    'RainToday']
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))]
            )
            logging.info("numerical columns scaling completed")
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="most_frequent")),
                    ('One_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numeric_features),
                    ('cat_pipeline', cat_pipeline, categorical_features)
                ]               
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self, raw_data_path):

        try:
            df = pd.read_csv(raw_data_path)

            logging.info("Reading raw data completed")
            
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = "RainTomorrow"
            numeric_features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 
                                'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 
                                'Humidity9am','Humidity3pm', 'Cloud9am', 
                                'Cloud3pm', 'Temp3pm']
            categorical_features = ['WindGustDir', 'WindDir9am', 'WindDir3pm',
                                    'RainToday']
            x = df.drop(columns=[target_column_name], axis=1)
            y = df[target_column_name]
            y.replace(['No','Yes'],[0,1], inplace=True)

            logging.info(
                "Applying preprocessing object on features."
            )

            input_feature_arr=preprocessing_obj.fit_transform(x).toarray()

            arr = np.c_[
                input_feature_arr, np.array(y)
            ]
            print(arr[0:1])

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (arr, self.data_transformation_config.preprocessor_obj_file_path,)
        
        except Exception as e:
            raise CustomException(e,sys)
