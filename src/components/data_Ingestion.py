from dataclasses import dataclass
import logging
import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    raw_data_path: str=os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion method or component")
        try:
            df=pd.read_csv('notebook/raw_data.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            return self.ingestion_config.raw_data_path
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__== "__main__":
    obj=DataIngestion()
    raw_data = obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    arr,_=data_transformation.initiate_data_transformation(raw_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(arr))
