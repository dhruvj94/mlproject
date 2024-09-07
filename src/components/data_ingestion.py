import os
import sys

from src.exception import Custom_Exception
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import Data_transformation_Config

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import Model_Trainer

@dataclass
class DataIngetionconfig():
    train_data_path: str=os.path.join('artifact','train.csv')
    test_data_path: str=os.path.join('artifact','test.csv')
    raw_data_path: str=os.path.join('artifact','data.csv')

class DataIngetion:
    def __init__(self):
        self.ingestion_config=DataIngetionconfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as DataFrame')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header =True)

            logging.info('train tsest split initiated')
            train_set,test_set= train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise Custom_Exception(e,sys)
        

if __name__ == "__main__":
    obj = DataIngetion()
    train_data,test_data= obj.initiate_data_ingestion()

    data_transformation1=DataTransformation()
    train_arr,test_arr,_=data_transformation1.initiate_data_transformation(train_data,test_data)

    modeltrainer= Model_Trainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))