import os
import sys
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logger
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Data Ingestion started")
        try:
            # Build a path to the CSV relative to project root (src/.. = project)
            project_root = Path(__file__).resolve().parents[2]
            source_csv = project_root / "notebook" / "data" / "stud.csv"

            if not source_csv.exists():
                raise FileNotFoundError(f"Raw CSV not found at: {source_csv}")

            df = pd.read_csv(source_csv)
            logger.info("Dataset read as dataframe with shape %s", df.shape)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logger.info("Raw data saved to %s", self.ingestion_config.raw_data_path)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logger.info("Data split into train (%d) and test (%d)", len(train_set), len(test_set))

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logger.info("Train saved to %s | Test saved to %s",
                        self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logger.exception("Error occurred during data ingestion")
            raise CustomException(e, sys) from e



# checks if logger is working
if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    
    DataTransformation=DataTransformation()
    train_array,test_array,_=DataTransformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    r2_square=modeltrainer.initiate_model_trainer(train_array,test_array)
    print("R2 square value :",r2_square)
