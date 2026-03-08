import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            # 1. Start Data Ingestion
            train_path, test_path = self.data_ingestion.intiate_data_ingestion()

            # 2. Start Data Transformation
            train_arr, test_arr, preprocessor_path = self.data_transformation.initiate_data_transformation(train_path, test_path)

            # 3. Start Model Training
            model_score = self.model_trainer.initiate_model_trainer(train_arr, test_arr)

            print(f"Training Completed. Best Model Score: {model_score}")
            return model_score

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = TrainPipeline()
    obj.run_pipeline()