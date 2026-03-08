import sys
import pandas as pd
import os
from src.exception import CustomException

from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__( self,
        gender:str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course:str,
        reading_score:int,
        writing_score:int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)

    def validate_data(self):
        try:
            # Check if scores are within the 0-100 range
            if not (0 <= self.reading_score <= 100):
                raise ValueError(f"Reading score {self.reading_score} is out of range (0-100)")
            
            if not (0 <= self.writing_score <= 100):
                raise ValueError(f"Writing score {self.writing_score} is out of range (0-100)")
                
            # You can also add checks for empty strings if needed
            if not self.gender or not self.race_ethnicity:
                 raise ValueError("Categorical fields cannot be empty")

        except Exception as e:
            raise CustomException(e, sys)
    

