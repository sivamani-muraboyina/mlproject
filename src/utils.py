import os 
import sys
import numpy as np
import pandas as pd
import dill 
from src.exception import CustomException 
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        # Get model names and values once
        model_names = list(models.keys())
        model_objects = list(models.values())

        for i in range(len(model_objects)):
            model = model_objects[i]
            model_name = model_names[i]
            para = params.get(model_name, {})  # safer in case no params provided

            # Run GridSearchCV only if parameters are provided
            if para:
                gs = GridSearchCV(model, para, cv=3)
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)

            # Fit model on training data
            model.fit(X_train, y_train)

            # Predict
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluate
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Save only test score (like you did)
            report[model_name] = test_model_score

        return report

    except Exception as e:
        
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
        