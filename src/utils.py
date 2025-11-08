import os
import sys
import dill
import pandas as pd
import numpy as np
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    """
    Save a Python object to a file using dill.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            dill.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys) from e


def evaluate_models(X_train, y_train, X_test, y_test, models: dict):
    """
    Fit each model, evaluate R2 on train/test, and return a report:
    { model_name: test_r2 }
    """
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred  = model.predict(X_test)

            train_r2 = r2_score(y_train, y_train_pred)
            test_r2  = r2_score(y_test, y_test_pred)

            # If you want, you can log/print train_r2 too
            report[model_name] = test_r2
        return report
    except Exception as e:
        raise CustomException(e, sys) from e
