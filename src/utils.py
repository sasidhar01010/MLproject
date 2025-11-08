import os
import sys
import dill
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path: str, obj: Any) -> None:
    """
    Save a Python object to a file using dill.
    Creates parent directories if needed.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            dill.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys) from e


def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: Dict[str, Any],
    params: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    """
    Fit/tune each model, evaluate R2 on test, and return a report:
        { model_name: test_r2 }

    Notes:
    - If a model has a non-empty param grid in `params`, GridSearchCV is used.
    - The corresponding entry in `models[model_name]` is UPDATED IN-PLACE
      to the best (fitted) estimator so callers can directly use it afterwards.
    """
    try:
        report: Dict[str, float] = {}

        for model_name, model in models.items():
            grid = params.get(model_name, {})

            # Tune if a grid is provided
            if grid:
                gs = GridSearchCV(model, grid, cv=3, n_jobs=-1)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                # Fit directly if no params to tune
                best_model = model
                best_model.fit(X_train, y_train)

            # Ensure the final picked model is fitted (gs already fits; otherwise we did above)
            # Compute test R2
            y_test_pred = best_model.predict(X_test)
            test_r2 = r2_score(y_test, y_test_pred)

            # Update outward-facing dict so caller can save the tuned/fitted model
            models[model_name] = best_model
            report[model_name] = float(test_r2)

        return report

    except Exception as e:
        raise CustomException(e, sys) from e
