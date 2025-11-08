import os
import sys
from dataclasses import dataclass

from sklearn.metrics import r2_score
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        train_array/test_array: numpy arrays with features in all columns except last,
        and target in the last column.
        """
        try:
            logger.info("Splitting training and testing input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # --- Models ---
            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(
                    random_state=42, n_jobs=-1, verbosity=0
                ),
                "CatBoost Regressor": CatBoostRegressor(
                    verbose=False, random_state=42
                ),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
            }

            # --- Param grids (keys MUST match `models`) ---
            params = {
                "Random Forest": {
                    "n_estimators": [64, 128, 256],
                    # "max_depth": [None, 10, 20],
                    # "min_samples_split": [2, 5],
                },
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                    # "max_depth": [None, 5, 10, 20],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.05, 0.01],
                    "subsample": [0.7, 0.85, 1.0],
                    "n_estimators": [64, 128, 256],
                },
                "Linear Regression": {},  # no tuning
                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 11],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2],
                },
                "XGB Regressor": {
                    "learning_rate": [0.1, 0.05, 0.01],
                    "n_estimators": [64, 128, 256],
                    # "max_depth": [3, 6, 10],
                },
                "CatBoost Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.1, 0.05, 0.01],
                    "iterations": [50, 100, 200],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.5, 0.1, 0.01],
                    "n_estimators": [64, 128, 256],
                },
            }

            logger.info("Evaluating and tuning models via GridSearch where applicable...")
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,     # will be updated in-place with best estimators
                params=params,
            )

            if not model_report:
                raise CustomException("Model evaluation returned an empty report")

            # Pick best by test R2
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]  # already the tuned estimator

            if best_model_score < 0.6:
                raise CustomException(
                    f"No best model found with r2_score >= 0.6. Best: "
                    f"{best_model_name}={best_model_score:.3f}"
                )

            logger.info(
                "Best model found: %s with r2_score: %.3f",
                best_model_name, best_model_score
            )

            # Save the tuned best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )
            logger.info(
                "Saved best model to %s",
                self.model_trainer_config.trained_model_file_path,
            )

            # Final test R2 (should match best_model_score but we recompute cleanly)
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logger.info("Final test R2: %.3f", r2_square)
            return r2_square

        except Exception as e:
            logger.exception("Error occurred during model training")
            raise CustomException(e, sys) from e
