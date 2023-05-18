import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import pandas as pd

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array.drop(['Price','Location'],axis=1),
                train_array['Price'],
                test_array.drop(['Price','Location'],axis=1),
                test_array['Price']
            )
            params = {'criterion': 'friedman_mse', 'learning_rate': 1, 'loss': 'squared_error', 'n_estimators': 100}
            model = GradientBoostingRegressor(**params)
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            print(f'R2 Score: {r2_score(y_test,y_pred)}')
        except Exception as e:
            raise CustomException(e,sys)