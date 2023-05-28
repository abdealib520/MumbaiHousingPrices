import os
import sys
from dataclasses import dataclass
from src.components.data_transformation import initiate_data_transformation
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import pandas as pd
import pickle
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from joblib import parallel_backend
from ray.util.joblib import register_ray
register_ray()

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,X_train,X_test,y_train,y_test):
        try:
            logging.info("Started Hyperparamter Tuning")
            X_train,X_test,y_train,y_test = initiate_data_transformation()
            def find_best_model_using_gridsearchcv(X,y):
                algos = {
                'gradient_booster': {
                'model': GradientBoostingRegressor(),
                'params': {
                    'loss': ['squared_error', 'absolute_error'],
                    'learning_rate': [0.1,1,1.5,2],
                    'n_estimators': [10,50,100,150,200],
                    'criterion': ['friedman_mse', 'squared_error']
                        }
                    }
                }
                scores = []
                cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
                for algo_name, config in algos.items():
                    gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
                    with parallel_backend('ray'): # We are using multithreading to speed up the process of training our models.
                        gs.fit(X,y)
                    scores.append({
                        'model': algo_name,
                        'best_score': gs.best_score_,
                        'best_params': gs.best_params_
                    })
                return pd.DataFrame(scores,columns=['model','best_score','best_params'])
            params = dict(find_best_model_using_gridsearchcv(X_train,y_train)['best_params'][0])
            model = GradientBoostingRegressor(**params)
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            logging.info("Finished Hyperparamter Tuning")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            with open('model.pkl', "wb") as file_obj:
                pickle.dump(model, file_obj)
            print(f'R2 Score: {r2_score(y_test,y_pred)}')
        except Exception as e:
            raise CustomException(e,sys)