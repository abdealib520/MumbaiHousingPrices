import asyncio
from src.components.data_ingestion import data_ingestion
from src.components.data_transformation import initiate_data_transformation
from src.components.model_trainer import ModelTrainer


X_train,X_test,y_train,y_test = initiate_data_transformation()
obj = ModelTrainer()
obj.initiate_model_trainer(X_train,X_test,y_train,y_test)