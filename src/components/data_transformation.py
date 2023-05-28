import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import numpy as np
import pandas as pd
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def convert_area(value):
    try:
        new_value1 = re.sub(re.compile('[^.0-9]'), '', value)
        new_value1 = float(new_value1)
    except:
        new_value1 = None
    return new_value1

class AreaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        X['Area'] = X['Area'].apply(convert_area)
        X['Area'] = pd.to_numeric(X['Area'])
        return X
class BHKTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        X['BHK'] = X['BHK'].apply(convert_area)
        X['BHK'] = pd.to_numeric(X['BHK'])
        return X
class LocationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        Location_stats = X['Location'].value_counts(ascending=False)
        Location_stats_less_than_10 = Location_stats[Location_stats<=10]
        X.Location = X.Location.apply(lambda x : 'other' if x in Location_stats_less_than_10 else x)
        dummies = pd.get_dummies(X.Location)
        X = pd.concat([X.drop('Location',axis='columns'),dummies.drop('other',axis='columns')],axis='columns')
        return X

def convert_price(value):
    if 'Cr' in value:
        new_value1 = re.sub(re.compile('[^.0-9]'), '', value)
        return float(new_value1)*100
    elif 'Lac' in value:
        new_value2 = re.sub(re.compile('[^.0-9]'), '', value)
        return float(new_value2)
    else:
        return None
class PriceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        X = X.apply(convert_price)
        X = np.array(X)
        X = X.reshape(-1,1)
        return X
class PriceLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        X = X.ravel()
        return np.log(X)


def initiate_data_transformation():
    try:
        logging.info('Started Data Transformation')
        df = pd.read_csv('artifacts/data.csv')
        df = df.drop('Amenities',axis=1)
        X = df.drop('Price',axis=1)
        X_CustomPipelineLocation = Pipeline(steps=[
            ('LocationTransformer',LocationTransformer()
        )])
        X = X_CustomPipelineLocation.transform(X)
        Locations = X.columns[2:]
        with open('Locations.pkl', "wb") as file_obj:
            pickle.dump(Locations, file_obj)
        y = df['Price']
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        X_CustomPipeline = Pipeline(steps=[
            ('AreaTransformer',AreaTransformer()),
            ('BHKTransformer',BHKTransformer())
        ])
        X_NumericPipeline = Pipeline(steps=[
            ('Simple Imputer',SimpleImputer(strategy='median'))
        ])
        y_CustomPipeline = Pipeline(steps=[
            ('Price Transformer',PriceTransformer())
        ])
        y_NumericPipeline = Pipeline(steps=[
            ('Simple Imputer',SimpleImputer(strategy='median')),
            ('Log Transformer',PriceLogTransformer())
        ])
        X_train = X_CustomPipeline.transform(X_train)
        X_train_final = X_NumericPipeline.fit_transform(X_train[['Area','BHK']])
        X_train['Area'] = X_train_final[:,0]
        X_train['BHK'] = X_train_final[:,1]

        X_test = X_CustomPipeline.transform(X_test)
        X_test_final = X_NumericPipeline.fit_transform(X_test[['Area','BHK']])
        X_test['Area'] = X_test_final[:,0]
        X_test['BHK'] = X_test_final[:,1]

        y_train = y_CustomPipeline.fit_transform(y_train)
        y_train = y_NumericPipeline.fit_transform(y_train)

        y_test = y_CustomPipeline.fit_transform(y_test)
        y_test = y_NumericPipeline.fit_transform(y_test)
        logging.info('Finished Data Transformation')
        return(
            X_train,X_test,
            y_train,y_test
        )
    except Exception as e:
        raise CustomException(e,sys)
