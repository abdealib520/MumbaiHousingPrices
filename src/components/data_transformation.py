import sys
from src.exception import CustomException
from src.logger import logging
import os
from category_encoders import TargetEncoder
import numpy as np
import pandas as pd

def target_encoding(df):
    tenc=TargetEncoder() 
    df_Location=tenc.fit_transform(df['Location'],df['Price'],smoothing = 10)
    df_Location.rename(columns = {'Location':'Location_Encoded'},inplace = True)
    df_new = df.join(df_Location)
    return df_new

def removeOutliers(df):
    df['Price_per_sqft'] = df['Price']/df['Area']
    df_Out = pd.DataFrame()
    for key,subdf in df.groupby('Location'):
        m = np.mean(subdf.Price_per_sqft)
        st = np.std(subdf.Price_per_sqft)
        reduced = subdf[(subdf.Price_per_sqft>(m-(st*3))) & (subdf.Price_per_sqft<=(m+(st*3)))]
        df_Out = pd.concat([df_Out,reduced],ignore_index=True)
    df_Out = df_Out.drop(['Price_per_sqft'],axis=1)
    return df_Out

class DataTransformation:
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            train_df = removeOutliers(train_df)
            test_df = removeOutliers(test_df)

            train_df['Price'] = np.log(train_df['Price'])
            test_df['Price'] = np.log(test_df['Price'])

            logging.info("Read train and test data completed")
            return(
                train_df,
                test_df
            )
        except Exception as e:
            raise CustomException(e,sys)
