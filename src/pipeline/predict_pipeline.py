import sys
import os
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            model=load_object(file_path=model_path)
            preds=model.predict(features)
            return int(np.exp(preds))
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,area: float,no_of_bedrooms: int,location: str,):
        self.area = area
        self.no_of_bedrooms = no_of_bedrooms
        self.location = location
    def get_data_as_dataframe(self):
        try:
            df = pd.read_csv('artifacts\data.csv')
            location_encoded = df.loc[df['Location'] == self.location]['Location_Encoded'].iloc[0]
            custom_dict = {
                'Area': [self.area],
                'No. of Bedrooms': [self.no_of_bedrooms],
                'Location_Encoded': [location_encoded]
            }
            return pd.DataFrame(custom_dict)
        except Exception as e:
            raise CustomException(e,sys)

def get_locations():
        df = pd.read_csv('artifacts\data.csv')
        return df['Location'].unique()

if __name__ == '__main__':
    custom_data = CustomData(720.0,1,'Kharghar')
    data = custom_data.get_data_as_dataframe()
    predict = PredictPipeline()
    prediction = predict.predict(data)
    print(prediction)
