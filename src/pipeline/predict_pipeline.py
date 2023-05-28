import sys
import os
import numpy as np
import pandas as pd
import pickle
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            with open('model.pkl', 'rb') as handle:
                model= pickle.load(handle)
            preds=model.predict(features)
            preds = np.round(np.exp(preds),0)
            preds = preds[0]
            if preds>100:
                preds = preds/100
                return f'₹ {preds} Cr.'
            else:
                return f'₹ {preds} Lakhs'
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,area: float,no_of_bedrooms: int,location: str,):
        self.area = area
        self.no_of_bedrooms = no_of_bedrooms
        self.location = location
    def get_data_as_dataframe(self):
        try:
            with open('Locations.pkl', 'rb') as handle:
                Locations = pickle.load(handle)
            predict_df = pd.DataFrame({
                'Area': [self.area],
                'BHK': [self.area]
            })
            for Location in Locations:
                if Location==self.location:
                    predict_df[Location] = 1
                else:
                    predict_df[Location] = 0
            return predict_df
        except Exception as e:
            raise CustomException(e,sys)

def get_locations():
        with open('Locations.pkl', 'rb') as handle:
                Locations = pickle.load(handle)
        return Locations

if __name__ == '__main__':
    custom_data = CustomData(720.0,1,'Kharghar')
    data = custom_data.get_data_as_dataframe()
    predict = PredictPipeline()
    prediction = predict.predict(data)
    print(prediction)
