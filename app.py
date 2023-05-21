import streamlit as st
from src.pipeline.predict_pipeline import PredictPipeline,CustomData,get_locations

st.set_page_config(page_title='Mumbai Housing Prices', layout='wide')
st.title('Mumbai House Price Prediction')

predict = PredictPipeline()

with st.form('form'):
    area = st.number_input(label='Area in sqft',min_value=150,step=50)
    no_of_bedrooms = st.number_input(label='No of Bedrooms',min_value=0,step=1)
    location = st.selectbox(label='Location',options=get_locations())
    submitted = st.form_submit_button('Predict')
    if submitted:
        custom_data = CustomData(float(area),int(no_of_bedrooms),location)
        data = custom_data.get_data_as_dataframe()
        prediction = predict.predict(data)
        st.write('The estimated price of this house is',prediction)
