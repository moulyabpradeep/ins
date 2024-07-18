from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd

model=load_model('insurance_dt_model')

input_dict = {'age':20, 'sex': 'female', 'bmi':20, 'children':2, 'smoker':'no','region':'southwest'}

 
input_df = pd.DataFrame([input_dict])
 
predictions_df = predict_model(estimator=model, data=input_df)
print(predictions_df)
print(predictions_df['prediction_label'])
pred=predictions_df.iloc[0]['prediction_label']
print(pred)

st.markdown(pred)