from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd

# Load the trained model
model = load_model('insurance_dt_model')

# Streamlit app title
st.title('Insurance Charges Prediction')

# Input fields
age = st.number_input('Age', min_value=0, max_value=120, value=20)
sex = st.selectbox('Sex', options=['male', 'female'])
bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=20.0)
children = st.number_input('Children', min_value=0, max_value=10, value=2)
smoker = st.selectbox('Smoker', options=['yes', 'no'])
region = st.selectbox('Region', options=['northeast', 'southeast', 'southwest', 'northwest'])

# Create input dictionary
input_dict = {
    'age': age,
    'sex': sex,
    'bmi': bmi,
    'children': children,
    'smoker': smoker,
    'region': region
}

# Convert input to DataFrame
input_df = pd.DataFrame([input_dict])


# Add a button to make predictions
if st.button('Predict'):
    # Make predictions
    predictions_df = predict_model(estimator=model, data=input_df)

    # Display the predictions DataFrame to debug
    st.write("Predictions DataFrame:", predictions_df)

    # Extract the prediction value
    prediction_column = 'Label' if 'Label' in predictions_df.columns else predictions_df.columns[-1]
    pred = predictions_df.iloc[0][prediction_column]

    # Ensure 'pred' is a float
    try:
        pred = float(pred)
        st.write(f'Predicted Insurance Charges: ${pred:.2f}')
    except ValueError:
        st.write(f'Error: Unable to convert prediction to a float. Received: {pred}')
