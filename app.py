import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pandas as pd
import pickle

#load the trained model
model = tf.keras.models.load_model('model.h5')

#load the encoders and sclar

with open('label_encode_gender.pkl','rb') as file:
    label_encode_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)
    
with open('scalar.pkl','rb') as file:
    scalar = pickle.load(file)
    
##streamlit app

st.title('Customer Churn Prediction')

#user input
geography = st.selectbox('Geography',('France','Spain','Germany'))
gender = st.selectbox('Gender',('Male','Female'))
age = st.slider('Age',18,100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encode_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

#onehot encode "Geography"
geo_encode = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encode, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

#combine onehot encoded "Geography" with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

#scale the input data
input_data_scaled = scalar.transform(input_data)

#make prediction
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

st.write(f'Customer Churn Probability: {prediction_prob:.2f}')

if prediction_prob > 0.5:
    st.write('Customer will not churn')
else:
    st.write('Customer will churn')