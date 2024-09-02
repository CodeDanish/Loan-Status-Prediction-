import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.preprocessing import LabelEncoder

loaded_model = pkl.load(open('classifier.pkl', 'rb'))
loan_data = pd.read_csv('loan_data_backup.csv')

def status_prediction(Gender, Married,  Dependents, Education, Self_Employed, Applicant_Income, Coapplicant_Income, Loan_Amount, Loan_Amount_Term, Credit_History, Property_Area):

    new_loan_data = loan_data.copy()

    if Gender == 'Male':
        Gender_val = 1
    else: 
        Gender_val = 0

    if Married == 'Yes':
        Married_val = 1
    else: 
        Married_val = 0

    if Education == 'Graduate':
        Education_val = 1
    else: 
        Education_val = 0

    if Self_Employed == 'Yes':
        Self_Employed_val = 1
    else: 
        Self_Employed_val = 0

    if Property_Area == 'Urban':
        Property_Area_val = 0
    elif Property_Area == 'Semiurban':
        Property_Area_val = 1
    else:
        Property_Area_val = 2

    loan_status = np.array([Gender_val, Married_val,  Dependents, Education_val, Self_Employed_val, Applicant_Income, Coapplicant_Income, Loan_Amount, Loan_Amount_Term, Credit_History, Property_Area_val])

    loan_status_reshape = loan_status.reshape(1,-1)

    prediction = loaded_model.predict(loan_status_reshape)
    return prediction[0]

# Streamlit UI
st.title('Loan Status Prediction')

# Getting the input data from user
# columns for input field
col1, col2, col3 =  st.columns(3)

# User input fields
with col1:
    Gender =  st.selectbox('Select Your Gender:', ['Male','Female'])

with col2:
    Married = st.selectbox('Are you Married', ['Yes','No'])

with col3:
    Dependents = st.selectbox('No.Of Dependents:', [0, 1, 2, 3, 4])

with col1:
    Education = st.selectbox('Are you Educated', ['Yes','No'])

with col2:
    Self_Employed = st.selectbox('Are you Self Employed:', ['Yes','No'])

with col3:
    Applicant_Income = st.number_input('Enter your Income',)

with col1:
    Coapplicant_Income = st.number_input('Enter Coapplicant Income:')

with col2:
    Loan_Amount = st.number_input('Enter your Loan Amount:')

with col3:
    Loan_Amount_Term = st.selectbox('Select Loan Amount Term:', pd.unique(loan_data['Loan_Amount_Term'].values))

with col1:
    Credit_History = st.selectbox('Select Credit History:', [0,1])

with col2:
    Property_Area = st.selectbox('Select Property Area:', pd.unique(loan_data['Property_Area'].values))

# Predict button
if st.button('Predict Loan Status'):
    # Make prediction
    prediction = status_prediction(Gender, Married, Dependents, Education, Self_Employed, Applicant_Income, Coapplicant_Income, Loan_Amount, Loan_Amount_Term, Credit_History, Property_Area)

    if prediction == 0:
        st.success(f'Application for your Loan approval has been Rejected')
    else:
        st.success(f'Application for your Loan approval has been Accepted')

