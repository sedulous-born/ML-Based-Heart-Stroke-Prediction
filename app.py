
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('heart_disease_data.csv')

# Sidebar for data exploration options
st.sidebar.title("Data Exploration")

if st.sidebar.checkbox('Show Data Summary'):
    st.write("## Data Summary")
    st.write(df.describe())

if st.sidebar.checkbox('Show Correlation Heatmap'):
    st.write("## Correlation Heatmap")
    corr = df.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    st.pyplot(plt)


# Split the data into features and target
X = df.drop(columns='target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Create a logistic regression model and train it
model = LogisticRegression()
model.fit(X_train, y_train)

# Model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit app for predictions
st.title('Heart Disease Prediction')

# User input for the features
age = st.number_input('Age', min_value=1, max_value=120, value=25)
sex = st.selectbox('Sex (1: Male, 0: Female)', [1, 0])
cp = st.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure', min_value=0, max_value=300, value=120)
chol = st.number_input('Cholesterol Level', min_value=0, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1: True, 0: False)', [1, 0])
restecg = st.selectbox('Resting Electrocardiographic Results (0-2)', [0, 1, 2])
thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=250, value=150)
exang = st.selectbox('Exercise Induced Angina (1: Yes, 0: No)', [1, 0])
oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox('Slope of the Peak Exercise ST Segment (0-2)', [0, 1, 2])
ca = st.selectbox('Number of Major Vessels (0-4)', [0, 1, 2, 3, 4])
thal = st.selectbox('Thalassemia (1-3)', [1, 2, 3])

# Make a prediction
if st.button('Predict'):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.write('Prediction: The patient is at risk of heart disease.')
    else:
        st.write('Prediction: The patient is not at risk of heart disease.')

# Display model accuracy
st.write(f'Model Accuracy: {accuracy * 100:.2f}%')
