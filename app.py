import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv(r"Crop_recommendation.csv")  # Use relative path
    crop_dict = {
        'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5,
        'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10,
        'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14,
        'lentil': 15, 'blackgram': 16, 'mungbean': 17, 'mothbeans': 18,
        'pigeonpeas': 19, 'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
    }
    data['crop_num'] = data['label'].map(crop_dict)
    data.drop(['label'], axis=1, inplace=True)
    return data, crop_dict

# Load and prepare data
data, crop_dict = load_data()
X = data.drop(['crop_num'], axis=1)
y = data['crop_num']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Crop Dictionary for decoding predictions
reverse_crop_dict = {v: k for k, v in crop_dict.items()}

# Function to make crop recommendation
def recommend_crop(n, p, k, temperature, humidity, ph, rainfall):
    # Prepare input as DataFrame for consistency
    input_data = pd.DataFrame([[n, p, k, temperature, humidity, ph, rainfall]],
                               columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    input_data_scaled = scaler.transform(input_data)
    crop_num = model.predict(input_data_scaled)[0]
    return reverse_crop_dict[crop_num]

# Streamlit App
st.title("🌾 Crop Recommendation System")
st.markdown("Enter the soil and environmental parameters to get a crop recommendation.")

# Display Model Accuracy
st.sidebar.header("Model Performance")
st.sidebar.write(f"🌟 **Model Accuracy:** {accuracy * 100:.2f}%")

# User Inputs
n = st.number_input("Nitrogen (N)", min_value=0, max_value=300, value=90)
p = st.number_input("Phosphorus (P)", min_value=0, max_value=300, value=42)
k = st.number_input("Potassium (K)", min_value=0, max_value=300, value=43)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.5)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0)
ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=220.0)

# Prediction Button
if st.button("Recommend Crop"):
    recommended_crop = recommend_crop(n, p, k, temperature, humidity, ph, rainfall)
    st.success(f"🌱 Recommended Crop: **{recommended_crop}**")

    # Optionally provide more info about the recommended crop
    st.write(f"You might want to consider additional factors such as local climate and soil type for best results.")
