import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import folium
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_folium import folium_static
from datetime import datetime, timedelta
from folium.plugins import HeatMap
import io

# OpenWeather API Key (Replace with your own key)
API_KEY = "2d6c23f739fdc7ba90b8754fd02dae9c"

# Load trained LightGBM model (NO₂ and NO only)
model_file = "lightgbm_air_pollution_model_no2_no.pkl"
lgb_models = joblib.load(model_file)

# Function to fetch real-time hourly weather data
def get_weather_data():
    url = f"http://api.openweathermap.org/data/2.5/weather?q=Codogno,it&units=metric&appid={API_KEY}"
    response = requests.get(url)
    data = response.json()

    weather_info = {
        "humidity": data["main"]["humidity"],
        "temperature": data["main"]["temp"],
        "precipitation": data.get("rain", {}).get("1h", 0),  # Rainfall in last hour
        "wind_speed": data["wind"]["speed"] * 3.6  # Convert from m/s to km/h
    }
    return weather_info

# Streamlit App Title
st.title("⏳ Hourly Air Pollution Forecast in Codogno, Italy 🇮🇹")
st.write("This app predicts **hourly NO₂ and NO levels** using **LightGBM AI models**.")

# Sidebar options for single prediction
st.sidebar.subheader("📅 Select Single Prediction Date and Hour")
date = st.sidebar.date_input("Date", datetime.today())
hour = st.sidebar.slider("Hour (0-23)", min_value=0, max_value=23, value=datetime.now().hour)

# Sidebar options for batch download
st.sidebar.subheader("📥 Select Date Range for Data Download")
start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=7))
end_date = st.sidebar.date_input("End Date", datetime.today())

# Convert selected date into features
day = date.day
month = date.month
year = date.year
weekday = date.weekday()

# Single Prediction Button
if st.sidebar.button("🔍 Predict for Selected Date & Hour"):
    # Fetch real-time weather data
    weather_data = get_weather_data()

    # LightGBM Predictions (Weather-Based AI)
    input_data = pd.DataFrame([[year, month, day, hour, weekday, 
                                weather_data["humidity"], weather_data["temperature"], 
                                weather_data["precipitation"], weather_data["wind_speed"] / 3.6]],  # Convert to m/s
                              columns=["Year", "Month", "Day", "Hour", "Weekday", "Humidity", "Temperature", 
                                       "Precipitation", "WindSpeed_m_s"])

    # Generate predictions from LightGBM
    lightgbm_preds = {target: lgb_models[target].predict(input_data)[0] for target in lgb_models}

    # Display Results
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="🌫️ NO₂ (µg/m³)", value=f"{lightgbm_preds['NO2_µg_m3']:.2f}")
    with col2:
        st.metric(label="💨 NO (µg/m³)", value=f"{lightgbm_preds['NO_µg_m3']:.2f}")

# Batch Download Button
if st.sidebar.button("📥 Download NO₂ & NO Data"):
    # Create DataFrame for Date Range
    date_range = pd.date_range(start=start_date, end=end_date, freq="H")
    
    batch_data = []
    for timestamp in date_range:
        year, month, day, hour, weekday = timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.weekday()
        
        # Fetch real-time weather data
        weather_data = get_weather_data()

        # Prepare input data
        input_data = pd.DataFrame([[year, month, day, hour, weekday, 
                                    weather_data["humidity"], weather_data["temperature"], 
                                    weather_data["precipitation"], weather_data["wind_speed"] / 3.6]],  # Convert to m/s
                                  columns=["Year", "Month", "Day", "Hour", "Weekday", "Humidity", "Temperature", 
                                           "Precipitation", "WindSpeed_m_s"])

        # Generate predictions
        predictions = {target: lgb_models[target].predict(input_data)[0] for target in lgb_models}

        # Append results
        batch_data.append({
            "DateTime": timestamp,
            "NO₂ (µg/m³)": predictions["NO2_µg_m3"],
            "NO (µg/m³)": predictions["NO_µg_m3"],
            "Humidity (%)": weather_data["humidity"],
            "Temperature (°C)": weather_data["temperature"],
            "Precipitation (mm)": weather_data["precipitation"],
            "Wind Speed (km/h)": weather_data["wind_speed"]
        })

    # Convert to DataFrame
    batch_df = pd.DataFrame(batch_data)

    # Convert dataframe to CSV
    csv_buffer = io.StringIO()
    batch_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    # Button to download CSV
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv_buffer.getvalue(),
        file_name=f"NO2_NO_Pollution_{start_date}_to_{end_date}.csv",
        mime="text/csv"
    )
