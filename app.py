import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import joblib
import json
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Set Streamlit page config first thing
st.set_page_config(page_title="EV Forecast", layout="wide")

# This function loads the data and the models and caches them
# It runs only once when the app is first started
@st.cache_resource
def load_resources():
    try:
        # The LSTM model is loaded from a directory, not a .pkl file
        model = load_model('ev_lstm_model.keras')
        
        # Load and prepare the CSV data
        df = pd.read_csv('Electric_vehicle_population_By_country.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df['County'] = df['County'].str.strip()
        df['State'] = df['State'].str.strip()
        df['Electric Vehicle (EV) Total'] = pd.to_numeric(df['Electric Vehicle (EV) Total'].astype(str).str.replace(',', ''), errors='coerce')
        
        # Filter for Washington and Passenger vehicles
        df_wa = df[(df['State'] == 'WA') & (df['Vehicle Primary Use'] == 'Passenger')].copy()
        
        # Get a list of all unique counties
        all_counties = sorted(df_wa['County'].unique())

        return model, df_wa, all_counties

    except Exception as e:
        st.error(f"Error loading resources. Please ensure `EV_DL_Model_Training.ipynb` has been run to generate the model files. Error: {e}")
        st.stop()

# Load all resources at the start of the app
model, df_wa, available_counties = load_resources()

# === Styling ===
st.markdown("""
    <style>
        body {
            background-color: #fcf7f7;
            color: #000000;
        }
        .stApp {
            background: linear-gradient(to right, #c2d3f2, #7f848a);
        }
    </style>
""", unsafe_allow_html=True)

# Stylized title using markdown + HTML
st.markdown("""
    <div style='text-align: center; font-size: 36px; font-weight: bold; color: #FFFFFF; margin-top: 20px;'>
        ⚡ EV Adoption Forecaster for a County in Washington State
    </div>
""", unsafe_allow_html=True)

# Welcome subtitle
st.markdown("""
    <div style='text-align: center; font-size: 22px; font-weight: bold; padding-top: 10px; margin-bottom: 25px; color: #FFFFFF;'>
        Welcome to the Electric Vehicle (EV) Adoption Forecasting Tool!
    </div>
""", unsafe_allow_html=True)

# Display image after titles
img = Image.open("image.jpg")
img = img.resize((1400, 700))   # width=600, height=350 (medium height)

st.image(img, caption="Electric Vehicle Manufacturing & Adoption")

# === Helper function to perform prediction for a given county ===
def predict_county_data(county_name, forecast_horizon, model):
    county_data = df_wa[df_wa['County'] == county_name].sort_values(by='Date').copy()
    county_data['Cumulative_EV'] = county_data['Electric Vehicle (EV) Total'].cumsum()

    if len(county_data) < 13:
        # Returning None, None, None, None to handle the case of insufficient data
        return None, None, None, None

    # Rescale the data for the LSTM model
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(county_data['Cumulative_EV'].values.reshape(-1, 1))

    # Create look-back dataset for evaluation
    look_back = 12
    X_full, y_full = create_dataset(scaled_data, look_back)
    X_full = np.reshape(X_full, (X_full.shape[0], X_full.shape[1], 1))
    
    # Calculate accuracy metrics
    full_predict = model.predict(X_full, verbose=0)
    full_predict_inv = scaler.inverse_transform(full_predict)
    y_full_inv = scaler.inverse_transform(y_full.reshape(-1, 1))
    
    rmse = np.sqrt(mean_squared_error(y_full_inv, full_predict_inv))
    r2 = r2_score(y_full_inv, full_predict_inv)

    # Generate future predictions
    predictions = []
    input_sequence = scaled_data[-look_back:]
    input_sequence = np.reshape(input_sequence, (1, look_back, 1))
    
    for _ in range(forecast_horizon):
        next_pred = model.predict(input_sequence, verbose=0)
        predictions.append(next_pred[0, 0])
        input_sequence = np.append(input_sequence[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)

    forecasted_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    
    # Create forecast DataFrame
    last_date = county_data['Date'].max()
    forecast_dates = [last_date + timedelta(days=30 * i) for i in range(1, forecast_horizon + 1)]
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Cumulative EV': forecasted_values,
        'County': county_name
    })

    return county_data, forecast_df, rmse, r2

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# --- Main Dashboard Layout ---

st.markdown("---")
st.subheader("🔍 Individual County Forecast")

# Single county selection
selected_county = st.selectbox("Select a County", options=available_counties)
forecast_horizon = st.slider("Forecast Horizon (Months)", 12, 36, 36, 12)

if selected_county:
    # Get the data and forecast for the single selected county
    historical_df, forecast_df, rmse, r2 = predict_county_data(selected_county, forecast_horizon, model)
    
    if historical_df is not None:
        st.write(f"Forecast starting from: **{historical_df['Date'].iloc[-1].strftime('%B %d, %Y')}**")
        # Combine historical and forecasted data for plotting
        historical_plot_df = historical_df[['Date', 'Cumulative_EV']].rename(columns={'Cumulative_EV': 'Cumulative EV'})
        historical_plot_df['County'] = selected_county
        combined_df = pd.concat([historical_plot_df, forecast_df], ignore_index=True)

        st.markdown("---")
        st.subheader(f"📈 EV Adoption Prediction for {selected_county} County")
        
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(historical_plot_df['Date'], historical_plot_df['Cumulative EV'], label=f'{selected_county} (Historical)', marker='o', linestyle='-')
        ax.plot(forecast_df['Date'], forecast_df['Cumulative EV'], label=f'{selected_county} (Forecasted)', linestyle='--', marker='o')

        ax.set_title("EV Adoption Trends: Historical + 3-Year Forecast", fontsize=16, color='white')
        ax.set_xlabel("Date", color='white')
        ax.set_ylabel("Cumulative EV Count", color='white')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#1c1c1c")
        fig.patch.set_facecolor('#1c1c1c')
        ax.tick_params(colors='white')
        ax.legend(title="County", bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)
        
        # Display accuracy and growth
        historical_end = historical_df['Cumulative_EV'].iloc[-1]
        forecasted_end = forecast_df['Cumulative EV'].iloc[-1]
        growth_pct = ((forecasted_end - historical_end) / historical_end) * 100 if historical_end > 0 else 0
        
        st.success(
            f"**{selected_county} County**\n\n"
            f"**Projected Growth:** From a historical count of **{int(historical_end):,}**, the count is projected to increase to "
            f"**{int(forecasted_end):,}**, showing a growth of **{growth_pct:.2f}%**."
        )

st.markdown("---")
st.subheader("📊 Comparison of Multiple Counties Forecasts")

# Let user select up to 3 counties
selected_counties_compare = st.multiselect("Select up to 3 Counties to Compare", options=available_counties, default=["King", "Snohomish", "Spokane"])
forecast_horizon_compare = st.slider("Forecast Horizon (Months) for Comparison", 12, 36, 36, 12)

if selected_counties_compare:
    comparison_data = {}
    growth_summary = {}
    
    for county in selected_counties_compare:
        historical_df, forecast_df, _, _ = predict_county_data(county, forecast_horizon_compare, model)
        if historical_df is not None:
            comparison_data[county] = {
                'historical': historical_df,
                'forecast': forecast_df
            }
            
            historical_end = historical_df['Cumulative_EV'].iloc[-1]
            forecasted_end = forecast_df['Cumulative EV'].iloc[-1]
            growth_pct = ((forecasted_end - historical_end) / historical_end) * 100 if historical_end > 0 else 0
            growth_summary[county] = {
                'historical_end': historical_end,
                'forecasted_end': forecasted_end,
                'growth_pct': growth_pct
            }

    if comparison_data:
        st.write(f"Forecast starting from: **{list(comparison_data.values())[0]['historical']['Date'].iloc[-1].strftime('%B %d, %Y')}**")
        
        fig, ax = plt.subplots(figsize=(14, 7))
        for county, data in comparison_data.items():
            ax.plot(data['historical']['Date'], data['historical']['Cumulative_EV'], label=f'{county} (Historical)', marker='o', linestyle='-')
            ax.plot(data['forecast']['Date'], data['forecast']['Cumulative EV'], label=f'{county} (Forecasted)', linestyle='--', marker='o')

        ax.set_title("EV Adoption Trends: Historical + 3-Year Forecast", fontsize=16, color='white')
        ax.set_xlabel("Date", color='white')
        ax.set_ylabel("Cumulative EV Count", color='white')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#1c1c1c")
        fig.patch.set_facecolor('#1c1c1c')
        ax.tick_params(colors='white')
        ax.legend(title="County", bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)

        growth_str = " ✅ **Comparison Successful!**"
        for county, summary in growth_summary.items():
            growth_str += f"\n\n**{county} County**\n"
            growth_str += f"Projected Growth: From a historical count of **{int(summary['historical_end']):,}**, the count is projected to increase to **{int(summary['forecasted_end']):,}**, showing a growth of **{summary['growth_pct']:.2f}%**."
        st.success(growth_str)

st.markdown("---")
