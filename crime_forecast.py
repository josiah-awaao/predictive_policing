# predictive_gui.py
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import joblib
import folium
from folium.plugins import MarkerCluster
import webbrowser
import os
from datetime import timedelta
from sklearn.metrics import classification_report, roc_auc_score

# Load model once
MODEL_PATH = "crime_predictor_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Please train and save the model first.")

# Fallback setup
weather_cols = ['temp', 'rhum', 'prcp', 'wspd']
FEATURES = [
    'Hour', 'DayOfWeek', 'Month', 'IsWeekend',
    'temp', 'rhum', 'prcp', 'wspd',
    'RecentArrests', 'RepeatOffenderSignal'
]

# Use sample datetime and sampled locations
last_hour = pd.Timestamp.now().floor('h')
future_hours = pd.date_range(start=last_hour + timedelta(hours=1), periods=12, freq='h')

latitudes = np.round(np.random.uniform(41.64, 42.02, 100), 2)
longitudes = np.round(np.random.uniform(-87.94, -87.52, 100), 2)
sampled_locations = pd.DataFrame({'LatGrid': latitudes, 'LonGrid': longitudes})

forecast_grid = pd.MultiIndex.from_product(
    [future_hours, sampled_locations['LatGrid'], sampled_locations['LonGrid']],
    names=['DateHour', 'LatGrid', 'LonGrid']
)
forecast_df = pd.DataFrame(index=forecast_grid).reset_index()

forecast_df['Hour'] = forecast_df['DateHour'].dt.hour
forecast_df['DayOfWeek'] = forecast_df['DateHour'].dt.dayofweek
forecast_df['Month'] = forecast_df['DateHour'].dt.month
forecast_df['IsWeekend'] = forecast_df['DayOfWeek'].isin([5, 6]).astype(int)

# Generate weather features
for col in weather_cols:
    forecast_df[col] = np.random.normal(loc=10.0, scale=5.0, size=len(forecast_df)).astype(np.float32)

forecast_df['RecentArrests'] = np.random.poisson(2, size=len(forecast_df))
forecast_df['RepeatOffenderSignal'] = np.random.poisson(5, size=len(forecast_df))

forecast_X = forecast_df[FEATURES]
forecast_df['PredictedCrimeProb'] = model.predict_proba(forecast_X)[:, -1]
top_preds = forecast_df.sort_values('PredictedCrimeProb', ascending=False).head(10)

# Output predictions
print("\nTop 10 Predicted Crime Locations in Next 12 Hours:")
print(top_preds[['DateHour', 'LatGrid', 'LonGrid', 'PredictedCrimeProb']])

# Generate map
m = folium.Map(location=[41.88, -87.63], zoom_start=12)
marker_cluster = MarkerCluster().add_to(m)
for _, row in top_preds.iterrows():
    folium.Marker(
        location=[row['LatGrid'], row['LonGrid']],
        popup=f"{row['DateHour']}\nProbability: {row['PredictedCrimeProb']:.2%}",
        icon=folium.Icon(color="red" if row['PredictedCrimeProb'] > 0.5 else "green")
    ).add_to(marker_cluster)

map_path = os.path.abspath("crime_forecast_map.html")
m.save(map_path)
webbrowser.open(f"file://{map_path}")
