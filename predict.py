import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("data.csv")

# Normalize data
scaler = MinMaxScaler()
data_norm = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SMA', 'RSI', 'MACD', 'upper_band', 'middle_band', 'lower_band', 'aroon_up', 'aroon_down', 'kicking', 'ATR', 'upper_band_supertrend', 'lower_band_supertrend']])

# Define time steps
timesteps = 100

# Create sequences of timesteps
def create_sequences(data, timesteps):
    X = []
    for i in range(timesteps, len(data)):
        X.append(data[i-timesteps:i])
    return np.array(X)

X_data = create_sequences(data_norm, timesteps)

# Load model
model = load_model('model.h5')
model.summary()

num_predictions = 365

# Make predictions for next num_predictions days
X_pred = X_data[-num_predictions:].reshape((num_predictions, timesteps, X_data.shape[2]))
y_pred = model.predict(X_pred)[:, 0]

# Inverse transform predictions
y_pred = scaler.inverse_transform(np.hstack([np.zeros((len(y_pred), 17)), np.array(y_pred).reshape(-1, 1)]))[:, -1]

# Generate date index for predictions
last_date = data['Date'].iloc[-1]
index = pd.date_range(last_date, periods=num_predictions, freq='D', tz='UTC').tz_localize(None)

# Save predictions in a CSV file
predictions = pd.DataFrame({'Date': index, 'Predicted Close': y_pred})
predictions.to_csv('predictions.csv', index=False)

print(predictions)
