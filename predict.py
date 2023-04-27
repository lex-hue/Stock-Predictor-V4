import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load data
data = pd.read_csv("data.csv")

# Normalize data
scaler = MinMaxScaler()
data_norm = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SMA', 'RSI', 'MACD', 'upper_band',
                                       'middle_band', 'lower_band', 'aroon_up', 'aroon_down', 'kicking', 'ATR', 'upper_band_supertrend', 'lower_band_supertrend']])

# Define time steps
timesteps = 100

# Create sequences of timesteps
def create_sequences(data, timesteps):
    X = []
    y = []
    for i in range(timesteps, len(data)):
        X.append(data[i-timesteps:i])
        y.append(data[i, 3])
    X = np.array(X)
    y = np.array(y)
    return X, y


X, y = create_sequences(data_norm, timesteps)

# Load model
model = load_model('model.h5')

# Make predictions
y_pred = model.predict(X)

# Scale back to original values
y_pred = scaler.inverse_transform(np.concatenate((np.zeros((timesteps, 1)), y_pred), axis=0))[:, 0]

# Plot and save predictions
import matplotlib.pyplot as plt

plt.plot(data['Date'], data['Close'], label='Actual')
plt.plot(data['Date'][timesteps:], y_pred[timesteps:], label='Predicted')
plt.legend()
plt.savefig('predictions.png')

# Save predictions to CSV
df = pd.DataFrame({'Date': data['Date'][timesteps:], 'Predicted': y_pred[timesteps:]})
df.to_csv('predictions.csv', index=False)
