import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model

print("TensorFlow version:", tf.__version__)

# Load data
data = pd.read_csv("data.csv")

# Normalize data
scaler = MinMaxScaler()
data_norm = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SMA', 'RSI', 'MACD', 'upper_band', 'middle_band', 'lower_band', 'aroon_up', 'aroon_down', 'kicking', 'ATR', 'upper_band_supertrend', 'lower_band_supertrend']])

# Define time steps
timesteps = 100

# Extract the last 60 days of data
last_60_days = data_norm[-60:]

# Create a sequence of input data to predict the next 30 days
X_test = np.array([last_60_days[i-timesteps:i] for i in range(timesteps, len(last_60_days)+1)])

# Load model
tf.config.run_functions_eagerly(True)
model = load_model('model.h5')

# Evaluate model
y_pred = model.predict(X_test)

# Inverse transform the predicted values
y_pred_inv = scaler.inverse_transform(np.hstack((X_test[:, -1, :-1], y_pred.reshape(-1, 1))))

# Get the predicted prices for the next 30 days
predicted_prices = y_pred_inv[:, -1]

# Print the predicted prices
for i, price in enumerate(predicted_prices):
    print(f"Day {i+1}: ${price:.2f}")
