import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import accuracy_score

print("TensorFlow version:", tf.__version__)

# Define reward function
def get_reward(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    reward = 1 / (1 + mse)  # Reward is inversely proportional to the MSE
    return reward

# Load data
data = pd.read_csv("data.csv")

# Split data into train and test sets
train_data = data.iloc[:int(0.8*len(data))]
test_data = data.iloc[int(0.8*len(data)):]

# Normalize data
scaler = MinMaxScaler()
train_data_norm = scaler.fit_transform(train_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SMA', 'RSI', 'MACD', 'upper_band',
                                       'middle_band', 'lower_band', 'aroon_up', 'aroon_down', 'kicking', 'ATR', 'upper_band_supertrend', 'lower_band_supertrend']])
test_data_norm = scaler.transform(test_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SMA', 'RSI', 'MACD', 'upper_band',
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
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data_norm, timesteps)
X_test, y_test = create_sequences(test_data_norm, timesteps)

# Build model
model = Sequential()

# Add first LSTM layer with 100 units
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

# Add second LSTM layer with 50 units
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

# Add third LSTM layer with 25 units
model.add(LSTM(units=25))
model.add(Dropout(0.2))

# Add first dense layer with 50 units
model.add(Dense(units=50, activation='relu'))

# Add second dense layer with 25 units
model.add(Dense(units=25, activation='relu'))

# Add output layer
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mse')


# Train model with RL
callback = Callback()
history = model.fit(X_train, y_train, epochs=360, batch_size=50,
                    validation_data=(X_test, y_test), callbacks=[callback])

# Save model
model.save('model.h5')
