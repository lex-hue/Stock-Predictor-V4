import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Lambda
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

print("TensorFlow version:", tf.__version__)

# Define reward function
def get_reward(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    acc = np.mean(y_true / y_pred)
    reward = (acc - mse)
    return reward

# Load data
data = pd.read_csv("data.csv")

# Split data into train and test sets
train_data = data.iloc[:int(0.8*len(data))]
test_data = data.iloc[int(0.8*len(data)):]

# Normalize data
scaler = MinMaxScaler()
train_data_norm = scaler.fit_transform(train_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SMA', 'RSI', 'MACD', 'upper_band', 'middle_band', 'lower_band', 'aroon_up', 'aroon_down', 'kicking', 'ATR', 'ADX', 'CCI', 'upper_band_supertrend', 'lower_band_supertrend', 'in_uptrend', 'supertrend_signal', 'EMA', 'STOCH_k', 'STOCH_d', 'obv', 'pct_change', 'money_change']])
test_data_norm = scaler.transform(test_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SMA', 'RSI', 'MACD', 'upper_band', 'middle_band', 'lower_band', 'aroon_up', 'aroon_down', 'kicking', 'ATR', 'ADX', 'CCI', 'upper_band_supertrend', 'lower_band_supertrend', 'in_uptrend', 'supertrend_signal', 'EMA', 'STOCH_k', 'STOCH_d', 'obv', 'pct_change', 'money_change']])

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
model.add(LSTM(units=300, return_sequences=True, input_shape=(timesteps, X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=200, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=130, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=100))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.summary()

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error', run_eagerly=True)

# Define callbacks
filepath="model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Train model
history = model.fit(X_train, y_train, epochs=150, batch_size=50, validation_data=(X_test, y_test), callbacks=[checkpoint, early_stopping])

# Evaluate model
model = load_model("model.h5")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_reward = get_reward(y_train, y_pred_train)
test_reward = get_reward(y_test, y_pred_test)

print("Train reward:", train_reward)
print("Test reward:", test_reward)
