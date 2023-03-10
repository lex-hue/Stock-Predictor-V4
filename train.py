import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load data
data = pd.read_csv("data.csv")

# Split data into train and test sets
train_data = data.iloc[:int(0.8*len(data))]
test_data = data.iloc[int(0.8*len(data)):]

# Normalize data
scaler = MinMaxScaler()
train_data_norm = scaler.fit_transform(train_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SMA', 'RSI', 'MACD', 'upper_band', 'middle_band', 'lower_band', 'aroon_up', 'aroon_down', 'kicking', 'ATR', 'upper_band_supertrend', 'lower_band_supertrend']])
test_data_norm = scaler.transform(test_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SMA', 'RSI', 'MACD', 'upper_band', 'middle_band', 'lower_band', 'aroon_up', 'aroon_down', 'kicking', 'ATR', 'upper_band_supertrend', 'lower_band_supertrend']])

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
model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mse')

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save model
joblib.dump(model, 'best_model.joblib', compress=True)
