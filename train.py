import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import accuracy_score

# Define reward function
def get_reward(y_true, y_pred):
    # Calculate accuracy score
    acc_score = accuracy_score(y_true, np.round(y_pred))
    # Return 1 as reward if accuracy is greater than 70%
    return 1 if acc_score > 0.7 else 0

# Define custom callback to calculate reward at end of each epoch
class RLCallback(Callback):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X)
        reward = get_reward(self.y, y_pred)
        logs['reward'] = reward

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

# Train model with RL
rl_callback = RLCallback(X_test, y_test)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[rl_callback])

# Save model
model.save('model.h5')
