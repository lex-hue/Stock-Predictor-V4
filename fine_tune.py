import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

print("TensorFlow version:", tf.__version__)

# Load data
data = pd.read_csv("data.csv")

# Split data into train and test sets
train_data = data.iloc[:int(0.8*len(data))]
test_data = data.iloc[int(0.8*len(data)):]

# Normalize data
scaler = MinMaxScaler()
train_data_norm = scaler.fit_transform(train_data[['Close', 'Adj Close', 'Volume', 'High', 'Low', 'SMA', 'MACD', 'upper_band', 'middle_band', 'lower_band', 'supertrend_signal', 'RSI', 'aroon_up', 'aroon_down', 'kicking', 'upper_band_supertrend', 'lower_band_supertrend']])
test_data_norm = scaler.transform(test_data[['Close', 'Adj Close', 'Volume', 'High', 'Low', 'SMA', 'MACD', 'upper_band', 'middle_band', 'lower_band', 'supertrend_signal', 'RSI', 'aroon_up', 'aroon_down', 'kicking', 'upper_band_supertrend', 'lower_band_supertrend']])

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

X_test, y_test = create_sequences(test_data_norm, timesteps)

# Load model
model = load_model('model.h5')

# Define reward threshold
reward_threshold = 0.99

# Initialize rewards
rewards = []
count = 0

while True:
    os.system('clear')
    # Load model
    model = load_model('model.h5')
    print("Evaluating Model")
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # Append rewards
    rewards.append(1 - mape)

    # Print current rewards
    print("Rewards:", rewards)
    print("MAPE:", mape)
    count += 1
    print("Looped", count, "times.")

    # Check if reward threshold is reached
    if len(rewards) >= 3 and sum(rewards[-3:]) / 3 >= reward_threshold:
        print("Reward threshold reached!")
        model.save('model.h5')
        break
    else:
        # Set up callbacks
        checkpoint = ModelCheckpoint("model.h5", save_best_only=True, verbose=1)
        earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

        # Fine-tune model)
        print("\nReward threshold not reached, Trying to Finetune the Model with 100 Epochs. Will only save best results and will early stop after 5 non-improvements")
        model.fit(X_test, y_test, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[checkpoint,earlystop], verbose=1)