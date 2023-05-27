import os
import signal
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

# Define custom Metric
def accuracy(y_true, y_pred):
    acc = tf.reduce_mean(tf.abs((y_true / y_pred) * 100))

    if tf.greater(acc, 100):
        acc = tf.constant(0, dtype=tf.float32)

    return acc

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
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data_norm, timesteps)
X_test, y_test = create_sequences(test_data_norm, timesteps)

# Load model
model = load_model('model.h5')
model.summary()

# Define reward threshold
reward_threshold = float(input("Enter the reward threshold (0 - 1, 0.9 recommended): "))

# Initialize rewards
rewards = []
mses = []
mapes = []
r2s = []
count = 0

# Function to handle SIGINT signal (CTRL + C)
def handle_interrupt(signal, frame):
    print("\nInterrupt received.")

    # Ask the user for confirmation
    user_input = input(f"Are you sure that you want to save the Model, Plot the Rewards and also End the Program? (yes/no): ")

    if user_input.lower() == 'yes':
        model.save('model.h5')

        # Plot results
        fig, axs = plt.subplots(4, 1, figsize=(10, 10))
        axs[0].plot(mses)
        axs[0].set_title('MSE')
        axs[1].plot(mapes)
        axs[1].set_title('MAPE')
        axs[2].plot(r2s)
        axs[2].set_title('R2')
        axs[3].plot(rewards)
        axs[3].set_title('Rewards')
        plt.tight_layout()
        plt.show()

        exit(0)

    else:
        # Plot results
        fig, axs = plt.subplots(4, 1, figsize=(10, 10))
        axs[0].plot(mses)
        axs[0].set_title('MSE')
        axs[1].plot(mapes)
        axs[1].set_title('MAPE')
        axs[2].plot(r2s)
        axs[2].set_title('R2')
        axs[3].plot(rewards)
        axs[3].set_title('Rewards')
        plt.tight_layout()
        plt.show()

        print("Continuing the Fine-tuning Process")

# Register the signal handler
signal.signal(signal.SIGINT, handle_interrupt)

while True:
    os.system('clear')
    # Load model
    model = load_model('model.h5')
    print("Evaluating Model")
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Append rewards
    reward = (((1 - mape) * 0.9) + (r2 * 1.1)) / 2
    rewards.append(reward)
    mses.append(mse)
    mapes.append(mape)
    r2s.append(r2)

    # Print current rewards
    print("Rewards:", rewards)
    print("MAPE:", mape)
    print("MSE:", mse)
    print("R2:", r2)
    count += 1
    print("Looped", count, "times.")

    # Check if reward threshold is reached
    if len(rewards) >= 1 and sum(rewards[-1:]) >= reward_threshold:
        print("Reward threshold reached!")
        model.save('model.h5')

        # Plot results
        fig, axs = plt.subplots(4, 1, figsize=(10, 10))
        axs[0].plot(mses)
        axs[0].set_title('MSE')
        axs[1].plot(mapes)
        axs[1].set_title('MAPE')
        axs[2].plot(r2s)
        axs[2].set_title('R2')
        axs[3].plot(rewards)
        axs[3].set_title('Rewards')
        plt.tight_layout()
        plt.show()

        break
    else:
        # Set up callbacks
        checkpoint = ModelCheckpoint("model.h5", save_best_only=True, verbose=1, mode="min")
        earlystop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

        # Fine-tune model)
        print("\nReward threshold not reached, Trying to Finetune the Model with 50 Epochs. Will only save best results and will early stop after 3 non-improvements")

        history = model.fit(X_train, y_train, epochs=50, batch_size=256, validation_data=(X_test, y_test), callbacks=[checkpoint, earlystop])
