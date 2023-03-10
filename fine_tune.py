import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

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

X_test, y_test = create_sequences(test_data_norm, timesteps)

# Load model
model = load_model('model.h5')

# Define accuracy threshold and reward threshold
accuracy_threshold = 0.895
reward_threshold = 0.03

# Initialize rewards
rewards = []

while True:
    # Evaluate model
    loss, mse = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test)
    accuracy = sum(np.round(y_pred) == y_test)/len(y_test)

    # Append accuracy to rewards
    rewards.append(accuracy)

    # Print current accuracy and rewards
    print("Accuracy:", accuracy)
    print("Rewards:", rewards)

    # Check if accuracy threshold is reached
    if accuracy >= accuracy_threshold:
        print("Accuracy threshold reached!")
        break

    # Check if reward threshold is reached
    if len(rewards) >= 3 and sum(rewards[-3:]) >= reward_threshold:
        print("Reward threshold reached!")
        model.save('model.h5')
        break

    # Fine-tune model if accuracy is not high enough
    model.fit(X_test, y_test, epochs=1, verbose=0)
