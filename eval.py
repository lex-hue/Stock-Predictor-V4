import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)

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

# Evaluate model
rmse_scores = []
r2_scores = []
rewards = []
for i in range(30):
    print(f"Evaluating model {i+1}/30")
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    rmse_scores.append(rmse)
    r2_scores.append(r2)
    if r2 > 0.9:
        rewards.append(1)
        model.save('model.h5')
    else:
        rewards.append(0)


# Print results
print(f"Mean RMSE: {np.mean(rmse_scores)}")
print(f"Mean R2: {np.mean(r2_scores)}")
print(f"Total Rewards: {sum(rewards)}")

# Plot results
fig, axs = plt.subplots(3, 1, figsize=(10,10))
axs[0].plot(rmse_scores)
axs[0].set_title('RMSE')
axs[1].plot(r2_scores)
axs[1].set_title('R2 Score')
axs[2].plot(rewards)
axs[2].set_title('Rewards')
plt.tight_layout()
plt.show()
