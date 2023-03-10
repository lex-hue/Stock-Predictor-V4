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

# Evaluate model
accuracies = []
rewards = []
mses = []
for i in range(30):
    loss, mse = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test)
    accuracy = sum(np.round(y_pred) == y_test)/len(y_test)
    accuracies.append(accuracy)
    mses.append(mse)
    if accuracy > 0.85:
        rewards.append(1)
    else:
        rewards.append(0)

# Plot results
fig, axs = plt.subplots(3, 1, figsize=(10,10))
axs[0].plot(mses)
axs[0].set_title('MSE')
axs[1].plot(accuracies)
axs[1].set_title('Accuracy')
axs[2].plot(rewards)
axs[2].set_title('Rewards')
plt.tight_layout()
plt.show()
