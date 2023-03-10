import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("data.csv")

# Normalize data
scaler = MinMaxScaler()
data_norm = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SMA', 'RSI', 'MACD', 'upper_band', 'middle_band', 'lower_band', 'aroon_up', 'aroon_down', 'kicking', 'ATR', 'upper_band_supertrend', 'lower_band_supertrend']])

# Define time steps
timesteps = 100

# Create sequences of timesteps
def create_sequences(data, timesteps):
    X = []
    for i in range(timesteps, len(data)):
        X.append(data[i-timesteps:i])
    return np.array(X)

X = create_sequences(data_norm, timesteps)

# Load model
model = load_model('model.h5')

# Ask user for number of days to predict
num_days = int(input("Enter the number of days to predict: "))

# Predict next n days
X_future = X[-1:, :, :]
for i in range(num_days):
    y_pred = model.predict(X_future)
    X_future = np.append(X_future[:, 1:, :], y_pred.reshape(1, 1, -1), axis=1)

# Inverse transform the data
X_future_inv = scaler.inverse_transform(X_future.reshape(-1, X_future.shape[-1]))
y_pred_inv = X_future_inv[:, 3]

# Plot the predicted prices
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data['Date'], data['Close'])
ax.plot(data['Date'].iloc[-1:].append(pd.date_range(start=data['Date'].iloc[-1]+pd.Timedelta(days=1), periods=num_days, freq='D')), y_pred_inv)
ax.legend(['Actual', 'Predicted'])
ax.set_title('Stock price prediction')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
plt.show()

# Ask user for reward
reward = float(input("How much would you like to reward the model (out of 1-10)? "))
if reward > 0:
    # Update model with reward
    model.reward(reward)
    model.save('model.h5')
    print("Model updated with reward %.1f." % reward)
else:
    print("Model not updated.")
