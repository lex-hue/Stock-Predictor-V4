import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("data.csv")

# Normalize data
scaler = MinMaxScaler()
data_norm = scaler.fit_transform(data[['Close', 'Adj Close', 'Volume', 'High', 'Low', 'SMA', 'MACD', 'upper_band', 'middle_band', 'lower_band', 'supertrend_signal', 'RSI', 'aroon_up', 'aroon_down', 'kicking', 'upper_band_supertrend', 'lower_band_supertrend']])

# Define time steps
timesteps = 100

# Create sequences of timesteps
def create_sequences(data, timesteps):
    X = []
    for i in range(timesteps, len(data)):
        X.append(data[i-timesteps:i])
    return np.array(X)

X_data = create_sequences(data_norm, timesteps)

# Load model
model = load_model('model.h5')
model.summary()

num_predictions = 30

# Make predictions for next num_predictions days
X_pred = X_data[-num_predictions:].reshape((num_predictions, timesteps, X_data.shape[2]))
y_pred = model.predict(X_pred)[:, 0]

# Inverse transform predictions
y_pred = scaler.inverse_transform(np.hstack([np.zeros((len(y_pred), data_norm.shape[1]-1)), np.array(y_pred).reshape(-1, 1)]))[:, -1]

# Generate date index for predictions
last_date = data['Date'].iloc[-1]
index = pd.date_range(last_date, periods=num_predictions, freq='D', tz='UTC').tz_localize(None)

# Calculate % change
y_pred_pct_change = (y_pred - y_pred[0]) / y_pred[0] * 100

# Save predictions and % change in a CSV file
predictions = pd.DataFrame({'Date': index, 'Predicted Close': y_pred, '% Change': y_pred_pct_change})
predictions.to_csv('predictions.csv', index=False)

print(predictions)

# Find the rows with the lowest and highest predicted close and the highest and lowest % change
min_close_row = predictions.iloc[predictions['Predicted Close'].idxmin()]
max_close_row = predictions.iloc[predictions['Predicted Close'].idxmax()]
max_pct_change_row = predictions.iloc[predictions['% Change'].idxmax()]
min_pct_change_row = predictions.iloc[predictions['% Change'].idxmin()]

# Print the rows with the lowest and highest predicted close and the highest and lowest % change
print(f"\n\nHighest predicted close:\n{max_close_row}\n")
print(f"Lowest predicted close:\n{min_close_row}\n")
print(f"Highest % change:\n{max_pct_change_row}\n")
print(f"Lowest % change:\n{min_pct_change_row}")

# Plot historical data and predictions
plt.plot(data['Close'].values, label='Actual Data')
plt.plot(np.arange(len(data), len(data)+num_predictions), y_pred, label='Predicted Data')

# Add red and green arrows for highest and lowest predicted close respectively, and highest and lowest percentage change
plt.annotate('↓', xy=(min_close_row.name - len(data), min_close_row['Predicted Close']), color='red', fontsize=16, arrowprops=dict(facecolor='red', shrink=0.05))
plt.annotate('↑', xy=(max_close_row.name - len(data), max_close_row['Predicted Close']), color='green', fontsize=16, arrowprops=dict(facecolor='green', shrink=0.05))
plt.annotate('↑', xy=(max_pct_change_row.name - len(data), y_pred.max()), color='green', fontsize=16, arrowprops=dict(facecolor='green', shrink=0.05))
plt.annotate('↓', xy=(min_pct_change_row.name - len(data), y_pred.min()), color='red', fontsize=16, arrowprops=dict(facecolor='red', shrink=0.05))

# Add legend and title
plt.legend()
plt.title('Predicted Close Prices')

# Show plot
plt.show()