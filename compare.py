import os
import pandas as pd
import matplotlib.pyplot as plt

# Load predicted and actual data
predicted_data = pd.read_csv("predictions.csv")
actual_data = pd.read_csv(os.path.join("data", "BTC-USD.csv"))

# Rename columns for clarity
predicted_data = predicted_data.rename(columns={'Predicted Close': 'Close'})
actual_data = actual_data.rename(columns={'Close': 'Actual Close'})

# Join predicted and actual data on the date column
combined_data = pd.merge(predicted_data, actual_data, on='Date')

# Calculate the absolute percentage error between the predicted and actual values
combined_data['Absolute % Error'] = abs(combined_data['Close'] - combined_data['Actual Close']) / combined_data['Actual Close'] * 100

# Calculate the mean absolute percentage error and print it
mape = combined_data['Absolute % Error'].mean()
print(f"Mean Absolute Percentage Error: {mape:.2f}%")

# Find the row with the highest and lowest absolute percentage error and print them
min_error_row = combined_data.iloc[combined_data['Absolute % Error'].idxmin()]
max_error_row = combined_data.iloc[combined_data['Absolute % Error'].idxmax()]
print(f"\nMost Accurate Prediction:\n{min_error_row}\n")
print(f"Least Accurate Prediction:\n{max_error_row}\n")

# Plot the predicted and actual close prices
plt.plot(combined_data['Date'], combined_data['Close'], label='Predicted Close')
plt.plot(combined_data['Date'], combined_data['Actual Close'], label='Actual Close')

# Add title and legend
plt.title('Predicted vs Actual Close Prices')
plt.legend()

# Show plot
plt.show()