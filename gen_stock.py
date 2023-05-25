import csv
import random
import datetime

def generate_stock_data_csv(file_path, num_lines, data_type):
    # Define the column names
    columns = ['Date', 'Close', 'Adj Close', 'Volume', 'High', 'Low', 'Open']

    # Generate stock data based on the selected data type
    data = []
    start_date = datetime.datetime(2022, 1, 1)
    for i in range(num_lines):
        if data_type == 'linear':
            close_price = 100.0 + i * 10
        elif data_type == 'exponential':
            close_price = 100.0 * (1.1 ** i)
        elif data_type == 'random':
            close_price = random.uniform(80, 120)
        elif data_type == 'trend':
            close_price = 100.0 + i * 5 * random.uniform(0.8, 1.2)
        else:
            raise ValueError("Invalid data type provided.")

        date = start_date + datetime.timedelta(days=i)
        adj_close = close_price
        volume = random.randint(100000, 1000000)
        high = close_price * random.uniform(1.01, 1.05)
        low = close_price * random.uniform(0.95, 0.99)
        open_price = close_price * random.uniform(0.98, 1.02)

        data.append([date.strftime('%Y-%m-%d'), close_price, adj_close, volume, high, low, open_price])

    # Save the generated data to a CSV file
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(data)

    print(f"Stock data CSV file '{file_path}' generated successfully.")


# Prompt the user for options
num_lines = int(input("Enter the number of lines: "))
data_type = input("Enter the data type (linear/exponential/random/trend): ")
file_path = "data/example.csv"

# Generate the stock data CSV file
generate_stock_data_csv(file_path, num_lines, data_type)
