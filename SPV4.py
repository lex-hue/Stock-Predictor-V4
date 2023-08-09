import argparse

def cpugpu():
    import tensorflow as tf
    # Check if GPU is available and print the list of GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print(f"Found GPU: {gpu}")
    else:
        print("No GPU devices found.")
    
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            # Specify the GPU device to use (e.g., use the first GPU)
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            
            # Test TensorFlow with a simple computation on the GPU
            with tf.device('/GPU:0'):
                a = tf.constant([1.0, 2.0, 3.0])
                b = tf.constant([4.0, 5.0, 6.0])
                c = a * b

            print("GPU is available and TensorFlow is using it.")
            print("Result of the computation on GPU:", c.numpy())
        except RuntimeError as e:
            print("Error while setting up GPU:", e)
    else:
        print("No GPU devices found, TensorFlow will use CPU.")

def install_dependencies():
    import os
    import platform
    import subprocess
    import time
    import sys

    print("\n--------------------------------------")
    system = platform.system()    
    print("OS: ", system)
    print("--------------------------------------\n")

    def install_dependencies():
        print("Installing Python dependencies...")
        start_time = time.time()
        packages = [
            "pandas",
            "ta",
            "numpy",
            "scikit-learn",
            "matplotlib",
        ]
        total_packages = len(packages)
        progress = 0
        for package in packages:
            progress += 1
            print(f"Installing {package}... ({progress}/{total_packages})")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", package], check=True
            )
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(
            f"Python dependencies installation complete (Time: {elapsed_time:.2f} seconds)"
        )

    if __name__ == "__main__":
        print("Welcome to the SPV4 installation!")
        print("This script will install all the necessary dependencies.\n")
        print("Prior to proceeding, ensure that you have the necessary programs installed to enable TensorFlow to utilize your GPU or GPUs. If you haven't done so yet, you may press CTRL + C now to halt the process.")

        time.sleep(5)

        def get_user_choice():
            while True:
                print("Select the version of TensorFlow you want to install:")
                print("1. TensorFlow with CPU support")
                print("2. TensorFlow with GPU support")
                choice = input("Enter 1 or 2 to make your selection: ")
                if choice in ["1", "2"]:
                    return choice
                print("Invalid choice. Please enter 1 or 2.")

        def install_tf(choice):
            if choice == "1":
                print("Installing TensorFlow with CPU support...")
                subprocess.run(["pip", "install", "tensorflow-cpu"])
                print("Installation of TensorFlow with CPU support completed.")
            elif choice == "2":
                print("Installing TensorFlow with GPU support...")
                subprocess.run(["pip", "install", "tensorflow"])
                print("Installation of TensorFlow with GPU support completed.")

        print("\nPython dependencies installation will now begin.")

        install_dependencies()

        print("Python dependencies installation completed successfully!\n")
        
        user_choice = get_user_choice()
        install_tf(user_choice)

        print("Creating 'data' directory...")
        os.makedirs("data", exist_ok=True)

        filename = os.path.join("data", "add csvs in this folder.txt")
        with open(filename, "w") as file:
            file.write("This is the 'add csvs in this folder.txt' file.")

        print("'data' directory and file created successfully!\n")
        print("SPV4 installation completed successfully!")

def prepare_data():
    print("Preprocessing and preparing the CSV data...")
    import os
    import pandas as pd
    import numpy as np
    import ta
    import matplotlib.pyplot as plt

    # List all CSV files in the "data" folder
    data_folder = "data"
    csv_files = [file for file in os.listdir(data_folder) if file.endswith(".csv")]

    # Print the available CSV files with numbers for selection
    print("Available CSV files:")
    for i, file in enumerate(csv_files):
        print(f"{i + 1}. {file}")

    # Ask for user input to select a CSV file
    selected_file_index = (
        int(input("Enter the number of the CSV file to preprocess: ")) - 1
    )
    selected_file = csv_files[selected_file_index]
    selected_file_path = os.path.join(data_folder, selected_file)

    # Preprocess the selected CSV file
    df = pd.read_csv(selected_file_path)

    # Calculate indicators using ta library
    df["SMA"] = ta.trend.sma_indicator(df["Close"], window=14)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    df["MACD"] = ta.trend.macd_diff(df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df_bollinger = ta.volatility.BollingerBands(df["Close"], window=20)
    df["upper_band"] = df_bollinger.bollinger_hband()
    df["middle_band"] = df_bollinger.bollinger_mavg()
    df["lower_band"] = df_bollinger.bollinger_lband()
    df["aroon_up"] = ta.trend.aroon_up(df["Close"], window=25)
    df["aroon_down"] = ta.trend.aroon_down(df["Close"], window=25)
    
    open_prices = df["Open"]
    close_prices = df["Close"]

    # Calculate the "Kicking" pattern feature using NumPy
    kicking_pattern = np.zeros_like(open_prices)

    # Loop through the data and check for "Kicking" pattern
    for i in range(1, len(open_prices)):
        if open_prices[i] < open_prices[i-1] and \
        open_prices[i] > close_prices[i-1] and \
        close_prices[i] > open_prices[i-1] and \
        close_prices[i] < close_prices[i-1] and \
        open_prices[i] - close_prices[i] > open_prices[i-1] - close_prices[i-1]:
            kicking_pattern[i] = 100  # Some positive value to indicate the pattern

    # Assign the kicking_pattern as a new column to the DataFrame
    df["kicking"] = kicking_pattern

    # Calculate ATR and SuperTrend
    def calculate_atr(high, low, close, window=14):
        true_ranges = np.maximum.reduce([high - low, np.abs(high - close.shift()), np.abs(low - close.shift())])
        atr = np.zeros_like(high)
        atr[window - 1] = np.mean(true_ranges[:window])
        for i in range(window, len(high)):
            atr[i] = (atr[i - 1] * (window - 1) + true_ranges[i]) / window
        return atr

    df["ATR"] = calculate_atr(df["High"], df["Low"], df["Close"], window=14)

    # Calculate Supertrend signals with reduced sensitivity and using all indicators
    df["upper_band_supertrend"] = df["High"] - (df["ATR"] * 2)
    df["lower_band_supertrend"] = df["Low"] + (df["ATR"] * 2)

    # Define conditions for uptrend and downtrend based on sensitivity to indicators
    uptrend_conditions = [
        (df["Close"] > df["lower_band_supertrend"]),
        (df["Close"] > df["SMA"]),
        (df["Close"] > df["middle_band"]),
        (df["Close"] > df["MACD"]),
        (df["RSI"] > 50),
        (df["aroon_up"] > df["aroon_down"]),
        (df["kicking"] == 1),  # Assuming "kicking" is an indicator where 1 indicates an uptrend.
        (df["Close"] > df["upper_band_supertrend"])
    ]

    downtrend_conditions = [
        (df["Close"] < df["upper_band_supertrend"]),
        (df["Close"] < df["SMA"]),
        (df["Close"] < df["middle_band"]),
        (df["Close"] < df["MACD"]),
        (df["RSI"] < 50),
        (df["aroon_up"] < df["aroon_down"]),
        (df["kicking"] == -1),  # Assuming "kicking" is an indicator where -1 indicates a downtrend.
        (df["Close"] < df["lower_band_supertrend"])
    ]

    # Set initial signal values to 0
    df["supertrend_signal"] = 0

    # Update signals based on sensitivity to indicators
    df.loc[np.any(uptrend_conditions, axis=0), "supertrend_signal"] = 1
    df.loc[np.any(downtrend_conditions, axis=0), "supertrend_signal"] = -1

    # Remove consecutive signals in the same direction (less sensitive)
    signal_changes = df["supertrend_signal"].diff().fillna(0)
    consecutive_mask = (signal_changes == 0) & (signal_changes.shift(-1) == 0)
    df.loc[consecutive_mask, "supertrend_signal"] = 0

    # Replace "False" with 0 and "True" with 1
    df = df.replace({False: 0, True: 1})

    # Fill missing values with 0
    df.fillna(0, inplace=True)

    # Concatenate the columns in the order you want
    df2 = pd.concat(
        [
            df["Date"],
            df["Close"],
            df["Adj Close"],
            df["Volume"],
            df["High"],
            df["Low"],
            df["SMA"],
            df["MACD"],
            df["upper_band"],
            df["middle_band"],
            df["lower_band"],
            df["supertrend_signal"],
            df["RSI"],
            df["aroon_up"],
            df["aroon_down"],
            df["kicking"],
            df["upper_band_supertrend"],
            df["lower_band_supertrend"],
        ],
        axis=1,
    )

    # Save the DataFrame to a new CSV file with indicators
    df2.to_csv("data.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(df["Close"], label="Close")
    ax1.plot(df["SMA"], label="SMA")
    ax1.fill_between(
        df.index, df["upper_band"], df["lower_band"], alpha=0.2, color="gray"
    )
    ax1.plot(df["upper_band"], linestyle="dashed", color="gray")
    ax1.plot(df["middle_band"], linestyle="dashed", color="gray")
    ax1.plot(df["lower_band"], linestyle="dashed", color="gray")
    ax1.scatter(
        df.index[df["supertrend_signal"] == 1],
        df["Close"][df["supertrend_signal"] == 1],
        marker="^",
        color="green",
        s=100,
    )
    ax1.scatter(
        df.index[df["supertrend_signal"] == -1],
        df["Close"][df["supertrend_signal"] == -1],
        marker="v",
        color="red",
        s=100,
    )
    ax1.legend()

    ax2.plot(df["RSI"], label="RSI")
    ax2.plot(df["aroon_up"], label="Aroon Up")
    ax2.plot(df["aroon_down"], label="Aroon Down")
    ax2.scatter(
        df.index[df["kicking"] == 100],
        df["High"][df["kicking"] == 100],
        marker="^",
        color="green",
        s=100,
    )
    ax2.scatter(
        df.index[df["kicking"] == -100],
        df["Low"][df["kicking"] == -100],
        marker="v",
        color="red",
        s=100,
    )
    ax2.legend()

    plt.xlim(df.index[0], df.index[-1])

    plt.show()

def train_model():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.metrics import mean_absolute_percentage_error
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    import sys

    print("TensorFlow version:", tf.__version__)

    cpugpu()

    # Define reward function
    def get_reward(y_true, y_pred):
        mape = mean_absolute_percentage_error(y_true, y_pred)
        reward = (1 - mape)
        return reward

    # Load data
    data = pd.read_csv("data.csv")

    # Split data into train and test sets
    train_data = data.iloc[:int(0.8*len(data))]
    test_data = data.iloc[int(0.8*len(data)):]

    # Normalize data
    scaler = MinMaxScaler()
    train_data_norm = scaler.fit_transform(train_data[[
                "Close",
                "Adj Close",
                "Volume",
                "High",
                "Low",
                "SMA",
                "MACD",
                "upper_band",
                "middle_band",
                "lower_band",
                "supertrend_signal",
                "RSI",
                "aroon_up",
                "aroon_down",
                "kicking",
                "upper_band_supertrend",
                "lower_band_supertrend",]])
    
    test_data_norm = scaler.transform(test_data[[
                "Close",
                "Adj Close",
                "Volume",
                "High",
                "Low",
                "SMA",
                "MACD",
                "upper_band",
                "middle_band",
                "lower_band",
                "supertrend_signal",
                "RSI",
                "aroon_up",
                "aroon_down",
                "kicking",
                "upper_band_supertrend",
                "lower_band_supertrend",]])

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
    model.add(LSTM(units=300, return_sequences=True, input_shape=(timesteps, X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=300, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=250, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=200, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=150))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.summary()

    # Compile model
    model.compile(optimizer='adam', loss='mse')

    # Define callbacks
    filepath="model.keras"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    epochs = 10
    batch_size = 50

    for i in range(epochs):
        if i == 1:
            model = load_model("model.keras")
        print("Epoch", i+1, "/", epochs)
        # Train the model for one epoch
        for a in range(0, len(X_train), batch_size):
            if a == 0:
                print(
                    "Batch", a+1, "/", len(X_train),
                    "(", ((a/len(X_train))*100), "% Done)"
                )
            else:
                sys.stdout.write('\033[F\033[K')
                print(
                    "Batch", a+1, "/", len(X_train),
                    "(", ((a/len(X_train))*100), "% Done)"
                )
            batch_X = X_train[a:a + batch_size]
            batch_y = y_train[a:a + batch_size]
            history = model.fit(
                batch_X, batch_y,
                batch_size=batch_size, epochs=1, verbose=0
            )

        # Evaluate the model on the test set
        y_pred_test = model.predict(X_test)
        sys.stdout.write('\033[F\033[K')
        test_reward = get_reward(y_test, y_pred_test)

        print("Test reward:", test_reward)

        if i == 0:
            best_reward1 = test_reward

        if test_reward >= best_reward1:
            best_reward1 = test_reward
            print("Model saved!")
            model.save("model.keras")

    if i == epochs - 1:
        model = load_model("model.keras")
        y_pred_test = model.predict(X_test)
        test_reward = get_reward(y_test, y_pred_test)
        test_loss = model.evaluate(X_test, y_test)

        print("Final test reward:", test_reward)
        print("Final test loss:", test_loss)

def evaluate_model():
    print("Evaluating the model...")
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    import pandas as pd
    import numpy as np
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import load_model
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
    import matplotlib.pyplot as plt

    print("TensorFlow version:", tf.__version__)

    cpugpu()

    # Load data
    data = pd.read_csv("data.csv")

    # Split data into train and test sets
    train_data = data.iloc[: int(0.8 * len(data))]
    test_data = data.iloc[int(0.8 * len(data)) :]

    # Normalize data
    scaler = MinMaxScaler()
    train_data_norm = scaler.fit_transform(
        train_data[
            [
                "Close",
                "Adj Close",
                "Volume",
                "High",
                "Low",
                "SMA",
                "MACD",
                "upper_band",
                "middle_band",
                "lower_band",
                "supertrend_signal",
                "RSI",
                "aroon_up",
                "aroon_down",
                "kicking",
                "upper_band_supertrend",
                "lower_band_supertrend",
            ]
        ]
    )
    test_data_norm = scaler.transform(
        test_data[
            [
                "Close",
                "Adj Close",
                "Volume",
                "High",
                "Low",
                "SMA",
                "MACD",
                "upper_band",
                "middle_band",
                "lower_band",
                "supertrend_signal",
                "RSI",
                "aroon_up",
                "aroon_down",
                "kicking",
                "upper_band_supertrend",
                "lower_band_supertrend",
            ]
        ]
    )

    # Define time steps
    timesteps = 100

    def create_sequences(data, timesteps):
        X = []
        y = []
        for i in range(timesteps, len(data)):
            X.append(data[i - timesteps : i])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    # Load model
    model = load_model("model.keras")

    # Evaluate model
    rmse_scores = []
    mape_scores = []
    rewards = []
    model = load_model("model.keras")
    X_test, y_test = create_sequences(test_data_norm, timesteps)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse_scores.append(rmse)
    mape_scores.append(mape)
    rewards.append(1 - mape)

    # Print results
    print(f"Mean RMSE: {np.mean(rmse_scores)}")
    print(f"Mean MAPE: {np.mean(mape_scores)}")
    print(f"Total Reward: {sum(rewards)}")


def fine_tune_model():
    print("Finetuning the model...")
    import os
    import signal

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    import sys
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import load_model
    from sklearn.metrics import (
        mean_squared_error,
        r2_score,
        mean_absolute_percentage_error,
    )
    import matplotlib.pyplot as plt
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

    print("TensorFlow version:", tf.__version__)

    cpugpu()

    # Define reward function
    def get_reward(y_true, y_pred):
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        reward = ((1 - mape) + r2) / 2
        return reward

    # Load data
    data = pd.read_csv("data.csv")

    # Split data into train and test sets
    train_data = data.iloc[: int(0.8 * len(data))]
    test_data = data.iloc[int(0.8 * len(data)) :]

    # Normalize data
    scaler = MinMaxScaler()
    train_data_norm = scaler.fit_transform(
        train_data[
            [
                "Close",
                "Adj Close",
                "Volume",
                "High",
                "Low",
                "SMA",
                "MACD",
                "upper_band",
                "middle_band",
                "lower_band",
                "supertrend_signal",
                "RSI",
                "aroon_up",
                "aroon_down",
                "kicking",
                "upper_band_supertrend",
                "lower_band_supertrend",
            ]
        ]
    )
    test_data_norm = scaler.transform(
        test_data[
            [
                "Close",
                "Adj Close",
                "Volume",
                "High",
                "Low",
                "SMA",
                "MACD",
                "upper_band",
                "middle_band",
                "lower_band",
                "supertrend_signal",
                "RSI",
                "aroon_up",
                "aroon_down",
                "kicking",
                "upper_band_supertrend",
                "lower_band_supertrend",
            ]
        ]
    )

    # Define time steps
    timesteps = 100

    # Create sequences of timesteps
    def create_sequences(data, timesteps):
        X = []
        y = []
        for i in range(timesteps, len(data)):
            X.append(data[i - timesteps : i])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_data_norm, timesteps)
    X_test, y_test = create_sequences(test_data_norm, timesteps)

    # Define reward threshold
    reward_threshold = float(
        input("Enter the reward threshold (0 - 1, 0.9 recommended): ")
    )

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
        user_input = input(
            f"Are you sure that you want to End the Program? (yes/no): "
        )

        if user_input.lower() == "yes":
            exit(0)

        else:
            print("Continuing the Fine-tuning Process")

    # Register the signal handler
    signal.signal(signal.SIGINT, handle_interrupt)

    while True:
        # Load model
        model = load_model("model.keras")
        print("\nEvaluating Model")
        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Append rewards
        reward = get_reward(y_test, y_pred)
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
            model.save("model.keras")

            break
        else:
            print("Training Model with 5 Epochs")
            epochs = 5
            batch_size = 50
            for i in range(epochs):
                print("Epoch", i, "/", epochs)
                # Train the model for one epoch
                for a in range(0, len(X_train), batch_size):
                    if a == 0:
                        print("Batch", a, "/", len(X_train), "(", ((a/len(X_train))*100), "% Done)")
                    else:
                        sys.stdout.write('\033[F\033[K')
                        print("Batch", a, "/", len(X_train), "(", ((a/len(X_train))*100), "% Done)")
                    batch_X = X_train[a:a + batch_size]
                    batch_y = y_train[a:a + batch_size]
                    history = model.fit(batch_X, batch_y, batch_size=batch_size, epochs=1, verbose=0)

                # Evaluate the model on the test set
                y_pred_test = model.predict(X_test)
                sys.stdout.write('\033[F\033[K')
                test_reward = get_reward(y_test, y_pred_test)

                print("Test reward:", test_reward)

                if i == 0 and count == 1:
                    best_reward1 = test_reward

                if test_reward >= best_reward1:
                    print("Model saved!")
                    model_saved = 1
                    best_reward1 = test_reward
                    model.save("model.keras")
                
                if test_reward >= reward_threshold:
                    print("Model reached reward threshold", test_reward, ". Saving and stopping epochs!")
                    model_saved = 1
                    model.save("model.keras")
                    break


def predict_future_data():
    print("Utilizing the model for predicting future data (30 days)...")
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import load_model
    import matplotlib.pyplot as plt

    cpugpu()

    # Load data
    data = pd.read_csv("data.csv")

    # Normalize data
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(
        data[
            [
                "Close",
                "Adj Close",
                "Volume",
                "High",
                "Low",
                "SMA",
                "MACD",
                "upper_band",
                "middle_band",
                "lower_band",
                "supertrend_signal",
                "RSI",
                "aroon_up",
                "aroon_down",
                "kicking",
                "upper_band_supertrend",
                "lower_band_supertrend",
            ]
        ]
    )

    # Define time steps
    timesteps = 100

    # Create sequences of timesteps
    def create_sequences(data, timesteps):
        X = []
        for i in range(timesteps, len(data)):
            X.append(data[i - timesteps : i])
        return np.array(X)

    X_data = create_sequences(data_norm, timesteps)

    # Load model
    model = load_model("model.keras")
    model.summary()

    num_predictions = 30

    # Make predictions for next num_predictions days
    X_pred = X_data[-num_predictions:].reshape(
        (num_predictions, timesteps, X_data.shape[2])
    )
    y_pred = model.predict(X_pred)[:, 0]

    # Inverse transform predictions
    y_pred = scaler.inverse_transform(
        np.hstack(
            [
                np.zeros((len(y_pred), data_norm.shape[1] - 1)),
                np.array(y_pred).reshape(-1, 1),
            ]
        )
    )[:, -1]

    # Generate date index for predictions
    last_date = data["Date"].iloc[-1]
    index = pd.date_range(
        last_date, periods=num_predictions, freq="D", tz="UTC"
    ).tz_localize(None)

    # Calculate % change
    y_pred_pct_change = (y_pred - y_pred[0]) / y_pred[0] * 100

    # Save predictions and % change in a CSV file
    predictions = pd.DataFrame(
        {"Date": index, "Predicted Close": y_pred, "% Change": y_pred_pct_change}
    )
    predictions.to_csv("predictions.csv", index=False)

    print(predictions)

    # Find the rows with the lowest and highest predicted close and the highest and lowest % change
    min_close_row = predictions.iloc[predictions["Predicted Close"].idxmin()]
    max_close_row = predictions.iloc[predictions["Predicted Close"].idxmax()]
    max_pct_change_row = predictions.iloc[predictions["% Change"].idxmax()]
    min_pct_change_row = predictions.iloc[predictions["% Change"].idxmin()]

    # Print the rows with the lowest and highest predicted close and the highest and lowest % change
    print(f"\n\nHighest predicted close:\n{max_close_row}\n")
    print(f"Lowest predicted close:\n{min_close_row}\n")
    print(f"Highest % change:\n{max_pct_change_row}\n")
    print(f"Lowest % change:\n{min_pct_change_row}")

    # Plot historical data and predictions
    plt.plot(data["Close"].values, label="Actual Data")
    plt.plot(
        np.arange(len(data), len(data) + num_predictions),
        y_pred,
        label="Predicted Data",
    )

    # Add red and green arrows for highest and lowest predicted close respectively, and highest and lowest percentage change
    plt.annotate(
        "↓",
        xy=(min_close_row.name - len(data), min_close_row["Predicted Close"]),
        color="red",
        fontsize=16,
        arrowprops=dict(facecolor="red", shrink=0.05),
    )
    plt.annotate(
        "↑",
        xy=(max_close_row.name - len(data), max_close_row["Predicted Close"]),
        color="green",
        fontsize=16,
        arrowprops=dict(facecolor="green", shrink=0.05),
    )
    plt.annotate(
        "↑",
        xy=(max_pct_change_row.name - len(data), y_pred.max()),
        color="green",
        fontsize=16,
        arrowprops=dict(facecolor="green", shrink=0.05),
    )
    plt.annotate(
        "↓",
        xy=(min_pct_change_row.name - len(data), y_pred.min()),
        color="red",
        fontsize=16,
        arrowprops=dict(facecolor="red", shrink=0.05),
    )

    # Add legend and title
    plt.legend()
    plt.title("Predicted Close Prices")

    # Show plot
    plt.show()


def compare_predictions():
    print("Comparing the predictions with the actual data...")
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    # Get a list of CSV files in the "data" folder
    data_folder = "data"
    csv_files = [file for file in os.listdir(data_folder) if file.endswith(".csv")]

    # Display the list of CSV files to the user
    print("Available CSV files:")
    for i, file in enumerate(csv_files):
        print(f"{i + 1}. {file}")

    # Ask the user to select a CSV file
    selected_file = None
    while selected_file is None:
        try:
            file_number = int(
                input(
                    "Enter the number corresponding to the CSV file you want to select: "
                )
            )
            if file_number < 1 or file_number > len(csv_files):
                raise ValueError()
            selected_file = csv_files[file_number - 1]
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    # Load predicted and actual data
    predicted_data = pd.read_csv("predictions.csv")
    actual_data = pd.read_csv(os.path.join(data_folder, selected_file))

    # Rename columns for clarity
    predicted_data = predicted_data.rename(columns={"Predicted Close": "Close"})
    actual_data = actual_data.rename(columns={"Close": "Actual Close"})

    # Join predicted and actual data on the date column
    combined_data = pd.merge(predicted_data, actual_data, on="Date")

    # Calculate the absolute percentage error between the predicted and actual values
    combined_data["Absolute % Error"] = (
        abs(combined_data["Close"] - combined_data["Actual Close"])
        / combined_data["Actual Close"]
        * 100
    )

    # Calculate the mean absolute percentage error and print it
    mape = combined_data["Absolute % Error"].mean()
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")

    # Find the row with the highest and lowest absolute percentage error and print them
    min_error_row = combined_data.iloc[combined_data["Absolute % Error"].idxmin()]
    max_error_row = combined_data.iloc[combined_data["Absolute % Error"].idxmax()]
    print(f"\nMost Accurate Prediction:\n{min_error_row}\n")
    print(f"Least Accurate Prediction:\n{max_error_row}\n")

    # Plot the predicted and actual close prices
    plt.plot(combined_data["Date"], combined_data["Close"], label="Predicted Close")
    plt.plot(combined_data["Date"], combined_data["Actual Close"], label="Actual Close")

    # Add title and legend
    plt.title("Predicted vs Actual Close Prices")
    plt.legend()

    # Show plot
    plt.show()


def gen_stock():
    print("Generating Stock Data...")
    import csv
    import random
    import datetime

    def generate_stock_data_csv(file_path, num_lines, data_type):
        # Define the column names
        columns = ["Date", "Close", "Adj Close", "Volume", "High", "Low", "Open"]

        # Generate stock data based on the selected data type
        data = []
        start_date = datetime.datetime(2022, 1, 1)
        for i in range(num_lines):
            if data_type == "linear":
                close_price = 100.0 + i * 10
            elif data_type == "exponential":
                close_price = 100.0 * (1.1**i)
            elif data_type == "random":
                if i > 0:
                    prev_close = data[i - 1][1]
                    close_price = prev_close * random.uniform(0.95, 1.05)
                else:
                    close_price = 100.0
            elif data_type == "trend":
                if i > 0:
                    prev_close = data[i - 1][1]
                    close_price = prev_close + random.uniform(-2, 2)
                else:
                    close_price = 100.0
            else:
                raise ValueError("Invalid data type provided.")

            date = start_date + datetime.timedelta(days=i)
            adj_close = close_price
            volume = random.randint(100000, 1000000)
            high = close_price * random.uniform(1.01, 1.05)
            low = close_price * random.uniform(0.95, 0.99)
            open_price = close_price * random.uniform(0.98, 1.02)

            data.append(
                [
                    date.strftime("%Y-%m-%d"),
                    close_price,
                    adj_close,
                    volume,
                    high,
                    low,
                    open_price,
                ]
            )

        # Save the generated data to a CSV file
        with open(file_path, "w", newline="") as file:
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

def update():
    import os

    def create_update_script():
        script_content = """
import os
import subprocess

def print_red(text):
    print(f"\033[91m{text}\033[0m")

def print_yellow(text):
    print(f"\033[93m{text}\033[0m")

def print_green(text):
    print(f"\033[92m{text}\033[0m")

# Path to the local commit_sha.sha file
local_sha_path = "commit_sha.sha"

# Path to the online commit_sha.sha file (replace with your online file path)
online_sha_path = "https://raw.githubusercontent.com/Qerim-iseni09/Stock-Predictor-V4/main/commit_sha.sha"
downloaded_sha = "commit_sha_online.sha"

def get_online_sha():
    # Retrieve the online commit SHA file
    subprocess.run(["wget", "-q", "-O", downloaded_sha, online_sha_path])
    with open(downloaded_sha, "r") as file:
        lines = file.readlines()
        return lines[0].strip(), lines[1].strip()
    subprocess.run(["rm", "-rf", downloaded_sha])

def check_for_updates():
    local_sha = ""
    online_sha = ""
    major_update = False

    if os.path.exists(local_sha_path):
        with open(local_sha_path, "r") as file:
            local_sha = file.readline().strip()

    try:
        online_sha, update_type = get_online_sha()
    except Exception as e:
        print_red("Failed to retrieve the online commit SHA.")
        print_red(str(e))
        return

    if local_sha != online_sha:
        if update_type == "(Major Update)":
            major_update = True
            print_red("Major update available!")
            print_red("Please consider updating to the latest version.")

        print_yellow("Do you want to update your repository?")
        confirmation = input("(yes/no): ")

        if confirmation.lower() == "yes":
            try:
                subprocess.run(["git", "pull"])
                print_green("Repository updated successfully!")

                if major_update:
                    print_red("This was a major update. Please review the changelog.")

                # Delete the script after successful update
                os.remove(__file__)
            except Exception as e:
                print_red("Failed to update the repository.")
                print_red(str(e))
        else:
            print_yellow("Skipping repository update. Please update manually.")
    else:
        print_green("No updates available. Repository is up to date.")

check_for_updates()
os.remove(__file__)
    """

        with open("update.py", "w") as file:
            file.write(script_content)

    def run_update_script():
        os.system("python update.py")

    create_update_script()
    run_update_script()

def do_all_actions():
    prepare_data()
    train_model()
    evaluate_model()
    fine_tune_model()
    evaluate_model()
    predict_future_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPV4 Script")
    parser.add_argument(
        "--update", action="store_true", help="Check updates for SPV4"
    )
    parser.add_argument(
        "--install", action="store_true", help="Install all dependencies for SPV4"
    )
    parser.add_argument(
        "--generate_stock", action="store_true", help="Generate Stock Data"
    )
    parser.add_argument(
        "--prepare_data",
        action="store_true",
        help="Preprocess and Prepare the CSV Data",
    )
    parser.add_argument("--train", action="store_true", help="Train the SPV4 Model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the Model")
    parser.add_argument("--fine_tune", action="store_true", help="Finetune the Model")
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Utilize the Model for Predicting Future Data (30 Days)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare the Predictions with the Actual Data",
    )
    parser.add_argument(
        "--do-all",
        action="store_true",
        help="Do all actions from above (No Install & Generating Stock Data)",
    )

    args = parser.parse_args()

    if args.do_all:
        do_all_actions()
    else:
        if args.install:
            install_dependencies()
        if args.update:
            update()
        if args.generate_stock:
            gen_stock()
        if args.prepare_data:
            prepare_data()
        if args.train:
            train_model()
        if args.eval:
            evaluate_model()
        if args.fine_tune:
            fine_tune_model()
        if args.predict:
            predict_future_data()
        if args.compare:
            compare_predictions()