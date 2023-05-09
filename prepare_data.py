import pandas as pd
import talib
import matplotlib.pyplot as plt

df = pd.read_csv('data/BTC-USD.csv')

df['SMA'] = talib.SMA(df['Close'], timeperiod=14)
df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
df['MACD'], _, _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(df['Close'], timeperiod=20)
df['aroon_up'], df['aroon_down'] = talib.AROON(df['High'], df['Low'], timeperiod=25)
df['kicking'] = talib.CDLKICKINGBYLENGTH(df['Open'], df['High'], df['Low'], df['Close'])

df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
df['upper_band_supertrend'] = df['High'] - (df['ATR'] * 2)
df['lower_band_supertrend'] = df['Low'] + (df['ATR'] * 2)
df['in_uptrend'] = df['Close'] > df['lower_band_supertrend']
df['supertrend_signal'] = df['in_uptrend'].diff().fillna(0)

df['EMA'] = talib.EMA(df['Close'], timeperiod=14)
df['STOCH_k'], df['STOCH_d'] = talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=14, slowk_period=3, slowd_period=3)
df['obv'] = talib.OBV(df['Close'], df['Volume'])

# Replace "False" with 0 and "True" with 1
df = df.replace({False: 0, True: 1})

# Calculate percent change and money change
df['pct_change'] = df['Close'].pct_change()
df['money_change'] = df['Close'] - df['Close'].shift(1)

# Fill missing values with 0
df.fillna(0, inplace=True)

# Save the DataFrame to a new CSV file with indicators
df.to_csv('data.csv', index=False)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.plot(df['Close'], label='Close')
ax1.plot(df['SMA'], label='SMA')
ax1.plot(df['EMA'], label='EMA')
ax1.fill_between(df.index, df['upper_band'], df['lower_band'], alpha=0.2, color='gray')
ax1.plot(df['upper_band'], linestyle='dashed', color='gray')
ax1.plot(df['middle_band'], linestyle='dashed', color='gray')
ax1.plot(df['lower_band'], linestyle='dashed', color='gray')
ax1.scatter(df.index[df['supertrend_signal'] == 1], df['Close'][df['supertrend_signal'] == 1], marker='^', color='green', s=100)
ax1.scatter(df.index[df['supertrend_signal'] == -1], df['Close'][df['supertrend_signal'] == -1], marker='v', color='red', s=100)
ax1.legend()

ax2.plot(df['RSI'], label='RSI')
ax2.plot(df['aroon_up'], label='Aroon Up')
ax2.plot(df['aroon_down'], label='Aroon Down')
ax2.scatter(df.index[df['kicking'] == 100], df['High'][df['kicking'] == 100], marker='^', color='green', s=100)
ax2.scatter(df.index[df['kicking'] == -100], df['Low'][df['kicking'] == -100], marker='v', color='red', s=100)
ax2.legend()

plt.xlim(df.index[0], df.index[-1])

plt.show()