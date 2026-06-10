import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data = pd.read_csv("stock_data.csv")

# Ensure dataset has enough rows
if len(data) < 100:
    raise ValueError("Dataset too small. Please use at least 100 rows of stock data.")

# Use only 'Close' column
close_prices = data['Close'].values.reshape(-1, 1)

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Create sequences (60-day lookback)
X, y = [], []
lookback = 60
for i in range(lookback, len(scaled_data)):
    X.append(scaled_data[i-lookback:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# Reshape input for LSTM: (samples, timesteps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

# Compile and train
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=25, batch_size=32, verbose=1)

# Predict
predicted = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted)

print("✅ Model trained successfully. Example predictions:")
print(predicted_prices[:5])
