import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def create_lstm_model(seq_length, forecast_horizon):
    """Create LSTM model architecture"""
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(seq_length, 1)))
    model.add(Dense(forecast_horizon))
    model.compile(optimizer='adam', loss='mse')
    return model

def create_sequences(data, seq_length):
    """Create sequences for LSTM training"""
    x = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
    return np.array(x)

def create_train_data(data, seq_length, forecast_horizon):
    """Create training data with sequences and targets"""
    x, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+forecast_horizon].flatten())
    return np.array(x), np.array(y)