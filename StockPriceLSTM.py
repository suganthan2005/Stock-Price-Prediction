import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

st.set_page_config(page_title="Stock Predictor", layout="wide")

st.sidebar.title("Stock Price Prediction")
page = st.sidebar.radio("Go to", ["Stock Data", "LSTM Model Training & Prediction"])

# Function to fetch stock data
def get_stock_data(ticker, period="10y"):
    data = yf.download(ticker, period=period, interval="1d")
    if data is None or data.empty:
        st.error(f"Insufficient data for {ticker}. Try another stock or a shorter period.")
        return None
    return data

# Function to create sequences
def create_sequences(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i])
        y.append(data[i])
    return np.array(X), np.array(y)

if page == "Stock Data":
    st.title("Real-Time Stock Data Visualization")
    
    ticker = st.text_input("Enter Stock Ticker (e.g., TSLA)", "TSLA").upper()
    
    if st.button("Fetch Data"):
        with st.spinner("Fetching stock data..."):
            data = get_stock_data(ticker)
            if data is not None:
                st.success(f"Data fetched for {ticker}")
                st.line_chart(data["Close"])
                st.write(data.tail())

elif page == "LSTM Model Training & Prediction":
    st.title("Train & Predict Stock Prices using LSTM")
    
    ticker = st.text_input("Enter Stock Ticker", "TSLA").upper()
    look_back = st.slider("Look-back period (days)", 30, 3650, 60)
    future_days = st.slider("Days to predict", 1, 30, 5)
    
    if st.button("Train LSTM Model & Predict"):
        with st.spinner("Training model..."):
            data = get_stock_data(ticker, period="10y")
            if data is None:
                st.stop()
            
            close_prices = data[['Close']]
            dates = data.index[-len(close_prices):]
            
            # Adjust look_back to ensure enough data
            max_lookback = min(len(data) - 1, 3650)
            look_back = min(look_back, max_lookback)
            
            # Normalize Data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(close_prices)
            
            # Create sequences
            X, y = create_sequences(scaled_data, look_back)
            
            # Check if dataset is too small
            if len(X) < 2:
                st.error("Not enough data for training. Reduce the lookback period.")
                st.stop()
            
            # Dynamically adjust test size
            test_size = min(0.2, max(1 / len(X), 0.1))  

            # Split Data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

            # Reshape input for LSTM (samples, time steps, features)
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            # LSTM Model
            model = keras.Sequential([
                keras.layers.LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)),  # Reduced units
                keras.layers.Dropout(0.2),
                keras.layers.LSTM(50, return_sequences=True),  # Reduced units
                keras.layers.Dropout(0.2),
                keras.layers.LSTM(25, return_sequences=False),  # Reduced units
                keras.layers.Dense(25, activation='relu'),
                keras.layers.Dense(1)
            ])

            # Compile model
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

            # Callbacks for early stopping and learning rate reduction
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)  # Reduced patience
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)  # Reduced patience

            # Train Model
            history = model.fit(
                X_train, y_train,
                epochs=30,  # Reduced epochs
                batch_size=16,  # Smaller batch size
                validation_data=(X_test, y_test),
                verbose=1,
                callbacks=[early_stopping, reduce_lr]
            )

            # Predictions
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            st.success(f"LSTM Model trained! MSE: {mse:.5f}")

            # Future Prediction
            last_known_data = scaled_data[-look_back:].reshape(1, look_back, 1)
            future_predictions = []
            future_dates = [dates[-1] + datetime.timedelta(days=i+1) for i in range(future_days)]
            
            for _ in range(future_days):
                pred = model.predict(last_known_data)
                future_predictions.append(pred[0, 0])
                last_known_data = np.roll(last_known_data, -1)
                last_known_data[0, -1, 0] = pred[0, 0]
            
            future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
            
            # Plot Training Loss
            fig_loss, ax_loss = plt.subplots(figsize=(8, 4))
            ax_loss.plot(history.history['loss'], label='Train Loss')
            ax_loss.plot(history.history['val_loss'], label='Validation Loss')
            ax_loss.set_title("Training Loss Curve")
            ax_loss.legend()
            st.pyplot(fig_loss)

            # Plot predictions
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(dates[-len(y_test):], y_test, label="Actual Price")
            ax.plot(dates[-len(y_test):], predictions, label="Predicted Price")
            ax.set_title("Stock Price Prediction")
            ax.legend()
            st.pyplot(fig)

            # Future Predictions Plot
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.plot(future_dates, future_predictions, marker='o', linestyle='dashed', alpha=0.7, label="Future Prediction")
            ax2.set_title(f"Next {future_days} Days Prediction")
            ax2.legend()
            st.pyplot(fig2)

            # Show predictions
            future_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_predictions.flatten()})
            st.write(future_df)
