from pyexpat import model
from joblib import PrintTime
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

#Set random seed
np.random.seed(42)

#Download Stock Data
def download_stock_data(stock_symbol, start_date, end_date):
    #Downsloads data from Yahoo Finance
    print(f"Downloading stock data for {stock_symbol} ...")
    stock_data = yf.download(stock_symbol, start = start_date, end = end_date)
    print("Download Complete!")
    return stock_data



#Exploratory Data Analysis
def perform_eda(stock_data):
    #Performs basic exploratory data analysis on stock data
    print("Performing EDA...")
    print(stock_data.head())
    print(stock_data.describe())


    #plot closing price over time
    stock_data['Close'].plot(title="SPY Closing Price Over time", figsize=(10,6))
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.show()

    #heatmap of corrleations
    plt.figure(figsize=(8,6))
    sns.heatmap(stock_data.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Corrleations")
    plt.show()


    #Feature Engineering
def add_features(stock_data):
    #adds technical indicators to stock data: Daily Return, Volatility
    print("Adding Features...")
    stock_data['Daily_Return'] = stock_data['Close'].pct_change()
    stock_data['Volatility'] = (stock_data['High'] - stock_data['Low']) /stock_data['Close']
    stock_data.fillna(method='ffill', inplace=True)
    return stock_data



#Preprocessing Data
def preprocessing_data(stock_data, sequence_length=50):
    #Preps the stock data for LSTM Modeling
    print("Processing Data...")
    scaler = MinMaxScaler()
    stock_data[['Close']] = scaler.fit_transform(stock_data[["Close"]])


    def create_sequences(data, seq_length):
        x,y = [], []
        for i in range(len(data) - seq_length):
            x.append(data[i : i + seq_length])
            y.append(data[i + seq_length])
        return np.array(x), np.array(y)

    x, y = create_sequences(stock_data['Close'].values, sequence_length)

    train_size = int(len(x) * .8)
    x_train, y_train = x[:train_size], y[:train_size]
    x_test, y_test = x[train_size:], y[train_size:]

    return x_train, y_train, x_test, y_test, scaler



#Build and Train Model
def buildAndTrainModel(x_train, y_train, x_test, y_test, sequence_length, lstm_units, learning_rate, epochs, batch_size):
    #Builds trains and returns an LSTM Model

    print("Building the LSTM Model")
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=(sequence_length, 1)),
        LSTM(lstm_units),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer='adam', loss="mse")
    print(model.summary())


    print("Training the Model...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size
    )

    plt.plot(history.history['loss'],label= 'Training Loss')
    plt.plot(history.history["val_loss"], label= "Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    return model


#Evaluate the model
def evaluate_model(model, x_test, y_test, scaler):
    #Evaluates the trained LSTM Model
    print("Evaluating the Model")
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1,1))
    y_test = scaler.inverse_transform(y_test.reshape(-1,1))

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mape = np.mean(np.abs((y_test - predictions) / y_test))


    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}")

    plt.figure(figsize=(10,6))
    plt.plot(y_test, label="Predicted Prices", color='red')
    plt.title("Actual vs Predicted SPY Prices")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


#Save the Model
def save_model(model, file_path):
    #Saves the trained LSTM model to a files
    print(f"Saving model to {file_path}")
    model.save(file_path)
    print("Model saved successfully")



if __name__ == "__main__":
    #Parameters
    STOCK_SYMBOL = "SPY"
    START_DATE = "2015-01-01"
    END_DATE = "2024-12-09"
    SEQUENCE_LENGTH = 50
    LSTM_UNITS = 64
    LEARNING_RATE = 0.001
    EPOCHS = 20
    BATCH_SIZE = 32

    #Step 1
    stock_data = download_stock_data(STOCK_SYMBOL, START_DATE, END_DATE)

    #Step 2
    perform_eda(stock_data)

    #Step 3
    stock_data = add_features(stock_data)

    #Step 4
    x_train, y_train, x_test, y_test, scaler = preprocessing_data(stock_data, SEQUENCE_LENGTH)

    #Step 5
    model = buildAndTrainModel(
        x_train, y_train, x_test, y_test,
        sequence_length=SEQUENCE_LENGTH,
        lstm_units=LSTM_UNITS,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    #Step 6
    evaluate_model(model, x_test, y_test, scaler)

    #Step 7
    save_model(model, "models/SPY_Stock_Prediction_Model.h5")
