import yfinance as yf
import keras.backend as K
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM, Dropout, BatchNormalization, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.models import Sequential
from keras.optimizer_v2.adam import Adam
from keras.optimizer_v2.rmsprop import RMSprop
import talib
import plotly.express as px
import plotly.graph_objects as go


def build_model():
    data = yf.download('0005.hk', "2017-09-28", "2021-09-24")
    data['H-L'] = data['High'] - data['Low']
    data['O-C'] = data['Close'] - data['Open']
    # data=pd.merge(data,data["Close"].pct_change(),left_index=True,right_index=True)
    data["% Change"] = data["Close"].shift(1).pct_change()
    data['3day MA'] = data['Close'].shift(1).rolling(window=3).mean()
    data['10day MA'] = data['Close'].shift(1).rolling(window=10).mean()
    data['30day MA'] = data['Close'].shift(1).rolling(window=30).mean()
    data['Std_dev'] = data['Close'].shift(1).rolling(5).std()
    data.dropna(inplace=True)
    print(data.isnull().sum())
    data_price = data["Close"]
    train = data.shift(1).dropna().values
    sc = MinMaxScaler(feature_range=(0, 1))
    train = sc.fit_transform(train)


    def processData(data, data_price, lb):
        X, Y = [], []
        for i in range(lb-1, len(data)-lb-1):
            X.append(data[i-(lb-1):i])
            Y.append(data_price[(i)])
        return np.array(X), np.array(Y)


    lb = 7
    X, y = processData(train, data_price, lb)
    X_train, X_test = X[:int(X.shape[0]*0.90)], X[int(X.shape[0]*0.90):]
    y_train, y_test = y[:int(y.shape[0]*0.90)].reshape(-1,
                                                    1), y[int(y.shape[0]*0.90):].reshape(-1, 1)
    # print(type(X_train))
    print(X[0])
    print(y[0])
    print(data.shape)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(
        X_train.shape[1], X_train.shape[2])))
    # model.add(Dropout(0.7))
    model.add(LSTM(64, return_sequences=False))
    # model.add(Dropout(0.5))
    model.add(Dense(32))
    model.add(Dense(1))
    model.compile(
        loss="mean_squared_error",
        optimizer='Adam'
    )
    history = model.fit(X_train,y_train,epochs=10,batch_size=20,validation_data=(X_test,y_test),shuffle=False)
    y_pred = model.predict(X_test)
    fig=go.Figure()
    fig.add_trace(go.Scatter(y=history.history['loss'],mode="lines",name="train"))
    fig.add_trace(go.Scatter(y=history.history['val_loss'],mode="lines",name="test"))
    return fig
    print(model.summary())
    print(y_pred)
