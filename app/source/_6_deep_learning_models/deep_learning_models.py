import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten, Conv1D, Lambda
from tensorflow.keras.optimizers import Adam


def get_model(name, trainX, trainY):
    
    if name == "CNN_LSTM":
        model, loss, val_loss = getLSTM_CNN(trainX, trainY)
        
    elif name == "CNN":
        model, loss, val_loss = getCNN(trainX, trainY)
        
    elif name == "LSTM":
        model, loss, val_loss = getLSTM(trainX, trainY)
        
    else:
        model, loss, val_loss = getMLP(trainX, trainY)
        
    return model, loss, val_loss
        

def getLSTM_CNN(trainX, trainY):
        
    model = Sequential([
        
        Conv1D(filters=60, kernel_size=5, strides=1, padding="causal", activation="relu"),
        
        LSTM(48, return_sequences=True),
        LSTM(48, return_sequences=False),
        
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        
        Dense(trainY.shape[1], kernel_initializer=tf.initializers.zeros),
        Lambda(lambda x: x * 400)
        
    ])
    
    model.compile(loss="mae", optimizer="adam")
    
    history = model.fit(trainX, trainY, epochs=15, batch_size=32, validation_split=0.2)
    
    return model, np.array(history.history["loss"][-3:]).mean(), np.array(history.history["val_loss"][-3:]).mean()


def getCNN(trainX, trainY):
    
    model = Sequential([
        
        Conv1D(filters=64, kernel_size=3, strides=1, padding="causal", activation="relu"),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(trainY.shape[1], kernel_initializer=tf.initializers.zeros),
        
    ])
    
    model.compile(optimizer='adam', loss='mae')

    history = model.fit(trainX, trainY, epochs=25, batch_size=32, validation_split=0.2)
    
    return model, np.array(history.history["loss"][-3:]).mean(), np.array(history.history["val_loss"][-3:]).mean()
    
    
def getMLP(trainX, trainY):
    
    model = Sequential([
        
        Flatten(),
        Dense(30, activation='relu'),
        Dense(30, activation='relu'),
        Dense(trainY.shape[1], kernel_initializer=tf.initializers.zeros),
        
    ])

    model.compile(optimizer='adam', loss='mae')

    history = model.fit(trainX, trainY, epochs=50, batch_size=32, validation_split=0.2)
    
    return model, np.array(history.history["loss"][-3:]).mean(), np.array(history.history["val_loss"][-3:]).mean()


def getLSTM(trainX, trainY):

    model = Sequential([
        
        LSTM(48, activation='relu', return_sequences=True),
        LSTM(48, activation='relu', return_sequences=False),
        Dense(32, activation='relu'),
        Dense(trainY.shape[1], kernel_initializer=tf.initializers.zeros),
    ])
    
    optimizer = Adam(learning_rate=0.003)
    model.compile(optimizer=optimizer, loss='mae')

    history = model.fit(trainX, trainY, epochs=30, batch_size=64, validation_split=0.2)
    
    return model, np.array(history.history["loss"][-3:]).mean(), np.array(history.history["val_loss"][-3:]).mean()

