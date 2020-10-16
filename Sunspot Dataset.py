# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 20:30:37 2020

@author: Shivansh
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
import numpy as np

df = pd.read_csv(r"Datasets\Sunspots.csv")

sunspots = df['Monthly Mean Total Sunspot Number'].to_numpy()
timestamp = df['Unnamed: 0'].to_numpy()



plt.plot(timestamp, sunspots)
plt.xlabel("Timestep")
plt.ylabel("Sunspots")
plt.show()


splitTime = 3000
time_train = timestamp[:splitTime]
x_train = sunspots[:splitTime]
time_valid = timestamp[splitTime:]
x_valid = sunspots[splitTime:]

window_size = 30
batch_size = 32
shuffle_buffer = 1000


def windowedDataset(series , windowSize , batchSize , shuffleSize):
    series = tf.expand_dims(series ,  axis = -1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window( size = windowSize+1 , shift = 1 , drop_remainder = True)
    dataset = dataset.flat_map( lambda w : w.batch( windowSize + 1 ) )
    dataset = dataset.shuffle( shuffleSize )
    dataset = dataset.map( lambda w : (w[:-1] , w[1:]))   
    dataset = dataset.batch(batchSize).prefetch(1)     
    return dataset

def modelForecast( model , series , windowSize ):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window( windowSize , shift = 1 , drop_remainder = True )
    dataset = dataset.flat_map( lambda x : x.batch(windowSize))
    dataset = dataset.batch(32).prefetch(1)
    
    forecast = model.predict(dataset)
    
    return forecast


tf.keras.backend.clear_session()


window_size = 64
batch_size = 256
train_set = windowedDataset(x_train, window_size, batch_size, shuffle_buffer)
print(train_set)
print(x_train.shape)

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.Dense(30, activation="relu"),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 400)
      ])

model.compile( loss = tf.keras.losses.Huber() , optimizer = tf.keras.optimizers.SGD(lr=1e-5) , metrics = ['mae'] )
history = model.fit(train_set, epochs = 500)  

forecasts = modelForecast( model , sunspots[...,np.newaxis], window_size)
forecasts = forecasts[splitTime - window_size:-1, -1, 0]


plt.plot(time_valid,x_valid)
plt.plot(time_valid,forecasts)

tf.keras.metrics.mean_absolute_error(x_valid, forecasts)



























