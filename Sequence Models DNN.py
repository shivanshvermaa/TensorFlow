# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 12:47:05 2020

@author: Shivansh
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

def windowedDataset( series , windowSize , batchSize , shuffleSize ):
    
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window( size = windowSize + 1 , shift = 1 , drop_remainder = True )
    dataset = dataset.flat_map( lambda window : window.batch(batchSize+1))
    dataset = dataset.shuffle(shuffleSize).map( lambda window: ( window[:-1] , window[-1] ) )
    dataset = dataset.batch(batchSize).prefetch(1)
    
    return dataset


def plotGraph( xvalues , yvalues ,  format='-' , start = 0 , end = None ):
    plt.plot(xvalues[start:end] , yvalues[start:end] , format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    
def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)  
baseline = 10
amplitude = 20
slope = 0.09
noise_level = 5

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=42)

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

dataset = windowedDataset( x_train , window_size , batch_size , shuffle_buffer_size)

model = keras.models.Sequential([
        keras.layers.Dense( 10 , input_shape = [window_size] , activation ='relu' ),
        keras.layers.Dense( 10 , activation = 'relu'),
        keras.layers.Dense(1)
        ])
    
model.compile(loss = 'mse'  , optimizer = tf.keras.optimizers.SGD(lr = 1e-6 , momentum = 0.9))

model.fit( dataset, epochs = 100 )


forecast = []

for time in range( len(series) - window_size ):
    forecast.append( model.predict( series[time:time + window_size][np.newaxis]))
    
forecast = forecast[split_time-window_size:]
result = np.array(forecast)[:,0,0]

plotGraph(time_valid,x_valid)
plotGraph(time_valid,result)

tf.keras.metrics.mean_absolute_error(x_valid,result)


