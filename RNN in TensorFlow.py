# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 20:51:19 2020

@author: Shivansh
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np


#From line 13 to 45 was way to understant what the tf.data.Dataset worked you can also uncomment
#and run those blocks Ps I use spyder
#dataset = tf.data.Dataset.range(10)
##printing the dataset for the series
#
##for val in dataset:
##    print( val.numpy() )
#
#dataset = dataset.window( 5 , shift = 1 )
#
##To see what the splitted data in a window looks like
#
##for window in dataset:
##    for val in window:
##        print( val.numpy() , end = " " )
##    print()
#
##Now we see that there is a remainder when the size is less than 5 so we will drop it
#dataset = tf.data.Dataset.range(10)
#dataset = dataset.window( 5 , shift = 1  , drop_remainder = True )
#
#for window in dataset:
#    for val in window:
#        print( val.numpy() , end = " " )
#    print()
#    
##Flattens the dataset into batches of 5 as a numpy array
#dataset = dataset.flat_map( lambda window:window.batch(5))
#dataset = dataset.map( lambda window: ( window[:-1],window[-1:] ) )
#dataset = dataset.shuffle(buffer_size = 10 )
#dataset = dataset.batch(2).prefetch(2)
#
#for x,y in dataset:
#    print( "X : values " , x.numpy() )
#    print( "Y : values " , y.numpy() )


#The below part generates a synthetic data for the time series analysis
#It was part of a course on courser by DeepLearning.ai where I learnt this


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
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
amplitude = 40
slope = 0.05
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
    

dataset = windowedDataset( x_train  , window_size , batch_size , shuffle_buffer_size )

tf.keras.backend.clear_session()
tf.random.set_seed(51)



model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda( lambda x : tf.expand_dims( x , axis = -1 ) , input_shape = [None]),
        tf.keras.layers.SimpleRNN( 40 , return_sequences = True ),
        tf.keras.layers.SimpleRNN( 40 ),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda( lambda x : x * 100.0)       
        ])
    
    
#Now if we want to change the LR rate of SGD on the fly then we can do the following
    
LR_Schedule = tf.keras.callbacks.LearningRateScheduler( lambda epoch: 1e-8 * 10**(epoch / 20) )

model.compile( loss = tf.keras.losses.Huber() , optimizer = tf.keras.optimizers.SGD( lr = 1e-8 , momentum = 0.9) , metrics = ['acc','mae'])

history = model.fit( dataset , callbacks = [LR_Schedule] , epochs = 100  )

#We need to find which Learning Rate worked for us by plotting the graph between the learning rate and the loss value

plt.plot( history.history["lr"] , history.history["loss"])
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.axis([1e-8, 1e-4, 0, 30])


#Therefore we find that the lowest loss value was at the value at 5e-5

tf.keras.backend.clear_session()

model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda( lambda x : tf.expand_dims( x , axis=-1 ) , input_shape = [None] ),
        tf.keras.layers.SimpleRNN( 40 ,  return_sequences = True),
        tf.keras.layers.SimpleRNN( 40 ),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda( lambda x : x*100.0)
        ])
    
model.compile( loss = tf.keras.losses.Huber() , optimizer = tf.keras.optimizers.SGD(lr=5e-5 , momentum = 0.9),metrics = ['mae'])

history  = model.fit( dataset ,epochs = 100 )


forecast = []


for time in range( len(series) - window_size ):
    forecast.append( model.predict( series[time:time+window_size][np.newaxis]))
    
forecast = forecast[split_time-window_size:]
result = np.array(forecast)[:,0,0]

plotGraph(time_valid,x_valid)
plotGraph(time_valid,result)

print( "Mean Absolute Error is : {} ".format(tf.keras.metrics.mean_absolute_error(x_valid,result)))

mae=history.history['mae']
loss=history.history['loss']

epochs=range(len(loss)) # Get number of epochs

plt.plot(epochs, mae, 'r')
plt.plot(epochs, loss, 'b')
plt.title('MAE and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["MAE", "Loss"])

plt.figure()
    
















































