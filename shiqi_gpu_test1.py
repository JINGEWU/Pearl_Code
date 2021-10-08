import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import h5py              as h5
from sklearn.preprocessing import normalize
import time
from datetime import datetime
from sklearn import model_selection
from sklearn.model_selection import KFold
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
from sklearn import model_selection
from sklearn import metrics

# New libraries
import sys
from tensorflow.python.client import device_lib 

# Testing the availability of GPUs
print(tf.__version__)
visible_devices = device_lib.list_local_devices()
print(f'visible_devices: ', visible_devices)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Debugging, where the operations are carried out
#tf.debugging.set_log_device_placement(True)
print(tf.config.list_physical_devices('GPU')) # Check if GPU exists
print(tf.test.is_built_with_cuda()) # Check was built with CUDA

Training = h5.File('training_data_10000.hdf5', 'r')
xTr = Training['inputs'][...]
yTr = Training['target'][...]
yTr = np.reshape(yTr,(-1,1))

Testing = h5.File('test_data_1000.hdf5', 'r')
xTe = Testing['inputs'][...]
yTe = Testing['target'][...]
yTe = np.reshape(yTe,(-1,1))

X_train, X_val, y_train, y_val = model_selection.train_test_split(xTr, yTr, test_size=0.2)

with tf.device('GPU:0'): # Force to be on GPU
    X_train = tf.convert_to_tensor(X_train)
    X_val = tf.convert_to_tensor(X_val)
    y_train = tf.convert_to_tensor(y_train)
    y_val = tf.convert_to_tensor(y_val)

print(f'X_train.device: ', X_train.device) # == device:GPU:0 or similar for TF version >= 2.3.0


#The single ANN run on CPU takes 4 or 3 seconds. Maybe no significant results from GPU.

modelGPU = tf.keras.models.Sequential()
# Define input layer
modelGPU.add(tf.keras.Input(shape=(256,)))
# Define hidden layer 1
modelGPU.add(tf.keras.layers.Dense(100, activation='tanh',name='dense_1'))
modelGPU.add(tf.keras.layers.Dense(200, activation='tanh',name='dense_2'))
# Define output layer
modelGPU.add(tf.keras.layers.Dense(1,name='output'))

modelGPU.compile(optimizer='Adam',loss='mse')
early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=2)
startTime = datetime.now()
print ("Start model fit")
history = modelGPU.fit(X_train, y_train, epochs=1000,validation_data=(X_val,y_val),
                      verbose=2,shuffle=False,callbacks=[early_stopping])
First_Trial = datetime.now() - startTime
print("\nTime taken: ", First_Trial)

#Ensemble Method on CPU takes 20 minutes. 
print("Start ensemble method >>>>>")
start_time_ensemble = time.time()
# kf = KFold(5,shuffle=True,random_state=42)
kf = KFold(2,shuffle=True,random_state=42)
fold = 0
oos_y = []
oos_pred = []
startTime = datetime.now()
for train,val in kf.split(xTr):
    fold+=1
    print(f'Fold#{fold}')
    
    x_train = xTr[train]
    y_train = yTr[train]
    x_val = xTr[val]
    y_val = yTr[val]
    
    model = Sequential()
    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
    model.add(Dense(100,input_dim=xTr.shape[1],kernel_initializer=initializer,activation='tanh'))
    model.add(Dense(100,kernel_initializer=initializer,activation='tanh'))
    model.add(Dense(1,kernel_initializer=initializer,activation='linear'))
    model.compile(loss='mean_squared_error',optimizer='adam')

    with tf.device('GPU:0'): # Force to be on GPU
        X_train = tf.convert_to_tensor(X_train)
        x_val = tf.convert_to_tensor(x_val)
        y_train = tf.convert_to_tensor(y_train)
        y_val = tf.convert_to_tensor(y_val)
        model.fit(x_train,y_train,validation_data=(x_val,y_val),verbose=0,epochs=500)
   
    # save model
    # filename = 'models/model_' + str(fold) +'.h5'
    # model.save(filename)
    pred = model.predict(x_val)
    oos_y.append(y_val)
    oos_pred.append(pred)
    
    # Measure this fold's MSE
    score = metrics.mean_squared_error(pred,y_val)
    print(f'Fold score (MSE):{score}')
print("End ensemble method >>>>>")

elapsed_time_ensemple = time.time() - start_time_ensemble
print("Ensemle time: ", elapsed_time_ensemple)

#exit execution
sys.exit()

