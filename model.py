# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from tensorflow.keras import layers

# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

df = pd.read_csv('Dataset.csv')

x = df.drop(columns=['state'])
y = df.drop(columns=['Fx','Fy','Px','Py'])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

model = tf.keras.Sequential()
model.add(layers.Dense(24, activation='relu', input_shape=(4,)))
# The new second layer may help the network learn more complex representations
# model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(24, activation='softmax'))
model.add(layers.Dense(1))

#Compile the model using a standard optimizer and loss function for regression
model.compile(optimizer='Adamax', loss='mse', metrics=['mae','accuracy'])

history_1 = model.fit(x_train, y_train, epochs=200, batch_size = 1,
                        validation_data=(x_test, y_test))

loss = history_1.history['loss']
val_loss = history_1.history['val_loss']

# Plot 1
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'g.', label='Training loss')
plt.plot(epochs, val_loss, 'b.', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Use the model to make predictions from our validation data
predictions = model.predict(x_test)

# Calculate and print the loss on our test dataset
loss = model.evaluate(x_test, y_test)

# Graph the predictions against the actual values
plt.clf()
plt.title('Comparison of predictions and actual values')
plt.plot(x_test, y_test, 'b.')
plt.plot(x_test, predictions, 'r.')
plt.xlabel('Test samples id')
plt.ylabel('Prediction-Actual Value')
plt.legend()
plt.show()

print(predictions)
print(y_test)

#model = tf.keras.Sequential()
#model = model

# Convert the model to the TensorFlow Lite format without quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model to disk
open("Vibration_Stat_Regr_model.tflite", "wb").write(tflite_model)

# Convert the model to the TensorFlow Lite format with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()

# Save the model to disk
open("Vibration_Stat_Regr_model_quantized.tflite", "wb").write(tflite_model)

#######################################################################

# Instantiate an interpreter for each model
sine_model = tf.lite.Interpreter('Vibration_Stat_Regr_model.tflite')
sine_model_quantized = tf.lite.Interpreter('Vibration_Stat_Regr_model_quantized.tflite')

import os
basic_model_size = os.path.getsize("Vibration_Stat_Regr_model.tflite")
print("Basic model is %d bytes" % basic_model_size)
quantized_model_size = os.path.getsize("Vibration_Stat_Regr_model_quantized.tflite")
print("Quantized model is %d bytes" % quantized_model_size)
difference = basic_model_size - quantized_model_size
print("Difference is %d bytes" % difference)


# test_df = pd.DataFrame(columns=["x", "y","px","py"], data=[[103.50,186.53,2,2]])
# test_pred = model.predict(test_df)
# print(test_pred)
