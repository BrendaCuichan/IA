from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as py

##agregando con capa densa 3 neuronas
# model = Sequential()
# layer =Dense(3) #primera capa
# print(layer.weights)
# x=tf.ones((1,4)) #forma de los datps de entrada
# print(x)
# y=layer(x)
# print(layer.weights)

# #ingresando neuronas
# model = Sequential()
# #x = tf.ones((1,4))  #forma de entrada
# model.add(Dense(6, activation='relu', input_shape= (4, ))) #imput shape /forma de los datos
# model.add(Dense(4, activation='relu'))
# model.add(Dense(2, activation='relu'))
# #y=model(x)
# model.summary()

#
# #CREACION DE UN MODELO randomico
#
# #Datos de entrenamiento
# x_data = np.random.random((2000, 10))
# y_data= np.random.randint(2,size=(2000,1))  #label (solo valores de 0 y 1
# # print(x_data[0])
#
# #datos de validacion
# x_val = np.random.random((200, 10))
# y_val= np.random.randint(2,size=(200,1))
#
# #datos de test
# x_test = np.random.random((200, 10))
# y_test= np.random.randint(2,size=(200,1))
#
# model = Sequential()
#
# #definicion del modelo
# model.add(Dense(64, activation='relu', input_shape=(10, )))
# model.add(Dense(32, activation='relu', use_bias= False))
# model.add(Dense(16, activation='relu', use_bias= False))
# model.add(Dense(8, activation='relu', use_bias= False))
# model.add(Dense(1, activation='relu', use_bias= False))
#
# # compilar el modelo , optimizar
# model.compile(optimizer='adam', loss= 'BinaryCrossentropy', metrics= 'BinaryCrossentropy')
#
# #entrnar el modelo
# h=model.fit(x_data, y_data, epochs=500, batch_size=64, validation_data= (x_val,y_val))
#
# #visualizar modelo
# #model.summary()
# # plot_model (model, to_file= 'RedNeronal1.pngh')
#
#
# #graficar el modelo
# py.figure(0)
# py.plot(h.history['binary_crossentropy'], 'r')
# py.plot(h.history['val_binary_crossentropy'], 'g')
# py.xlabel('Num Epocas')
# py.ylabel('Metricas')
#
#
#
#
# # graficar las perdidas
# py.figure(1)
# py.plot(h.history['loss'], 'c')
# py.plot(h.history['val_loss'], 'm')
# py.xlabel('Num Epocas')
# py.ylabel('Perdidas y metricas')
# py.show()

from keras.layers import Flatten
#importar de MNIST___________________________________________
from tensorflow.keras.datasets import mnist
#CREACION DE UN MODELO

#Datos de entrenamiento
(x_data, y_data), (x_test, y_test) = mnist.load_data()
num_clases = 10  #numero de clases
print(x_data[0])# dato
print(x_data.ndim) #dimesion
print(x_data.shape) #forma
print(x_data.dtype) #tipo de dato
print(y_data[0]) #que imagen

#visualizar la imagen
# py.imshow(x_data[0], cmap=py.cm.binary)
# py.show()

#trandrma datos para el tenserflow
x_data = x_data.astype('float32') #transforma a float para que use kersas
x_data = x_data / 255 #matriz flotante de 0 a 1
x_data = x_data.reshape(x_data.shape[0],28,28,1)#cambiar su forma

x_test= x_test.astype('float32')
x_test = x_test / 255
x_test= x_test.reshape(x_test.shape[0],28,28,1)


y_data = tf.keras.utils.to_categorical(y_data, num_clases) #trasdorma a 0 y  1
y_test = tf.keras.utils.to_categorical(y_test, num_clases)


#activar modelo
model = Sequential()
model.add(Flatten(input_shape=(28,28,1)))
#definicion del modelo flaten
model.add(Dense(68, activation='relu', ))
model.add(Dense(20, activation='relu'))
model.add(Dense(num_clases, activation='softmax'))
#model.summary()
# compilar el modelo , optimizar
model.compile(optimizer='Adam', loss= 'categorical_crossentropy', metrics= 'categorical_crossentropy')

# #entrenar el modelo
h=model.fit(x_data, y_data, epochs=50, batch_size=64, validation_data=(x_test, y_test))

#imprimir

py.figure(0)
py.plot(h.history['loss'], 'r')
py.plot(h.history['val_loss'], 'g')
py.xlabel('No epocas')
py.ylabel('perdidas')
py.show()