from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as py

from keras.layers import Flatten
#importar de MNIST___________________________________________
from tensorflow.keras.datasets import mnist
#CREACION DE UN MODELO

#Datos de entrenamiento
(x_data, y_data), (x_test, y_test) = mnist.load_data()
num_clases = 10  #numero de clases
# print(x_data[0])# dato
# print(x_data.ndim) #dimesion
# print(x_data.shape) #forma
# print(x_data.dtype) #tipo de dato
# print(y_data[0]) #que imagen

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

#guarda un conjuntod de datos a verificar
datos_indice = []
datos_label = []
for i in range(20):
    indice= np.random.randint(300,500)
    print(y_test[indice])
    datos_indice.append(indice)
    datos_label.append(y_test[indice])
    py.imshow(x_test[indice],cmap=py.cm.binary)
    py.show()

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

#ecualuar
(loss_eval,metricas_eval)= model.evaluate(x_test,y_test)
#predicion
print("Exactitud de la Evalucacion: MEtricas [{}] Perdidas[{}]".format(metricas_eval,loss_eval))
#predecir
pred= model.predict(x_test) #devuelve un vector de cuaes son la y que redice


for i in range(20):
    indice =datos_indice[i]
    Yreal = datos_label[i]
    print("Y real es: [{}];Y calculado es [{}]".format(Yreal,np.argmax(pred[indice]))) #rgmax tranforma a ceros y unos