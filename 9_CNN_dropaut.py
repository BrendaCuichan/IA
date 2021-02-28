from keras.datasets import cifar10
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from  keras.layers import Conv2D, MaxPool2D ,Flatten ,Dense, Dropout
from keras.optimizers import SGD
from keras.regularizers import L1
import matplotlib.pyplot as plt
import tensorflow as ts
import os

#Cargar datos
def CargarDatos():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = np.array(x_train, dtype=np.float) #imagenes convertimos en un array
    x_train = x_train/255.0
    x_test = np.array(x_test, dtype=np.float)
    x_test = x_test / 255.0

    # tranformando a categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test

#funcion para crear modelo CNN
def CrearModeloCNN():
    model = Sequential()

    # keras no nos permite manipular solo dice cuantos filsttros quiere utilazr y ese parametro debe
    # ser multiplo de 32
    # tamaño del kernel
    # firna de las entradas 32*32*3

    # etapa de extraxion de cararcteristicas
    # cifar 10 su imagenes tiene dos dimenciones,( filtro, capo de recepcio= de 3 neuronas,stride=cuadritos solpados enttre
    # una neurona, padin el borde= ningu, vald una capa de ceros
    model.add(Conv2D(32, (3, 3), 1, activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3))) #+ kernerl --> kernel_regularizar=L2(0.0001) regularizador
    model.add(Conv2D(32, (3, 3), 1, activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(MaxPool2D(2, 2)) #maxpuling reducion de los datos
    model.add(Dropout(0.35))

    model.add(Conv2D(64, (3, 3), 1, activation='relu', kernel_initializer='he_uniform', padding='same',input_shape=(32, 32, 3)))
    model.add(Conv2D(64, (3, 3), 1, activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(MaxPool2D(2, 2))

    model.add(Conv2D(128, (3, 3), 1, activation='relu', kernel_initializer='he_uniform', padding='same',input_shape=(32, 32, 3)))
    model.add(Conv2D(128, (3, 3), 1, activation='relu', kernel_initializer='he_uniform', padding='same',input_shape=(32, 32, 3)))
    model.add(MaxPool2D(2, 2))


    #el bacth normalization --->>
    model.add(Conv2D(128, (3, 3), 1, activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(128, (3, 3), 1, activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(MaxPool2D(2, 2))

    # Etapa de clasificación

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    optimizador = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=optimizador, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def CurvaEvaluacion(history):
    plt.figure(0)
    plt.plot(history.history['accuracy'], 'r')
    plt.plot(history.history['loss'], 'g')
    plt.xlabel("No. de Epocas")
    plt.ylabel("Perdidas")
    plt.title("Perdidas vs Precision")
    plt.show()

x_train, y_train, x_test, y_test = CargarDatos()
model = CrearModeloCNN()
#implementar los collbasck para evitar el sobrenetrenaiento._________________________________________
#definir la ruta y el directorio
check_path= "CNN/Train/cp.ckpt"
check_dir= os.path.dirname(check_path) #ruta a guardar
#crear un colbalk
cp_callback= ts.keras.callbacks.ModelCheckpoint(filepath= check_path, save_weights_only=True, verbose=1,save_freq=10*64) #guarda cada cierta epoca los pesos en frecuencia qeu guarde cada 10 epocas y se multripli por bact zice

#parar el entranamiento
erly_estop = ts.keras.callbacks.EarlyStopping(monitor='loss', patience= 10) #despues de 10 repetiticas variaciones cosidera parar

history = model.fit(x_train, y_train, epochs=1, batch_size=64, validation_data=(x_test, y_test),callbacks= [cp_callback, erly_estop]) #entrenar el modelo


#evaluar el modelo de 3 bloques con un modelo de 2 bloques, de un bloque.

perdidas, presicion = model.evaluate(x_test, y_test)
print("Exactitud de  la Evaluacion: Metricas[{}] Perdidas[{}]".format(presicion, perdidas))
CurvaEvaluacion(history)


##compra la ejecucion de tres bloques sin droutout y con drout oup(0.05), 20% y al 35%