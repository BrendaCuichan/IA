from keras.datasets import cifar10
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from  keras.layers import Conv2D, MaxPool2D ,Flatten ,Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt

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
    model.add(Conv2D(32, (3, 3), 1, activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), 1, activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(MaxPool2D(2, 2)) #maxpuling reducion de los datos
    model.add(Conv2D(64, (3, 3), 1, activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=(32, 32, 3)))
    model.add(Conv2D(64, (3, 3), 1, activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=(32, 32, 3)))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(128, (3, 3), 1, activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=(32, 32, 3)))
    model.add(Conv2D(128, (3, 3), 1, activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=(32, 32, 3)))
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
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test)) #entrenar el modelo
perdidas, presicion = model.evaluate(x_test, y_test)
model.save('CNN1.h5')
print("Exactitud de  la Evaluacion: Metricas[{}] Perdidas[{}]".format(presicion, perdidas))
CurvaEvaluacion(history)

