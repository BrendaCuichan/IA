from keras.datasets import cifar10
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from  keras.layers import Conv2D, MaxPool2D ,Flatten ,Dense
from keras.optimizers import SGD , Adam
import matplotlib.pyplot as plt
from keras.regularizers import L2
from sklearn.metrics  import classification_report
import tensorflow as ts


#Cargar datos
from tensorflow import strided_slice


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

#funcion para crear modelo CNN de dos bloques
def CrearModeloCNN():
    model = Sequential()
    model.add(Conv2D(input_shape=(32, 32, 3), kernel_size=(2, 2), padding='same',strides=(2, 2), filters=32))
    model.add(MaxPool2D(pool_size=(2, 2),strides=(1,1),padding='same')) #maxpuling reducion de los datos
    model.add(Conv2D(kernel_size=(2,2),padding='same',strides=(2,2),filters=64))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same'))



    # Etapa de clasificación

    model.add(Flatten()) #capa plana
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    adam=Adam()
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model
def CurvaEvaluacion(history):
    plt.figure(0)
    plt.plot(history.history['accuracy'], 'r')
    plt.plot(history.history['loss'], 'g')
    plt.xlabel("No. de Epocas")
    plt.ylabel("Perdidas")
    plt.title("Perdidas vs Precision(TRAIN)")
    plt.show()

def CurvaEvaluacion(model_history):
    plt.style.use('ggplot')
    plt.figure(0)
    plt.plot(model_history.history['loss'],label='Perdidas de entrenamiento')
    plt.plot(model_history.history['val_loss'], label='Perdidas de validación')
    plt.plot(model_history.history['accuracy'], label='Exactitud de entrenamiento')
    plt.plot(model_history.history['val_accuracy'],label='Exactitud de validación')
    plt.title("Perdidas y Exactitus de entrenamieto")
    plt.xlabel("Epocas No.")
    plt.ylabel("Perdida/Exactitud")
    plt.legend()
    plt.show()


CIFAR_10_CLASSES = ['airplane', #avion
                    'automobile',#automovil
                    'bird', #Ave
                    'cat',#gato
                    'deer',#venado
                    'dog',#perro
                    'frog',#rana
                    'horse',#caballo
                    'ship',#barco
                    'truck']#camio
x_train, y_train, x_test, y_test = CargarDatos()
model = CrearModeloCNN()
early_stop = ts.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test)) #entrenar el modelo
perdidas, presicion = model.evaluate(x_test, y_test)
model.save('CNN1.h5')
print("Exactitud de  la Evaluacion: Metricas[{}] Perdidas[{}]".format(presicion, perdidas))
CurvaEvaluacion(history)

predictions= model.predict(x_test,batch_size=64)

print(classification_report(y_test.argmax(axis=1),target_names=CIFAR_10_CLASSES)) #repote de clasificacion
