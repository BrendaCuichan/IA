from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import boston_housing
import matplotlib.pyplot as plt
import tensorflow as ts
import os
'''
#importar datos de base de datos
(x_data, y_data), (x_test, y_test) = boston_housing.load_data() #data para entrenamiento y test para probar y val para validar

##mostrar datos de entradas y salidas
# print("Formatos de entras dimension[{}] tipo[{}] forma[{}]".format(x_data.ndim, x_data.dtype, x_data.shape)) #froma de los datos de entrada
# print("Formatos de salida dimension[{}] tipo[{}] forma[{}]".format(y_data.ndim, y_data.dtype, y_data.shape))#forma de datos de salida
# print("Formatos de entras dimension[{}] tipo[{}] forma[{}]".format(x_test.ndim, x_test.dtype, x_test.shape))
# print("Formatos de salidas dimension[{}] tipo[{}] forma[{}]".format(y_test.ndim, y_test.dtype, y_test.shape))

x_val= x_data[250:,] #se ecoje solo un conjunto de datos
y_val= y_data[250:,]
#print(x_val[5])

#activar modelo
model = Sequential()

#definicion del model
model.add(Dense(13, input_dim=13, kernel_initializer='normal',activation='relu' )) #inicializador = ayudam a incial los pesos
model.add(Dense(6,kernel_initializer='normal', activation='relu'))
model.add(Dense(1,activation='relu'))

#compilar
model.compile(optimizer='rmsprop', loss= 'MeanAbsoluteError', metrics= 'MeanAbsolutePercentageError') #mean porcetaje de valor absolute para ver como va el modelo durante el entranamiento

#entrenar
hist=model.fit(x_data, y_data, epochs=500, validation_data=(x_test, y_test))

# #guardar los pesos despues de entrenar
# model.save_weights('/')

#regresiom
predicion=model.predict(x_val)
print("y real [{}] Ycalculado[{}]".format(y_val[10],predicion[10]))

#guarda pesos
model.save_weights('./Predicion')
#guardad modelo
model.save('myModelo/Regresion')

# #ecualuar
# (loss_eval, metricas_eval) = model.evaluate(x_val, y_val)
# print("Exactitud de la Evalucacion: Metricas [{}] Perdidas[{}]".format(metricas_eval,loss_eval))
#
#
# plt.title('Funcion de perdidas')
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.xlabel('Epocas')
# plt.ylabel('Loss')
# plt.show()


# #predecir
# prediccion= model.predict(x_val)
# #print("Y REAL[{}] Y CAlCULAdo[{}]".format((y_eval[30],prediccion[30])))
'''''

##GURAR PESOS AUTOMATICAMENTE ___ COM REGRECION___________________________________________________
#importar datos de base de datos
(x_data, y_data), (x_test, y_test) = boston_housing.load_data() #data para entrenamiento y test para probar y val para validar

x_val= x_data[250:,] #se ecoje solo un conjunto de datos
y_val= y_data[250:,]
#print(x_val[5])

#definir la ruta y el directorio
check_path= "Train/cp.ckpt"
check_dir= os.path.dirname(check_path) #ruta a guardar
#crear un colbalk
cp_callback= ts.keras.callbacks.ModelCheckpoint(filepath= check_path, save_weights_only=True, verbose=1,save_freq=10*32) #guarda cada cierta epoca los pesos en frecuencia qeu guarde cada 10 epocas y se multripli por bact zice

#parar el entranamiento
erly_estop = ts.keras.callbacks.EarlyStopping(monitor='loss', patience= 10) #despues de 10 repetiticas variaciones cosidera parar


#activar modelo
model = Sequential()

#definicion del model                batch sise cuando datos va a coger
model.add(Dense(13, input_dim=13, kernel_initializer='normal',activation='relu' )) #inicializador = ayudam a incial los pesos
model.add(Dense(6,kernel_initializer='normal', activation='relu'))
model.add(Dense(1,activation='relu'))

#compilar
model.compile(optimizer='Adam', loss= 'MeanSquaredError', metrics= 'MeanAbsolutePercentageError') #mean porcetaje de valor absolute para ver como va el modelo durante el entranamiento

#entrenar
hist=model.fit(x_data, y_data, epochs=500, batch_size=32, validation_data=(x_test, y_test), callbacks= [cp_callback, erly_estop]) #guarda los pesos

#ecualuar
(loss_eval, metricas_eval) = model.evaluate(x_val, y_val)
print("Exactitud de la Evalucacion: Metricas [{}] Perdidas[{}]".format(metricas_eval,loss_eval))

plt.title('Funcion de perdidas')
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.xlabel('Epocas')
plt.ylabel('Loss')
plt.show()

plt.title('Metricas')
plt.plot(hist.history['mean_absolute_percentage_error'])
plt.plot(hist.history['val_mean_absolute_percentage_error'])
plt.xlabel('Epocas')
plt.ylabel('Metricas')
plt.show()


