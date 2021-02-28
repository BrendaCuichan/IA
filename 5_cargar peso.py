from keras.models import Sequential
from keras.layers import Dense
import tensorflow as ts
from keras.datasets import boston_housing
import matplotlib.pyplot as plt
import  os
import tensorflow as ts
#
#
# (x_data, y_data), (x_test, y_test) = boston_housing.load_data() #data para entrenamiento y test para probar y val para validar
# x_val= x_data[250:,] #se ecoje solo un conjunto de datos
# y_val= y_data[250:,]
#
#
# #cargar modelo guragadao
# nuevo_modelo=ts.keras.models.load_model('myModelo/Regresion') #cargar modelo
#
# #cargar los pesos guardados
# nuevo_modelo.load_weights('./Predicion')
#
# #ecualuar
# (loss_eval, metricas_eval) = nuevo_modelo.evaluate(x_val, y_val)
# print("Exactitud de la Evalucacion: Metricas [{}] Perdidas[{}]".format(metricas_eval,loss_eval))
#
# #regresiom
# predicion=nuevo_modelo.predict(x_val)
# print("y real [{}] Ycalculado[{}]".format(y_val[10],predicion[10]))
#
#
#
# #
# # plt.title('Funcion de perdidas')
# # plt.plot(hist.history['loss'])
# # plt.plot(hist.history['val_loss'])
# # plt.xlabel('Epocas')
# # plt.ylabel('Loss')
# # plt.show()
#
#
# # #predecir
# # prediccion= model.predict(x_val)
# # #print("Y REAL[{}] Y CAlCULAdo[{}]".format((y_eval[30],prediccion[30])))

# CARGAR PESO CON CALLBACKS_____________________________________________________________________________


(x_data, y_data), (x_test, y_test) = boston_housing.load_data() #data para entrenamiento y test para probar y val para validar
x_val= x_data[250:,] #se ecoje solo un conjunto de datos
y_val= y_data[250:,]


#cargar modelo guragadao
nuevo_modelo = ts.keras.models.load_model('myModelo/Regresion') #cargar modelo

#cargar los pesos guardados
# nuevo_modelo.load_weights('./Predicion')
check_path= "Train/cp.ckpt"
check_dir= os.path.dirname(check_path) #ruta a guardar

#tomar el ultimo peso
ultimo= ts.train.latest_checkpoint(check_dir)

#cargar de todos los cobals los peso
nuevo_modelo.load_weights(ultimo)

#ecualuar
(loss_eval, metricas_eval) = nuevo_modelo.evaluate(x_val, y_val)
print("Exactitud de la Evalucacion: Metricas [{}] Perdidas[{}]".format(metricas_eval,loss_eval))

#regresiom
predicion=nuevo_modelo.predict(x_val)
print("y real [{}] Ycalculado[{}]".format(y_val[10],predicion[10]))


#trsdomar el dato que sean plano
pred = predicion.flatten()
plt.scatter(y_val, pred)


plt.title('PRrediciones')
plt.xlabel('Valores Reales')
plt.ylabel('Valores Calculados')
plt.xlim([0,plt.xlim()[1]])#prediciendo correctamete
plt.ylim([0,plt.ylim()[1]])#prediciendo correctamete
_=plt.plot([-100,100],[-100,100])
plt.show()
