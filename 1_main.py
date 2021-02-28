from keras.models import Sequential
#improtar la capa densa
from keras.layers import Dense
import tensorflow as tf
'''''
model=Sequential()

#crecionde de la capa  densa
#model=Sequential([Dense(2,activation="relu",name="layer1"),Dense(3,activation="relu",name="Layers2"),Dense(4,name="Layer3")])

layer=Dense(3) # pesos
#model.add(layer)

#crea los pesos a la primera que se llama a un entrad
x= tf.ones((1,4)) #crea matriz de 1+4 tipo flotante
print(x)
y=layer(x)

print(layer.weights) # **/
model.summary()#cual es la forma desde la entra desde el principio
'''
model=Sequential()
model=Sequential(Dense(2,activation="relu",input_shape=(4, ))) #num neurona, funcion, la forma)
model.summary()