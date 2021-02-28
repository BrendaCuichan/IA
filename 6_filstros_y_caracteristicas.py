from sklearn.datasets import load_sample_images
import numpy as np
import tensorflow as ts
import matplotlib.pyplot as plt
#cargar el dataset de imagenes
imagenes= load_sample_images().images

#variables para cargar una imagen
img1= imagenes[0]
img2= imagenes[1]

#mostrar imagenes
plt.imshow(img1)
plt.show()
plt.imshow(img2)
plt.show()

#formato del dato
print("Forato de las imagenes --> dim[{}] Shape[{}] type[{}] ".format(img1.ndim, img1.shape , img1.dtype))
#covertir las imagenes en un arreglo de numeros flotantes
dataset = np.array(imagenes,dtype=np.float)
print("Forato de las imagenes --> dim[{}] Shape[{}] type[{}] ".format(dataset.ndim, dataset.shape , dataset.dtype))

#que sea compatible con la versio  1
ts.compat.v1.disable_eager_execution()

#delaramos variables
tam, alto, ancho, canales= dataset.shape

#declarar uuna matriz de ceros
filtros= np.zeros(shape=(5,5,canales,2), dtype=np.float)
# print(filtros[0])

#filstro vertical colocar 1 al final
filtros[:,3,:,0] =1
#filtro horizontal
filtros[3,:,:,1]=1

#compatibilidad
x= ts.compat.v1.placeholder(ts.float32, shape= (None, alto, ancho, canales))
#crear una capa convulucional
conv1= ts.nn.conv2d(x,filtros,strides=[1,5,5,1],padding= "VALID")

# ejecutar la convulucion uno creando una secion creando una capa covuluciona
with ts.compat.v1.Session() as Sesion:
    resultado = Sesion.run(conv1, feed_dict= {x: dataset}) #colocar los resultados y que ejecute con la convulucion 1
#
plt.imshow (resultado[0,:,:,0])
plt.show()

plt.imshow (resultado[0,:,:,1])
plt.show()

plt.imshow (resultado[1,:,:,0])
plt.show()

plt.imshow (resultado[1,:,:,1])
plt.show()

#
