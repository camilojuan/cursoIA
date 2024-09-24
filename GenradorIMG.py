import numpy as np
from io import BytesIO
import requests
from matplotlib import pyplot as plt
from PIL import Image
import random as rd

# URL de la imagen de la Mona Lisa
url = "https://media.npr.org/assets/img/2012/02/02/mona-lisa_custom-31a0453b88a2ebcb12c652bce5a1e9c35730a132-s1100-c50.jpg"

# Descarga de la imagen desde la web
rta = requests.get(url)
imagen = Image.open(BytesIO(rta.content))

# Conversión de la imagen a un array de Numpy
imgArrayOriginal = np.array(imagen)

# Muestra la imagen original
plt.imshow(imgArrayOriginal)
plt.title('Imagen Original')
plt.show()  # Se usa plt.show() en lugar de %matplotlib inline

# Parámetros iniciales para la mutación
porcentajeMuta = 1  # Tolerancia a genes con un porcentaje de error
iteraciones = 5  # Número de iteraciones para el proceso de mutación

# Conversión de la imagen original a un array con tipo int64
imgArrayOriginal = np.array(imagen, dtype='int64')

# Inicialización de arrays auxiliares
prubimgArrayOriginal = imgArrayOriginal
valor = [0] * len(imgArrayOriginal) * len(imgArrayOriginal[1])
nuevaimg = np.array(imagen)
a = np.array(imagen)

# Generación de imagen aleatoria (se genera una imagen con colores RGB aleatorios)
imgAleatoria = np.array(imagen)
for i in range(len(imgAleatoria)):
    for j in range(len(imgAleatoria[1])):
        for ij in range(3):
            imgAleatoria[i][j][ij] = rd.randint(0, 255)

# Visualiza la imagen aleatoria generada
plt.imshow(imgAleatoria)
plt.title('Imagen Aleatoria Generada')
plt.show()

# Proceso de mutación basado en el error relativo
for p in range(iteraciones):
    for i in range(len(imgAleatoria)):
        for j in range(len(imgAleatoria[1])):

            # Caracterización del canal de color rojo (R)
            if imgArrayOriginal[i][j][0] != 0:  # Evita divisiones por cero
                if (abs(imgAleatoria[i][j][0] - imgArrayOriginal[i][j][0]) / imgArrayOriginal[i][j][0]) > (1 - porcentajeMuta / 100):
                    if (imgAleatoria[i][j][0] - imgArrayOriginal[i][j][0]) < 0:
                        imgAleatoria[i][j][0] += abs((imgAleatoria[i][j][0] - imgArrayOriginal[i][j][0]) / 2)
                    else:
                        imgAleatoria[i][j][0] -= abs((imgAleatoria[i][j][0] - imgArrayOriginal[i][j][0]) / 2)
            else:
                imgAleatoria[i][j][0] -= abs((imgAleatoria[i][j][0] - imgArrayOriginal[i][j][0]) / 2)

            # Caracterización del canal de color verde (G)
            if imgArrayOriginal[i][j][1] != 0:  # Evita divisiones por cero
                if (abs(imgAleatoria[i][j][1] - imgArrayOriginal[i][j][1]) / imgArrayOriginal[i][j][1]) > (1 - porcentajeMuta / 100):
                    if (imgAleatoria[i][j][1] - imgArrayOriginal[i][j][1]) < 0:
                        imgAleatoria[i][j][1] += abs((imgAleatoria[i][j][1] - imgArrayOriginal[i][j][1]) / 2)
                    else:
                        imgAleatoria[i][j][1] -= abs((imgAleatoria[i][j][1] - imgArrayOriginal[i][j][1]) / 2)
            else:
                imgAleatoria[i][j][1] -= abs((imgAleatoria[i][j][1] - imgArrayOriginal[i][j][1]) / 2)

            # Caracterización del canal de color azul (B)
            if imgArrayOriginal[i][j][2] != 0:  # Evita divisiones por cero
                if (abs(imgAleatoria[i][j][2] - imgArrayOriginal[i][j][2]) / imgArrayOriginal[i][j][2]) > (1 - porcentajeMuta / 100):
                    if (imgAleatoria[i][j][2] - imgArrayOriginal[i][j][2]) < 0:
                        imgAleatoria[i][j][2] += abs((imgAleatoria[i][j][2] - imgArrayOriginal[i][j][2]) / 2)
                    else:
                        imgAleatoria[i][j][2] -= abs((imgAleatoria[i][j][2] - imgArrayOriginal[i][j][2]) / 2)
            else:
                imgAleatoria[i][j][2] -= abs((imgAleatoria[i][j][2] - imgArrayOriginal[i][j][2]) / 2)

# Visualiza la imagen mutada después de todas las iteraciones
plt.imshow(imgAleatoria)
plt.title('Imagen después de mutaciones')
plt.show()
