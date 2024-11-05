import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Cargar el dataset
data = pd.read_csv('dataset_presion_arterial.csv')

# Dividir los datos en características (X) y etiquetas (y)
X = data['Edad'].values
y = data['Presion_Arterial'].values

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state =42)

# Construir el modelo de la red neuronal
model = tf.keras.models.Sequential([
    # Capa de entrada con 64 neuronas y función de activación ReLU
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),

    # Capa oculta con 32 neuronas y función de activación ReLU
    tf.keras.layers.Dense(32, activation='relu'),

    # Capa de salida con una neurona (regresión, sin función de activación específica)
    tf.keras.layers.Dense(1)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=500, batch_size=32)

# Evaluar el modelo en los datos de prueba y obtener la pérdida
loss = model.evaluate(X_test, y_test)

# Imprimir la pérdida en los datos de prueba
print(f'Pérdida en datos de prueba: {loss}')

# Crear una nueva edad como un arreglo numpy
new_age = np.array([40])

# Realizar una predicción de la presión arterial para la nueva edad utilizando el modelo
predicted_blood_pressure = model.predict(new_age)

# Imprimir la presión arterial predicha en la consola
print(f'Presión arterial predicha: {predicted_blood_pressure[0][0]}')

data.plot(kind='scatter', x='Edad', y='Presion_Arterial', figsize=(10 ,6))

# Visualizar la pérdida durante el entrenamiento
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Pérdida en entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Pérdida durante el entrenamiento')
plt.show()

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Visualizar las predicciones
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, label='Datos reales', color='blue')
plt.scatter(X_test, y_pred, label='Predicciones', color='red', marker='x')
plt.xlabel('Edad')
plt.ylabel('Presión Arterial')
plt.legend()
plt.title('Predicciones vs. Datos reales')
plt.show()

