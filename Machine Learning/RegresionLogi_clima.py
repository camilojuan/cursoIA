# Importa la biblioteca pandas como 'pd' para el manejo de datos tabulares.
import pandas as pd

# Importa la biblioteca statsmodels como 'sm' para análisis estadístico y modelado.
import statsmodels.api as sm

# Importa la función 'train_test_split' de la biblioteca sklearn.model_selection para dividir los datos en conjuntos de entrenamiento y prueba.
from sklearn.model_selection import train_test_split

# Importa la clase 'LogisticRegression' de la biblioteca sklearn.linear_model para crear y ajustar un modelo de regresión logística.
from sklearn.linear_model import LogisticRegression

# Importa la biblioteca Matplotlib como 'plt' para crear gráficos y visualizaciones.
import matplotlib.pyplot as plt

# Importa el módulo 'metrics' de sklearn para evaluar el rendimiento del modelo.
from sklearn import metrics

# Importa la biblioteca NumPy como 'np' para operaciones y cálculos numéricos.
import numpy as np

# Importa el módulo 'warnings' para manejar advertencias.
import warnings

print('>>> Librerías importadas')

datos = pd.read_csv('weather_dataset.csv', header=0)
datos = datos.drop(['Weather'], axis=1)
datos

# Crear la variable 'y' que almacena la columna de los datos de salida
y = datos[['Category']]

# Crear la variable 'x' que almacena un subconjunto de columnas del DataFrame 'datos' que se utilizarán como características
""" X = datos.iloc[:, [0,1,2,3,4,5]] //Alternativa"""
x = datos[datos.columns[0:5]]

# Crear un modelo de regresión logística
logit_model = sm.Logit(y, x)

# Ajustar el modelo a los datos
result = logit_model.fit()

# Imprimir un resumen detallado del modelo ajustado
print(result.summary())

# x_train: Conjunto de características de entrenamiento.
# x_test: Conjunto de características de prueba.
# y_train: Variable objetivo de entrenamiento.
# y_test: Variable objetivo de prueba.

# Dividir los datos en conjuntos de entrenamiento y prueba - 30% Test y 70% Training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Crear una instancia del modelo de regresión logística
logistic_model = LogisticRegression()

# Ajustar el modelo a los datos de entrenamiento después de convertir y_train en un array unidimensional
# Esto es necesario para evitar el warning "DataConversionWarning"
y_train_1d = y_train.values.ravel()

"""
El ajuste del modelo de datos es el proceso en el que el modelo aprende a partir de los
datos de entrenamiento para hacer predicciones precisas en nuevos datos.
Implica encontrar los parámetros óptimos y capturar patrones en los datos.
El objetivo final es tener un modelo que pueda realizar predicciones útiles y precisas en situaciones del mundo real.
"""
logistic_model.fit(x_train, y_train_1d)

print('>>> Modelo entrenado')

# Realizar predicciones en el conjunto de prueba utilizando el modelo logistic_model
y_pred = logistic_model.predict(x_test)

# Contar la cantidad de predicciones de lluvia y no lluvia
rain_predicted = np.sum(y_pred == 1)
no_rain_predicted = np.sum(y_pred == 0)

# Imprimir la cantidad de predicciones de lluvia y no lluvia
print('RESULTADOS DE DATOS ANALIZADOS DEL CONJUNTO DE PRUEBA')
print(f"Número de casos en donde se predice lluvia: {str(rain_predicted)}")
print(f"Número de casos en donde no se predice lluvia: {str(no_rain_predicted)}")

# Calcular la matriz de confusión utilizando la biblioteca metrics
metricas = metrics.confusion_matrix(y_test, y_pred)

# Crear una visualización de la matriz de confusión
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=metricas, display_labels=[False, True])

# Mostrar la visualización de la matriz de confusión
cm_display.plot()

# Mostrar la gráfica
plt.show()

# Imprimir la exactitud del modelo comparando las etiquetas reales 'y_test' con las predicciones 'y_pred'
print('Exactitud de predicciones:', metrics.accuracy_score(y_test, y_pred))
print('Precisión de predicciones:', metrics.precision_score(y_test, y_pred))
print('Sensibilidad de predicciones:', metrics.recall_score(y_test, y_pred))
print('Puntuación F de predicciones:', metrics.f1_score(y_test, y_pred))

# Valores de entrada para la predicción
hour, relative_humidity, wind_speed, visibility_km, pressure_kPa = 7,70,33,25,99.14

# Realizar la predicción utilizando el modelo de regresión logística con supresión de advertencia
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    prediction = logistic_model.predict([[hour, relative_humidity, wind_speed, visibility_km, pressure_kPa]])

# Determinar la descripción de la predicción en función del resultado
prediction_description = 'Lluvia' if prediction[0] == 1 else 'No lluvia'

# Mostrar la predicción con información sobre los valores ingresados
print(f"De acuerdo a los valores ingresados, la predicción es: {prediction_description}")

def relation_x_y(X, Y):
    """
    Muestra un gráfico de dispersión de dos conjuntos de datos y calcula la correlación entre ellos.

    Parámetros:
    X (array-like): Un conjunto de datos (por ejemplo, una lista o un arreglo NumPy).
    Y (array-like): Otro conjunto de datos del mismo tamaño que X.

    Retorna:
    float: El coeficiente de correlación de Pearson entre X y Y, que mide la relación lineal entre los dos conjuntos de datos.

    El coeficiente de correlación de Pearson varía entre -1 y 1, donde:
    -1: Hay una correlación negativa perfecta (a medida que una variable aumenta, la otra disminuye).
    0: No hay correlación lineal.
    1: Hay una correlación positiva perfecta (a medida que una variable aumenta, la otra también aumenta).
    """
    # Crea un gráfico de dispersión de los conjuntos de datos X e Y con transparencia (alpha)
    plt.scatter(X, Y, alpha=0.25)

    # Agrega un título al gráfico.
    plt.title("Dispersión entre X e Y")

    # Muestra el gráfico en pantalla.
    plt.show()

    # Calcula el coeficiente de correlación de Pearson entre X e Y y lo retorna.
    correlation = np.corrcoef(X, Y)[0, 1]
    return correlation
print("Coeficiente de correlación ", relation_x_y(datos[datos.columns[1]], datos[datos.columns[2]]))