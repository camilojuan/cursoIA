# @title
# Importa la biblioteca NumPy bajo el alias 'np'.
import numpy as np

# Importa la clase 'LinearRegression' del módulo 'linear_model' de la biblioteca scikit-learn.
from sklearn.linear_model import LinearRegression

# Importa la biblioteca pandas bajo el alias 'pd' para el manejo de datos tabulares.
import pandas as pd

# Importa la biblioteca Matplotlib bajo el alias 'plt' para crear gráficos y visualizaciones.
import matplotlib.pyplot as plt

# Importa el módulo 'warnings' para manejar advertencias.
import warnings

def create_scatter_plot(data, x_label, y_label, plot_title='Gráfico de Dispersión', figure_size=(8, 5)):
    """
    Crea un gráfico de dispersión a partir de los datos proporcionados y las etiquetas de los ejes.

    Args:
    data (pd.DataFrame): Un DataFrame de pandas que contiene los datos a graficar.
    x_label (str): La etiqueta para el eje X.
    y_label (str): La etiqueta para el eje Y.
    plot_title (str): El título del gráfico.
    figure_size (tuple, opcional): El tamaño de la figura (ancho, alto). Por defecto, es (8, 5) pulgadas.

    Returns:
    None: La función muestra el gráfico, pero no devuelve ningún valor.
    """
    data.plot(kind='scatter', x=x_label, y=y_label, figsize=figure_size)

    # Establece el título del gráfico.
    plt.title(plot_title)

    # Muestra el gráfico.
    plt.show()

def train_linear_regression(data, feature_col, target_col):
    """
    Entrena un modelo de regresión lineal.

    Args:
        data (pandas.DataFrame): El DataFrame que contiene los datos de entrenamiento.
        feature_col (str): El nombre de la columna que se utilizará como característica.
        target_col (str): El nombre de la columna que se utilizará como objetivo.

    Returns:
        sklearn.linear_model.LinearRegression: El modelo de regresión lineal entrenado.
    """
    # Crea una instancia del modelo de regresión lineal.
    reg = LinearRegression()

    # Ajusta el modelo a los datos.
    reg.fit(data[[feature_col]], data[target_col])

    return reg

def make_prediction(model, value):
    """
    Realiza una predicción utilizando un modelo de regresión lineal.

    Args:
        model (sklearn.linear_model.LinearRegression): El modelo de regresión lineal previamente entrenado.
        value (float): El valor de la característica para el cual se desea hacer una predicción.

    Returns:
        numpy.ndarray: El resultado de la predicción.
    """
    # Realiza una predicción utilizando el modelo entrenado para el valor de la característica proporcionado.
    prediction = model.predict(np.array([[value]]))
    return prediction

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

def linear_regression(x, y):
    """
    Realiza una regresión lineal simple utilizando el método de los mínimos cuadrados.

    Parámetros:
    x (list/array-like): Lista o arreglo de datos independientes (predictores).
    y (list/array-like): Lista o arreglo de datos dependientes (objetivos).

    Retorna:
    tuple: Una tupla que contiene los siguientes valores:
        - a1: Coeficiente de regresión (pendiente de la recta de regresión).
        - a0: Término independiente (ordenada al origen de la recta de regresión).
        - Syx: Desviación estándar residual.
        - r2: Coeficiente de determinación (R^2).

    El coeficiente de regresión (a1) representa la pendiente de la recta de regresión.
    El término independiente (a0) representa la ordenada al origen de la recta de regresión.
    Syx es la desviación estándar residual que mide la dispersión de los datos respecto a la recta de regresión.
    r2 es el coeficiente de determinación, que indica cuánta variación en y es explicada por la regresión lineal.
    """

    sumx, sumy, sumxy, sumx2, Sr, St = 0, 0, 0, 0, 0, 0
    n = len(x)

    # Calcula sumatorias necesarias para el cálculo de a1 y a0.
    for i in range(n):
        sumx = sumx + x[i]
        sumy = sumy + y[i]
        sumxy = sumxy + x[i] * y[i]
        sumx2 = sumx2 + x[i] ** 2

    # Calcula los promedios de x y y.
    promx = sumx / n
    promy = sumy / n

    # Calcula los coeficientes a1 y a0 de la recta de regresión.
    a1 = (n * sumxy - sumx * sumy) / (n * sumx2 - sumx ** 2)
    a0 = promy - a1 * promx

    # Calcula las sumatorias necesarias para el cálculo de Syx y r2.
    for i in range(n):
        St = St + (y[i] - promy) ** 2
        Sr = Sr + (y[i] - a1 * x[i] - a0) ** 2

    # Calcula Syx (desviación estándar residual) y r2 (coeficiente de determinación).
    Syx = (Sr / (n - 2)) ** 0.5
    r2 = (St - Sr) / St

    # Retorna los coeficientes y medidas de calidad del modelo de regresión.
    return a1, a0, Syx, r2
# Define la variable 'path' que contiene la ubicación del archivo CSV que se va a cargar.
file_path = 'dataset_presion_arterial.csv'

# Entrada (edad)
value_to_predict = 20

# Carga los datos desde el archivo CSV en un DataFrame.
data = pd.read_csv(file_path, header=0)

# Muestra el contenido del DataFrame 'data' en la salida.
data

# Calcula estadísticas descriptivas para todas las columnas del DataFrame, incluyendo tanto las numéricas como las no numéricas.
data.describe()

# Crea el gráfico de dispersión.
create_scatter_plot(data, data.columns[0], data.columns[1])

# Entrena un modelo de regresión lineal usando 'feature_col' como característica y 'target_col' como objetivo.
feature_col = data.columns[0]
target_col = data.columns[1]
model = train_linear_regression(data, feature_col, target_col)

# Realiza una predicción para un valor específico de la columna característica
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    prediction = make_prediction(model, value_to_predict)

print('>>> Sistema entrenado')

# # Imprime los coeficientes del modelo.
# print("Coeficiente (m):", model.coef_)
# print("Ordenada al origen (b):", model.intercept_)

# Imprime el coeficiente de correlación entre dos variables y muestra un histograma de una de las variables.
print("Coeficiente de correlación ", relation_x_y(data[data.columns[0]], data[data.columns[1]]))

# Convierte la primera y segunda columna del DataFrame 'data' en arreglos NumPy.
x_data = np.array(data[data.columns[0]])
y_data = np.array(data[data.columns[1]])

# Llama a la función de regresión y desempaqueta los valores de a1, a0, Syx y r2.
a1, a0, Syx, r2 = linear_regression(x_data, y_data)

# Imprime el coeficiente de regresión (pendiente de la recta de regresión).
print("a1 : ", a1)

# Imprime el término independiente (ordenada al origen de la recta de regresión).
print("a0 : ", a0)

# Imprime la desviación estándar residual.
print("Syx : ", Syx)

# Imprime el coeficiente de determinación (R^2) que indica cuánta variación en Y es explicada por la regresión lineal.
print("r2 : ", r2)

# Crea un objeto polinómico de primer grado con los coeficientes a1 y a0.
polynomial = np.poly1d([a1, a0])

# Imprime la ecuación de la recta de regresión en formato matemático.
print("Ecuación de la recta de regresión: ", polynomial)
print()

# Imprime la ecuación de la recta de regresión en un formato más legible.
equation_str = f"y = {a1}x + {a0}"
print("Ecuación en formato legible:")
print(equation_str)

# Configura y muestra un gráfico de dispersión y una línea de regresión.

# Crea una figura para el gráfico con un tamaño de 7x7 pulgadas.
plt.figure(figsize=(7, 7))

# Dibuja los puntos de datos en el gráfico como círculos ("o") y etiqueta los puntos como "Datos".
plt.plot(x_data, y_data, "o", label="Datos")

# Dibuja la línea de regresión en el gráfico utilizando los coeficientes a1 y a0.
plt.plot(x_data, a1 * np.array(x_data) + a0, label="Polinomio")

# Etiqueta los ejes
plt.xlabel(data.columns[0])
plt.ylabel(data.columns[1])

# Agrega una cuadrícula al gráfico.
plt.grid()

# Agrega una leyenda en la esquina inferior derecha del gráfico utilizando la etiqueta "Datos" y la etiqueta del objeto polinómico.
plt.legend(loc=4)

# Muestra el gráfico en pantalla.
plt.show()

# Imprime la predicción
print("A los {} años se tendrá una presión sanguínea de {:.0f}".format(value_to_predict, *prediction))