import tkinter as tk
from tkinter import ttk, messagebox
import random as rd

# Función para generar la población inicial.
def getPoblacion(opcion1, opcion2):
    return [getIndividuo(opcion1, opcion2) for i in range(cantInterGen)]
# Cada individuo es una lista de valores aleatorios entre opcion1 y opcion2.
def getIndividuo(opcion1, opcion2):
    return [rd.choice([opcion1, opcion2]) for x in range(cantIndividuos)]

# Se suman usando informacionTab.
def calcularPesos(individuo):
    peso = 0
    for i in range(len(individuo)):
        if individuo[i] == 5:
            peso += informacionTab[i][0] # Suma el peso del elemento correspondiente
    return peso

# Se suman usando informacionTab.
def calcularValor(individuo):
    valor = 0
    for i in range(len(individuo)):
        if individuo[i] == 5:
            valor += informacionTab[i][1]
    return valor

# Función para seleccionar los individuos con menor peso.
def seleccionarMenor(lista):
    lista2 = []
    cont = 0
    for i in lista:
        if i[0] <= maxPeso: # Condición para que el peso sea menor o igual al máximo permitido
            if cont < Mejores: # Se seleccionan los 'Mejores' individuos
                lista2.append([i])
                cont += 1
    return lista2

# Cada individuo de la nueva generación toma una parte de los padres según un punto de cruce aleatorio.
def cruce(lPoblacion, lPadres):
    for i in range(len(lPoblacion) - 1):
        puntoCruce = rd.randint(0, cantIndividuos - 1)
        # El primer segmento del nuevo individuo proviene del primer padre
        lPoblacion[i][2][:puntoCruce] = lPadres[0][0][2][:puntoCruce]
        # El segundo segmento proviene del segundo padre
        lPoblacion[i][2][puntoCruce:] = lPadres[1][0][2][puntoCruce:]
    return lPoblacion

# Hay una probabilidad de mutar a un valor nuevo.
def mutacion(listaCruzados):
    m = []
    for i in range(len(listaCruzados)):
        datos = listaCruzados[i][2] # Obtener el individuo a mutar
        for j in range(len(datos)):
            if rd.random() <= probabilidadMutacion:
                nuevo_valor = rd.choice([5, 7]) # Se elige un nuevo valor para el gen
                while nuevo_valor == datos[j]:
                    nuevo_valor = rd.choice([5, 7]) # Asegurarse de que el nuevo valor sea diferente
                datos[j] = nuevo_valor # Agregar
        m.append(datos)
    return m

# Función para mostrar la matriz en una tabla en Tkinter
def mostrar_matriz(poblacion, text_widget):
    text_widget.delete(1.0, tk.END)
    text_widget.insert(tk.END, "--- Población inicial ---\n")
    for individuo in poblacion:
        fila = " | ".join(map(str, individuo)) + "\n"
        text_widget.insert(tk.END, fila)

# Función que ejecuta el algoritmo genético
def ejecutar_algoritmo():
    #declarar variables globalmente
    global cantIndividuos, cantInterGen, Mejores, probabilidadMutacion, maxPeso
    global informacionTab, listaPoblacion, iteraciones

    try:
        # Variables iniciales del algoritmo genético
        cantIndividuos = int(entry_individuos.get())
        cantInterGen = int(entry_generaciones.get())
        Mejores = int(entry_mejores.get())
        iteraciones = int(entry_iteraciones.get())
        probabilidadMutacion = float(entry_mutacion.get())
        maxPeso = int(entry_max_peso.get())

        # Información de la tabla de pesos y valores [peso, valor] por cada gen
        informacionTab = [[12, 4], [2, 2], [1, 2], [1, 1], [4, 10]]

        listaPoblacion = getPoblacion(5, 7)
        mostrar_matriz(listaPoblacion, text_output)
        for _ in range(iteraciones):
            # Calcular el peso y valor de cada individuo en la población
            listaValorados = [(calcularPesos(i), calcularValor(i), i) for i in listaPoblacion]
            # Ordenar los individuos por su valor (mayor valor primero)
            listaOrdenados = sorted(listaValorados, reverse=True)
            # Seleccionar los mejores individuos (padres)
            listaSeleccionados = seleccionarMenor(listaOrdenados)
            # Realizar el cruce para generar la nueva población
            listaPoblacionCruzada = cruce(listaValorados, listaSeleccionados)
            # Aplicar mutación a la nueva población cruzada
            listaPoblacion = mutacion(listaPoblacionCruzada)

        # Al final de las iteraciones, se evalúa la nueva población
        listaValores = [(calcularPesos(i), calcularValor(i), i) for i in listaPoblacion]
        listaOrdenados = sorted(listaValores, reverse=True)
        listaMejores = seleccionarMenor(listaOrdenados)

        # Mostrar el mejor individuo en la tabla
        text_output.insert(tk.END, f"\nMejor individuo:\n{listaMejores[0][0][2]}\n")
        text_output.insert(tk.END, f"Peso: {listaMejores[0][0][0]} kg\nValor: {listaMejores[0][0][1]}\n")
        
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error: {e}")

# Función para centrar la ventana en la pantalla
def centrar_ventana(root):
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')

def limpiar_campos():
    # Limpiar los campos de texto
    entry_individuos.delete(0, tk.END)
    entry_generaciones.delete(0, tk.END)
    entry_mejores.delete(0, tk.END)
    entry_iteraciones.delete(0, tk.END)
    entry_mutacion.delete(0, tk.END)
    entry_max_peso.delete(0, tk.END)
    text_output.delete(1.0, tk.END)  # Limpiar el área de salida de texto

# Configuración de la interfaz de Tkinter
root = tk.Tk()
root.title("Algoritmo Genético")
root.geometry("700x600")

# Usando ttk para mejorar el diseño
frame = ttk.Frame(root, padding="10")
frame.pack(expand=True)

# Widgets de entrada con ttk
ttk.Label(frame, text="Individuos por generación:").grid(row=0, column=0, padx=5, pady=5)
entry_individuos = ttk.Entry(frame)
entry_individuos.grid(row=0, column=1, padx=5, pady=5)

ttk.Label(frame, text="Cantidad de generaciones:").grid(row=1, column=0, padx=5, pady=5)
entry_generaciones = ttk.Entry(frame)
entry_generaciones.grid(row=1, column=1, padx=5, pady=5)

ttk.Label(frame, text="Mejores individuos:").grid(row=2, column=0, padx=5, pady=5)
entry_mejores = ttk.Entry(frame)
entry_mejores.grid(row=2, column=1, padx=5, pady=5)

ttk.Label(frame, text="Iteraciones:").grid(row=3, column=0, padx=5, pady=5)
entry_iteraciones = ttk.Entry(frame)
entry_iteraciones.grid(row=3, column=1, padx=5, pady=5)

ttk.Label(frame, text="Probabilidad de mutación:").grid(row=4, column=0, padx=5, pady=5)
entry_mutacion = ttk.Entry(frame)
entry_mutacion.grid(row=4, column=1, padx=5, pady=5)

ttk.Label(frame, text="Peso máximo:").grid(row=5, column=0, padx=5, pady=5)
entry_max_peso = ttk.Entry(frame)
entry_max_peso.grid(row=5, column=1, padx=5, pady=5)

# Área de salida de texto para mostrar la matriz
text_output = tk.Text(frame, height=15, width=50, wrap=tk.NONE)
text_output.grid(row=6, column=0, columnspan=2, padx=5, pady=5)

# Botón para ejecutar el algoritmo
ttk.Button(frame, text="Ejecutar", command=ejecutar_algoritmo).grid(row=7, column=0, columnspan=2, padx=5, pady=20)
# Botón para limpiar campos
ttk.Button(frame, text="Limpiar", command=limpiar_campos).grid(row=7, column=1, padx=5, pady=20)
# Centrar la ventana en la pantalla
centrar_ventana(root)

root.mainloop()
