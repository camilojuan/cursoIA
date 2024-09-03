import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
from clips import Environment

# Crear el entorno CLIPS
env = Environment()

# Definir las plantillas en CLIPS
try:
    env.build("(deftemplate fallo (slot sintoma (type STRING)))")
    env.build("(deftemplate diagnostico (slot mensaje (type STRING)))")
except Exception as e:
    print(f"Error al cargar plantillas en CLIPS: {e}")

# Definir las reglas iniciales de diagnóstico en CLIPS
rules = [
    '(defrule rule1 (fallo (sintoma "pantalla-negra")) => (assert (diagnostico (mensaje "Posible fallo en la tarjeta gráfica o monitor desconectado"))))',
    '(defrule rule2 (fallo (sintoma "no-arranca")) => (assert (diagnostico (mensaje "Posible fallo en la fuente de alimentación o placa base"))))',
    '(defrule rule3 (fallo (sintoma "sonidos-extraños")) => (assert (diagnostico (mensaje "Posible fallo en el disco duro o ventiladores dañados"))))',
    '(defrule rule4 (fallo (sintoma "reinicio-repetitivo")) => (assert (diagnostico (mensaje "Posible sobrecalentamiento del procesador o fallo de memoria RAM"))))',
    '(defrule rule5 (fallo (sintoma "congelamiento")) => (assert (diagnostico (mensaje "Posible falta de memoria RAM o fallo en el sistema operativo"))))',
    '(defrule rule6 (fallo (sintoma "red-lenta")) => (assert (diagnostico (mensaje "Posible problema con la tarjeta de red o congestión en la red"))))',
    '(defrule rule7 (fallo (sintoma "pantalla-azul")) => (assert (diagnostico (mensaje "Posible fallo en el controlador de hardware o problema de hardware"))))'
]

# Cargar las reglas en CLIPS
for rule in rules:
    try:
        env.build(rule)
    except Exception as e:
        print(f"Error al cargar regla en CLIPS: {e}")

# Asignación de valores numéricos a cada síntoma para el sistema difuso
sintoma_valores = {
    "pantalla-negra": (3, 6),
    "no-arranca": (5, 8),
    "sonidos-extraños": (4, 7),
    "reinicio-repetitivo": (6, 7),
    "congelamiento": (5, 6),
    "red-lenta": (2, 4),
    "pantalla-azul": (7, 9)
}

# Definir las entradas difusas
fallas = ctrl.Antecedent(np.arange(0, 10, 1), 'Fallas')
gravedad = ctrl.Antecedent(np.arange(0, 10, 1), 'Gravedad')

# Definir la salida difusa
decision = ctrl.Consequent(np.arange(0, 100, 1), 'Decision')

# Definir las funciones de membresía para cada entrada
fallas['pocas'] = fuzz.trimf(fallas.universe, [0, 0, 4])
fallas['moderadas'] = fuzz.trimf(fallas.universe, [2, 5, 8])
fallas['muchas'] = fuzz.trimf(fallas.universe, [6, 9, 9])

gravedad['baja'] = fuzz.trimf(gravedad.universe, [0, 0, 4])
gravedad['media'] = fuzz.trimf(gravedad.universe, [2, 5, 8])
gravedad['alta'] = fuzz.trimf(gravedad.universe, [6, 9, 9])

decision['no_cambiar'] = fuzz.trimf(decision.universe, [0, 0, 40])
decision['considerar'] = fuzz.trimf(decision.universe, [30, 50, 70])
decision['cambiar'] = fuzz.trimf(decision.universe, [60, 100, 100])

# Crear las reglas difusas
rule1 = ctrl.Rule(fallas['pocas'] & gravedad['baja'], decision['no_cambiar'])
rule2 = ctrl.Rule(fallas['pocas'] & gravedad['media'], decision['considerar'])
rule3 = ctrl.Rule(fallas['pocas'] & gravedad['alta'], decision['considerar'])
rule4 = ctrl.Rule(fallas['moderadas'] & gravedad['baja'], decision['considerar'])
rule5 = ctrl.Rule(fallas['moderadas'] & gravedad['media'], decision['cambiar'])
rule6 = ctrl.Rule(fallas['moderadas'] & gravedad['alta'], decision['cambiar'])
rule7 = ctrl.Rule(fallas['muchas'] & gravedad['baja'], decision['considerar'])
rule8 = ctrl.Rule(fallas['muchas'] & gravedad['media'], decision['cambiar'])
rule9 = ctrl.Rule(fallas['muchas'] & gravedad['alta'], decision['cambiar'])

# Crear el sistema de control difuso
decision_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
sistema = ctrl.ControlSystemSimulation(decision_ctrl)

# Función para añadir síntomas al sistema experto
def agregar_fallo(sintoma):
    env.assert_string(f'(fallo (sintoma "{sintoma}"))')

# Función para obtener el diagnóstico del sistema experto
def obtener_diagnostico():
    env.run()
    diagnosticos = []
    for fact in env.facts():
        if fact.template.name == "diagnostico":
            diagnosticos.append(fact['mensaje'])
    return diagnosticos

# Función para manejar el clic del botón "Diagnosticar"
def diagnosticar():
    seleccionados = [var.get() for var in sintomas_vars if var.get()]
    if seleccionados:
        diagnosticos_completos = []
        for sintoma in seleccionados:
            agregar_fallo(sintoma)
            diagnosticos = obtener_diagnostico()
            diagnosticos_completos.extend(diagnosticos)

            # Asignar valores difusos
            if sintoma in sintoma_valores:
                num_fallas, nivel_gravedad = sintoma_valores[sintoma]

                # Asignar valores al sistema difuso
                sistema.input['Fallas'] = num_fallas
                sistema.input['Gravedad'] = nivel_gravedad

                # Computar la decisión
                sistema.compute()

                # Obtener el resultado difuso
                decision_final = sistema.output['Decision']
                
                conclusion = f'\nLa decisión con "{sintoma}" es: {decision_final:.2f}%\n'
                if decision_final <= 30:
                    conclusion += 'No es necesario cambiar el PC.'
                elif 30 < decision_final <= 60:
                    conclusion += 'Considera la posibilidad de cambiar el PC pronto.'
                else:
                    conclusion += 'Es recomendable cambiar el PC.'

                messagebox.showinfo("Decisión", conclusion)
                decision.view(sim=sistema)
                plt.show()
            else:
                messagebox.showwarning("Advertencia", f"El síntoma '{sintoma}' no tiene una regla asociada en el sistema difuso.")
        
        if diagnosticos_completos:
            resultado = "\n".join(diagnosticos_completos)
            messagebox.showinfo("Diagnóstico", resultado)
        else:
            messagebox.showwarning("Advertencia", "No se encontró diagnóstico para los síntomas proporcionados.")
        
        env.reset()
    else:
        messagebox.showwarning("Advertencia", "Por favor, seleccione al menos un síntoma.")

# Crear la ventana principal
root = tk.Tk()
root.title("Sistema Experto de Diagnóstico de Fallas de Computador")

# Crear la lista de síntomas y sus variables asociadas
sintomas = ["pantalla-negra", "no-arranca", "sonidos-extraños", "reinicio-repetitivo", "congelamiento", "red-lenta", "pantalla-azul"]
sintomas_vars = [tk.StringVar(value="") for _ in sintomas]

# Crear los checkboxes para cada síntoma
tk.Label(root, text="Seleccione los síntomas del computador:").pack(pady=10)
for sintoma, var in zip(sintomas, sintomas_vars):
    chk = tk.Checkbutton(root, text=sintoma.replace("-", " ").capitalize(), variable=var, onvalue=sintoma, offvalue="")
    chk.pack(anchor=tk.W)

# Crear el botón de diagnosticar
tk.Button(root, text="Diagnosticar", command=diagnosticar).pack(pady=20)

# Ejecutar la aplicación
root.mainloop()
