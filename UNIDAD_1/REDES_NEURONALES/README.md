# Redes Neuronales Artificiales (RNA)

Este módulo contiene la implementación de modelos de **Redes Neuronales Artificiales (Artificial Neural Networks - ANN)** aplicados a tareas de **regresión supervisada**, utilizando arquitecturas profundas desarrolladas con TensorFlow/Keras.  

El objetivo principal es comparar el rendimiento predictivo en dos escenarios distintos:

- Predicción del **nivel de estrés en adolescentes**
- Predicción de **salarios en el sector tecnológico**

---

# 📁 Estructura de la Carpeta

La carpeta contiene los notebooks de desarrollo, además de los informes exportados en PDF con resultados, métricas y visualizaciones.

## Análisis de Estrés

### `red_estres.ipynb`
Cuaderno de Jupyter que incluye:

- limpieza y preprocesamiento de datos,
- normalización de variables,
- construcción de la red neuronal,
- entrenamiento,
- validación y evaluación del modelo.

### `red_estres.pdf`
Informe exportado con:

- métricas de desempeño,
- curvas de pérdida,
- gráficos comparativos,
- análisis predictivo del modelo de estrés adolescente.

---

## Análisis de Salarios

### `red_salarios.ipynb`
Notebook enfocado en:

- codificación de variables categóricas,
- escalamiento de datos,
- diseño de la arquitectura neuronal,
- entrenamiento profundo para regresión salarial.

### `red_salarios.pdf`
Documento con:

- resultados visuales,
- análisis estadístico,
- comparación entre valores reales y predichos,
- evaluación del desempeño del modelo.

---

# 🧠 Arquitectura del Modelo

Los modelos fueron implementados utilizando redes neuronales densas (*Fully Connected Networks*) con aprendizaje supervisado.

## Componentes principales utilizados

- **Capas densas (Dense Layers)**  
  Encargadas de aprender relaciones no lineales complejas entre variables.

- **Función de activación ReLU**  
  Introducida para mejorar la capacidad de aprendizaje profundo y evitar problemas de gradientes pequeños.

- **Capa de salida lineal**  
  Utilizada para tareas de regresión continua.

- **Optimizador Adam**  
  Algoritmo adaptativo de descenso de gradiente ampliamente utilizado en Deep Learning.

- **Función de pérdida MSE (Mean Squared Error)**  
  Empleada para minimizar el error cuadrático medio entre valores reales y predichos.

---

# ⚙️ Tecnologías Utilizadas

- Python 3.10
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- scikit-learn

---

# 📊 Variables Analizadas

## Dataset de Estrés Adolescente

Incluye variables relacionadas con:

- horas de sueño,
- uso de redes sociales,
- actividad física,
- rendimiento académico,
- hábitos diarios,
- nivel de ansiedad y estrés.

---

## Dataset de Salarios Tech

Incluye atributos como:

- experiencia laboral,
- país,
- nivel educativo,
- tipo de empleo,
- tecnologías utilizadas,
- modalidad de trabajo,
- salario anual.


# 📈 Resultados Esperados

Los modelos permiten:

- analizar relaciones no lineales complejas,
- realizar predicciones continuas mediante Deep Learning,
- comparar desempeño entre distintos conjuntos de datos,
- evaluar capacidad de generalización mediante métricas de regresión.