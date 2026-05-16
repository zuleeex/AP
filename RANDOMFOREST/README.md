# Random Forest (Bosques Aleatorios)

Este módulo contiene la implementación del algoritmo de ensamble **Random Forest**, utilizando técnicas de *Bagging* (Bootstrap Aggregating) para reducir la varianza de los árboles individuales. El modelo está entrenado para predecir y clasificar niveles de estrés y ansiedad.

---

## 📁 Estructura de la Carpeta

* **`INFORME SOBRE RANDOM FOREST.pdf`**: Documento técnico principal que detalla la metodología, el análisis del error Out-Of-Bag (OOB) y las conclusiones estadísticas.
* **`datosestresyansiedad.xlsx`**: Matriz de datos original en formato Excel utilizada para el análisis y entrenamiento de los modelos.
* **Scripts de Código**:
    * `RandomForest_Estres.R`: Código fuente en R con la sintonización de los parámetros `ntree` y `mtry`.
    * `randomforest_estres.html`: Reporte interactivo autogenerado (Knit) para visualizar el flujo del análisis desde el navegador.
    * `RandomForest_Estres.ipynb`: Cuaderno de Jupyter con el ciclo del modelado desarrollado en Python.
    * `RandomForest_Estres_Python.pdf`: Reporte exportado con los resultados y métricas del modelo ajustado en Python.

---

## 📊 Gráficos Analíticos e Interpretación
Los siguientes recursos visuales (incluidos en la carpeta) corresponden a las salidas clave del modelo:

* **`01_eda_completo.png`**: Análisis Exploratorio de Datos (EDA) que muestra la distribución de las variables y correlaciones.
* **`02_resultados_completos.png`**: Comparativa global del rendimiento del modelo frente a los datos de prueba.
* **`03_prediccion_individual.png`**: Evaluación detallada del comportamiento de las predicciones en casos puntuales.
* **`04_prediccion_escenarios.png`**: Simulaciones del nivel de estrés variando las condiciones del entorno.
* **`05_sensibilidad.png`**: Gráfico de importancia de variables, identificando los factores que más influyen en el estrés y la ansiedad (basado en la pureza de Gini / IncNodePurity).

---

## 🛠️ Tecnologías Utilizadas

* **R**: Paquetes `randomForest`, `caret` y `tidyverse`.
* **Python**: Librerías `scikit-learn` (`RandomForestClassifier` / `RandomForestRegressor`), `pandas` y `matplotlib`.
