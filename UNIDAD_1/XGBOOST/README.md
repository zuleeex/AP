# Extreme Gradient Boosting (XGBoost)

Este módulo contiene la implementación avanzada del algoritmo **XGBoost**, un modelo basado en ensambles de árboles de decisión optimizados mediante el descenso de gradiente (*Gradient Boosting*). Está enfocado en maximizar la capacidad predictiva tanto para tareas de clasificación (niveles de estrés) como de regresión (salarios).

---

## 📁 Estructura de la Carpeta

La carpeta está organizada con los códigos fuentes en R y Python, además de sus respectivos reportes y visualizaciones:

* **`INFORME DE XGBoost.pdf`**: Documento técnico principal que detalla el marco metodológico, la estrategia de particionamiento, la sintonización fina de hiperparámetros y el análisis comparativo final.
* **Análisis de Estrés**:
    * `XGBoost_Estres.pdf`: Reporte exportado con las gráficas de rendimiento y evaluación del caso de estrés.
    * `XGBoost_EstresAdolescentes.ipynb`: Cuaderno de Jupyter con el preprocesamiento de datos y desarrollo del modelo en Python.
    * `XGBoost_estres.R`: Script de código fuente en R utilizando optimización estocástica.
    * `XGBoost_estres_R.html`: Reporte interactivo dinámico (Knit) que permite inspeccionar todo el flujo de R y los gráficos analíticos directamente desde el navegador.
* **Análisis de Salarios**:
    * `XGBoost_Salarios.ipynb`: Cuaderno de Jupyter enfocado en la codificación de variables categóricas, entrenamiento y validación del modelo de ingresos en Python.
    * `XGBoost_Salarios.pdf`: Documento con los resultados visuales, curvas de aprendizaje y métricas del modelo de salarios.

---

## 🛠️ Hiperparámetros Clave Sintonizados

El algoritmo fue optimizado mediante las siguientes palancas de regularización y aprendizaje:
* **`eta` (Learning Rate)**: Reduce el peso de cada nuevo árbol para controlar el paso en el descenso de gradiente y evitar el sobreajuste.
* **`max_depth`**: Controla la profundidad máxima de los árboles base (árboles débiles o *weak learners*).
* **`subsample` & `colsample_bytree`**: Técnicas de submuestreo estocástico por filas y columnas para añadir aleatoriedad y robustez al ensamble.

---

## 🚀 Instrucciones de Revisión

1. **Lectura Rápida (.html)**: Puede abrir directamente el archivo `XGBoost_estres_R.html` en su navegador web favorito para revisar las salidas gráficas y el código estructurado en R sin ejecutar software adicional.
2. **Entorno R**: Ejecute `XGBoost_estres.R` en RStudio asegurándose de transformar primero el conjunto de datos a matrices optimizadas del tipo `xgb.DMatrix`.
3. **Entorno Python**: Suba los cuadernos `.ipynb` a Google Colab o ejecútelos localmente instalando la librería oficial de `xgboost` mediante el gestor de paquetes de Python.
