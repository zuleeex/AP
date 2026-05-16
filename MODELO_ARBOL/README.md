# Modelos basados en Árboles de Decisión

Este módulo contiene la implementación, entrenamiento y análisis predictivo utilizando algoritmos de **Árboles de Decisión**, aplicados tanto a tareas de regresión como de clasificación para el análisis de niveles de estrés y salarios.

---

## 📁 Estructura de la Carpeta

* **`INFORME ARBOL DE REGRESION.pdf`**: Documento técnico principal que detalla el marco estadístico, las reglas de división obtenidas, el análisis de los nodos y la interpretación de los resultados.
* **Análisis de Estrés**:
    * `arbolestres.R`: Script de código fuente en R para el modelado y la visualización gráfica de las ramificaciones del árbol.
    * `arbolestres.ipynb`: Cuaderno de Jupyter con el preprocesamiento de datos y desarrollo del modelo en Python.
* **Análisis de Salarios**:
    * `arbolsalarios.R`: Script de desarrollo en R enfocado en la predicción/estimación del comportamiento de los ingresos.
    * `arbolsalarios.ipynb`: Cuaderno de Jupyter utilizando librerías de Python para el ajuste del árbol de regresión.

---

## 🛠️ Tecnologías y Librerías Utilizadas

### 📊 Entorno R
* **Paquetes Clave**: `rpart` (para la construcción del árbol), `rpart.plot` (para gráficos interactivos y estéticos de las ramificaciones) y `caret`.
* **Instalación**:
    ```R
    install.packages(c("rpart", "rpart.plot", "caret"))
    ```

### 🐍 Entorno Python
* **Librerías Clave**: `scikit-learn` (`DecisionTreeRegressor` / `DecisionTreeClassifier`), `pandas`, `numpy` y `matplotlib`.
* **Instalación**:
    ```bash
    pip install scikit-learn pandas numpy matplotlib
    ```

---

## 🚀 Instrucciones de Ejecución

1. **En RStudio**: Abra los archivos `arbolestres.R` o `arbolsalarios.R` para examinar el comportamiento del particionamiento recursivo. Use la librería `rpart.plot` integrada en los scripts para visualizar cómo se dividen las variables.
2. **En Jupyter / Python**: Cargue `arbolestres.ipynb` o `arbolsalarios.ipynb` en VS Code o Google Colab para evaluar las métricas de rendimiento del modelo e inspeccionar la profundidad óptima del árbol.
