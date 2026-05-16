# Regresión Lineal Múltiple (RLM)

Este módulo contiene el desarrollo analítico y estadístico mediante modelos de **Regresión Lineal Múltiple**. A diferencia de otros módulos, este enfoque se centra en la optimización de ecuaciones analíticas, técnicas de validación como *Bootstrap* y métodos de selección de variables mediante regularización.

---

## 📁 Estructura de la Carpeta

* **`Informe Regresion Lineal Multiple.pdf`**: Documento técnico principal que detalla el marco teórico, el cumplimiento de supuestos estadísticos y la interpretación matemática de los coeficientes.
* **`datosestresyansiedad.xlsx`**: Conjunto de datos original en formato Excel utilizado como base para todo el modelado.
* **Cuadernos de Jupyter (Python)**:
    * `RLM_estres_ansiedad.ipynb`: Implementación inicial y ajuste básico del modelo lineal para evaluar el impacto de la ansiedad en el estrés.
    * `RLM_mejor_ecuacion_estres.ipynb`: Algoritmos de selección de características (*Feature Selection*) para identificar los predictores con mayor significancia estadística.
    * `Regularizacion_estres.ipynb`: Aplicación de técnicas de regularización (Ridge, Lasso o ElasticNet) para penalizar coeficientes y reducir el sobreajuste.
    * `RLM_Bootstrap.ipynb`: Implementación de técnicas de remuestreo (Bootstrapping) para validar la estabilidad de los parámetros de regresión de manera no paramétrica.

---

## 🔍 Supuestos Estadísticos Evaluados
Dentro de los cuadernos se incluye la validación matemática de:
1. **Linealidad** de los parámetros.
2. **Homocedasticidad** (análisis de varianza de los residuos).
3. **Normalidad** en la distribución de los errores.
4. **No Multicolinealidad** mediante la evaluación del Factor de Inflación de la Varianza (VIF).
