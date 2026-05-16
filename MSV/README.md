# Proyecto: Modelación Predictiva con Máquinas de Soporte Vectorial (SVM)

Este repositorio contiene el desarrollo práctico, scripts de código y los informes estadísticos correspondientes a la implementación del algoritmo **Support Vector Machines (SVM)** para tareas de clasificación y regresión.

---

## 📁 Estructura del Proyecto

El directorio principal está organizado en subcarpetas temáticas para facilitar la revisión del código y los entregables:

* **`INFORME COMPLETO MSV`**: Documento técnico principal que consolida el marco teórico, la metodología, el análisis de resultados y las conclusiones estadísticas globales.
* 📁 **`Trabajo_Salarios`**: Contiene la modelación predictiva aplicada al análisis de ingresos.
    * `svm_salarios.R`: Script de código fuente en R.
    * `svm_salarios.html` y carpeta `_files`: Reporte interactivo autogenerado (Knit) para visualización en navegador web.
    * `SVMSalarios.ipynb`: Cuaderno de Jupyter con la experimentación en Python.
    * `SVM_Salarios_Python.pdf`: Reporte exportado con las métricas y gráficos del modelo en Python.
* 📁 **`Trabajo_Estres`**: Contiene el análisis predictivo enfocado en salud mental y niveles de ansiedad.
    * `svm_estres_ansiedad.R`: Script de desarrollo en R.
    * `svm_estres_ansiedad.html` y carpeta `_files`: Reporte dinámico en formato HTML ejecutable desde el navegador.
    * `MSV_Estres.ipynb`: Cuaderno de Jupyter con la implementación del modelo en Python.
    * `MSV_Estres_Python.pdf`: Documento con los resultados visuales del análisis de estrés.

---

## 🛠️ Requisitos del Entorno

### 🐍 Entorno Python
* **Librerías principales**: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `jupyter`.
* **Instalación**:
    ```bash
    pip install scikit-learn pandas numpy matplotlib seaborn jupyter
    ```

### 📊 Entorno R
* **Paquetes requeridos**: `e1071` (o `kernlab`), `tidyverse`, `caret`, `rmarkdown`.
* **Instalación**:
    ```R
    install.packages(c("e1071", "tidyverse", "caret", "rmarkdown"))
    ```

---

## 🚀 Instrucciones de Ejecución

1.  **Reportes HTML**: Puede hacer doble clic sobre `svm_salarios.html` o `svm_estres_ansiedad.html` para inspeccionar el flujo de trabajo de R y sus respectivas gráficas directamente desde cualquier navegador, sin necesidad de ejecutar código.
2.  **Scripts de R**: Abra los archivos `.R` en RStudio, asegúrese de ajustar el directorio de trabajo a la ubicación de sus datos y ejecute las líneas de forma secuencial.
3.  **Cuadernos de Jupyter**: Cargue los archivos `.ipynb` en su entorno local (Jupyter Lab / VS Code) o en Google Colab para ejecutar las celdas del modelado en Python.
