#=====================================================
# ÁRBOLES DE REGRESIÓN — Dataset: Salarios Globales
# Variable respuesta: salario (numérica continua)
#=====================================================

# ── 1. LIBRERÍAS ─────────────────────────────────────────────────────────────
library(dplyr)       # Manipulación de datos
library(tidyr)       # Limpieza y transformación
library(ggplot2)     # Visualizaciones
library(ggpubr)      # Gráficos publicables
library(skimr)       # Estadística descriptiva completa
library(readxl)      # Importar archivos Excel
library(tree)        # Árbol base (paquete 'tree')
library(rpart)       # Árbol alternativo más flexible
library(rpart.plot)  # Visualización de árboles rpart
library(MLmetrics)   # Métricas de evaluación (RMSE, MSE)

# ── 2. IMPORTAR Y CONOCER LOS DATOS ──────────────────────────────────────────
salarios <- read_excel("salarios.xlsx", sheet = "salarios")
head(salarios)
dim(salarios)   # 250,000 filas × 10 columnas

# ── 3. PREPARACIÓN DE VARIABLES ───────────────────────────────────────────────
# Convertir variables categóricas a factor
salarios$titulo_profesional <- as.factor(salarios$titulo_profesional)
salarios$nivel_educativo    <- as.factor(salarios$nivel_educativo)
salarios$industria          <- as.factor(salarios$industria)
salarios$tamaño_empresa     <- as.factor(salarios$tamaño_empresa)
salarios$ubicación          <- as.factor(salarios$ubicación)
salarios$trabajo_remoto     <- as.factor(salarios$trabajo_remoto)

str(salarios)

# ── 4. ESTADÍSTICA DESCRIPTIVA ───────────────────────────────────────────────
skim(salarios)
summary(salarios)

# ── 5. MUESTRA DE TRABAJO (para agilizar el árbol 'tree') ─────────────────────
# Con 250,000 filas el paquete tree puede ser lento;
# usamos una muestra estratificada de 10,000 obs. para exploración rápida.
set.seed(123)
idx_muestra <- sample(1:nrow(salarios), size = 10000)
salarios_muestra <- salarios[idx_muestra, ]

# ── 6. DIVISIÓN: TRAIN (80%) y TEST (20%) ─────────────────────────────────────
set.seed(123)
train      <- sample(1:nrow(salarios_muestra), size = nrow(salarios_muestra) * 0.8)
data_train <- salarios_muestra[train, ]
data_test  <- salarios_muestra[-train, ]
dim(data_train)  # 8,000 filas
dim(data_test)   # 2,000 filas

# ── 7. ÁRBOL DE REGRESIÓN INICIAL (valores por defecto) ───────────────────────
set.seed(123)
arbol_regresion1 <- tree(
  formula = salario ~ .,
  data    = data_train,
  split   = "deviance"   # Minimiza el RSS en cada división
)

summary(arbol_regresion1)
arbol_regresion1

# Visualización
par(mfrow = c(1, 1))
plot(x = arbol_regresion1, type = "proportional")
text(x = arbol_regresion1, splits = TRUE, pretty = 0,
     cex = 0.80, col = "firebrick")
title(main = "Árbol de Regresión Inicial — Salarios")

# Correlación en datos de entrenamiento
y_hat <- predict(object = arbol_regresion1, newdata = data_train)
cor(y_hat, data_train$salario)

# RMSE en datos de prueba
predicciones <- predict(arbol_regresion1, newdata = data_test)
test_rmse    <- sqrt(mean((predicciones - data_test$salario)^2))
paste("Error de test del árbol inicial (RMSE):", round(test_rmse, 2))

# ── 8. PREDICCIÓN CON NUEVOS DATOS ───────────────────────────────────────────
nuevos_datos <- data.frame(
  titulo_profesional = factor("Data Analyst",
                              levels = levels(salarios$titulo_profesional)),
  anios_experiencia  = 5,
  nivel_educativo    = factor("Master",
                              levels = levels(salarios$nivel_educativo)),
  numero_habilidades = 8,
  industria          = factor("Healthcare",
                              levels = levels(salarios$industria)),
  tamaño_empresa     = factor("Large",
                              levels = levels(salarios$tamaño_empresa)),
  ubicación          = factor("USA",
                              levels = levels(salarios$ubicación)),
  trabajo_remoto     = factor("Yes",
                              levels = levels(salarios$trabajo_remoto)),
  certificaciones    = 2
)

prediccion_nueva <- predict(arbol_regresion1, newdata = nuevos_datos)
paste("Salario predicho:", round(prediccion_nueva, 2))

# ── 9. ÁRBOL CONTROLADO (parámetros personalizados) ───────────────────────────
set.seed(123)
arbol_regresion <- tree(
  formula = salario ~ .,
  data    = data_train,
  split   = "deviance",
  mincut  = 5,    # mín. obs. en cada nodo hijo
  minsize = 10,   # mín. obs. para que un nodo se divida
  mindev  = 0.01  # umbral de devianza para nueva división
)

summary(arbol_regresion)
arbol_regresion

# ── 10. VISUALIZACIONES DEL ÁRBOL CONTROLADO ──────────────────────────────────
dev.off()  # limpiar dispositivo gráfico
par(mar = c(4, 4, 2, 1))

# Versión proporcional (ramas ∝ devianza)
plot(x = arbol_regresion, type = "proportional")
text(x = arbol_regresion, splits = TRUE, all = TRUE,
     pretty = 0, cex = 0.75, col = "firebrick")
title(main = "Árbol Controlado — Ramas Proporcionales")

# Versión uniforme (más limpia)
plot(x = arbol_regresion, type = "uniform")
text(x = arbol_regresion, splits = TRUE, all = FALSE,
     pretty = 0, cex = 0.80, col = "firebrick")
title(main = "Árbol Controlado — Ramas Uniformes")

# ── 11. PODA POR COSTE-COMPLEJIDAD (Cross-Validation) ─────────────────────────
set.seed(123)
cv_arbol <- cv.tree(
  object = arbol_regresion,
  FUN    = prune.tree,   # podado para regresión
  K      = 10            # 10-fold Cross-Validation
)

cv_arbol

# Curva de error por tamaño del árbol
plot(
  x    = cv_arbol$size,
  y    = cv_arbol$dev,
  type = "b",
  pch  = 19,
  col  = "steelblue",
  xlab = "Número de nodos terminales",
  ylab = "Devianza (Error CV)",
  main = "Selección del Tamaño Óptimo del Árbol — Salarios"
)

# Tamaño óptimo
size_optimo <- rev(cv_arbol$size)[which.min(rev(cv_arbol$dev))]
paste("Tamaño óptimo de nodos terminales:", size_optimo)

# ── 12. ÁRBOL PODADO ──────────────────────────────────────────────────────────
arbol_podado <- prune.tree(
  tree = arbol_regresion,
  best = size_optimo
)

# Visualización del árbol podado
plot(x = arbol_podado, type = "uniform")
text(x = arbol_podado, splits = TRUE, all = TRUE,
     pretty = 0, cex = 0.80, col = "firebrick")
title(main = "Árbol de Regresión Podado — Salarios")

# ── 13. COMPARACIÓN: ÁRBOL ORIGINAL vs PODADO ────────────────────────────────
pred_original <- predict(arbol_regresion, newdata = data_test)
pred_podado   <- predict(arbol_podado,    newdata = data_test)

rmse <- function(real, pred) sqrt(mean((real - pred)^2))

paste("RMSE Árbol Original:", round(rmse(data_test$salario, pred_original), 2))
paste("RMSE Árbol Podado:  ", round(rmse(data_test$salario, pred_podado),   2))

# ── 14. ÁRBOL CON rpart (sobre muestra completa) ──────────────────────────────
# rpart es más eficiente y puede trabajar con los 250,000 registros
set.seed(123)
train_full <- sample(1:nrow(salarios), size = nrow(salarios) * 0.8)

m1 <- rpart(
  formula = salario ~ .,
  data    = salarios,
  subset  = train_full,
  method  = "anova"
)
m1

# Visualización rpart
rpart.plot(m1, main = "Árbol rpart — Salarios (Modelo Base)")

# Curva de complejidad-error
plotcp(m1)

# ── 15. MODELO rpart SIN PODA (exploración completa) ─────────────────────────
m2 <- rpart(
  formula = salario ~ .,
  data    = salarios,
  subset  = train_full,
  method  = "anova",
  control = list(cp = 0, xval = 10)
)
plotcp(m2)
abline(v = 12, lty = "dashed")

# Tabla de CP
m1$cptable

# ── 16. MODELO rpart ÓPTIMO ────────────────────────────────────────────────────
optimal_tree <- rpart(
  formula = salario ~ .,
  data    = salarios,
  subset  = train_full,
  method  = "anova",
  control = list(minsplit = 6, maxdepth = 7, cp = 0.01)
)

# RMSE del modelo óptimo
pred_opt <- predict(optimal_tree, newdata = salarios[train_full, ])
rmse_gof <- MSE(y_pred = pred_opt,
                y_true  = salarios$salario[train_full])^(1/2)
paste("RMSE Modelo rpart Óptimo (train):", round(rmse_gof, 2))

# Visualización final
rpart.plot(
  optimal_tree,
  main  = "Árbol de Regresión Óptimo — Salario",
  extra = 101   # N observaciones + valor medio en cada hoja
)