#=====================================================
# ÁRBOLES DE REGRESIÓN — Dataset: Estrés en Adolescentes
# Variable respuesta: stress_level (1–10)
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
estres <- read_excel("estres.xlsx")
head(estres)
dim(estres)  # 1200 filas × 13 columnas

# ── 3. PREPARACIÓN DE VARIABLES ───────────────────────────────────────────────
# Convertir variables categóricas a factor
estres$gender                 <- as.factor(estres$gender)
estres$platform_usage         <- as.factor(estres$platform_usage)
estres$social_interaction_level <- as.factor(estres$social_interaction_level)
estres$depression_label       <- as.factor(estres$depression_label)

str(estres)

# ── 4. ESTADÍSTICA DESCRIPTIVA ───────────────────────────────────────────────
skim(estres)
summary(estres)

# ── 5. DIVISIÓN: TRAIN (80%) y TEST (20%) ─────────────────────────────────────
set.seed(123)
train     <- sample(1:nrow(estres), size = nrow(estres) * 0.8)
data_train <- estres[train, ]
data_test  <- estres[-train, ]
dim(data_train)  # 960 filas
dim(data_test)   #  240 filas

# ── 6. ÁRBOL DE REGRESIÓN INICIAL (valores por defecto) ───────────────────────
set.seed(123)
arbol_regresion1 <- tree(
  formula = stress_level ~ .,
  data    = data_train,
  split   = "deviance"  # Minimiza el RSS en cada división
)

summary(arbol_regresion1)
arbol_regresion1

# Visualización del árbol inicial
par(mfrow = c(1, 1))
plot(x = arbol_regresion1, type = "proportional")
text(x = arbol_regresion1, splits = TRUE, pretty = 0,
     cex = 0.80, col = "firebrick")
title(main = "Árbol de Regresión Inicial — Estrés")

# Correlación en datos de entrenamiento
y_hat <- predict(object = arbol_regresion1, newdata = data_train)
cor(y_hat, data_train$stress_level)

# RMSE en datos de prueba
predicciones <- predict(arbol_regresion1, newdata = data_test)
test_rmse    <- sqrt(mean((predicciones - data_test$stress_level)^2))
paste("Error de test del árbol inicial (RMSE):", round(test_rmse, 2))

# ── 7. PREDICCIÓN CON NUEVOS DATOS ───────────────────────────────────────────
nuevos_datos <- data.frame(
  age                      = 17,
  gender                   = factor("female", levels = c("female", "male")),
  daily_social_media_hours = 5.0,
  platform_usage           = factor("TikTok", levels = c("Both", "Instagram", "TikTok")),
  sleep_hours              = 6.5,
  screen_time_before_sleep = 2.0,
  academic_performance     = 3.0,
  physical_activity        = 1.0,
  social_interaction_level = factor("medium", levels = c("high", "low", "medium")),
  anxiety_level            = 6,
  addiction_level          = 5,
  depression_label         = factor("0", levels = c("0", "1"))
)

prediccion_nueva <- predict(arbol_regresion1, newdata = nuevos_datos)
paste("Nivel de estrés predicho:", round(prediccion_nueva, 2))

# ── 8. ÁRBOL CONTROLADO (parámetros personalizados) ───────────────────────────
set.seed(123)
arbol_regresion <- tree(
  formula = stress_level ~ .,
  data    = data_train,
  split   = "deviance",
  mincut  = 5,    # mín. obs. en cada nodo hijo
  minsize = 10,   # mín. obs. para que un nodo se divida
  mindev  = 0.01  # umbral de devianza para nueva división
)

summary(arbol_regresion)
arbol_regresion

# ── 9. VISUALIZACIONES DEL ÁRBOL CONTROLADO ───────────────────────────────────
dev.off()  # limpiar dispositivo gráfico (ejecutar 1 o 2 veces si hace falta)
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

# ── 10. PODA POR COSTE-COMPLEJIDAD (Cross-Validation) ─────────────────────────
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
  main = "Selección del Tamaño Óptimo del Árbol — Estrés"
)

# Tamaño óptimo
size_optimo <- rev(cv_arbol$size)[which.min(rev(cv_arbol$dev))]
paste("Tamaño óptimo de nodos terminales:", size_optimo)

# ── 11. ÁRBOL PODADO ──────────────────────────────────────────────────────────
arbol_podado <- prune.tree(
  tree = arbol_regresion,
  best = size_optimo   # usar el tamaño óptimo hallado por CV
)

# Visualización del árbol podado
plot(x = arbol_podado, type = "uniform")
text(x = arbol_podado, splits = TRUE, all = TRUE,
     pretty = 0, cex = 0.80, col = "firebrick")
title(main = "Árbol de Regresión Podado — Estrés")

# ── 12. COMPARACIÓN: ÁRBOL ORIGINAL vs PODADO ────────────────────────────────
pred_original <- predict(arbol_regresion, newdata = data_test)
pred_podado   <- predict(arbol_podado,    newdata = data_test)

rmse <- function(real, pred) sqrt(mean((real - pred)^2))

paste("RMSE Árbol Original:", round(rmse(data_test$stress_level, pred_original), 4))
paste("RMSE Árbol Podado:  ", round(rmse(data_test$stress_level, pred_podado), 4))

# ── 13. ÁRBOL CON rpart (más flexible y visual) ────────────────────────────────
m1 <- rpart(
  formula = stress_level ~ .,
  data    = estres,
  subset  = train,
  method  = "anova"   # regresión
)
m1

# Visualización del árbol rpart
rpart.plot(m1, main = "Árbol rpart — Estrés (Modelo Base)")

# Curva de complejidad-error
plotcp(m1)

# ── 14. MODELO rpart SIN PODA (exploración completa) ─────────────────────────
m2 <- rpart(
  formula = stress_level ~ .,
  data    = estres,
  subset  = train,
  method  = "anova",
  control = list(cp = 0, xval = 10)
)
plotcp(m2)
abline(v = 12, lty = "dashed")

# Tabla de CP para elegir el mejor parámetro
m1$cptable

# ── 15. MODELO rpart ÓPTIMO ────────────────────────────────────────────────────
optimal_tree <- rpart(
  formula = stress_level ~ .,
  data    = estres,
  subset  = train,
  method  = "anova",
  control = list(minsplit = 6, maxdepth = 7, cp = 0.01)
)

# RMSE del modelo óptimo
pred_opt  <- predict(optimal_tree, newdata = estres[train, ])
rmse_gof  <- MSE(y_pred = pred_opt, y_true = estres$stress_level[train])^(1/2)
paste("RMSE Modelo rpart Óptimo (train):", round(rmse_gof, 4))

# Visualización final
rpart.plot(
  optimal_tree,
  main   = "Árbol de Regresión Óptimo — Nivel de Estrés",
  extra  = 101,  # muestra N observaciones y valor medio en cada hoja
  col    = "firebrick"
)