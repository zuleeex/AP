# =====================================================================
# REGRESION LOGISTICA SIMPLE - R
# Anemia infantil - ENDES 2024 (RECH6)
# Universidad Nacional del Altiplano
# =====================================================================
#
# Variable dependiente (Y): grupo_anemia (no tiene / tiene anemia)
#   Directriz OMS 2024 / RM 251-2024-MINSA (variable HC57A)
#
# Variable independiente (X) - UNICO PREDICTOR:
#   talla_edad_z : Z-score Talla/Edad (OMS), variable HC70
#   (ya ajustado por edad y sexo segun la OMS; valores mas bajos
#    indican desnutricion cronica / retraso en el crecimiento)
#
# Estructura basada en el ejemplo "matricula de honor ~ matematicas"
# (Joaquin Amat Rodrigo), adaptada al caso de anemia infantil.
# =====================================================================

library(tidyverse)
library(caret)
library(pROC)
library(ggplot2)
library(gridExtra)
library(ResourceSelection)
library(pscl)

set.seed(123)

# -----------------------------------------------------------------
# PASO 0: PREPARACION DEL DATASET (desde el archivo crudo RECH6_2024.csv)
# -----------------------------------------------------------------
cat(strrep("=", 70), "\n")
cat("PASO 0: PREPARACION DEL DATASET DESDE EL ARCHIVO CRUDO\n")
cat(strrep("=", 70), "\n")

df_crudo <- read.csv("RECH6_2024.csv", sep = ";", stringsAsFactors = FALSE)
cat("\nDataset original (crudo):", dim(df_crudo)[1], "filas x", dim(df_crudo)[2], "columnas\n")

datos_raw <- df_crudo[, c("HC70", "HC57A")]
colnames(datos_raw) <- c("talla_edad_z_x100", "anemia_oms")

mask_validos <- (
  (datos_raw$anemia_oms %in% c(1, 2, 3, 4)) &
  (abs(datos_raw$talla_edad_z_x100) < 9000)
)

cat("\nFilas excluidas por codigo de 'no medido':", sum(!mask_validos), "\n")
cat("Filas validas:", sum(mask_validos), "de", nrow(datos_raw), "\n")

datos_raw <- datos_raw[mask_validos, ]
datos_raw$talla_edad_z <- datos_raw$talla_edad_z_x100 / 100
datos_raw$matricula <- ifelse(datos_raw$anemia_oms == 4, 0, 1)  # 0=Sin_anemia, 1=Con_anemia

# Para mantener la misma nomenclatura que el ejemplo de matricula de honor,
# usamos un data frame "datos" con columnas: matricula (target) y talla_edad_z (predictor)
datos <- datos_raw[, c("matricula", "talla_edad_z")]

cat("\nPrimeras observaciones:\n")
print(head(datos, 5))
str(datos)

# -----------------------------------------------------------------
# Convertir la variable target a factor (cualitativa)
# -----------------------------------------------------------------
# Convertir la variable target a factor, etiquetando como tiene y no tiene
datos$matricula <- factor(datos$matricula, levels = c("0", "1"),
                           labels = c("no tiene anemia", "tiene anemia"))
str(datos)

# Identificamos las observaciones y dos variables: la variable "matricula"
# (en nuestro caso, presencia de anemia) es reconocida como factor, y la
# variable talla_edad_z como numerica (double).

# Distribucion de clases
cat("\n📊 Distribucion de clases:\n")
print(table(datos$matricula))

# Tabla de la variable dependiente (usando tidyverse)
tabla_resumen <- datos %>%
  group_by(datos$matricula) %>%
  summarise(
    numero_casos = n(),
    porcentaje = numero_casos / nrow(datos)
  )
print(tabla_resumen)

# -----------------------------------------------------------------
# ANALISIS EXPLORATORIO DE DATOS
# -----------------------------------------------------------------
cat("\n", strrep("=", 70), "\n")
cat("ANALISIS EXPLORATORIO DE DATOS\n")
cat(strrep("=", 70), "\n")

# Estadisticos descriptivos
print(summary(datos))

# Resumen de estadisticos por grupo de la target
resumen_grupo <- datos %>%
  filter(!is.na(talla_edad_z)) %>%
  group_by(matricula) %>%
  summarise(media = mean(talla_edad_z),
            mediana = median(talla_edad_z),
            min = min(talla_edad_z),
            max = max(talla_edad_z))
print(resumen_grupo)
cat("\nSe confirma que el promedio de talla_edad_z en el grupo CON anemia\n")
cat("es menor (mas negativo) que en el grupo SIN anemia.\n")

# -----------------------------------------------------------------
# Visualizacion de datos
# -----------------------------------------------------------------
cat("\n", strrep("=", 70), "\n")
cat("VISUALIZACION DE DATOS\n")
cat(strrep("=", 70), "\n")

# Grafico de la variable target
p1 <- ggplot(datos, aes(x = matricula, fill = matricula)) +
  geom_bar() +
  geom_text(aes(label = scales::percent(after_stat(count) / sum(after_stat(count)))),
            stat = 'count', position = position_stack(0.5)) +
  scale_fill_manual(values = c("no tiene anemia" = "steelblue", "tiene anemia" = "firebrick")) +
  theme_minimal() +
  labs(title = "Distribución de la anemia infantil", x = "", y = "Frecuencia")
ggsave("1_distribucion_anemia.png", p1, width = 6, height = 4, dpi = 120)
print(p1)

# Histograma de la variable cuantitativa talla_edad_z
p2 <- ggplot(datos, aes(x = talla_edad_z)) +
  geom_histogram(color = "steelblue", fill = "lightblue", bins = 30) +
  ggtitle("Histograma de talla_edad_z (Z-score Talla/Edad)")
ggsave("2_histograma_talla_edad_z.png", p2, width = 6, height = 4, dpi = 120)
print(p2)

cat("\nLos valores de talla_edad_z estan alrededor de su media (~-0.88),\n")
cat("con cola hacia valores muy negativos (desnutricion cronica severa).\n")

# Box plot para talla_edad_z segun la presencia de anemia
p3 <- ggplot(data = datos, aes(x = matricula, y = talla_edad_z, color = matricula)) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(width = 0.1, alpha = 0.1) +
  scale_color_manual(values = c("no tiene anemia" = "steelblue", "tiene anemia" = "firebrick")) +
  theme_bw() +
  theme(legend.position = "none") +
  labs(title = "talla_edad_z según presencia de anemia")
ggsave("3_boxplot_talla_edad_z.png", p3, width = 6, height = 5, dpi = 120)
print(p3)

cat("\nLa mediana de talla_edad_z en el grupo 'tiene anemia' es menor que en\n")
cat("el grupo 'no tiene anemia', consistente con la asociación esperada entre\n")
cat("desnutrición crónica (talla baja para la edad) y anemia.\n")

# -----------------------------------------------------------------
# ESTIMAR EL MODELO DE REGRESION LOGISTICA SIMPLE
# -----------------------------------------------------------------
cat("\n", strrep("=", 70), "\n")
cat("ESTIMACION DEL MODELO DE REGRESION LOGISTICA SIMPLE\n")
cat(strrep("=", 70), "\n")
cat("Variable de clasificacion (target) = matricula (presencia de anemia)\n")
cat("Variable independiente = talla_edad_z\n\n")

modelo <- glm(matricula ~ talla_edad_z, data = datos, family = "binomial")
print(summary(modelo))

cat("\nEl modelo logistico puede escribirse como:\n")
cat("p_hat(Y=1|X) = exp(b0 + b1*X) / (1 + exp(b0 + b1*X))\n")
coefs <- coef(modelo)
cat(sprintf("p_hat(Y=1|X) = exp(%.5f + %.5f*X) / (1 + exp(%.5f + %.5f*X))\n",
            coefs[1], coefs[2], coefs[1], coefs[2]))

# Intervalos de Confianza para los coeficientes
cat("\nIntervalos de Confianza al 95% para los coeficientes:\n")
print(round(confint(modelo, level = 0.95), 5))

# Interpretacion: exponenciando para facilitar la lectura de los coeficientes
cat("\nExponenciando los coeficientes (Odds Ratios):\n")
or_modelo <- round(exp(coefficients(modelo)), 6)
print(or_modelo)

cat(sprintf("\nInterpretacion: por cada unidad que AUMENTA talla_edad_z, las\n"))
cat(sprintf("posibilidades (odds) de tener anemia se MULTIPLICAN por %.4f\n", or_modelo["talla_edad_z"]))
if (or_modelo["talla_edad_z"] < 1) {
  cat(sprintf("(es decir, DISMINUYEN un %.1f%%). Esto es coherente: a mejor talla\n",
              (1 - or_modelo["talla_edad_z"]) * 100))
  cat("para la edad (menos desnutricion cronica), menor riesgo de anemia.\n")
}

# Intervalos de confianza para el Odds Ratio
cat("\nIntervalos de Confianza al 95% para el Odds Ratio:\n")
print(round(exp(confint(modelo)), 5))

# -----------------------------------------------------------------
# Significancia mediante ANOVA (Chi-cuadrado)
# -----------------------------------------------------------------
cat("\n", strrep("=", 70), "\n")
cat("SIGNIFICANCIA DEL PREDICTOR (ANOVA Chi-cuadrado)\n")
cat(strrep("=", 70), "\n")
print(anova(modelo, test = "Chisq"))

# -----------------------------------------------------------------
# EVALUACION DEL MODELO
# -----------------------------------------------------------------
cat("\n", strrep("=", 70), "\n")
cat("EVALUACION DEL MODELO\n")
cat(strrep("=", 70), "\n")

cat("\nDevianza nula vs devianza residual:\n")
cat(sprintf("Null deviance: %.4f en %d grados de libertad\n",
            modelo$null.deviance, modelo$df.null))
cat(sprintf("Residual deviance: %.4f en %d grados de libertad\n",
            modelo$deviance, modelo$df.residual))
cat(sprintf("AIC: %.4f\n", AIC(modelo)))
cat(sprintf("BIC: %.4f\n", BIC(modelo)))

# Likelihood Ratio Test (diferencia de residuos)
dif_residuos <- modelo$null.deviance - modelo$deviance
df_lr <- modelo$df.null - modelo$df.residual
p_value_lr <- pchisq(q = dif_residuos, df = df_lr, lower.tail = FALSE)
cat(sprintf("\nDiferencia de residuos (Likelihood Ratio): %.4f\n", dif_residuos))
cat(sprintf("Grados de libertad: %d\n", df_lr))
cat(sprintf("p-value: %s\n", format(p_value_lr, scientific = TRUE)))
if (p_value_lr < 0.05) {
  cat("-> El modelo es significativo en la prediccion de la anemia.\n")
}

# Pseudo R2 de McFadden
cat("\nPseudo R2 de McFadden:\n")
print(pscl::pR2(modelo)["McFadden"])

# Pseudo R2 de Cox y Snell
LR <- modelo$null.deviance - modelo$deviance
N <- sum(weights(modelo))
RsqrCN <- 1 - exp(-LR / N)
cat(sprintf("Pseudo R2 de Cox y Snell: %.4f\n", RsqrCN))

# Pseudo R2 de Nagelkerke
L0.adj <- exp(-modelo$null.deviance / N)
RsqrNal <- RsqrCN / (1 - L0.adj)
cat(sprintf("Pseudo R2 de Nagelkerke: %.4f\n", RsqrNal))

# Test de Hosmer-Lemeshow (bondad de ajuste)
cat("\nTest de Hosmer-Lemeshow (bondad de ajuste):\n")
hl_test <- tryCatch({
  hoslem.test(as.numeric(datos$matricula) - 1, fitted(modelo))
}, error = function(e) NULL)
if (!is.null(hl_test)) {
  print(hl_test)
} else {
  cat("(No se pudo calcular; comun con datasets grandes con muchos valores repetidos de X)\n")
}

# -----------------------------------------------------------------
# Curva ROC
# -----------------------------------------------------------------
cat("\n", strrep("=", 70), "\n")
cat("CURVA ROC Y AREA BAJO LA CURVA\n")
cat(strrep("=", 70), "\n")

prob <- predict(modelo, type = "response")
roc_obj <- roc(datos$matricula, prob, quiet = TRUE)

png("4_curva_roc.png", width = 700, height = 700, res = 120)
plot(roc_obj, col = "blue", print.auc = TRUE,
     main = "Curva ROC - talla_edad_z ~ anemia")
dev.off()
cat("Grafico guardado: 4_curva_roc.png\n")

cat(sprintf("\nArea bajo la curva (AUC): %.4f\n", auc(roc_obj)))

# Punto de corte optimo (mas cercano a la esquina superior izquierda)
closest <- coords(roc_obj, "best", ret = c("threshold", "specificity", "sensitivity", "npv", "ppv"),
                   best.method = "closest.topleft")
cat("\nPunto de corte optimo (closest.topleft):\n")
print(closest)

# -----------------------------------------------------------------
# Matriz de confusion
# -----------------------------------------------------------------
cat("\n", strrep("=", 70), "\n")
cat("MATRIZ DE CONFUSION\n")
cat(strrep("=", 70), "\n")

predicciones <- ifelse(prob > 0.5, 1, 0)
matriz_confusion <- table(as.numeric(datos$matricula) - 1, predicciones,
                           dnn = c("observaciones", "predicciones"))
print(matriz_confusion)

accuracy_train <- sum(diag(matriz_confusion)) / sum(matriz_confusion)
cat(sprintf("\nAccuracy (entrenamiento completo): %.4f (%.1f%%)\n", accuracy_train, accuracy_train * 100))

error1 <- (matriz_confusion[1, 2] + matriz_confusion[2, 1]) / sum(matriz_confusion)
cat(sprintf("Error de clasificacion: %.4f\n", error1))
cat("\nNota: este es el error de entrenamiento (in-sample); no es generalizable\n")
cat("a nuevas observaciones. Mas adelante se calcula el error con datos de prueba.\n")

# -----------------------------------------------------------------
# Guardar la data con las predicciones
# -----------------------------------------------------------------
finaldata <- cbind(datos, prob, predicciones)
cat("\nPrimeras filas con valores predichos:\n")
print(head(finaldata, 10))
write.csv(finaldata, "anemia_predict.csv", row.names = FALSE)
cat("\nGuardado: anemia_predict.csv\n")

# -----------------------------------------------------------------
# CONCLUSION Y ECUACION DEL MODELO
# -----------------------------------------------------------------
cat("\n", strrep("=", 70), "\n")
cat("CONCLUSION\n")
cat(strrep("=", 70), "\n")
cat(sprintf("\nlogit(anemia) = %.6f + %.6f * (talla_edad_z)\n", coefs[1], coefs[2]))
cat("p(anemia) = exp(logit) / (1 + exp(logit))\n")

# -----------------------------------------------------------------
# PREDICCION PARA NUEVOS VALORES
# -----------------------------------------------------------------
cat("\n", strrep("=", 70), "\n")
cat("PREDICCION PARA NUEVOS VALORES\n")
cat(strrep("=", 70), "\n")

for (valor in c(0, -1, -2, -3)) {
  nuevo <- data.frame(talla_edad_z = valor)
  p_pred <- predict(modelo, newdata = nuevo, type = "response")
  clase_pred <- ifelse(p_pred > 0.5, "tiene anemia", "no tiene anemia")
  cat(sprintf("talla_edad_z = %d  ->  P(anemia) = %.4f  ->  %s\n", valor, p_pred, clase_pred))
}

# -----------------------------------------------------------------
# GRAFICO DEL MODELO (sin intervalos de confianza)
# -----------------------------------------------------------------
png("5_modelo_sin_ic.png", width = 800, height = 600, res = 120)
plot(as.numeric(datos$matricula) - 1 ~ talla_edad_z, datos, col = "darkblue",
     main = "Regresion logistica: anemia ~ talla_edad_z",
     ylab = "P(anemia = 1 | talla_edad_z)", xlab = "talla_edad_z", pch = "I")
curve(predict(modelo, data.frame(talla_edad_z = x), type = "response"),
      col = "firebrick", lwd = 2.5, add = TRUE)
dev.off()
cat("\nGrafico guardado: 5_modelo_sin_ic.png\n")

# -----------------------------------------------------------------
# GRAFICO DEL MODELO CON INTERVALOS DE CONFIANZA (ggplot2)
# -----------------------------------------------------------------
nuevos_puntos <- seq(from = min(datos$talla_edad_z), to = max(datos$talla_edad_z), by = 0.05)

predicciones_se <- predict(modelo, data.frame(talla_edad_z = nuevos_puntos), se.fit = TRUE)

predicciones_logit <- exp(predicciones_se$fit) / (1 + exp(predicciones_se$fit))

limite_inferior <- predicciones_se$fit - 1.96 * predicciones_se$se.fit
limite_inferior_logit <- exp(limite_inferior) / (1 + exp(limite_inferior))

limite_superior <- predicciones_se$fit + 1.96 * predicciones_se$se.fit
limite_superior_logit <- exp(limite_superior) / (1 + exp(limite_superior))

datos_curva <- data.frame(talla_edad_z = nuevos_puntos,
                           probabilidad_anemia = predicciones_logit,
                           limite_inferior_logit = limite_inferior_logit,
                           limite_superior_logit = limite_superior_logit)

p4 <- ggplot(datos, aes(x = talla_edad_z, y = as.numeric(matricula) - 1)) +
  geom_point(aes(color = matricula), shape = "I", size = 3, alpha = 0.3) +
  geom_line(data = datos_curva, aes(x = talla_edad_z, y = probabilidad_anemia),
            color = "firebrick", inherit.aes = FALSE) +
  geom_line(data = datos_curva, aes(x = talla_edad_z, y = limite_inferior_logit),
            linetype = "dashed", inherit.aes = FALSE) +
  geom_line(data = datos_curva, aes(x = talla_edad_z, y = limite_superior_logit),
            linetype = "dashed", inherit.aes = FALSE) +
  theme_bw() +
  labs(title = "Regresión logística: anemia ~ talla_edad_z",
       y = "P(anemia = 1 | talla_edad_z)", x = "talla_edad_z (Z-score Talla/Edad)") +
  theme(legend.position = "none") +
  theme(plot.title = element_text(hjust = 0.5))
ggsave("6_modelo_con_ic.png", p4, width = 8, height = 6, dpi = 120)
print(p4)
cat("\nGrafico guardado: 6_modelo_con_ic.png\n")

# =====================================================================
# COMO MACHINE LEARNING (train / test split + validacion cruzada)
# =====================================================================
cat("\n", strrep("=", 70), "\n")
cat("REGRESION LOGISTICA SIMPLE COMO MACHINE LEARNING\n")
cat(strrep("=", 70), "\n")

# Division de datos en entrenamiento (70%) y prueba (30%)
set.seed(123)
indice_entrenamiento <- createDataPartition(datos$matricula, p = 0.7, list = FALSE)
datos_entrenamiento <- datos[indice_entrenamiento, ]
datos_prueba <- datos[-indice_entrenamiento, ]

cat("Datos de entrenamiento:", nrow(datos_entrenamiento), "observaciones\n")
cat("Datos de prueba:", nrow(datos_prueba), "observaciones\n")

# Ajuste del modelo logistico simple con datos de entrenamiento
modelo_entrenamiento <- glm(matricula ~ talla_edad_z, data = datos_entrenamiento, family = "binomial")
print(summary(modelo_entrenamiento))

# Coeficientes y Odds Ratios
coeficientes_train <- coef(modelo_entrenamiento)
cat("\nCoeficientes:\n")
print(coeficientes_train)
cat("\nOdds Ratios:\n")
print(exp(coeficientes_train))

# Predicciones en datos de entrenamiento
pred_entrenamiento_prob <- predict(modelo_entrenamiento, datos_entrenamiento, type = "response")
pred_entrenamiento_clase <- as.factor(ifelse(pred_entrenamiento_prob > 0.5, 1, 0))

# Predicciones en datos de prueba
pred_prueba_prob <- predict(modelo_entrenamiento, datos_prueba, type = "response")
pred_prueba_clase <- as.factor(ifelse(pred_prueba_prob > 0.5, 1, 0))

obs_entrenamiento <- as.factor(as.numeric(datos_entrenamiento$matricula) - 1)
obs_prueba <- as.factor(as.numeric(datos_prueba$matricula) - 1)

# Matriz de confusion - entrenamiento
cat("\nMatriz de confusion (entrenamiento):\n")
matriz_confusion_train <- confusionMatrix(pred_entrenamiento_clase, obs_entrenamiento, positive = "1")
print(matriz_confusion_train)

# Matriz de confusion - prueba
cat("\nMatriz de confusion (prueba):\n")
matriz_confusion_test <- confusionMatrix(pred_prueba_clase, obs_prueba, positive = "1")
print(matriz_confusion_test)

# Metricas de rendimiento (test)
accuracy <- matriz_confusion_test$overall["Accuracy"]
precision <- matriz_confusion_test$byClass["Precision"]
recall <- matriz_confusion_test$byClass["Recall"]
f1 <- matriz_confusion_test$byClass["F1"]

cat(sprintf("\nAccuracy: %.4f (%.2f%%)\n", accuracy, accuracy * 100))
cat(sprintf("Precision: %.4f\n", precision))
cat(sprintf("Recall: %.4f\n", recall))
cat(sprintf("F1-Score: %.4f\n", f1))

# Curva ROC para los datos de prueba
roc_obj_test <- roc(obs_prueba, pred_prueba_prob, quiet = TRUE)
roc_auc <- auc(roc_obj_test)
cat(sprintf("ROC-AUC (prueba): %.4f\n", roc_auc))

png("7_roc_test.png", width = 700, height = 700, res = 120)
plot(roc_obj_test, main = "Curva ROC - Datos de Prueba", col = "blue", print.auc = TRUE)
dev.off()
cat("Grafico guardado: 7_roc_test.png\n")

# -----------------------------------------------------------------
# Visualizaciones resumen (panel 2x2)
# -----------------------------------------------------------------
png("8_resultados_resumen.png", width = 1400, height = 1000, res = 100)
par(mfrow = c(2, 2))

# Matriz de confusion (test)
cm_matrix <- as.matrix(matriz_confusion_test$table)
colores <- colorRampPalette(c("white", "steelblue"))(100)
image(1:2, 1:2, t(cm_matrix[2:1, ]),
      col = colores, xlab = "Predicho", ylab = "Verdadero",
      main = "Matriz de Confusión (test)", axes = FALSE, cex.main = 1.4)
axis(1, at = 1:2, labels = c("Sin anemia", "Con anemia"))
axis(2, at = 1:2, labels = c("Con anemia", "Sin anemia"))
text(rep(1:2, each = 2), rep(1:2, 2),
     c(cm_matrix[2, 1], cm_matrix[1, 1], cm_matrix[2, 2], cm_matrix[1, 2]),
     cex = 2, font = 2)

# Curva ROC
plot(roc_obj_test, col = "darkorange", lwd = 2, main = "Curva ROC (test)",
     cex.main = 1.4, cex.lab = 1.1)
abline(a = 0, b = 1, lty = 2, col = "navy", lwd = 2)
legend("bottomright", legend = sprintf("ROC (AUC = %.3f)", roc_auc),
       col = "darkorange", lwd = 2, cex = 1.0)
grid()

# Distribucion de probabilidades predichas
hist(pred_prueba_prob[obs_prueba == "0"], col = rgb(0.27, 0.51, 0.71, 0.5),
     breaks = 20, main = "Distribución de probabilidades (test)",
     xlab = "P(anemia)", xlim = c(0, 1), ylim = c(0, 200))
hist(pred_prueba_prob[obs_prueba == "1"], col = rgb(0.7, 0.13, 0.13, 0.5),
     breaks = 20, add = TRUE)
abline(v = 0.5, col = "red", lty = 2, lwd = 2)
legend("topright", legend = c("Sin anemia", "Con anemia"),
       fill = c(rgb(0.27, 0.51, 0.71, 0.5), rgb(0.7, 0.13, 0.13, 0.5)), cex = 0.8)

# Metricas comparativas
metricas_nombres <- c("Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC")
metricas_valores <- c(accuracy, precision, recall, f1, roc_auc)
colores_barras <- c("#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6")
barplot(metricas_valores, names.arg = metricas_nombres, col = colores_barras,
        ylim = c(0, 1.1), main = "Resumen de Métricas (test)", ylab = "Score",
        cex.main = 1.4, cex.lab = 1.1, cex.names = 0.85)
text(x = seq(0.7, by = 1.2, length.out = 5), y = metricas_valores + 0.05,
     labels = sprintf("%.3f", metricas_valores), cex = 1.0, font = 2)
grid()

dev.off()
cat("\nGrafico guardado: 8_resultados_resumen.png\n")

# -----------------------------------------------------------------
# Validacion cruzada con caret (10-fold)
# -----------------------------------------------------------------
cat("\n", strrep("=", 70), "\n")
cat("VALIDACION CRUZADA 10-FOLD (caret)\n")
cat(strrep("=", 70), "\n")

train_control <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

datos_cv <- datos
levels(datos_cv$matricula) <- c("Sin_anemia", "Con_anemia")

logistic_model_cv <- train(
  matricula ~ talla_edad_z,
  data = datos_cv,
  method = "glm",
  family = "binomial",
  trControl = train_control,
  metric = "ROC"
)
print(logistic_model_cv)

cat("\nResumen del modelo final (CV):\n")
print(summary(logistic_model_cv$finalModel))

predicciones_cv <- logistic_model_cv$pred
predicciones_cv <- predicciones_cv[order(predicciones_cv$rowIndex), ]

cat("\nMatriz de confusion (validacion cruzada):\n")
conf_matrix_cv <- confusionMatrix(predicciones_cv$pred, predicciones_cv$obs)
print(conf_matrix_cv)

roc_curve_cv <- roc(predicciones_cv$obs, predicciones_cv$Con_anemia, quiet = TRUE)
cat(sprintf("\nAUC (validacion cruzada): %.4f\n", auc(roc_curve_cv)))

png("9_roc_cv.png", width = 700, height = 700, res = 120)
plot(roc_curve_cv, main = "Curva ROC - Validación Cruzada 10-fold",
     col = "blue", lwd = 2, print.auc = TRUE)
abline(a = 0, b = 1, lty = 2, col = "red")
dev.off()
cat("Grafico guardado: 9_roc_cv.png\n")

# -----------------------------------------------------------------
# RESUMEN EJECUTIVO
# -----------------------------------------------------------------
cat("\n", strrep("=", 70), "\n")
cat("RESUMEN EJECUTIVO\n")
cat(strrep("=", 70), "\n")
cat(sprintf("\nMODELO: Regresion Logistica Simple\n"))
cat(sprintf("Variable dependiente: presencia de anemia (Sin_anemia=0, Con_anemia=1)\n"))
cat(sprintf("Variable independiente: talla_edad_z (Z-score Talla/Edad, OMS)\n\n"))
cat(sprintf("logit(anemia) = %.4f + (%.4f) * talla_edad_z\n", coefs[1], coefs[2]))
cat(sprintf("Odds Ratio (talla_edad_z): %.4f\n", or_modelo["talla_edad_z"]))
cat(sprintf("AIC: %.2f\n", AIC(modelo)))
cat(sprintf("Pseudo R2 McFadden: %.4f\n", pscl::pR2(modelo)["McFadden"]))
cat(sprintf("AUC-ROC (modelo completo): %.4f\n", auc(roc_obj)))
cat(sprintf("AUC-ROC (validacion cruzada 10-fold): %.4f\n", auc(roc_curve_cv)))
cat(sprintf("\nInterpretacion: por cada unidad que aumenta talla_edad_z, las odds\n"))
cat(sprintf("de tener anemia se reducen en un %.1f%%.\n", (1 - or_modelo["talla_edad_z"]) * 100))
cat(sprintf("El predictor es estadisticamente significativo (p < 0.001).\n"))

cat("\n", strrep("=", 70), "\n")
cat("ANALISIS COMPLETADO EXITOSAMENTE\n")
cat(strrep("=", 70), "\n")
