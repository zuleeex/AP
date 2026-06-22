# =====================================================================
# ANALISIS DISCRIMINANTE CUADRATICO (QDA) - VERSION R
#========================================================
#
# VARIABLE DEPENDIENTE (Y) - grupo_anemia: 2 categorias
#   - Sin_anemia : ninos clasificados sin anemia (Directriz OMS 2024 / RM 251-2024-MINSA)
#   - Con_anemia : ninos con anemia (Leve, Moderada o Grave)
#
# VARIABLES INDEPENDIENTES (X) - las mismas que en el LDA, para comparar
# directamente ambos modelos:
#   - edad_meses    : Edad del nino en meses (HC1)
#   - peso_kg       : Peso en kilogramos (HC2)
#   - talla_cm      : Talla en centimetros (HC3)
#   - talla_edad_z  : Z-score Talla/Edad segun OMS (HC70)
#   - peso_edad_z   : Z-score Peso/Edad segun OMS (HC71)
#

library(MASS)        # lda(), qda()
library(ggplot2)
library(gridExtra)
library(ggpubr)
library(biotools)    # boxM()
library(MVN)         # normalidad multivariante
library(caret)       # confusionMatrix, createDataPartition
library(pROC)        # curva ROC
library(dplyr)
library(reshape2)
library(knitr)
library(klaR)         # partimat()

set.seed(123)
options(scipen = 999)

# =====================================================================
# PASO 0.1: PREPARACION DEL DATASET (a partir del archivo crudo RECH6_2024.csv)
# =====================================================================
# Mismo criterio de limpieza que en el LDA, basado en el diccionario INEI:
# se excluyen codigos de "no medido" (9999, 9998, 9997, 9996, 9) y se
# convierten las variables a su unidad real. Asi, LDA y QDA usan exactamente
# el mismo dataset y son directamente comparables.
# =====================================================================
cat("======================================================================\n")
cat("PASO 0.1: PREPARACION DEL DATASET DESDE EL ARCHIVO CRUDO\n")
cat("======================================================================\n")

df_crudo <- read.csv("RECH6_2024.csv", sep = ";", stringsAsFactors = FALSE)
cat("Dataset original (crudo):", dim(df_crudo), "\n")

datos <- df_crudo[, c("HC1", "HC2", "HC3", "HC57A", "HC70", "HC71")]
colnames(datos) <- c("edad_meses", "peso_kg_x10", "talla_cm_x10",
                      "anemia_oms", "talla_edad_z_x100", "peso_edad_z_x100")

mask_validos <- (
  (datos$peso_kg_x10 < 999) &
    (datos$talla_cm_x10 < 9999) &
    (datos$anemia_oms %in% c(1, 2, 3, 4)) &
    (abs(datos$talla_edad_z_x100) < 9000) &
    (abs(datos$peso_edad_z_x100) < 9000)
)
cat("\nFilas excluidas por codigo de 'no medido':", sum(!mask_validos), "\n")
cat("Filas validas:", sum(mask_validos), "de", nrow(datos), "\n")

datos <- datos[mask_validos, ]
datos$peso_kg <- datos$peso_kg_x10 / 10
datos$talla_cm <- datos$talla_cm_x10 / 10
datos$talla_edad_z <- datos$talla_edad_z_x100 / 100
datos$peso_edad_z <- datos$peso_edad_z_x100 / 100
datos$grupo_anemia <- ifelse(datos$anemia_oms == 4, "Sin_anemia", "Con_anemia")

datos <- datos[, c("edad_meses", "peso_kg", "talla_cm", "talla_edad_z",
                    "peso_edad_z", "anemia_oms", "grupo_anemia")]
datos$grupo_anemia <- as.factor(datos$grupo_anemia)

X_cols <- c("edad_meses", "peso_kg", "talla_cm", "talla_edad_z", "peso_edad_z")
colores_clase <- c("Sin_anemia" = "#4C72B0", "Con_anemia" = "#DD8452")

cat("\nDataset final:", dim(datos), "\n")
print(table(datos$grupo_anemia))


# =====================================================================
# 1. CARGA Y EXPLORACION DE DATOS
# =====================================================================
cat("\n======================================================================\n")
cat("1. CARGA Y EXPLORACION DE DATOS\n")
cat("======================================================================\n")

str(datos)
cat("\nValores nulos:\n")
print(colSums(is.na(datos[, X_cols])))
cat("\nEstadisticas descriptivas:\n")
print(summary(datos[, X_cols]))

# Distribucion de la variable objetivo
p_dist <- ggplot(datos, aes(x = grupo_anemia, fill = grupo_anemia)) +
  geom_bar() +
  geom_text(stat = "count", aes(label = paste0(after_stat(count), "\n(",
                                                 round(after_stat(count)/sum(after_stat(count))*100,1), "%)")),
            vjust = -0.3) +
  scale_fill_manual(values = colores_clase) +
  labs(title = "Distribucion de la variable objetivo (grupo_anemia)",
       x = "", y = "Numero de ninos") +
  theme_bw() + theme(legend.position = "none")
print(p_dist)
ggsave("1_distribucion_Y_R.png", p_dist, width = 5, height = 4)


# =====================================================================
# 2. SELECCION DE PREDICTORES Y VISUALIZACION
# =====================================================================
cat("\n======================================================================\n")
cat("2. SELECCION DE PREDICTORES Y VISUALIZACION\n")
cat("======================================================================\n")

cat("Predictores seleccionados:", X_cols, "\n")
cat("Variable objetivo: grupo_anemia\n")
print(table(datos$grupo_anemia))

# Histogramas superpuestos por clase
p1 <- ggplot(datos, aes(x = edad_meses, fill = grupo_anemia)) +
  geom_histogram(position = "identity", alpha = 0.55, bins = 25) +
  scale_fill_manual(values = colores_clase) + theme_bw()
p2 <- ggplot(datos, aes(x = peso_kg, fill = grupo_anemia)) +
  geom_histogram(position = "identity", alpha = 0.55, bins = 25) +
  scale_fill_manual(values = colores_clase) + theme_bw()
p3 <- ggplot(datos, aes(x = talla_cm, fill = grupo_anemia)) +
  geom_histogram(position = "identity", alpha = 0.55, bins = 25) +
  scale_fill_manual(values = colores_clase) + theme_bw()
p4 <- ggplot(datos, aes(x = talla_edad_z, fill = grupo_anemia)) +
  geom_histogram(position = "identity", alpha = 0.55, bins = 25) +
  scale_fill_manual(values = colores_clase) + theme_bw()
p5 <- ggplot(datos, aes(x = peso_edad_z, fill = grupo_anemia)) +
  geom_histogram(position = "identity", alpha = 0.55, bins = 25) +
  scale_fill_manual(values = colores_clase) + theme_bw()

ggarrange(p1, p2, p3, p4, p5, nrow = 3, ncol = 2, common.legend = TRUE)
ggsave("2_histogramas_R.png", width = 10, height = 9)

# Dispersion por pares
pairs(x = datos[, X_cols],
      col = colores_clase[datos$grupo_anemia], pch = 19,
      main = "Dispersion entre predictores")


# =====================================================================
# 3. SUPUESTO DE NORMALIDAD
# =====================================================================
cat("\n======================================================================\n")
cat("3. SUPUESTO DE NORMALIDAD\n")
cat("======================================================================\n")

# --- Normalidad univariante: Shapiro-Wilk ---
cat("\n--- Test de Shapiro-Wilk por variable y clase ---\n")
datos_tidy <- melt(datos[, c(X_cols, "grupo_anemia")], id.vars = "grupo_anemia")

shapiro_resultados <- datos_tidy %>%
  group_by(grupo_anemia, variable) %>%
  summarise(p_value_Shapiro = {
    muestra <- value
    if (length(muestra) > 500) muestra <- sample(muestra, 500)
    round(shapiro.test(muestra)$p.value, 6)
  }, .groups = "drop")
print(kable(shapiro_resultados))

# QQ-plots
par(mfcol = c(2, 5))
for (col in X_cols) {
  for (grp in levels(datos$grupo_anemia)) {
    x <- datos[datos$grupo_anemia == grp, col]
    qqnorm(x, main = paste(grp, col), pch = 19,
           col = colores_clase[grp], cex = 0.5)
    qqline(x, col = "black")
  }
}

# --- Normalidad multivariante: Mardia, Henze-Zirkler, Royston ---
# NOTA DE COMPATIBILIDAD: paquete MVN version 6.x usa "mvn_test" (no "mvnTest")
# y summary(resultado, select = "mvn") en vez de resultado$multivariateNormality.
cat("\n--- Normalidad multivariante por grupo ---\n")
for (grp in levels(datos$grupo_anemia)) {
  cat("\n=== Grupo:", grp, "===\n")
  Xg <- datos[datos$grupo_anemia == grp, X_cols]
  set.seed(1)
  if (nrow(Xg) > 1500) Xg <- Xg[sample(nrow(Xg), 1500), ]

  res_mardia <- mvn(data = Xg, mvn_test = "mardia")
  cat("Mardia:\n")
  print(summary(res_mardia, select = "mvn"))

  res_hz <- mvn(data = Xg, mvn_test = "hz")
  cat("Henze-Zirkler:\n")
  print(summary(res_hz, select = "mvn"))
}


# =====================================================================
# 4. HOMOGENEIDAD DE MATRICES DE COVARIANZA (TEST M DE BOX)
# =====================================================================
cat("\n======================================================================\n")
cat("4. HOMOGENEIDAD DE MATRICES DE COVARIANZA (TEST M DE BOX)\n")
cat("======================================================================\n")
cat("Ho: las matrices de covarianza son iguales entre grupos\n")
cat("(Si se rechaza Ho, el QDA es mas apropiado que el LDA)\n\n")

box_result <- boxM(data = datos[, X_cols], grouping = datos$grupo_anemia)
print(box_result)

if (box_result$p.value < 0.001) {
  cat("\n-> Se RECHAZA Ho (criterio conservador p<0.001).\n")
  cat("   Las matrices de covarianza NO son homogeneas -> EL QDA ES EL MODELO ADECUADO.\n")
} else {
  cat("\n-> No se rechaza Ho. LDA seria suficiente.\n")
}

# Visualizacion de matrices de correlacion por clase
par(mfrow = c(1, 2))
for (grp in levels(datos$grupo_anemia)) {
  Xg <- datos[datos$grupo_anemia == grp, X_cols]
  corr_g <- cor(Xg)
  image(1:ncol(corr_g), 1:ncol(corr_g), corr_g[, ncol(corr_g):1],
        col = colorRampPalette(c("firebrick", "white", "steelblue"))(50),
        axes = FALSE, xlab = "", ylab = "", main = paste("Correlacion -", grp))
  axis(1, at = 1:ncol(corr_g), labels = X_cols, las = 2, cex.axis = 0.7)
  axis(2, at = 1:ncol(corr_g), labels = rev(X_cols), las = 2, cex.axis = 0.7)
}
par(mfrow = c(1, 1))


# =====================================================================
# 5. AJUSTE DEL MODELO QDA
# =====================================================================
cat("\n======================================================================\n")
cat("5. AJUSTE DEL MODELO QDA\n")
cat("======================================================================\n")

index <- createDataPartition(y = datos$grupo_anemia, p = 0.70, list = FALSE)
train <- datos[index, ]
test  <- datos[-index, ]
cat("Train:", nrow(train), " | Test:", nrow(test), "\n")

modelo_qda <- qda(grupo_anemia ~ edad_meses + peso_kg + talla_cm +
                     talla_edad_z + peso_edad_z, data = train)
print(modelo_qda)

cat("\nMatrices de covarianza por grupo (calculadas manualmente, QDA usa una por clase):\n")
for (grp in levels(train$grupo_anemia)) {
  cat("\n--- Covarianza:", grp, "---\n")
  print(round(cov(train[train$grupo_anemia == grp, X_cols]), 3))
}


# =====================================================================
# 6. EVALUACION DEL MODELO: MATRIZ DE CONFUSION Y METRICAS
# =====================================================================
cat("\n======================================================================\n")
cat("6. EVALUACION DEL MODELO: MATRIZ DE CONFUSION Y METRICAS\n")
cat("======================================================================\n")

pred_train_qda <- predict(modelo_qda, newdata = train)
pred_test_qda <- predict(modelo_qda, newdata = test)

train_error_qda <- mean(train$grupo_anemia != pred_train_qda$class) * 100
test_error_qda <- mean(test$grupo_anemia != pred_test_qda$class) * 100

cat("Training error QDA :", round(train_error_qda, 2), "%  |  Training accuracy:",
    round(100 - train_error_qda, 2), "%\n")
cat("Test error QDA     :", round(test_error_qda, 2), "%  |  Test accuracy    :",
    round(100 - test_error_qda, 2), "%\n")

tabla_qda <- table(test$grupo_anemia, pred_test_qda$class,
                    dnn = c("Clase real", "Clase predicha"))
cat("\nMatriz de confusion (test):\n")
print(tabla_qda)
print(confusionMatrix(pred_test_qda$class, test$grupo_anemia))

# Curva ROC
par(pty = "s")
roc_qda <- roc(test$grupo_anemia, pred_test_qda$posterior[, "Con_anemia"],
               levels = c("Sin_anemia", "Con_anemia"))
plot(roc_qda, col = "#DD8452", main = "Curva ROC - QDA", lwd = 2.5)
cat("\nAUC (QDA) =", round(auc(roc_qda), 4), "\n")


# =====================================================================
# 7. ESPACIO DE DECISION (PARTITION PLOT)
# =====================================================================
cat("\n======================================================================\n")
cat("7. ESPACIO DE DECISION (partimat - regiones de clasificacion QDA)\n")
cat("======================================================================\n")

partimat(grupo_anemia ~ edad_meses + peso_kg + talla_cm + talla_edad_z + peso_edad_z,
         data = train, method = "qda", prec = 100,
         image.colors = c("snow2", "moccasin"), col.mean = "firebrick")


# =====================================================================
# 8. COMPARACION QDA vs LDA
# =====================================================================
cat("\n======================================================================\n")
cat("8. COMPARACION QDA vs LDA\n")
cat("======================================================================\n")

modelo_lda <- lda(grupo_anemia ~ edad_meses + peso_kg + talla_cm +
                     talla_edad_z + peso_edad_z, data = train)
pred_test_lda <- predict(modelo_lda, newdata = test)
test_error_lda <- mean(test$grupo_anemia != pred_test_lda$class) * 100

roc_lda <- roc(test$grupo_anemia, pred_test_lda$posterior[, "Con_anemia"],
               levels = c("Sin_anemia", "Con_anemia"))

cat("=== Comparacion QDA vs LDA - Conjunto de Test ===\n")
cat("QDA - Error:", round(test_error_qda, 2), "%  | Accuracy:", round(100 - test_error_qda, 2),
    "%  | AUC:", round(auc(roc_qda), 4), "\n")
cat("LDA - Error:", round(test_error_lda, 2), "%  | Accuracy:", round(100 - test_error_lda, 2),
    "%  | AUC:", round(auc(roc_lda), 4), "\n")

par(pty = "s")
plot(roc_qda, col = "#DD8452", main = "Comparacion ROC: QDA vs LDA", lwd = 2.5)
plot(roc_lda, col = "#4C72B0", add = TRUE, lwd = 2.5)
legend("bottomright",
       legend = c(paste("QDA  AUC =", round(auc(roc_qda), 3)),
                   paste("LDA  AUC =", round(auc(roc_lda), 3))),
       col = c("#DD8452", "#4C72B0"), lwd = 2.5)


# =====================================================================
# 9. VALIDACION CRUZADA (K-FOLD)
# =====================================================================
cat("\n======================================================================\n")
cat("9. VALIDACION CRUZADA (10-FOLD)\n")
cat("======================================================================\n")

qda_cv <- train(grupo_anemia ~ edad_meses + peso_kg + talla_cm +
                   talla_edad_z + peso_edad_z, data = datos,
                 method = "qda", trControl = trainControl(method = "cv", number = 10))
cat("\n--- QDA ---\n")
print(qda_cv)

lda_cv <- train(grupo_anemia ~ edad_meses + peso_kg + talla_cm +
                   talla_edad_z + peso_edad_z, data = datos,
                 method = "lda", trControl = trainControl(method = "cv", number = 10))
cat("\n--- LDA ---\n")
print(lda_cv)


# =====================================================================
# 10. PREDICCION PARA UN NUEVO CASO
# =====================================================================
cat("\n======================================================================\n")
cat("10. PREDICCION PARA UN NUEVO CASO\n")
cat("======================================================================\n")

nuevo_caso <- data.frame(
  edad_meses = 24,
  peso_kg = 11.5,
  talla_cm = 84,
  talla_edad_z = -1.2,
  peso_edad_z = -0.8
)
cat("\nNuevo caso:\n")
print(nuevo_caso)

cat("\nPrediccion QDA:\n")
print(predict(modelo_qda, newdata = nuevo_caso))
