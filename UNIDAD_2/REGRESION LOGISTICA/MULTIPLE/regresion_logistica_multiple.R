# =====================================================================
# REGRESION LOGISTICA MULTIPLE - R
# Anemia infantil - ENDES 2024 (RECH6)
# Universidad Nacional del Altiplano
# =====================================================================
#
# Variable dependiente (Y): grupo_anemia (0 = Sin_anemia, 1 = Con_anemia)
#   Directriz OMS 2024 / RM 251-2024-MINSA (variable HC57A)
#
# Variables independientes (X), todas cuantitativas:
#   edad_meses    : Edad del nino en meses          (HC1)
#   peso_kg       : Peso en kilogramos               (HC2)
#   talla_cm      : Talla en centimetros             (HC3)
#   talla_edad_z  : Z-score Talla/Edad (OMS)          (HC70)
#   peso_edad_z   : Z-score Peso/Edad (OMS)           (HC71)
#
# Se excluye la hemoglobina como predictor porque es la variable que
# define la anemia (evitar circularidad / fuga de informacion).
# =====================================================================

# -----------------------------------------------------------------
# 0. Librerias
# -----------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(car)        # VIF
library(pROC)        # Curva ROC / AUC
library(caret)        # train/test split, seleccion de variables
library(pscl)        # Pseudo R2 (McFadden)

set.seed(123)  # misma semilla usada en el analisis LDA

# -----------------------------------------------------------------
# PASO 0.1: PREPARACION DEL DATASET (desde el archivo crudo RECH6_2024.csv)
# -----------------------------------------------------------------
cat(strrep("=", 70), "\n")
cat("PASO 0.1: PREPARACION DEL DATASET DESDE EL ARCHIVO CRUDO\n")
cat(strrep("=", 70), "\n")

df_crudo <- read.csv("RECH6_2024.csv", sep = ";", stringsAsFactors = FALSE)
cat("\nDataset original (crudo):", dim(df_crudo)[1], "filas x", dim(df_crudo)[2], "columnas\n")

datos <- df_crudo[, c("HC1", "HC2", "HC3", "HC57A", "HC70", "HC71")]
colnames(datos) <- c("edad_meses", "peso_kg_x10", "talla_cm_x10", "anemia_oms",
                      "talla_edad_z_x100", "peso_edad_z_x100")

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
datos$peso_kg      <- datos$peso_kg_x10 / 10
datos$talla_cm     <- datos$talla_cm_x10 / 10
datos$talla_edad_z <- datos$talla_edad_z_x100 / 100
datos$peso_edad_z  <- datos$peso_edad_z_x100 / 100
datos$grupo_anemia <- ifelse(datos$anemia_oms == 4, "Sin_anemia", "Con_anemia")

X_cols <- c("edad_meses", "peso_kg", "talla_cm", "talla_edad_z", "peso_edad_z")
Y_col  <- "grupo_anemia"

datos <- datos[, c(X_cols, "anemia_oms", Y_col)]

cat("\nDataset final, listo para la regresion logistica:\n")
print(dim(datos))
cat("\nDistribucion de la variable dependiente:\n")
print(table(datos[[Y_col]]))
print(round(prop.table(table(datos[[Y_col]])) * 100, 1))

# Codificacion: Con_anemia = 1 (evento), Sin_anemia = 0 (referencia)
datos$grupo_anemia <- factor(datos$grupo_anemia, levels = c("Sin_anemia", "Con_anemia"))
datos$grupo_anemia_num <- ifelse(datos$grupo_anemia == "Con_anemia", 1, 0)

# -----------------------------------------------------------------
# PASO 1: Revision de los datos
# -----------------------------------------------------------------
cat("\n", strrep("=", 70), "\n")
cat("PASO 1: REVISION DE LOS DATOS\n")
cat(strrep("=", 70), "\n")

str(datos)
cat("\nEstadisticos descriptivos de los predictores:\n")
print(round(sapply(datos[X_cols], summary), 2))

cat("\nValores nulos por columna:\n")
print(colSums(is.na(datos[c(X_cols, "grupo_anemia")])))

# -----------------------------------------------------------------
# Exploracion grafica
# -----------------------------------------------------------------
cat("\n", strrep("=", 70), "\n")
cat("EXPLORACION GRAFICA\n")
cat(strrep("=", 70), "\n")

p_dist <- ggplot(datos, aes(x = grupo_anemia, fill = grupo_anemia)) +
  geom_bar() +
  scale_fill_manual(values = c("Sin_anemia" = "steelblue", "Con_anemia" = "firebrick")) +
  labs(title = "Distribucion de la variable dependiente", x = "", y = "Frecuencia") +
  theme_minimal()
ggsave("1_distribucion_grupo_anemia.png", p_dist, width = 6, height = 4, dpi = 120)
print(p_dist)

# Histogramas de los predictores por grupo
png("2_histogramas_predictores.png", width = 1400, height = 900, res = 120)
par(mfrow = c(2, 3))
for (col in X_cols) {
  hist(datos[datos$grupo_anemia == "Sin_anemia", col], col = rgb(0.27, 0.51, 0.71, 0.5),
       main = col, xlab = col, breaks = 30, freq = FALSE,
       xlim = range(datos[[col]]))
  hist(datos[datos$grupo_anemia == "Con_anemia", col], col = rgb(0.7, 0.13, 0.13, 0.5),
       breaks = 30, freq = FALSE, add = TRUE)
  legend("topright", legend = c("Sin_anemia", "Con_anemia"),
         fill = c(rgb(0.27, 0.51, 0.71, 0.5), rgb(0.7, 0.13, 0.13, 0.5)), cex = 0.7)
}
dev.off()
cat("Grafico guardado: 2_histogramas_predictores.png\n")

# Correlacion de cada predictor con la variable dependiente (numerica)
correlaciones <- sapply(datos[X_cols], function(x) cor(x, datos$grupo_anemia_num))
cat("\nCorrelacion de cada predictor con grupo_anemia (1 = Con_anemia):\n")
print(round(sort(correlaciones), 4))

# -----------------------------------------------------------------
# Matriz de correlacion (heatmap simple) + boxplots por clase
# -----------------------------------------------------------------
png("3_matriz_correlacion.png", width = 900, height = 800, res = 120)
corr_matrix <- cor(datos[c(X_cols, "grupo_anemia_num")])
heatmap(corr_matrix, symm = TRUE, main = "Matriz de correlacion",
        col = colorRampPalette(c("steelblue", "white", "firebrick"))(50))
dev.off()
cat("Grafico guardado: 3_matriz_correlacion.png\n")

png("4_boxplots_por_clase.png", width = 1400, height = 900, res = 120)
par(mfrow = c(2, 3))
for (col in X_cols) {
  boxplot(datos[[col]] ~ datos$grupo_anemia, main = col, xlab = "", ylab = col,
          col = c("steelblue", "firebrick"))
}
dev.off()
cat("Grafico guardado: 4_boxplots_por_clase.png\n")

# -----------------------------------------------------------------
# PASO 2: Division train / test (80% / 20%, estratificado)
# -----------------------------------------------------------------
cat("\n", strrep("=", 70), "\n")
cat("PASO 2: DIVISION TRAIN / TEST (80% / 20%)\n")
cat(strrep("=", 70), "\n")

set.seed(123)
indices_train <- createDataPartition(datos$grupo_anemia, p = 0.8, list = FALSE)
train <- datos[indices_train, ]
test  <- datos[-indices_train, ]

cat("Train:", nrow(train), "observaciones\n")
cat("Test :", nrow(test), "observaciones\n")
cat("\nProporcion de clases en train:\n")
print(round(prop.table(table(train$grupo_anemia)), 3))
cat("Proporcion de clases en test:\n")
print(round(prop.table(table(test$grupo_anemia)), 3))

# -----------------------------------------------------------------
# Escalado de variables (Min-Max), ajustado SOLO con train
# -----------------------------------------------------------------
minmax_params <- lapply(train[X_cols], function(x) c(min = min(x), max = max(x)))

escalar_minmax <- function(df, params) {
  df_scaled <- df
  for (col in names(params)) {
    rango <- params[[col]]["max"] - params[[col]]["min"]
    df_scaled[[col]] <- (df[[col]] - params[[col]]["min"]) / rango
  }
  df_scaled
}

train_scaled <- escalar_minmax(train, minmax_params)
test_scaled  <- escalar_minmax(test, minmax_params)

# -----------------------------------------------------------------
# SUPUESTO: Deteccion de multicolinealidad (VIF)
# -----------------------------------------------------------------
cat("\n", strrep("=", 70), "\n")
cat("DETECCION DE MULTICOLINEALIDAD (VIF)\n")
cat(strrep("=", 70), "\n")

formula_completa <- as.formula(paste("grupo_anemia_num ~", paste(X_cols, collapse = " + ")))
modelo_vif_aux <- lm(formula_completa, data = train_scaled)
vif_vals <- vif(modelo_vif_aux)

vif_df <- data.frame(Variable = names(vif_vals), VIF = as.numeric(vif_vals))
vif_df <- vif_df[order(-vif_df$VIF), ]
cat("\nFactor de Inflacion de Varianza (VIF):\n")
print(vif_df, row.names = FALSE)
cat("\nInterpretacion:\n")
cat("  VIF < 5: No hay multicolinealidad\n")
cat("  5 <= VIF < 10: Multicolinealidad moderada\n")
cat("  VIF >= 10: Multicolinealidad alta (considerar eliminar)\n")

png("5_vif.png", width = 900, height = 500, res = 120)
colores_vif <- ifelse(vif_df$VIF < 5, "forestgreen", ifelse(vif_df$VIF < 10, "orange", "firebrick"))
par(mar = c(4, 7, 3, 1))
barplot(vif_df$VIF, names.arg = vif_df$Variable, horiz = TRUE, col = colores_vif,
        main = "Factor de Inflacion de Varianza (VIF)", xlab = "VIF", las = 1)
abline(v = c(5, 10), col = c("orange", "red"), lty = 2, lwd = 2)
dev.off()
cat("Grafico guardado: 5_vif.png\n")
cat("\nNota: peso_kg, talla_cm y edad_meses suelen mostrar VIF alto entre si,\n")
cat("porque el cuerpo crece con la edad. Los z-scores ya estan ajustados por\n")
cat("edad/sexo segun la OMS y normalmente presentan menor VIF.\n")

# -----------------------------------------------------------------
# PASO 3: Entrenamiento del modelo de Regresion Logistica (modelo completo)
# -----------------------------------------------------------------
cat("\n", strrep("=", 70), "\n")
cat("PASO 3: ENTRENAMIENTO DEL MODELO COMPLETO (glm binomial)\n")
cat(strrep("=", 70), "\n")

modelo_completo <- glm(formula_completa, data = train_scaled, family = binomial(link = "logit"))
cat("\nResumen del modelo completo:\n")
print(summary(modelo_completo))

# Odds Ratios e IC 95%
or_completo <- exp(cbind(OR = coef(modelo_completo), confint(modelo_completo)))
cat("\nOdds Ratios (modelo completo):\n")
print(round(or_completo, 4))

# Pseudo R2 de McFadden
pr2 <- pR2(modelo_completo)
cat("\nPseudo R2 de McFadden:", round(pr2["McFadden"], 4), "\n")

# -----------------------------------------------------------------
# Evaluacion del modelo completo en test
# -----------------------------------------------------------------
cat("\n", strrep("=", 70), "\n")
cat("EVALUACION DEL MODELO COMPLETO (test)\n")
cat(strrep("=", 70), "\n")

prob_test_completo <- predict(modelo_completo, newdata = test_scaled, type = "response")
pred_test_completo <- ifelse(prob_test_completo > 0.5, "Con_anemia", "Sin_anemia")
pred_test_completo <- factor(pred_test_completo, levels = c("Sin_anemia", "Con_anemia"))

cm_completo <- confusionMatrix(pred_test_completo, test$grupo_anemia, positive = "Con_anemia")
print(cm_completo)

roc_completo <- roc(test$grupo_anemia_num, prob_test_completo, quiet = TRUE)
auc_completo <- auc(roc_completo)
cat("\nAUC-ROC (modelo completo):", round(auc_completo, 4), "\n")

# -----------------------------------------------------------------
# PASO 4: Seleccion de variables (Stepwise por AIC, ambas direcciones)
# -----------------------------------------------------------------
cat("\n", strrep("=", 70), "\n")
cat("PASO 4: SELECCION DE VARIABLES (stepwise AIC)\n")
cat(strrep("=", 70), "\n")

modelo_nulo <- glm(grupo_anemia_num ~ 1, data = train_scaled, family = binomial)

modelo_step <- step(modelo_completo,
                     scope = list(lower = modelo_nulo, upper = modelo_completo),
                     direction = "both", trace = 0)

cat("\nModelo seleccionado por stepwise (AIC):\n")
print(summary(modelo_step))

variables_finales <- names(coef(modelo_step))[-1]  # sin el intercepto
cat("\nVariables finales seleccionadas:", paste(variables_finales, collapse = ", "), "\n")

# -----------------------------------------------------------------
# PASO 5: Modelo final - evaluacion completa
# -----------------------------------------------------------------
cat("\n", strrep("=", 70), "\n")
cat("PASO 5: EVALUACION DEL MODELO FINAL (seleccionado por AIC)\n")
cat(strrep("=", 70), "\n")

prob_test_final <- predict(modelo_step, newdata = test_scaled, type = "response")
pred_test_final <- ifelse(prob_test_final > 0.5, "Con_anemia", "Sin_anemia")
pred_test_final <- factor(pred_test_final, levels = c("Sin_anemia", "Con_anemia"))

cm_final <- confusionMatrix(pred_test_final, test$grupo_anemia, positive = "Con_anemia")
cat("\nMatriz de confusion (modelo final):\n")
print(cm_final)

roc_final <- roc(test$grupo_anemia_num, prob_test_final, quiet = TRUE)
auc_final <- auc(roc_final)
cat("\nAUC-ROC (modelo final):", round(auc_final, 4), "\n")

# Odds Ratios del modelo final
or_final <- exp(cbind(OR = coef(modelo_step), confint(modelo_step)))
cat("\nOdds Ratios (modelo final):\n")
print(round(or_final, 4))

# Validacion cruzada 10-fold (sobre el dataset completo, mismas variables)
cat("\nValidacion cruzada 10-fold (Accuracy):\n")
ctrl <- trainControl(method = "cv", number = 10)
formula_final <- as.formula(paste("grupo_anemia ~", paste(variables_finales, collapse = " + ")))
datos_scaled_completo <- escalar_minmax(datos, minmax_params)
cv_modelo <- train(formula_final, data = datos_scaled_completo, method = "glm",
                    family = "binomial", trControl = ctrl)
print(cv_modelo)

# -----------------------------------------------------------------
# Curva ROC (grafico)
# -----------------------------------------------------------------
png("6_curva_roc.png", width = 700, height = 700, res = 120)
plot(roc_final, col = "firebrick", lwd = 2,
     main = paste0("Curva ROC - Modelo final (AUC = ", round(auc_final, 3), ")"))
abline(a = 0, b = 1, col = "gray", lty = 2)
dev.off()
cat("Grafico guardado: 6_curva_roc.png\n")

# Matriz de confusion (grafico)
png("7_matriz_confusion.png", width = 600, height = 500, res = 120)
cm_table <- as.table(cm_final$table)
fourfoldplot(cm_table, color = c("steelblue", "firebrick"),
             main = "Matriz de Confusion - Modelo Final")
dev.off()
cat("Grafico guardado: 7_matriz_confusion.png\n")

# -----------------------------------------------------------------
# PASO 6: Ecuacion del modelo e interpretacion de Odds Ratios
# -----------------------------------------------------------------
cat("\n", strrep("=", 70), "\n")
cat("ECUACION DEL MODELO FINAL\n")
cat(strrep("=", 70), "\n")

coefs_final <- coef(modelo_step)
ecuacion <- paste0("log(p/(1-p)) = ", round(coefs_final[1], 4))
for (i in 2:length(coefs_final)) {
  ecuacion <- paste0(ecuacion, " + (", round(coefs_final[i], 4), ") * ", names(coefs_final)[i])
}
cat("\n", ecuacion, "\n")

cat("\nInterpretacion de Odds Ratios:\n")
for (var in variables_finales) {
  or_var <- exp(coefs_final[var])
  if (or_var > 1) {
    cambio <- (or_var - 1) * 100
    cat(sprintf("  - %s: por cada unidad de aumento (escala normalizada 0-1), las odds de anemia aumentan %.1f%%\n", var, cambio))
  } else {
    cambio <- (1 - or_var) * 100
    cat(sprintf("  - %s: por cada unidad de aumento (escala normalizada 0-1), las odds de anemia disminuyen %.1f%%\n", var, cambio))
  }
}

# -----------------------------------------------------------------
# PASO 7: Prediccion para un nuevo caso
# -----------------------------------------------------------------
cat("\n", strrep("=", 70), "\n")
cat("PASO 7: PREDICCION PARA UN NUEVO CASO\n")
cat(strrep("=", 70), "\n")

nuevo_caso <- data.frame(
  edad_meses   = 24,
  peso_kg      = 11.5,
  talla_cm     = 84.0,
  talla_edad_z = -1.2,
  peso_edad_z  = -0.8
)
cat("\nNuevo caso a clasificar:\n")
print(nuevo_caso)

nuevo_caso_scaled <- escalar_minmax(nuevo_caso, minmax_params)
prob_nuevo <- predict(modelo_step, newdata = nuevo_caso_scaled, type = "response")
clase_nuevo <- ifelse(prob_nuevo > 0.5, "Con_anemia", "Sin_anemia")

cat("\nProbabilidad de Con_anemia:", round(prob_nuevo, 4), "\n")
cat("Clase predicha:", clase_nuevo, "\n")

# -----------------------------------------------------------------
# RESUMEN EJECUTIVO
# -----------------------------------------------------------------
cat("\n", strrep("=", 70), "\n")
cat("RESUMEN EJECUTIVO\n")
cat(strrep("=", 70), "\n")

cat(sprintf("\nMODELO FINAL: glm binomial seleccionado por stepwise (AIC)\n"))
cat(sprintf("Variables incluidas (%d): %s\n", length(variables_finales), paste(variables_finales, collapse = ", ")))
cat(sprintf("AIC del modelo: %.2f\n", AIC(modelo_step)))
cat(sprintf("Pseudo R2 McFadden: %.4f\n", pR2(modelo_step)["McFadden"]))
cat(sprintf("Accuracy (test): %.4f\n", cm_final$overall["Accuracy"]))
cat(sprintf("Sensibilidad (Con_anemia): %.4f\n", cm_final$byClass["Sensitivity"]))
cat(sprintf("Especificidad (Sin_anemia): %.4f\n", cm_final$byClass["Specificity"]))
cat(sprintf("AUC-ROC: %.4f\n", auc_final))
cat("\nVariables mas importantes (por |coeficiente|):\n")
coefs_ordenados <- sort(abs(coefs_final[-1]), decreasing = TRUE)
for (i in seq_along(coefs_ordenados)) {
  var <- names(coefs_ordenados)[i]
  cat(sprintf("  %d. %s: beta = %.4f, OR = %.4f\n", i, var, coefs_final[var], exp(coefs_final[var])))
}

cat("\n", strrep("=", 70), "\n")
cat("ANALISIS COMPLETADO EXITOSAMENTE\n")
cat(strrep("=", 70), "\n")

# -----------------------------------------------------------------
# Guardar resultados (opcional)
# -----------------------------------------------------------------
# saveRDS(modelo_step, "modelo_logistico_final.rds")
# saveRDS(minmax_params, "minmax_params.rds")
# write.csv(or_final, "odds_ratios_modelo_final.csv")
