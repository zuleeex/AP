# =====================================================================
# ANALISIS DISCRIMINANTE LINEAL (LDA) - VERSION R
# Dataset: Anemia infantil ENDES 2024 - RECH6

# VARIABLE DEPENDIENTE (Y) - grupo_anemia: 2 categorias
#   - Sin_anemia : ninos clasificados sin anemia (Directriz OMS 2024 / RM 251-2024-MINSA)
#   - Con_anemia : ninos con anemia (Leve, Moderada o Grave)
#
# VARIABLES INDEPENDIENTES (X) - todas cuantitativas:
#   - edad_meses    : Edad del nino en meses (HC1)
#   - peso_kg       : Peso en kilogramos (HC2)
#   - talla_cm      : Talla en centimetros (HC3)
#   - talla_edad_z  : Z-score Talla/Edad segun OMS (HC70)
#   - peso_edad_z   : Z-score Peso/Edad segun OMS (HC71)
# =====================================================================

library(MASS)        # lda()
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

set.seed(123)
options(scipen = 999)

# =====================================================================
# PASO 0.1: PREPARACION DEL DATASET (a partir del archivo crudo RECH6_2024.csv)
# =====================================================================
# El archivo de ENDES viene separado por ";" y usa CODIGOS NUMERICOS para
# "valor no medido", definidos por el propio diccionario de variables del
# INEI (archivo Diccionario_RECH6.pdf). Estos codigos NO son datos reales
# y deben excluirse antes de cualquier analisis estadistico:
#
#   Variable | Significado                  | Rango valido | Codigo "no medido"
#   ---------|-------------------------------|--------------|--------------------
#   HC2      | Peso en kg (x10, 1 decimal)   | 15:500       | 9999
#   HC3      | Talla en cm (x10, 1 decimal)  | 400:1500     | 9999
#   HC70     | Z-score Talla/Edad (OMS)      | -450:500     | 9996,9997,9998
#   HC71     | Z-score Peso/Edad (OMS)       | -450:500     | 9996,9997,9998
#   HC57A    | Nivel de anemia (OMS 2024)    | 1:4          | 9 (No sabe)
#
# Lo unico que se hace aqui es:
#  1) Excluir las filas con codigo de "no medido" (no se alteran datos reales).
#  2) Convertir a unidades reales (HC2/10=kg, HC3/10=cm, HC70/100 y HC71/100=Z-score).
#  3) Agrupar Grave+Moderada+Leve en una sola clase "Con_anemia" (las 3 clases
#     por separado quedan muy chicas: 18, 1371 y 4371 casos vs 12598 sin anemia).
# =====================================================================
cat("======================================================================\n")
cat("PASO 0.1: PREPARACION DEL DATASET DESDE EL ARCHIVO CRUDO\n")
cat("======================================================================\n")

df_crudo <- read.csv("RECH6_2024.csv", sep = ";", stringsAsFactors = FALSE)
cat("Dataset original (crudo):", dim(df_crudo), "\n")

datos <- df_crudo[, c("HC1", "HC2", "HC3", "HC57A", "HC70", "HC71")]
colnames(datos) <- c("edad_meses", "peso_kg_x10", "talla_cm_x10",
                      "anemia_oms", "talla_edad_z_x100", "peso_edad_z_x100")

# --- Paso A: excluir codigos de "no medido" segun el diccionario INEI ---
mask_validos <- (
  (datos$peso_kg_x10 < 999) &                          # 9999 = no medido
    (datos$talla_cm_x10 < 9999) &                        # 9999 = no medido
    (datos$anemia_oms %in% c(1, 2, 3, 4)) &               # excluye 9 = No sabe
    (abs(datos$talla_edad_z_x100) < 9000) &               # excluye 9996/9997/9998
    (abs(datos$peso_edad_z_x100) < 9000)                  # excluye 9996/9997/9998
)
cat("\nFilas excluidas por codigo de 'no medido':", sum(!mask_validos), "\n")
cat("Filas que quedan con datos validos:", sum(mask_validos), "de", nrow(datos), "\n")

datos <- datos[mask_validos, ]

# --- Paso B: convertir a unidades reales ---
datos$peso_kg <- datos$peso_kg_x10 / 10
datos$talla_cm <- datos$talla_cm_x10 / 10
datos$talla_edad_z <- datos$talla_edad_z_x100 / 100
datos$peso_edad_z <- datos$peso_edad_z_x100 / 100

# --- Paso C: agrupar anemia en 2 categorias (variable dependiente) ---
datos$grupo_anemia <- ifelse(datos$anemia_oms == 4, "Sin_anemia", "Con_anemia")

datos <- datos[, c("edad_meses", "peso_kg", "talla_cm", "talla_edad_z",
                    "peso_edad_z", "anemia_oms", "grupo_anemia")]
datos$grupo_anemia <- as.factor(datos$grupo_anemia)

cat("\nDataset final, listo para el analisis discriminante:\n")
cat(dim(datos), "\n")
cat("\nDistribucion de la variable dependiente:\n")
print(table(datos$grupo_anemia))
print(round(prop.table(table(datos$grupo_anemia)) * 100, 2))

X_cols <- c("edad_meses", "peso_kg", "talla_cm", "talla_edad_z", "peso_edad_z")

# =====================================================================
# PASO 1: REVISION DE LOS DATOS YA PREPARADOS
# =====================================================================
cat("\n======================================================================\n")
cat("PASO 1: REVISION DE LOS DATOS\n")
cat("======================================================================\n")

str(datos)
cat("\nDimensiones:", dim(datos), "\n")
cat("\nDistribucion de grupos:\n")
print(table(datos$grupo_anemia))
print(round(prop.table(table(datos$grupo_anemia)) * 100, 2))

cat("\nResumen de los predictores:\n")
print(summary(datos[, X_cols]))


# =====================================================================
# EXPLORACION GRAFICA
# =====================================================================
cat("\n======================================================================\n")
cat("EXPLORACION GRAFICA\n")
cat("======================================================================\n")

p1 <- ggplot(datos, aes(x = edad_meses, fill = grupo_anemia)) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 30) + theme_bw()
p2 <- ggplot(datos, aes(x = peso_kg, fill = grupo_anemia)) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 30) + theme_bw()
p3 <- ggplot(datos, aes(x = talla_cm, fill = grupo_anemia)) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 30) + theme_bw()
p4 <- ggplot(datos, aes(x = talla_edad_z, fill = grupo_anemia)) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 30) + theme_bw()
p5 <- ggplot(datos, aes(x = peso_edad_z, fill = grupo_anemia)) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 30) + theme_bw()

ggarrange(p1, p2, p3, p4, p5, nrow = 3, ncol = 2, common.legend = TRUE)
ggsave("1_histogramas_R.png", width = 10, height = 9)

# Dispersion por pares
pairs(x = datos[, X_cols],
      col = c("firebrick", "steelblue")[datos$grupo_anemia], pch = 19,
      main = "Dispersion entre predictores")


# =====================================================================
# PROBABILIDADES PREVIAS
# =====================================================================
cat("\n======================================================================\n")
cat("PROBABILIDADES PREVIAS (Prior probabilities)\n")
cat("======================================================================\n")
print(round(prop.table(table(datos$grupo_anemia)), 4))


# =====================================================================
# PASO 2: DIVIDIR LOS DATOS (TRAIN/TEST)
# =====================================================================
cat("\n======================================================================\n")
cat("PASO 2: DIVISION TRAIN / TEST (70% / 30%)\n")
cat("======================================================================\n")

index <- createDataPartition(y = datos$grupo_anemia, p = 0.70, list = FALSE)
train <- datos[index, ]
test  <- datos[-index, ]
cat("Train:", nrow(train), " | Test:", nrow(test), "\n")


# =====================================================================
# SUPUESTO 1: NORMALIDAD UNIVARIANTE (Shapiro-Wilk)
# =====================================================================
cat("\n======================================================================\n")
cat("SUPUESTO: NORMALIDAD UNIVARIANTE (Shapiro-Wilk)\n")
cat("======================================================================\n")
cat("Ho: la variable se distribuye normalmente dentro de cada grupo\n\n")

# Shapiro requiere n <= 5000; se toma una submuestra de 500 por grupo
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
    qqnorm(x, main = paste(grp, col), pch = 19, col = "steelblue", cex = 0.5)
    qqline(x, col = "red")
  }
}


# =====================================================================
# SUPUESTO 2: NORMALIDAD MULTIVARIANTE (Mardia, Henze-Zirkler, Royston)
# =====================================================================
cat("\n======================================================================\n")
cat("SUPUESTO: NORMALIDAD MULTIVARIANTE\n")
cat("======================================================================\n")

# NOTA DE COMPATIBILIDAD: el paquete MVN cambio su sintaxis en la version 6.x.
# El argumento ahora se llama "mvn_test" (antes "mvnTest") y los resultados se
# extraen con summary(resultado, select = "mvn") en vez de resultado$multivariateNormality.
# Este codigo funciona con packageVersion("MVN") 6.x. Si tienes una version 5.x,
# cambia mvn_test por mvnTest y usa resultado$multivariateNormality directamente.

# Se usa una submuestra por costo computacional (igual criterio que en Python)
set.seed(1)
muestra_mvn <- datos[sample(nrow(datos), min(2000, nrow(datos))), X_cols]

result_mardia <- mvn(data = muestra_mvn, mvn_test = "mardia")
cat("\n--- Test de Mardia ---\n")
print(summary(result_mardia, select = "mvn"))

result_hz <- mvn(data = muestra_mvn, mvn_test = "hz")
cat("\n--- Test de Henze-Zirkler ---\n")
print(summary(result_hz, select = "mvn"))

result_royston <- mvn(data = muestra_mvn, mvn_test = "royston")
cat("\n--- Test de Royston ---\n")
print(summary(result_royston, select = "mvn"))


# =====================================================================
# SUPUESTO 3: HOMOGENEIDAD DE MATRICES DE COVARIANZA (M de Box)
# =====================================================================
cat("\n======================================================================\n")
cat("SUPUESTO: HOMOGENEIDAD DE VARIANZAS-COVARIANZAS (Test M de Box)\n")
cat("======================================================================\n")
cat("Ho: las matrices de covarianza son iguales entre grupos\n")
cat("(Es el supuesto clave que justifica el uso de LDA)\n\n")

box_result <- boxM(data = datos[, X_cols], grouping = datos$grupo_anemia)
print(box_result)

if (box_result$p.value < 0.001) {
  cat("\n-> Se RECHAZA Ho (criterio conservador p<0.001).\n")
  cat("   Las matrices de covarianza NO son homogeneas. El LDA sigue siendo valido\n")
  cat("   como primera aproximacion, pero conviene tener en cuenta esta limitacion\n")
  cat("   al interpretar los resultados.\n")
} else {
  cat("\n-> No se rechaza Ho. LDA es adecuado.\n")
}


# =====================================================================
# PASO 3: ENTRENAR EL MODELO LDA (Procedimiento de Fisher)
# =====================================================================
cat("\n======================================================================\n")
cat("PASO 3: CALCULO DE LA FUNCION DISCRIMINANTE (LDA - Fisher)\n")
cat("======================================================================\n")

modelo_lda <- lda(grupo_anemia ~ edad_meses + peso_kg + talla_cm +
                     talla_edad_z + peso_edad_z, data = train)
print(modelo_lda)

cat("\nProporcion de traza (poder discriminante de cada funcion):\n")
print(modelo_lda$svd^2 / sum(modelo_lda$svd^2))


# =====================================================================
# LAMBDA DE WILKS (significancia de la funcion discriminante)
# =====================================================================
cat("\n======================================================================\n")
cat("LAMBDA DE WILKS\n")
cat("======================================================================\n")

# svd al cuadrado son los autovalores (lambda) de W^-1 B en la solucion de Fisher
lambda1 <- modelo_lda$svd[1]^2
wilks_lambda <- 1 / (1 + lambda1)
n_train <- nrow(train)
p <- length(X_cols)
g <- nlevels(train$grupo_anemia)

gl_wilks <- p * (g - 1)
chi2_wilks <- -(n_train - 1 - (p + g)/2) * log(wilks_lambda)
p_wilks <- 1 - pchisq(chi2_wilks, gl_wilks)

cat("Autovalor (lambda1) =", round(lambda1, 4), "\n")
cat("Correlacion canonica =", round(sqrt(lambda1/(1+lambda1)), 4), "\n")
cat("Lambda de Wilks =", round(wilks_lambda, 5), "\n")
cat("Chi-cuadrado aprox. =", round(chi2_wilks, 3), " gl =", gl_wilks, "\n")
cat("p-value =", format.pval(p_wilks, digits = 6), "\n")


# =====================================================================
# PASO 4: EVALUAR LOS MODELOS - MATRIZ DE CONFUSION
# =====================================================================
cat("\n======================================================================\n")
cat("PASO 4: EVALUACION DE MODELOS EN TEST\n")
cat("======================================================================\n")

# ---- LDA ----
cat("\n--- LDA ---\n")
pred_lda <- predict(modelo_lda, newdata = test)
tabla_lda <- table(test$grupo_anemia, pred_lda$class,
                    dnn = c("Clase real", "Clase predicha"))
print(tabla_lda)
print(confusionMatrix(pred_lda$class, test$grupo_anemia))

error_lda <- mean(test$grupo_anemia != pred_lda$class) * 100
cat("Error de clasificacion LDA (test) =", round(error_lda, 2), "%\n")

# ---- Validacion cruzada con caret (10-fold) ----
cat("\n--- Validacion cruzada 10-fold (LDA) ---\n")
lda_cv <- train(grupo_anemia ~ edad_meses + peso_kg + talla_cm +
                   talla_edad_z + peso_edad_z, data = datos,
                 method = "lda", trControl = trainControl(method = "cv", number = 10))
print(lda_cv)


# =====================================================================
# PASO 5: CURVA ROC Y AUC
# =====================================================================
cat("\n======================================================================\n")
cat("PASO 5: CURVA ROC\n")
cat("======================================================================\n")

par(pty = "s")
roc_lda <- roc(test$grupo_anemia, pred_lda$posterior[, "Con_anemia"],
               levels = c("Sin_anemia", "Con_anemia"))

plot(roc_lda, col = "firebrick", main = "Curva ROC: LDA", print.auc = TRUE)

cat("AUC LDA =", round(auc(roc_lda), 4), "\n")


# =====================================================================
# PASO 6: PREDICCION PARA UN NUEVO CASO
# =====================================================================
cat("\n======================================================================\n")
cat("PASO 6: PREDICCION PARA UN NUEVO CASO\n")
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

cat("\nPrediccion LDA:\n")
print(predict(modelo_lda, newdata = nuevo_caso))


# =====================================================================
# VISUALIZACION DE REGIONES DE CLASIFICACION (klaR)
# =====================================================================
# install.packages("klaR")
library(klaR)
partimat(grupo_anemia ~ edad_meses + peso_kg + talla_cm + talla_edad_z + peso_edad_z,
         data = train, method = "lda", prec = 100,
         image.colors = c("snow2", "skyblue2"), col.mean = "firebrick")
