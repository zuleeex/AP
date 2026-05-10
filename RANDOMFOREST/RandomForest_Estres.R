###### RANDOMFOREST
# Regresión: predicción del nivel de estrés en adolescentes
library(dplyr)
library(tidyr)
library(ggplot2)
library(tidymodels)
library(ranger)
library(rpart)
library(rpart.plot)
library(doParallel)
library(recipes)
library(workflows)
library(readxl)
library(tibble)

# CARGA Y PREPARACION DE DATOS
#=============================
datos <- read_excel("datosestresyansiedad.xlsx")
head(datos)

# Convertir variables categóricas a factor
datos$gender                 <- as.factor(datos$gender)
datos$platform_usage         <- as.factor(datos$platform_usage)
datos$social_interaction_level <- as.factor(datos$social_interaction_level)
datos$depression_label       <- as.factor(datos$depression_label)
str(datos)

# PARTICION DE LA DATA
# creando data train (80%) y test 20%
set.seed(123)
train <- sample(1:nrow(datos), size = nrow(datos) * 0.8)
data_train <- datos[train, ]
data_test  <- datos[-train, ]

cat("Train:", nrow(data_train), "    Test:", nrow(data_test), "\n")

# MODELO BASE (referencia)
# ========================

# El modelo base usa solo 12 árboles
modelo_base <- ranger(
  formula   = stress_level ~ .,
  data      = data_train,
  num.trees = 12,
  seed      = 123
)

print(modelo_base)
ncol(datos)
summary(modelo_base)

# Error de test del modelo
pred_base <- predict(modelo_base, data = data_test)$predictions
rmse_base <- sqrt(mean((pred_base - data_test$stress_level)^2))

cat("RMSE Modelo Base (12 árboles):", round(rmse_base, 2), "\n\n")

# OPTIMIZACION: NUM.TREES (OOB)
# ================================
# Se evalúan dos estrategias de validación para encontrar el número óptimo de árboles.
# 1) empleando el OBB(Out-of-Bag) error (root mean squared error)

num_trees_range <- seq(1, 400, 20) # 20 valores
results_oob <- tibble(
  n_arboles  = num_trees_range,
  train_rmse = NA_real_,
  oob_rmse   = NA_real_
)

for (i in seq_along(num_trees_range)) {
  modelo <- ranger(
    formula   = stress_level ~ .,
    data      = data_train,
    num.trees = num_trees_range[i],
    oob.error = TRUE,
    seed      = 123
  )
  
  pred_train  <- predict(modelo, data = data_train)$predictions
  train_error <- sqrt(mean((pred_train - data_train$stress_level)^2))
  oob_error   <- sqrt(modelo$prediction.error)
  
  results_oob$train_rmse[i] <- train_error
  results_oob$oob_rmse[i]   <- oob_error
}

optimal_trees_oob <- results_oob$n_arboles[which.min(results_oob$oob_rmse)]

# Gráfico
p1 <- ggplot(results_oob, aes(x = n_arboles)) +
  geom_line(aes(y = train_rmse, color = "Train RMSE"), linewidth = 1) +
  geom_line(aes(y = oob_rmse,   color = "OOB RMSE"),   linewidth = 1) +
  geom_vline(xintercept = optimal_trees_oob, linetype = "dashed",
             color = "red", alpha = 0.7) +
  labs(
    title = "Optimización: Número de Árboles (OOB)",
    x     = "Número de árboles",
    y     = "RMSE",
    color = ""
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

print(p1)
cat("Óptimo num.trees (OOB):", optimal_trees_oob, "\n\n")


# 2)  OPTIMIZACIÓN: NUM.TREES (K-FOLD CV)
# Validación empleando K-fold cross-validation (k=5)

num_trees_range <- seq(1, 400, 10)
results_cv_trees <- tibble(
  n_arboles  = num_trees_range,
  train_rmse = NA_real_,
  cv_rmse    = NA_real_
)

for (i in seq_along(num_trees_range)) {
  modelo <- rand_forest(mode = "regression", trees = num_trees_range[i]) %>%
    set_engine("ranger", seed = 123)
  
  set.seed(1234)
  cv_folds <- vfold_cv(data_train, v = 5)
  
  cv_fit <- fit_resamples(
    preprocessor = stress_level ~ .,
    object       = modelo,
    resamples    = cv_folds,
    metrics      = metric_set(rmse)
  )
  
  cv_rmse <- collect_metrics(cv_fit)$mean[1]
  
  modelo_fit  <- fit(modelo, stress_level ~ ., data = data_train)
  pred_train  <- predict(modelo_fit, new_data = data_train)$.pred
  train_rmse  <- sqrt(mean((pred_train - data_train$stress_level)^2))
  
  results_cv_trees$train_rmse[i] <- train_rmse
  results_cv_trees$cv_rmse[i]    <- cv_rmse
}

optimal_trees_cv <- results_cv_trees$n_arboles[which.min(results_cv_trees$cv_rmse)]

p2 <- ggplot(results_cv_trees, aes(x = n_arboles)) +
  geom_line(aes(y = train_rmse, color = "Train RMSE"), linewidth = 1) +
  geom_line(aes(y = cv_rmse,    color = "CV RMSE"),    linewidth = 1) +
  geom_vline(xintercept = optimal_trees_cv, linetype = "dashed",
             color = "red", alpha = 0.7) +
  labs(
    title = "Optimización: Número de Árboles (5-Fold CV)",
    x     = "Número de árboles",
    y     = "RMSE",
    color = ""
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

print(p2)
cat("Óptimo num.trees (CV):", optimal_trees_cv, "\n\n")


####  OPTIMIZACIÓN: MTRY (OOB)

# mtry controla cuántas variables se consideran
# en cada split — es el hiperparámetro más influyente en Random Forest.

# Valores evaluados
mtry_range <- 1:(ncol(data_train) - 1)
results_oob_mtry <- tibble(
  mtry       = mtry_range,
  train_rmse = NA_real_,
  oob_rmse   = NA_real_
)

for (i in seq_along(mtry_range)) {
  modelo <- ranger(
    formula   = stress_level ~ .,
    data      = data_train,
    num.trees = 50,
    mtry      = mtry_range[i],
    oob.error = TRUE,
    seed      = 123
  )
  
  pred_train  <- predict(modelo, data = data_train)$predictions
  train_error <- sqrt(mean((pred_train - data_train$stress_level)^2))
  oob_error   <- sqrt(modelo$prediction.error)
  
  results_oob_mtry$train_rmse[i] <- train_error
  results_oob_mtry$oob_rmse[i]   <- oob_error
}

optimal_mtry_oob <- results_oob_mtry$mtry[which.min(results_oob_mtry$oob_rmse)]

p3 <- ggplot(results_oob_mtry, aes(x = mtry)) +
  geom_line(aes(y = train_rmse, color = "Train RMSE"), linewidth = 1) +
  geom_line(aes(y = oob_rmse,   color = "OOB RMSE"),   linewidth = 1) +
  geom_vline(xintercept = optimal_mtry_oob, linetype = "dashed",
             color = "red", alpha = 0.7) +
  labs(
    title = "Optimización: mtry (OOB)",
    x     = "mtry",
    y     = "RMSE",
    color = ""
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

print(p3)
cat("Óptimo mtry (OOB):", optimal_mtry_oob, "\n\n")


# GRID SEARCH CON VALIDACIÓN CRUZADA

# Definir modelo con hiperparámetros a optimizar
modelo_tune <- rand_forest(
  mode  = "regression",
  mtry  = tune(),
  trees = tune()
) %>%
  set_engine(
    engine    = "ranger",
    max.depth = tune(),
    seed      = 123
  )

# Preprocesado
transformer <- recipe(stress_level ~ ., data = data_train) %>%
  step_dummy(all_nominal_predictors())

# Workflow
workflow_modelado <- workflow() %>%
  add_recipe(transformer) %>%
  add_model(modelo_tune)

# Grid de hiperparámetros
n_pred <- ncol(data_train) - 1   # número de predictores disponibles
hiperpar_grid <- expand_grid(
  trees     = c(50, 100, 500, 1000),
  mtry      = c(3, 5, 7, n_pred),
  max.depth = c(1, 3, 10, 20)
)

# Validación cruzada
set.seed(1234)
cv_folds <- vfold_cv(data_train, v = 5, strata = stress_level)

# Paralelización
cl <- makePSOCKcluster(parallel::detectCores() - 1)
registerDoParallel(cl)

cat("Ejecutando Grid Search con", nrow(hiperpar_grid), "combinaciones...\n")

grid_fit <- tune_grid(
  object    = workflow_modelado,
  resamples = cv_folds,
  metrics   = metric_set(rmse),
  grid      = hiperpar_grid,
  control   = control_grid(verbose = TRUE)
)

stopCluster(cl)

# Mejores hiperparámetros
best_params <- show_best(grid_fit, metric = "rmse", n = 5)
print(best_params)

# MODELO FINAL
cat("\n=== Entrenando Modelo Final ===\n")

mejores_hiperpar <- select_best(grid_fit, metric = "rmse")

# IMPORTANTE: se guarda el workflow completo (no pull_workflow_fit)
# para que el preprocesador (step_dummy) se aplique al predecir
workflow_final <- finalize_workflow(
  workflow_modelado,
  parameters = mejores_hiperpar
) %>%
  fit(data = data_train)

# Predicciones en test (el workflow aplica las dummies automáticamente)
predicciones <- predict(workflow_final, new_data = data_test)

resultados_test <- predicciones %>%
  bind_cols(data_test %>% select(stress_level))

rmse_final <- rmse(
  data     = resultados_test,
  truth    = stress_level,
  estimate = .pred,
  na_rm    = TRUE
)

cat("RMSE Final en Test:", round(rmse_final$.estimate, 2), "\n")
cat("Mejora vs Modelo Base:",
    round((1 - rmse_final$.estimate / rmse_base) * 100, 1), "%\n")

cat("\n=== Importancia de Predictores ===\n")

# Workflow propio con importance = "permutation"
workflow_importance <- workflow() %>%
  add_recipe(transformer) %>%
  add_model(
    rand_forest(mode = "regression") %>%
      set_engine(
        engine     = "ranger",
        importance = "permutation",
        seed       = 123
      ) %>%
      finalize_model(mejores_hiperpar)
  ) %>%
  fit(data = data_train)


# Extraer importancia desde el motor ranger dentro del workflow
importancia_pred <- extract_fit_engine(workflow_importance)$variable.importance %>%
  enframe(name = "predictor", value = "importancia") %>%
  arrange(desc(importancia))

print(importancia_pred)

# Gráfico
p_importance <- ggplot(
  importancia_pred,
  aes(x = reorder(predictor, importancia), y = importancia, fill = importancia)
) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Importancia de Predictores (Permutación)",
    x     = "Predictor",
    y     = "Importancia"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

print(p_importance)

# COMPARACIÓN CON ÁRBOL DE DECISIÓN
# ============================================================================
cat("\n=== Comparación con Árbol de Decisión (rpart) ===\n")

# Modelo rpart
arbol_optimal <- rpart(
  formula = stress_level ~ .,
  data    = data_train,
  method  = "anova",
  control = list(minsplit = 6, maxdepth = 7, cp = 0.01)
)

pred_arbol <- predict(arbol_optimal, newdata = data_test)
rmse_arbol <- sqrt(mean((pred_arbol - data_test$stress_level)^2))

# Comparación
comparacion <- tibble(
  Modelo         = c("Árbol de Decisión", "Random Forest"),
  RMSE_Test      = c(rmse_arbol, rmse_final$.estimate),
  Mejora_vs_Base = c(
    (1 - rmse_arbol / rmse_base) * 100,
    (1 - rmse_final$.estimate / rmse_base) * 100
  )
)

print(comparacion)

# Visualizar árbol
rpart.plot(arbol_optimal, main = "Árbol de Decisión Óptimo")

# Predicción con nuevos datos
nuevos_datos <- data.frame(
  age                      = 16,
  gender                   = factor("female", levels = c("female", "male")),
  daily_social_media_hours = 5.0,
  platform_usage           = factor("TikTok", levels = c("Both", "Instagram", "TikTok")),
  sleep_hours              = 6.0,
  screen_time_before_sleep = 2.0,
  academic_performance     = 3.0,
  physical_activity        = 1.0,
  social_interaction_level = factor("medium", levels = c("high", "low", "medium")),
  anxiety_level            = 6,
  addiction_level          = 5,
  depression_label         = factor("0", levels = c("0", "1"))
)

predict(arbol_optimal, newdata = nuevos_datos)