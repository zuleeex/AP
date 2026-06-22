# =============================================================================
#  Teen Mental Health - Comparativa de modelos de ensamblado para regresión
#  Variable objetivo: stress_level (1–10)
#  Dataset: datosestresyansiedad.xlsx (1200 obs, 13 variables)
#  Versión científica: partición estratificada, CV repetida, IC bootstrap
#  en test, diagnóstico de residuos e importancias comparadas.
# =============================================================================

# 0. Librerías ---------------------------------------------------------------
pkgs_requeridos <- c("dplyr","tidyr","tibble","purrr","skimr","ggplot2","ggpubr",
                     "tidymodels","xgboost","gbm","randomForest","rpart",
                     "rpart.plot","doParallel","Metrics","readxl")
faltan <- pkgs_requeridos[!vapply(pkgs_requeridos, requireNamespace,
                                  logical(1), quietly = TRUE)]
if (length(faltan)) stop("Instala primero: ", paste(faltan, collapse = ", "))

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(tibble); library(purrr)
  library(skimr); library(ggplot2); library(ggpubr)
  library(tidymodels)
  library(xgboost); library(gbm); library(randomForest)
  library(rpart);   library(rpart.plot)
  library(doParallel)
  library(Metrics)   # OJO: Metrics::rmse/mae tapan a yardstick::rmse/mae;
  # en metric_set() usamos yardstick:: explícito.
  library(readxl)
})

# Semilla única para todo el flujo (reproducibilidad).
SEED <- 2024
set.seed(SEED)

# 1. Carga y exploración (EDA) -----------------------------------------------
datos_raw <- read_excel("datosestresyansiedad.xlsx")

# Convertir variables categóricas a factor
datos <- datos_raw %>%
  mutate(
    gender                 = factor(gender),
    platform_usage         = factor(platform_usage),
    social_interaction_level = factor(social_interaction_level)
  )

# Diagnóstico básico de calidad de datos
skim(datos)

stopifnot(sum(is.na(datos)) == 0)        # asumimos NA controlados

# Distribución de la variable objetivo: stress_level (1–10)
ggplot(datos, aes(stress_level)) +
  geom_histogram(bins = 10, fill = "steelblue", color = "white") +
  scale_x_continuous(breaks = 1:10) +
  labs(title = "Distribución de stress_level", x = "stress_level", y = "frecuencia") +
  theme_bw()

# Correlaciones numéricas
vars_num <- datos %>% select(where(is.numeric))
if (requireNamespace("corrplot", quietly = TRUE)) {
  corrplot::corrplot(cor(vars_num), method = "color", type = "upper",
                     tl.col = "black", addCoef.col = "black", number.cex = 0.6)
} else {
  message("Paquete corrplot no instalado; se omite el plot de correlaciones.")
  print(round(cor(vars_num), 2))
}

# 2. Partición estratificada por cuartiles de stress_level -------------------
# En regresión, estratificar por la variable respuesta evita que un fold/test
# concentre solo valores altos o bajos.
set.seed(SEED)
split_obj  <- initial_split(datos, prop = 0.8, strata = stress_level)
data_train <- training(split_obj)
data_test  <- testing(split_obj)

# Matrices numéricas para XGBoost (one-hot encoding de factores)
# model.matrix excluye el intercepto y genera dummies automáticamente.
X_train <- model.matrix(stress_level ~ . - 1, data = data_train)
y_train <- data_train$stress_level
X_test  <- model.matrix(stress_level ~ . - 1, data = data_test)
y_test  <- data_test$stress_level
dtrain  <- xgb.DMatrix(data = X_train, label = y_train)
dtest   <- xgb.DMatrix(data = X_test,  label = y_test)

# 3. Función de métricas con IC bootstrap ------------------------------------
# Las métricas puntuales en un único test set son estimaciones ruidosas.
# Devolvemos punto + IC95% por bootstrap de los pares (y, yhat).
boot_metrics <- function(y, yhat, B = 1000, alpha = 0.05) {
  n <- length(y)
  reps <- replicate(B, {
    idx <- sample.int(n, n, replace = TRUE)
    c(rmse = Metrics::rmse(y[idx], yhat[idx]),
      mae  = Metrics::mae (y[idx], yhat[idx]),
      r2   = 1 - sum((y[idx] - yhat[idx])^2) /
        sum((y[idx] - mean(y[idx]))^2))
  })
  data.frame(
    metric = c("RMSE", "MAE", "R2"),
    est    = c(Metrics::rmse(y, yhat),
               Metrics::mae (y, yhat),
               1 - sum((y - yhat)^2) / sum((y - mean(y))^2)),
    lower  = apply(reps, 1, quantile, probs = alpha / 2),
    upper  = apply(reps, 1, quantile, probs = 1 - alpha / 2)
  )
}

# 4. Modelos baseline --------------------------------------------------------

## 4.1 rpart con poda por cp óptimo
set.seed(SEED)
arbol <- rpart(stress_level ~ ., data = data_train, method = "anova",
               control = rpart.control(cp = 0.001, xval = 10))
cp_opt <- arbol$cptable[which.min(arbol$cptable[, "xerror"]), "CP"]
modelo_rpart <- prune(arbol, cp = cp_opt)
rpart.plot(modelo_rpart, main = sprintf("rpart podado (cp=%.4f)", cp_opt))

pred_rpart <- predict(modelo_rpart, newdata = data_test)

## 4.2 Bagging (mtry = p)
set.seed(SEED)
modelo_bag <- randomForest(stress_level ~ ., data = data_train,
                           mtry = ncol(data_train) - 1,
                           ntree = 500, importance = TRUE)
pred_bag <- predict(modelo_bag, newdata = data_test)

## 4.3 Random Forest con mtry escogido por error OOB (no a ojo)
mtry_grid  <- 2:(ncol(data_train) - 1)
oob_errors <- sapply(mtry_grid, function(m) {
  set.seed(SEED)
  rf <- randomForest(stress_level ~ ., data = data_train, mtry = m, ntree = 500)
  tail(rf$mse, 1)
})
mtry_opt <- mtry_grid[which.min(oob_errors)]
cat("mtry óptimo por OOB:", mtry_opt, "\n")

set.seed(SEED)
modelo_rf <- randomForest(stress_level ~ ., data = data_train,
                          mtry = mtry_opt, ntree = 500,
                          nodesize = 3, importance = TRUE)
pred_rf <- predict(modelo_rf, newdata = data_test)

## 4.4 GBM con n.trees por CV interna
set.seed(SEED)
modelo_gbm <- gbm(stress_level ~ ., data = data_train,
                  distribution = "gaussian",
                  n.trees = 5000, interaction.depth = 5,
                  shrinkage = 0.01, n.minobsinnode = 10,
                  bag.fraction = 0.5, cv.folds = 5)
best_iter_gbm <- gbm.perf(modelo_gbm, method = "cv", plot.it = FALSE)
cat("GBM mejor iter (CV):", best_iter_gbm, "\n")

pred_gbm <- predict(modelo_gbm, newdata = data_test, n.trees = best_iter_gbm)

# 5. XGBoost ----------------------------------------------------------------

## 5.1 Modelo base con parámetros razonables y early stopping
params_base <- list(
  objective = "reg:squarederror", eval_metric = "rmse",
  eta = 0.05, max_depth = 5, min_child_weight = 3,
  subsample = 0.8, colsample_bytree = 0.8, gamma = 0
)
set.seed(SEED)
cv_base <- xgb.cv(params = params_base, data = dtrain,
                  nrounds = 2000, nfold = 5,
                  early_stopping_rounds = 50,
                  print_every_n = 200, verbose = 1)

# best_iteration puede venir NULL según la versión de xgboost; lo
# recuperamos del evaluation_log de forma segura.
nrounds_base <- cv_base$best_iteration
if (is.null(nrounds_base) || is.na(nrounds_base) || nrounds_base < 1) {
  nrounds_base <- which.min(cv_base$evaluation_log$test_rmse_mean)
}
stopifnot(is.numeric(nrounds_base), nrounds_base >= 1)
cat(sprintf("XGB base - nrounds óptimo: %d\n", nrounds_base))

set.seed(SEED)
modelo_xgb <- xgb.train(params = params_base, data = dtrain,
                        nrounds = nrounds_base,
                        watchlist = list(train = dtrain, test = dtest),
                        verbose = 0)

pred_xgb <- predict(modelo_xgb, dtest)

## 5.2 Diagnóstico: curvas de aprendizaje por eta
eta_range <- c(0.001, 0.01, 0.05, 0.1, 0.3)
df_eta <- map_dfr(eta_range, function(e) {
  set.seed(SEED)
  xgb.cv(data = dtrain,
         params = list(objective = "reg:squarederror",
                       eta = e, max_depth = 6, subsample = 0.8),
         nrounds = 1500, nfold = 5,
         metrics = "rmse", verbose = 0)$evaluation_log %>%
    select(iter, test_rmse_mean) %>%
    mutate(eta = factor(e))
})
ggplot(df_eta, aes(iter, test_rmse_mean, color = eta)) +
  geom_line(linewidth = 0.7) +
  labs(title = "Curvas CV-RMSE según learning rate (stress_level)",
       x = "iteración", y = "CV RMSE") +
  theme_bw() + theme(legend.position = "bottom")

## 5.3 Búsqueda de hiperparámetros con tidymodels
##     - CV repetida (3 x 5) para reducir varianza en el ranking
##     - Latin Hypercube en lugar de grid factorial (mejor cobertura)
##     - La receta incluye one-hot encoding de variables categóricas
modelo_tm <- boost_tree(
  mode           = "regression",
  trees          = tune(),
  tree_depth     = tune(),
  learn_rate     = tune(),
  loss_reduction = tune(),
  min_n          = tune(),
  mtry           = tune(),
  sample_size    = tune()
) %>% set_engine("xgboost")

# La receta maneja las variables categóricas con step_dummy (one-hot)
# y elimina columnas constantes si las hubiera (step_zv).
receta <- recipe(stress_level ~ ., data = data_train) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors())

wf <- workflow() %>% add_recipe(receta) %>% add_model(modelo_tm)

params_tune <- extract_parameter_set_dials(wf) %>%
  finalize(data_train %>% select(-stress_level))

set.seed(SEED)
cv_folds <- vfold_cv(data_train, v = 5, repeats = 3, strata = stress_level)

set.seed(SEED)
# grid_latin_hypercube fue deprecado en dials >= 1.3 a favor de grid_space_filling.
lhs_grid <- if (exists("grid_space_filling", where = asNamespace("dials"))) {
  dials::grid_space_filling(params_tune, size = 40, type = "latin_hypercube")
} else {
  dials::grid_latin_hypercube(params_tune, size = 40)
}

# detectCores() puede devolver NA en algunos entornos (contenedores, IDEs):
# protegemos con na.rm y un tope razonable.
n_cores <- suppressWarnings(parallel::detectCores())
if (is.na(n_cores) || n_cores < 2) n_cores <- 2
n_workers <- max(1L, min(n_cores - 1L, 4L))

cl <- NULL
ok_paralelo <- tryCatch({
  cl <- makePSOCKcluster(n_workers)
  registerDoParallel(cl)
  TRUE
}, error = function(e) {
  message("Paralelización no disponible (", conditionMessage(e), "); sigo en serie.")
  FALSE
})

set.seed(SEED)
grid_fit <- tune_grid(
  object    = wf,
  resamples = cv_folds,
  metrics   = metric_set(yardstick::rmse, yardstick::mae, yardstick::rsq),
  grid      = lhs_grid,
  control   = control_grid(save_pred = TRUE, verbose = FALSE)
)

if (ok_paralelo && !is.null(cl)) { stopCluster(cl); registerDoSEQ() }

print(show_best(grid_fit, metric = "rmse", n = 5))

mejores_hp <- select_best(grid_fit, metric = "rmse")
print(mejores_hp)

modelo_final_fit <- finalize_workflow(wf, mejores_hp) %>% fit(data = data_train)

pred_tm <- predict(modelo_final_fit, new_data = data_test)$.pred

# 6. Comparativa final con IC bootstrap --------------------------------------
preds_list <- list(
  rpart           = pred_rpart,
  Bagging         = pred_bag,
  RandomForest    = pred_rf,
  GBM             = pred_gbm,
  XGBoost         = pred_xgb,
  `XGBoost tuned` = pred_tm
)

comparativa <- imap_dfr(preds_list, function(p, n) {
  m <- boot_metrics(y_test, p); m$Modelo <- n; m
})

comparativa_wide <- comparativa %>%
  mutate(text = sprintf("%.3f [%.3f, %.3f]", est, lower, upper)) %>%
  select(Modelo, metric, text) %>%
  pivot_wider(names_from = metric, values_from = text)
print(comparativa_wide)

comparativa %>%
  filter(metric == "RMSE") %>%
  ggplot(aes(reorder(Modelo, est), est)) +
  geom_col(fill = "steelblue") +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2) +
  coord_flip() +
  labs(x = NULL, y = "RMSE en test (IC 95% bootstrap)",
       title = "Comparativa de modelos - stress_level") +
  theme_bw()

# 7. Diagnóstico de residuos del mejor modelo --------------------------------
rmse_x_modelo <- sapply(preds_list, function(p) Metrics::rmse(y_test, p))
mejor <- names(which.min(rmse_x_modelo))
cat("Mejor modelo en test:", mejor, "\n")

mejor_pred <- preds_list[[mejor]]
residuos   <- y_test - mejor_pred

p_a <- ggplot(data.frame(y_test, mejor_pred),
              aes(mejor_pred, y_test)) +
  geom_point(alpha = .6, color = "steelblue") +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(x = "Predicho", y = "Real",
       title = paste("Real vs Predicho -", mejor)) + theme_bw()
p_b <- ggplot(data.frame(mejor_pred, residuos),
              aes(mejor_pred, residuos)) +
  geom_point(alpha = .6, color = "steelblue") +
  geom_hline(yintercept = 0, color = "red") +
  labs(x = "Predicho", y = "Residuo",
       title = "Residuos vs Predicho") + theme_bw()
p_c <- ggplot(data.frame(residuos), aes(sample = residuos)) +
  stat_qq() + stat_qq_line(color = "red") +
  labs(title = "QQ-plot de residuos") + theme_bw()
ggarrange(p_a, p_b, p_c, nrow = 1)

# Test de normalidad y de sesgo medio:
shapiro_p <- shapiro.test(residuos)$p.value
t_p       <- t.test(residuos)$p.value
cat(sprintf("Residuos -> p Shapiro=%.3f | p t.test(mu=0)=%.3f\n",
            shapiro_p, t_p))

# 8. Importancia comparada RF vs XGBoost -------------------------------------
imp_rf_df <- as.data.frame(importance(modelo_rf)) %>%
  rownames_to_column("variable") %>%
  transmute(variable, RF = `%IncMSE`)

imp_xgb_df <- xgb.importance(model = modelo_xgb,
                             feature_names = colnames(X_train)) %>%
  as.data.frame() %>%
  transmute(variable = Feature, XGB = Gain * 100)

imp_join <- full_join(imp_rf_df, imp_xgb_df, by = "variable") %>%
  pivot_longer(-variable, names_to = "modelo", values_to = "importancia") %>%
  replace_na(list(importancia = 0))

ggplot(imp_join, aes(reorder(variable, importancia), importancia, fill = modelo)) +
  geom_col(position = "dodge") + coord_flip() +
  labs(x = NULL, y = "Importancia (%)",
       title = "Importancia de variables: RF vs XGBoost (stress_level)") +
  theme_bw() + theme(legend.position = "bottom")

# =============================================================================
# 9. Nested CV: estimación honesta del error de generalización  ---------------
# =============================================================================
# Justificación: tune_grid() + evaluación en data_test usa data_test de forma
# implícita para "elegir el ganador". Eso sesga la estimación a la baja. La CV
# anidada separa estrictamente el tuning (folds internos) de la evaluación
# (folds externos), por lo que el RMSE medio externo es un estimador insesgado
# de cómo generalizaría el PROCEDIMIENTO completo (no un modelo concreto).
# Coste: ~outer_v * (inner_v * grid_size) fits. Aquí: 5 * (3 * 20) = 300 fits.

set.seed(SEED)
outer_folds <- vfold_cv(datos, v = 5, strata = stress_level)

fit_outer_fold <- function(split) {
  train_in <- analysis(split)
  test_in  <- assessment(split)
  
  inner_folds <- vfold_cv(train_in, v = 3, strata = stress_level)
  inner_params <- extract_parameter_set_dials(wf) %>%
    finalize(train_in %>% select(-stress_level))
  inner_grid <- if (exists("grid_space_filling", where = asNamespace("dials"))) {
    dials::grid_space_filling(inner_params, size = 20, type = "latin_hypercube")
  } else {
    dials::grid_latin_hypercube(inner_params, size = 20)
  }
  
  inner_fit <- tune_grid(
    object    = wf,
    resamples = inner_folds,
    metrics   = metric_set(yardstick::rmse),
    grid      = inner_grid,
    control   = control_grid(verbose = FALSE)
  )
  best_in   <- select_best(inner_fit, metric = "rmse")
  fitted    <- finalize_workflow(wf, best_in) %>% fit(data = train_in)
  preds     <- predict(fitted, new_data = test_in)$.pred
  list(rmse = Metrics::rmse(test_in$stress_level, preds),
       hp   = best_in)
}

cat("Lanzando nested CV (puede tardar unos minutos)...\n")

# Re-activamos paralelización si estaba disponible
if (ok_paralelo) {
  cl2 <- tryCatch(makePSOCKcluster(n_workers), error = function(e) NULL)
  if (!is.null(cl2)) registerDoParallel(cl2)
}

nested_out <- lapply(outer_folds$splits, fit_outer_fold)

if (exists("cl2") && !is.null(cl2)) { stopCluster(cl2); registerDoSEQ() }

rmses_nested <- vapply(nested_out, `[[`, numeric(1), "rmse")

cat(sprintf(
  "\n>> Nested CV RMSE: %.3f ± %.3f  (min=%.3f, max=%.3f, n=%d folds)\n",
  mean(rmses_nested), sd(rmses_nested),
  min(rmses_nested), max(rmses_nested), length(rmses_nested)))

# Estabilidad de hiperparámetros entre folds: si varían mucho, el procedimiento
# es inestable y conviene aumentar el grid o las repeticiones internas.
hp_table <- do.call(rbind, lapply(nested_out, `[[`, "hp"))
cat("\nHiperparámetros elegidos en cada fold externo:\n")
print(hp_table)

# Comparación contra el RMSE de test "single split" — la diferencia es el sesgo
# optimista del enfoque clásico:
cat(sprintf(
  "\nRMSE single test split (XGB tuned): %.3f\nRMSE nested CV (medio):     %.3f\nDiferencia (sesgo optimista estimado): %.3f\n",
  Metrics::rmse(y_test, pred_tm),
  mean(rmses_nested),
  Metrics::rmse(y_test, pred_tm) - mean(rmses_nested)))