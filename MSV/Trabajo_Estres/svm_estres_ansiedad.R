# =============================================================================
# MÓDULO 0 ── Librerías y semilla
# =============================================================================

pkgs_requeridos <- c(
  "dplyr","tidyr","tibble","purrr","skimr","ggplot2","ggpubr",
  "readxl",
  "tidymodels",
  "e1071",
  "kernlab",
  "randomForest",
  "xgboost",
  "doParallel",
  "Metrics"
)

faltan <- pkgs_requeridos[!vapply(pkgs_requeridos, requireNamespace,
                                  logical(1), quietly = TRUE)]
if (length(faltan)) {
  stop("Instala primero con: install.packages(c(",
       paste0('"', faltan, '"', collapse = ", "), "))")
}

suppressPackageStartupMessages({
  library(dplyr);      library(tidyr);   library(tibble); library(purrr)
  library(skimr);      library(ggplot2); library(ggpubr)
  library(readxl)
  library(tidymodels)
  library(e1071)
  library(kernlab)
  library(randomForest)
  library(xgboost)
  library(doParallel)
  library(Metrics)
})

SEED <- 2024
set.seed(SEED)

cat("
=========================================================
  MÓDULO 0 completo: paquetes cargados, semilla =", SEED, "
=========================================================\n")


# =============================================================================
# MÓDULO 1 ── Carga, EDA y partición estratificada
# =============================================================================

# Carga del archivo Excel con datos de salud mental adolescente
datos <- read_excel("datosestresyansiedad.xlsx",
                    sheet = "Teen_Mental_Health_Dataset")

cat("\nDimensiones del dataset:", nrow(datos), "filas x", ncol(datos), "columnas\n")

# Preprocesamiento: convertir variables categóricas a numéricas/factor
datos <- datos %>%
  mutate(
    gender = as.integer(factor(gender)),                         # female=1, male=2
    platform_usage = as.integer(factor(platform_usage)),         # Both=1, Instagram=2, TikTok=3
    social_interaction_level = as.integer(factor(
      social_interaction_level, levels = c("low","medium","high")))  # low=1, med=2, high=3
  )

cat("\n--- EDA básico ---\n")
skim(datos)
stopifnot(sum(is.na(datos)) == 0)

# Variable objetivo: stress_level (continua 1–10)
ggplot(datos, aes(stress_level)) +
  geom_histogram(bins = 10, fill = "steelblue", color = "white", alpha = .85) +
  labs(title = "Distribución de stress_level (variable objetivo)",
       x = "Nivel de estrés (1–10)", y = "Frecuencia") +
  theme_bw()

if (requireNamespace("corrplot", quietly = TRUE)) {
  corrplot::corrplot(cor(datos), method = "color", type = "upper",
                     tl.col = "black", addCoef.col = "black", number.cex = 0.6)
} else {
  print(round(cor(datos), 2))
}

set.seed(SEED)
split_obj  <- initial_split(datos, prop = 0.8, strata = stress_level)
data_train <- training(split_obj)
data_test  <- testing(split_obj)

cat(sprintf("\nTrain: %d obs | Test: %d obs\n", nrow(data_train), nrow(data_test)))

y_train <- data_train$stress_level
y_test  <- data_test$stress_level

cat("MÓDULO 1 completo.\n")


# =============================================================================
# MÓDULO 2 ── SVM Lineal
# =============================================================================

cat("\n=== MÓDULO 2: SVM Lineal ===\n")

set.seed(SEED)
svm_lineal <- svm(stress_level ~ ., data = data_train,
                  kernel = "linear",
                  cost   = 1,
                  scale  = TRUE)

cat("Resumen del modelo lineal:\n")
print(svm_lineal)
cat(sprintf("Vectores de soporte usados: %d de %d observaciones (%.1f%%)\n",
            svm_lineal$tot.nSV, nrow(data_train),
            100 * svm_lineal$tot.nSV / nrow(data_train)))

pred_lineal <- predict(svm_lineal, newdata = data_test)

cat(sprintf("SVM Lineal  → RMSE: %.3f | MAE: %.3f\n",
            Metrics::rmse(y_test, pred_lineal),
            Metrics::mae (y_test, pred_lineal)))

cat("MÓDULO 2 completo.\n")


# =============================================================================
# MÓDULO 3 ── SVM Polinomial
# =============================================================================

cat("\n=== MÓDULO 3: SVM Polinomial ===\n")

resultados_poly <- data.frame()

for (d in c(2, 3, 4)) {
  set.seed(SEED)
  m <- svm(stress_level ~ ., data = data_train,
           kernel = "polynomial", degree = d,
           cost = 1, gamma = 0.1, coef0 = 0,
           scale = TRUE)
  p        <- predict(m, newdata = data_test)
  rmse_val <- Metrics::rmse(y_test, p)
  mae_val  <- Metrics::mae (y_test, p)
  resultados_poly <- rbind(resultados_poly,
                           data.frame(degree = d, SV = m$tot.nSV,
                                      RMSE = rmse_val, MAE = mae_val))
}

cat("Comparativa por grado del polinomio:\n")
print(resultados_poly)

best_d <- resultados_poly$degree[which.min(resultados_poly$RMSE)]
set.seed(SEED)
svm_poly <- svm(stress_level ~ ., data = data_train,
                kernel = "polynomial", degree = best_d,
                cost = 1, gamma = 0.1, coef0 = 0,
                scale = TRUE)
pred_poly <- predict(svm_poly, newdata = data_test)

cat(sprintf("SVM Poly (d=%d) → RMSE: %.3f | MAE: %.3f\n",
            best_d,
            Metrics::rmse(y_test, pred_poly),
            Metrics::mae (y_test, pred_poly)))

cat("MÓDULO 3 completo.\n")


# =============================================================================
# MÓDULO 4 ── SVM Radial base (RBF)
# =============================================================================

cat("\n=== MÓDULO 4: SVM Radial (RBF) base ===\n")

set.seed(SEED)
svm_radial_base <- svm(stress_level ~ ., data = data_train,
                       kernel = "radial",
                       cost  = 1,
                       gamma = 1 / ncol(data_train),
                       scale = TRUE)

pred_radial_base <- predict(svm_radial_base, newdata = data_test)

cat(sprintf("SVM Radial base → RMSE: %.3f | MAE: %.3f | SV: %d\n",
            Metrics::rmse(y_test, pred_radial_base),
            Metrics::mae (y_test, pred_radial_base),
            svm_radial_base$tot.nSV))

gamma_range <- c(0.001, 0.01, 0.1, 0.5, 1, 5)
df_gamma <- map_dfr(gamma_range, function(g) {
  set.seed(SEED)
  m <- svm(stress_level ~ ., data = data_train,
           kernel = "radial", cost = 1, gamma = g, scale = TRUE)
  p <- predict(m, newdata = data_test)
  data.frame(gamma = g,
             RMSE  = Metrics::rmse(y_test, p),
             SV    = m$tot.nSV)
})

cat("\nSensibilidad al parámetro gamma (C=1):\n")
print(df_gamma)

p_gamma <- ggplot(df_gamma, aes(log10(gamma), RMSE)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(size = 3, color = "steelblue") +
  labs(title = "SVM Radial: RMSE en test según γ (C=1) — stress_level",
       x = "log10(γ)", y = "RMSE") +
  theme_bw()
print(p_gamma)

cat("MÓDULO 4 completo.\n")


# =============================================================================
# MÓDULO 5 ── SVM Radial TUNEADO con tune() de e1071
# =============================================================================

cat("\n=== MÓDULO 5: SVM Radial tuneado con e1071::tune() ===\n")
cat("(puede tardar 1-3 minutos según equipo)\n")

set.seed(SEED)
tune_result <- tune(svm,
                    stress_level ~ .,
                    data   = data_train,
                    kernel = "radial",
                    scale  = TRUE,
                    ranges = list(
                      cost  = c(0.1, 1, 5, 10, 50, 100),
                      gamma = c(0.001, 0.01, 0.05, 0.1, 0.5)
                    ),
                    tunecontrol = tune.control(sampling = "cross",
                                               cross    = 5))

cat("\nResultado del tuning:\n")
print(tune_result)

# e1071::tune() reporta MSE (no RMSE) en regresión
cat(sprintf("Mejor C=%.3f | Mejor gamma=%.4f | MSE CV=%.4f (RMSE≈%.4f)\n",
            tune_result$best.parameters$cost,
            tune_result$best.parameters$gamma,
            tune_result$best.performance,
            sqrt(tune_result$best.performance)))

plot(tune_result, main = "e1071::tune() - Error CV por C y gamma")

svm_radial_tuned  <- tune_result$best.model
pred_radial_tuned <- predict(svm_radial_tuned, newdata = data_test)

cat(sprintf("SVM Radial tuned (e1071) → RMSE: %.3f | MAE: %.3f | SV: %d\n",
            Metrics::rmse(y_test, pred_radial_tuned),
            Metrics::mae (y_test, pred_radial_tuned),
            svm_radial_tuned$tot.nSV))

cat("MÓDULO 5 completo.\n")


# =============================================================================
# MÓDULO 6 ── SVM Radial con tidymodels (parsnip + kernlab)
# =============================================================================

cat("\n=== MÓDULO 6: SVM Radial con tidymodels ===\n")
cat("(puede tardar 2-5 minutos según equipo)\n")

receta_svm <- recipe(stress_level ~ ., data = data_train) %>%
  step_normalize(all_numeric_predictors())

svm_spec <- svm_rbf(
  mode      = "regression",
  cost      = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab")

wf_svm <- workflow() %>%
  add_recipe(receta_svm) %>%
  add_model(svm_spec)

set.seed(SEED)
cv_folds_svm <- vfold_cv(data_train, v = 5, repeats = 3, strata = stress_level)

set.seed(SEED)
svm_grid <- grid_latin_hypercube(
  cost(range = c(-6, 7)),          # log2: ~0.016 – 128
  rbf_sigma(range = c(-4, 0)),     # log10: 0.0001 – 1
  size = 30
)

n_cores   <- suppressWarnings(parallel::detectCores())
if (is.na(n_cores) || n_cores < 2) n_cores <- 2
n_workers <- max(1L, min(n_cores - 1L, 4L))

cl_svm <- tryCatch({
  cl_tmp <- makePSOCKcluster(n_workers)
  registerDoParallel(cl_tmp)
  cl_tmp
}, error = function(e) {
  message("Sin paralelización: ", conditionMessage(e))
  NULL
})

set.seed(SEED)
svm_grid_fit <- tune_grid(
  object    = wf_svm,
  resamples = cv_folds_svm,
  metrics   = metric_set(yardstick::rmse, yardstick::mae, yardstick::rsq),
  grid      = svm_grid,
  control   = control_grid(save_pred = TRUE, verbose = FALSE)
)

if (!is.null(cl_svm)) { stopCluster(cl_svm); registerDoSEQ() }

cat("\nMejores combinaciones (RMSE):\n")
print(show_best(svm_grid_fit, metric = "rmse", n = 5))

autoplot(svm_grid_fit)

mejores_svm <- select_best(svm_grid_fit, metric = "rmse")
cat(sprintf("Mejores HP → cost=%.4f | rbf_sigma=%.6f\n",
            mejores_svm$cost, mejores_svm$rbf_sigma))

modelo_svm_final <- finalize_workflow(wf_svm, mejores_svm) %>%
  fit(data = data_train)

pred_svm_tm <- predict(modelo_svm_final, new_data = data_test)$.pred

cat(sprintf("SVM Radial (tidymodels) → RMSE: %.3f | MAE: %.3f\n",
            Metrics::rmse(y_test, pred_svm_tm),
            Metrics::mae (y_test, pred_svm_tm)))

cat("MÓDULO 6 completo.\n")


# =============================================================================
# MÓDULO 7 ── Comparativa final con IC Bootstrap
# =============================================================================

cat("\n=== MÓDULO 7: Comparativa final con IC Bootstrap ===\n")

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

preds_svm <- list(
  "SVM Lineal"         = pred_lineal,
  "SVM Poly"           = pred_poly,
  "SVM Radial base"    = pred_radial_base,
  "SVM Radial e1071"   = pred_radial_tuned,
  "SVM Radial tidy"    = pred_svm_tm
)

comparativa_svm <- imap_dfr(preds_svm, function(p, n) {
  m <- boot_metrics(y_test, p); m$Modelo <- n; m
})

comparativa_wide <- comparativa_svm %>%
  mutate(texto = sprintf("%.3f [%.3f, %.3f]", est, lower, upper)) %>%
  select(Modelo, metric, texto) %>%
  pivot_wider(names_from = metric, values_from = texto)

cat("\n--- Comparativa SVM: punto e IC95% Bootstrap ---\n")
print(comparativa_wide)

comparativa_svm %>%
  filter(metric == "RMSE") %>%
  ggplot(aes(reorder(Modelo, est), est)) +
  geom_col(fill = "steelblue", alpha = .8) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.25) +
  coord_flip() +
  labs(x = NULL, y = "RMSE en test (IC 95% Bootstrap)",
       title = "Comparativa de variantes SVM — stress_level") +
  theme_bw()

cat("MÓDULO 7 completo.\n")


# =============================================================================
# MÓDULO 8 ── Diagnóstico de residuos del mejor SVM
# =============================================================================

cat("\n=== MÓDULO 8: Diagnóstico de residuos ===\n")

rmse_x_svm <- sapply(preds_svm, function(p) Metrics::rmse(y_test, p))
mejor_svm  <- names(which.min(rmse_x_svm))
mejor_pred <- preds_svm[[mejor_svm]]
residuos   <- y_test - mejor_pred

cat("Mejor variante SVM:", mejor_svm, "\n")

p_a <- ggplot(data.frame(y_test, mejor_pred),
              aes(mejor_pred, y_test)) +
  geom_point(alpha = .6, color = "steelblue") +
  geom_abline(slope = 1, intercept = 0, color = "black") +
  labs(x = "Predicho", y = "Real",
       title = paste("Real vs Predicho —", mejor_svm)) + theme_bw()

p_b <- ggplot(data.frame(mejor_pred, residuos),
              aes(mejor_pred, residuos)) +
  geom_point(alpha = .6, color = "steelblue") +
  geom_hline(yintercept = 0, color = "black") +
  labs(x = "Predicho", y = "Residuo",
       title = "Residuos vs Predicho") + theme_bw()

p_c <- ggplot(data.frame(residuos), aes(sample = residuos)) +
  stat_qq(color = "steelblue") + stat_qq_line(color = "black") +
  labs(title = "QQ-plot de residuos") + theme_bw()

ggarrange(p_a, p_b, p_c, nrow = 1)

shapiro_p <- shapiro.test(residuos)$p.value
t_p       <- t.test(residuos)$p.value
cat(sprintf("Residuos → p Shapiro=%.4f | p t.test(mu=0)=%.4f\n",
            shapiro_p, t_p))

cat("MÓDULO 8 completo.\n")


# =============================================================================
# MÓDULO 9 ── SVM vs RF vs XGBoost: comparativa cruzada
# =============================================================================

cat("\n=== MÓDULO 9: SVM vs RF vs XGBoost ===\n")

# Variables predictoras (sin stress_level)
vars_pred <- setdiff(names(data_train), "stress_level")

mtry_grid_rf  <- 2:(length(vars_pred))
oob_errors_rf <- sapply(mtry_grid_rf, function(m) {
  set.seed(SEED)
  rf <- randomForest(stress_level ~ ., data = data_train, mtry = m, ntree = 500)
  tail(rf$mse, 1)
})
mtry_opt_rf <- mtry_grid_rf[which.min(oob_errors_rf)]
cat("RF mtry óptimo:", mtry_opt_rf, "\n")

set.seed(SEED)
modelo_rf_comp <- randomForest(stress_level ~ ., data = data_train,
                               mtry = mtry_opt_rf, ntree = 500,
                               nodesize = 3, importance = TRUE)
pred_rf_comp <- predict(modelo_rf_comp, newdata = data_test)

X_train_xgb <- data_train %>% select(-stress_level) %>% data.matrix()
X_test_xgb  <- data_test  %>% select(-stress_level) %>% data.matrix()
dtrain_xgb  <- xgb.DMatrix(data = X_train_xgb, label = y_train)
dtest_xgb   <- xgb.DMatrix(data = X_test_xgb,  label = y_test)

params_xgb <- list(
  objective = "reg:squarederror", eval_metric = "rmse",
  eta = 0.05, max_depth = 5, min_child_weight = 3,
  subsample = 0.8, colsample_bytree = 0.8, gamma = 0
)
set.seed(SEED)
cv_xgb <- xgb.cv(params = params_xgb, data = dtrain_xgb,
                 nrounds = 2000, nfold = 5,
                 early_stopping_rounds = 50,
                 print_every_n = 500, verbose = 1)
nr_xgb <- cv_xgb$best_iteration
if (is.null(nr_xgb) || is.na(nr_xgb) || nr_xgb < 1)
  nr_xgb <- which.min(cv_xgb$evaluation_log$test_rmse_mean)

set.seed(SEED)
modelo_xgb_comp <- xgb.train(params = params_xgb, data = dtrain_xgb,
                             nrounds = nr_xgb, verbose = 0)
pred_xgb_comp <- predict(modelo_xgb_comp, dtest_xgb)

preds_todos <- list(
  "SVM Lineal"               = pred_lineal,
  "SVM Radial (e1071 tuned)" = pred_radial_tuned,
  "SVM Radial (tidy)"        = pred_svm_tm,
  "Random Forest"            = pred_rf_comp,
  "XGBoost"                  = pred_xgb_comp
)

comparativa_total <- imap_dfr(preds_todos, function(p, n) {
  m <- boot_metrics(y_test, p); m$Modelo <- n; m
})

tabla_final <- comparativa_total %>%
  mutate(texto = sprintf("%.3f [%.3f, %.3f]", est, lower, upper)) %>%
  select(Modelo, metric, texto) %>%
  pivot_wider(names_from = metric, values_from = texto)

cat("\n=== TABLA FINAL: SVM vs RF vs XGBoost (IC 95% Bootstrap) ===\n")
print(tabla_final)

comparativa_total %>%
  filter(metric == "RMSE") %>%
  mutate(familia = case_when(
    grepl("SVM", Modelo)    ~ "SVM",
    grepl("Forest", Modelo) ~ "RandomForest",
    TRUE                    ~ "XGBoost"
  )) %>%
  ggplot(aes(reorder(Modelo, est), est, fill = familia)) +
  geom_col(alpha = .85) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.25) +
  scale_fill_manual(values = c("SVM"          = "steelblue",
                               "RandomForest"  = "seagreen",
                               "XGBoost"       = "darkorange")) +
  coord_flip() +
  labs(x = NULL, y = "RMSE en test (IC 95% Bootstrap)",
       title = "SVM vs Random Forest vs XGBoost — stress_level",
       fill = "Familia") +
  theme_bw() + theme(legend.position = "bottom")

cat("MÓDULO 9 completo.\n")


# =============================================================================
# MÓDULO 10 ── Vectores de Soporte: análisis e interpretación
# =============================================================================

cat("\n=== MÓDULO 10: Análisis de Vectores de Soporte ===\n")

idx_sv  <- svm_radial_tuned$index
sv_data <- data_train[idx_sv, ]
no_sv   <- data_train[-idx_sv, ]

cat(sprintf(
  "Vectores de soporte: %d (%.1f%%)  |  No SV: %d (%.1f%%)\n",
  nrow(sv_data), 100 * nrow(sv_data) / nrow(data_train),
  nrow(no_sv),   100 * nrow(no_sv)   / nrow(data_train)))

sv_data$tipo <- "Vector de Soporte"
no_sv$tipo   <- "No Soporte"
datos_tipo   <- rbind(sv_data, no_sv)

ggplot(datos_tipo, aes(stress_level, fill = tipo)) +
  geom_histogram(bins = 10, position = "identity", alpha = 0.55) +
  scale_fill_manual(values = c("Vector de Soporte" = "steelblue",
                               "No Soporte"        = "darkorange")) +
  labs(title = "Distribución de stress_level: Vectores de Soporte vs Resto",
       subtitle = "Los SV tienden a estar en zonas de mayor incertidumbre",
       x = "stress_level", y = "Frecuencia", fill = "") +
  theme_bw() + theme(legend.position = "bottom")

cat("\nEstadísticos stress_level — Vectores de Soporte:\n")
print(summary(sv_data$stress_level))
cat("Estadísticos stress_level — Resto de observaciones:\n")
print(summary(no_sv$stress_level))

df_sv_cost <- map_dfr(c(0.1, 0.5, 1, 5, 10, 50, 100), function(c_val) {
  set.seed(SEED)
  m <- svm(stress_level ~ ., data = data_train,
           kernel = "radial",
           cost  = c_val,
           gamma = tune_result$best.parameters$gamma,
           scale = TRUE)
  p      <- predict(m, newdata = data_test)
  rmse_v <- Metrics::rmse(y_test, p)
  data.frame(C = c_val,
             SV_pct = 100 * m$tot.nSV / nrow(data_train),
             RMSE   = rmse_v)
})

cat("\nRelación C → % Vectores de Soporte → RMSE test:\n")
print(df_sv_cost)

p_sv1 <- ggplot(df_sv_cost, aes(log10(C), SV_pct)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(size = 3, color = "steelblue") +
  labs(title = "C ↑  →  SV ↓  (margen más estrecho)",
       x = "log10(C)", y = "% Vectores de Soporte") +
  theme_bw()

p_sv2 <- ggplot(df_sv_cost, aes(log10(C), RMSE)) +
  geom_line(color = "darkorange", linewidth = 1) +
  geom_point(size = 3, color = "darkorange") +
  labs(title = "RMSE en test según C — stress_level",
       x = "log10(C)", y = "RMSE") +
  theme_bw()

ggarrange(p_sv1, p_sv2, nrow = 1)

cat("MÓDULO 10 completo.\n")


cat("
=================================================================
  SCRIPT COMPLETO EJECUTADO
  Módulos: 0-10 (SVM Lineal, Poly, Radial, Tuning, Tidymodels,
           Comparativa, Residuos, SVM vs RF vs XGB, SV Analysis)
  Base de datos: datosestresyansiedad.xlsx
  Variable objetivo: stress_level (cuantitativa 1-10)
=================================================================\n")
