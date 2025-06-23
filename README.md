# ----------------------------
# 1. Load Libraries
# ----------------------------
packages <- c("tidyverse", "caret", "xgboost", "randomForest", "e1071", "nnet",
              "class", "ggplot2", "reshape2", "data.table", "Metrics", "rpart", "gridExtra")
lapply(packages, require, character.only = TRUE)

# ----------------------------
# 2. Load Dataset
# ----------------------------
data <- read.csv("AI_Realistic_Merged_Dataset.csv")

# ----------------------------
# 3. Data Cleaning & Prep
# ----------------------------
data <- data %>% mutate(across(where(is.character), as.factor))
data$Suitability <- as.numeric(factor(data$Suitability, levels = c("Poor", "Moderate", "Good")))
data <- na.omit(data)
nzv <- nearZeroVar(data)
if (length(nzv) > 0) data <- data[, -nzv]
if ("well_id" %in% colnames(data)) data <- data %>% select(-well_id)

set.seed(42)
train_index <- createDataPartition(data$Suitability, p = 0.8, list = FALSE)
train <- data[train_index, ]
test  <- data[-train_index, ]
train_y <- train$Suitability
test_y  <- test$Suitability
train_x <- train %>% select(-Suitability)
test_x  <- test %>% select(-Suitability)

train_matrix <- model.matrix(~ . -1, data = train_x)
test_matrix  <- model.matrix(~ . -1, data = test_x)

# ----------------------------
# 4. Evaluation Function
# ----------------------------
eval_model <- function(preds, actuals, label = "Model") {
  rmse <- sqrt(mean((preds - actuals)^2))
  r2 <- cor(preds, actuals)^2
  data.frame(Model = label, RMSE = rmse, R2 = r2)
}

results <- list()
predictions_list <- list()

# ----------------------------
# 5. Models
# ----------------------------

# Linear Regression
lm_model <- lm(Suitability ~ ., data = train)
lm_preds <- predict(lm_model, newdata = test)
results[["Linear Regression"]] <- eval_model(lm_preds, test_y, "Linear Regression")
predictions_list[["Linear Regression"]] <- lm_preds

# Decision Tree
dt_model <- rpart(Suitability ~ ., data = train)
dt_preds <- predict(dt_model, test)
results[["Decision Tree"]] <- eval_model(dt_preds, test_y, "Decision Tree")
predictions_list[["Decision Tree"]] <- dt_preds

# Random Forest
rf_model <- randomForest(x = train_matrix, y = train_y, ntree = 500)
rf_preds <- predict(rf_model, newdata = test_matrix)
results[["Random Forest"]] <- eval_model(rf_preds, test_y, "Random Forest")
predictions_list[["Random Forest"]] <- rf_preds

# XGBoost
dtrain <- xgb.DMatrix(data = train_matrix, label = train_y)
dtest  <- xgb.DMatrix(data = test_matrix)
params <- list(objective = "reg:squarederror", eta = 0.1, max_depth = 6,
               subsample = 0.8, colsample_bytree = 0.8)
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 100)
xgb_preds <- predict(xgb_model, dtest)
results[["XGBoost"]] <- eval_model(xgb_preds, test_y, "XGBoost")
predictions_list[["XGBoost"]] <- xgb_preds

# SVM
svm_model <- svm(Suitability ~ ., data = train)
svm_preds <- predict(svm_model, test)
results[["SVM"]] <- eval_model(svm_preds, test_y, "SVM")
predictions_list[["SVM"]] <- svm_preds

# KNN
knn_preds <- knn(train = train_matrix, test = test_matrix, cl = train_y, k = 5)
knn_preds <- as.numeric(knn_preds)
results[["KNN"]] <- eval_model(knn_preds, test_y, "KNN")
predictions_list[["KNN"]] <- knn_preds

# Neural Network
nn_model <- nnet(Suitability ~ ., data = train, size = 3, linout = TRUE, trace = FALSE, maxit = 300)
nn_preds <- predict(nn_model, test)
results[["Neural Net"]] <- eval_model(nn_preds, test_y, "Neural Net")
predictions_list[["Neural Net"]] <- nn_preds

# ----------------------------
# 6. Combine & Print Results
# ----------------------------
results_df <- do.call(rbind, results)
print("Model Performance Summary:")
print(results_df)

# ----------------------------
# 7. Plots
# ----------------------------

# RMSE vs R2 plot
ggplot(results_df, aes(x = RMSE, y = R2, label = Model, color = Model)) +
  geom_point(size = 4) +
  geom_text(nudge_y = 0.01) +
  theme_minimal() +
  labs(title = "RMSE vs R² Comparison", x = "RMSE", y = "R²")

# RMSE Bar Plot
ggplot(results_df, aes(x = reorder(Model, RMSE), y = RMSE, fill = Model)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Model RMSE Comparison", y = "RMSE", x = "")

# R² Bar Plot
results_df_clean <- results_df[!is.na(results_df$R2), ]
ggplot(results_df_clean, aes(x = reorder(Model, R2), y = R2, fill = Model)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Model R² Comparison", y = "R²", x = "")

# Feature Importance (Random Forest)
rf_imp <- importance(rf_model)
rf_imp_df <- data.frame(Feature = rownames(rf_imp), Importance = rf_imp[, 1])
ggplot(rf_imp_df[order(-rf_imp_df$Importance)[1:15], ], aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 15 Feature Importances (Random Forest)", y = "Importance", x = "") +
  theme_minimal()

# Correlation Heatmap
cor_matrix <- cor(train_matrix)
melted_cor <- melt(cor_matrix)
ggplot(melted_cor, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, name = "Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90), axis.text.y = element_text(size = 6)) +
  labs(title = "Correlation Heatmap")

# Scatter Plots: Actual vs Predicted
for (model_name in names(predictions_list)) {
  df_plot <- data.frame(Actual = test_y, Predicted = predictions_list[[model_name]])
  p <- ggplot(df_plot, aes(x = Actual, y = Predicted)) +
    geom_point(alpha = 0.6, color = "darkgreen") +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
    theme_minimal() +
    labs(title = paste("Actual vs Predicted -", model_name))
  print(p)
}
saveRDS
