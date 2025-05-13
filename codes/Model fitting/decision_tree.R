# Libraries
library(dplyr)
library(caret)
library(smotefamily)
library(rpart)
library(MLmetrics)
library(fastDummies)
library(ggplot2)
library(pROC)

# Set seed for reproducibility
set.seed(42)

# Load Data
df <- read.csv("thyroid_cancer_risk_data.csv")
df = df[-16]  # Drop risk category
View(df)

# Binary columns (including Diagnosis)
binary_cols <- c("Gender", "Family_History", "Radiation_Exposure", 
                 "Iodine_Deficiency", "Smoking", "Obesity", "Diabetes", "Diagnosis")

# Categorical columns for dummy encoding
one_hot_cols <- c("Country", "Ethnicity")

# Encode binary variables
df <- df %>%
  mutate(across(all_of(binary_cols), ~ recode(., 
                                              "No" = 0, "Yes" = 1,
                                              "Male" = 0, "Female" = 1,
                                              "Benign" = 0, "Malignant" = 1))) %>%
  mutate(across(all_of(binary_cols), as.numeric))

# Store target separately
y <- as.factor(df$Diagnosis)

# Remove Diagnosis before one-hot encoding
df <- df %>% select(-Diagnosis)

# One-hot encode categorical predictors
df <- fastDummies::dummy_cols(df, select_columns = one_hot_cols,
                              remove_selected_columns = TRUE,
                              remove_first_dummy = TRUE)

# Final feature matrix
X <- df

# Stratified Train-Test Split
set.seed(42)
train_idx <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_idx, ]
y_train <- y[train_idx]
X_test <- X[-train_idx, ]
y_test <- y[-train_idx]

cat("Class distribution (Train):\n")
print(table(y_train))

# Ensure all features are numeric (including dummy vars)
X_train_numeric <- X_train %>% mutate(across(everything(), as.numeric))

# Check for NAs
cat("Missing values in X_train_numeric:\n")
print(sum(is.na(X_train_numeric)))  

# Impute NAs if necessary
X_train_numeric[is.na(X_train_numeric)] <- 0  

sapply(X_train_numeric, class)

# Apply SMOTE
data_smote <- SMOTE(X_train_numeric, y_train, K = 5, dup_size = 0)$data

# Extract SMOTE'd features and target
X_train_smote <- data_smote %>% select(-class)
y_train_smote <- as.factor(data_smote$class)

# Re-align X_test columns to match X_train after SMOTE
# Add any missing dummy columns to X_test
missing_cols <- setdiff(colnames(X_train_smote), colnames(X_test))
for (col in missing_cols) {
  X_test[[col]] <- 0
}

# Drop any extra columns that exist in X_test but not in X_train
extra_cols <- setdiff(colnames(X_test), colnames(X_train_smote))
X_test <- X_test[, !(names(X_test) %in% extra_cols)]

# Reorder test columns to match training columns
X_test <- X_test[, colnames(X_train_smote)]

# Ensure numeric columns for test set too
X_test <- X_test %>% mutate(across(everything(), as.numeric))

# Reapply scaler to make sure test set is standardized
num_cols <- sapply(X_train_smote, is.numeric)
scaler <- preProcess(X_train_smote[, num_cols], method = c("center", "scale"))
X_train_smote[, num_cols] <- predict(scaler, X_train_smote[, num_cols])
X_test[, num_cols] <- predict(scaler, X_test[, num_cols])

# Class balance
ggplot(data.frame(Class = y_train_smote), aes(x = Class)) +
  geom_bar(fill = "skyblue") +
  ggtitle("SMOTE Class Distribution") +
  theme_minimal()

# Basic Decision Tree
tree_base <- rpart(y_train_smote ~ ., data = data.frame(X_train_smote, y_train_smote), method = "class")

# Evaluate
evaluate_model <- function(model, X_train, y_train, X_test, y_test, name = "Model") {
  preds <- predict(model, X_test, type = "class")
  preds <- factor(preds, levels = levels(y_test))  # Ensure matching factor levels
  
  acc <- Accuracy(preds, y_test)
  f1 <- F1_Score(y_test, preds, positive = "1")
  cat(paste0(name, " Accuracy: ", round(acc, 3), "\n"))
  cat(paste0(name, " F1 Score: ", round(f1, 3), "\n"))
  
  return(list(accuracy = acc, f1 = f1))
}

tree_metrics <- evaluate_model(tree_base, X_train_smote, y_train_smote, X_test, y_test, "Decision Tree")

confusionMatrix(preds, y_test)

# View the tree
rpart.plot(tree_base)

# Tuned Decision Tree 
ctrl <- trainControl(method = "cv", number = 5)

tree_tuned <- train(x = X_train_smote, y = y_train_smote,
                    method = "rpart",
                    trControl = ctrl,
                    tuneGrid = expand.grid(cp = 0.001),
                    control = rpart.control(maxdepth = 10))

# Evaluate tuned model
tree_tuned_metrics <- evaluate_model(tree_tuned, X_train_smote, y_train_smote, X_test, y_test, "Decision Tree (Tuned)")

# Plot important variables
plot(varImp(tree_tuned), top = 10, main = "Top 10 Important - Tree")

# ROC Curve
probs <- predict(tree_base, X_test, type = "prob")[, "1"]
roc_obj <- roc(y_test, probs)
auc(roc_obj)
plot(roc_obj, main = "ROC Curve - Decision Tree")
