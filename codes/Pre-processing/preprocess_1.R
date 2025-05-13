# Load libraries
library(dplyr)
library(caret)
library(smotefamily)
library(e1071)
library(rpart)
library(naivebayes)
library(MLmetrics)

setwd("C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Final project")

df=read.csv("C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Final project/thyroid_cancer_risk_data.csv")

# Check for missing values
cat("Missing Values:\n")
print(colSums(is.na(df)))

# Check for duplicates
cat("\nNumber of duplicate rows:", sum(duplicated(df)), "\n")

# Check categorical columns
cat_vars <- names(df)[sapply(df, is.character) | sapply(df, is.factor)]
categories <- lapply(df[cat_vars], unique)
print(categories)

# Specify categorical encoding
one_hot_cols <- c("Country", "Ethnicity")
binary_cols <- c("Gender", "Family_History", "Radiation_Exposure", 
                 "Iodine_Deficiency", "Smoking", "Obesity", "Diabetes", "Diagnosis")

# Manual binary mapping
df <- df %>%
  mutate(across(all_of(binary_cols), ~ recode(., "No"=0, "Yes"=1, "Male"=0, "Female"=1, "Benign"=0, "Malignant"=1)))

# One-hot encoding (avoid dummy variable trap by removing first level)
df <- df %>%
  mutate(across(all_of(one_hot_cols), as.factor)) %>%
  fastDummies::dummy_cols(select_columns = one_hot_cols, remove_selected_columns = TRUE, remove_first_dummy = TRUE)

# Split features and response
y <- as.factor(df$Diagnosis)
X <- df %>% select(-Diagnosis)

# Train-Test Split (stratified)
set.seed(42)
train_idx <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_idx, ]
y_train <- y[train_idx]
X_test <- X[-train_idx, ]
y_test <- y[-train_idx]

# Check class balance
cat("Class distribution (Train):\n")
print(table(y_train))

# Apply SMOTE
data_smote <- SMOTE(X_train, y_train, K = 5, dup_size = 0)$data
X_train_smote <- data_smote %>% select(-class)
y_train_smote <- as.factor(data_smote$class)

# Check new class distribution
cat("\nClass distribution after SMOTE:\n")
print(table(y_train_smote))

# Standardize numeric features
num_cols <- c("Age", "TSH_Level", "T4_Level", "T3_Level", "Nodule_Size")
scaler <- preProcess(X_train_smote[, num_cols], method = c("center", "scale"))
X_train_smote[, num_cols] <- predict(scaler, X_train_smote[, num_cols])
X_test[, num_cols] <- predict(scaler, X_test[, num_cols])

# ---------------------
# 🔮 MODELING SECTION
# ---------------------

# SVM
svm_model <- svm(x = X_train_smote, y = y_train_smote, probability = TRUE)
svm_pred <- predict(svm_model, X_test)

# Classification Tree
tree_model <- rpart(y_train_smote ~ ., data = data.frame(X_train_smote, y_train_smote), method = "class")
tree_pred <- predict(tree_model, X_test, type = "class")

# Naive Bayes
nb_model <- naive_bayes(x = X_train_smote, y = y_train_smote)
nb_pred <- predict(nb_model, X_test)

# ---------------------
# 📊 EVALUATION
# ---------------------
evaluate_model <- function(true, pred, model_name) {
  cat(paste0("\n", model_name, " Evaluation:\n"))
  print(confusionMatrix(pred, true))
}

evaluate_model(y_test, svm_pred, "SVM")
evaluate_model(y_test, tree_pred, "Classification Tree")
evaluate_model(y_test, nb_pred, "Naive Bayes")


