# Load required libraries
library(caret)
library(e1071)
library(adabag)
library(dplyr)

# Set the working directory (adjust to your path)
setwd("C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Final project")

# Load your dataset
df <- read.csv("thyroid_cancer_risk_data.csv")

# Preprocess the data (example steps: dropping ID, recoding, and handling NAs)
df = df[-1]  # Drop ID column
df = df[-15] # Drop Risk Category

# Check column names to verify if 'Diagnosis' exists
colnames(df)

# Check categorical variables
cat_vars <- names(df)[sapply(df, is.character) | sapply(df, is.factor)]
categories <- lapply(df[cat_vars], unique)
print(categories)

# Define columns
one_hot_cols <- c("Country", "Ethnicity")
binary_cols <- c("Gender", "Family_History", "Radiation_Exposure", 
                 "Iodine_Deficiency", "Smoking", "Obesity", "Diabetes", "Diagnosis")

# Manual binary mapping
df <- df %>%
  mutate(across(all_of(binary_cols), ~ recode(., 
                                              "No" = 0, "Yes" = 1, 
                                              "Male" = 0, "Female" = 1,
                                              "Benign" = 0, "Malignant" = 1))) %>%
  mutate(across(all_of(binary_cols), as.numeric)) 


# Split features and response
y <- as.factor(df$Diagnosis)
X <- df %>% select(-Diagnosis)

# Stratified Train-Test Split (80-20)
set.seed(42)
train_idx <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_idx, ]
y_train <- y[train_idx]
X_test <- X[-train_idx, ]
y_test <- y[-train_idx]

# Train Control
ctrl <- trainControl(method = "cv", number = 5)

# AdaBoost Model (using the adabag package)
adaboost_model <- boosting(Diagnosis ~ ., data = data.frame(X_train, Diagnosis = y_train), 
                           boos = TRUE, 
                           mfinal = 50)  # Number of boosting iterations

# Evaluate the model
adaboost_pred_train <- predict(adaboost_model, newdata = X_train)
adaboost_pred_test <- predict(adaboost_model, newdata = X_test)

# Train Metrics
train_accuracy <- mean(adaboost_pred_train$class == y_train)
test_accuracy <- mean(adaboost_pred_test$class == y_test)

cat("Train Accuracy: ", train_accuracy, "\n")
cat("Test Accuracy: ", test_accuracy, "\n")

# For more detailed evaluation, you can use the caret's confusionMatrix function:
train_cm <- confusionMatrix(adaboost_pred_train$class, y_train)
test_cm <- confusionMatrix(adaboost_pred_test$class, y_test)

cat("Train Confusion Matrix:\n")
print(train_cm)

cat("Test Confusion Matrix:\n")
print(test_cm)

# If you want to see model performance on other metrics like F1, precision, etc.:
train_metrics <- get_metrics(y_train, adaboost_pred_train$class, "AdaBoost", "Train")
test_metrics <- get_metrics(y_test, adaboost_pred_test$class, "AdaBoost", "Test")

cat("Train Metrics:\n")
print(train_metrics)

cat("Test Metrics:\n")
print(test_metrics)

# Helper function to calculate evaluation metrics
get_metrics <- function(true, pred, model_name, set_type) {
  acc <- Accuracy(pred, true)
  prec <- Precision(pred, true, positive = "1")
  rec <- Recall(pred, true, positive = "1")
  f1 <- F1_Score(pred, true, positive = "1")
  data.frame(Model = model_name, Set = set_type, Accuracy = acc, Precision = prec, Recall = rec, F1 = f1)
}