library(dplyr)
library(caret)
library(smotefamily)
library(e1071)
library(MLmetrics)
library(fastDummies)
library(kernlab)
library(ggplot2)
library(reshape2)

set.seed(42)

# Set working directory
setwd("C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Final project")

# Load data
df <- read.csv("thyroid_cancer_risk_data.csv")

df = df[-16]  # Drop risk category
df = df[-1]   # Drop ID

# Data Check
cat("Missing Values:\n")
print(colSums(is.na(df)))

cat("\nNumber of duplicate rows:", sum(duplicated(df)), "\n")

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
  mutate(across(all_of(binary_cols), as.numeric))  # Ensure binary as numeric

# One-hot encode (removes multicollinearity)
df <- df %>%
  mutate(across(all_of(one_hot_cols), as.factor)) %>%
  fastDummies::dummy_cols(select_columns = one_hot_cols, remove_selected_columns = TRUE, remove_first_dummy = TRUE)

View(df)

# Split features and response
y <- as.factor(df$Diagnosis)
X <- df %>% select(-Diagnosis)

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

# Standardize numeric features
num_cols <- c("Age", "TSH_Level", "T4_Level", "T3_Level", "Nodule_Size")
scaler <- preProcess(X_train_smote[, num_cols], method = c("center", "scale"))
X_train_smote[, num_cols] <- predict(scaler, X_train_smote[, num_cols])
X_test[, num_cols] <- predict(scaler, X_test[, num_cols])

print(X_train_smote)
print(X_test)

# Helper to get metrics (unchanged)
get_metrics <- function(true, pred, model_name, set_type) {
  acc <- Accuracy(pred, true)
  prec <- Precision(pred, true, positive = "1")
  rec <- Recall(pred, true, positive = "1")
  f1 <- F1_Score(pred, true, positive = "1")
  data.frame(Model = model_name, Set = set_type, Accuracy = acc, Precision = prec, Recall = rec, F1 = f1)
}

# Updated evaluation for base SVM (e1071)
evaluate_svm_model <- function(model, x_train, y_train, x_test, y_test, model_name) {
  # Predict using type = "class"
  pred_train <- predict(model, x_train, type = "raw")
  pred_test <- predict(model, x_test, type = "raw")
  
  # Ensure predictions are factors with correct levels
  pred_train <- factor(pred_train, levels = levels(y_train))
  pred_test <- factor(pred_test, levels = levels(y_test))
  
  rbind(
    get_metrics(y_train, pred_train, model_name, "Train"),
    get_metrics(y_test, pred_test, model_name, "Test")
  )
}

# Linear, Polynomial and RBF kernels with small dataset

# Set seed for reproducibility
set.seed(42)

# Sample a subset 
sample_idx <- sample(nrow(X_train_smote), 5000)
X_train_large <- X_train_smote[sample_idx, ]
y_train_large <- y_train_smote[sample_idx]

# TrainControl for tuning 
ctrl <- trainControl(method = "cv", number = 5)

# Linear SVM Model
svm_linear_tuned <- train(
  x = X_train_large, y = y_train_large,
  method = "svmLinear",
  trControl = ctrl,
  preProcess = c("center", "scale"),
  tuneLength = 3
)

# Polynomial SVM Model 
svm_poly_tuned <- train(
  x = X_train_large, y = y_train_large,
  method = "svmPoly",
  trControl = ctrl,
  preProcess = c("center", "scale"),
  tuneGrid = expand.grid(
    degree = c(2, 3),
    scale = c(0.001, 0.01, 0.1), 
    C = c(0.5, 1, 2)
  )
)

# RBF SVM Model 
svm_rbf_tuned <- train(
  x = X_train_large, y = y_train_large,
  method = "svmRadial",
  trControl = ctrl,
  preProcess = c("center", "scale"),
  tuneGrid = expand.grid(
    sigma = c(0.001, 0.01, 0.1),  
    C = c(0.5, 1, 2)
  )
)

# Evaluate the models
svm_linear_metrics <- evaluate_svm_model(svm_linear_tuned, X_train_large, y_train_large, X_test, y_test, "SVM (Tuned Linear)")
svm_poly_metrics <- evaluate_svm_model(svm_poly_tuned, X_train_large, y_train_large, X_test, y_test, "SVM (Tuned Poly)")
svm_rbf_metrics <- evaluate_svm_model(svm_rbf_tuned, X_train_large, y_train_large, X_test, y_test, "SVM (Tuned RBF)")

# Combine results
svm_results_all <- rbind(svm_linear_metrics, svm_poly_metrics, svm_rbf_metrics)
print(svm_results_all)

# Plot F1 for all models
ggplot(svm_results_all, aes(x = Model, y = F1, fill = Set)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "SVM Comparison (F1 Score)", y = "F1 Score") +
  coord_flip()

#Variable Importance

# Get variable importance from linear SVM (caret)
imp <- varImp(svm_linear_tuned)$importance
print(imp)
imp$Variable <- rownames(imp)
colnames(imp)
imp$Overall <- rowMeans(imp[, c("X0", "X1")])

# Group dummy variables to their original variable
imp$Group <- gsub("_.*", "", imp$Variable)  # e.g., Country_US -> Country

print(imp$Group)

# Group by variable name
grouped_imp <- imp %>%
  group_by(Group) %>%
  summarise(GroupImportance = mean(Overall)) %>%
  arrange(desc(GroupImportance))

print(grouped_imp)

#Plot variable importance

ggplot(grouped_imp, aes(x = reorder(Group, GroupImportance), y = GroupImportance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Grouped Variable Importance for Linear SVM", x = "Variable", y = "Importance")

#Top important variables 

important_vars <- grouped_imp %>%
  filter(GroupImportance > 10) %>%
  pull(Group)

cat("Selected variables:\n")
print(important_vars)

# Get feature names that match the selected groups
selected_features <- imp %>%
  filter(gsub("(_.*)", "", Variable) %in% important_vars) %>%
  pull(Variable)

print(selected_features)

#Subset

X_train_selected  <- X_train_smote[, selected_features, drop = FALSE]
X_test_selected<- X_test[, selected_features, drop = FALSE]

common_vars <- intersect(colnames(X_train_selected), colnames(X_test_selected))
X_train_selected <- X_train_selected[, common_vars]
X_test_selected <- X_test_selected[, common_vars]

# Linear SVM
svm_linear_selected <- train(
  x = X_train_selected, y = y_train_smote,
  method = "svmLinear",
  trControl = ctrl,
  preProcess = c("center", "scale"),
  tuneLength = 3
)


# Get variable importance from polynomial SVM (caret)
imp_p <- varImp(svm_poly_tuned)$importance
print(imp_p)
imp_p$Variable <- rownames(imp_p)
colnames(imp_p)
imp_p$Overall <- rowMeans(imp_p[, c("X0", "X1")])

# Group dummy variables to their original variable
imp_p$Group <- gsub("_.*", "", imp_p$Variable)  # e.g., Country_US -> Country

print(imp_p$Group)

# Group by variable name
grouped_imp_p <- imp_p %>%
  group_by(Group) %>%
  summarise(GroupImportance = mean(Overall)) %>%
  arrange(desc(GroupImportance))

print(grouped_imp_p)

#Plot variable importance

ggplot(grouped_imp_p, aes(x = reorder(Group, GroupImportance), y = GroupImportance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Grouped Variable Importance for Polynomial SVM", x = "Variable", y = "Importance")

#Top important variables 

important_vars_p <- grouped_imp_p %>%
  filter(GroupImportance > 10) %>%
  pull(Group)

cat("Selected variables:\n")
print(important_vars_p)

# Get feature names that match the selected groups
selected_features_p <- imp_p %>%
  filter(gsub("(_.*)", "", Variable) %in% important_vars_p) %>%
  pull(Variable)

print(selected_features_p)

#Subset

X_train_selected_p  <- X_train_smote[, selected_features_p, drop = FALSE]
X_test_selected_p <- X_test[, selected_features_p, drop = FALSE]

common_vars_p <- intersect(colnames(X_train_selected_p), colnames(X_test_selected_p))
X_train_selected_p <- X_train_selected_p[, common_vars_p]
X_test_selected_p <- X_test_selected_p[, common_vars_p]


# Polynomial SVM
svm_poly_selected <- train(
  x = X_train_selected_p, y = y_train_smote,
  method = "svmPoly",
  trControl = ctrl,
  preProcess = c("center", "scale"),
  tuneGrid = expand.grid(
    degree = c(2, 3),
    scale = c(0.001, 0.01, 0.1),
    C = c(0.5, 1, 2)
  )
)

# Get variable importance from RBF SVM (caret)
imp_r <- varImp(svm_rbf_tuned)$importance
print(imp_r)
imp_r$Variable <- rownames(imp_r)
colnames(imp_r)
imp_r$Overall <- rowMeans(imp_r[, c("X0", "X1")])

# Group dummy variables to their original variable
imp$Group_r <- gsub("_.*", "", imp_r$Variable)  # e.g., Country_US -> Country

print(imp_r$Group)

# Group by variable name
grouped_imp_r <- imp_r %>%
  group_by(Group) %>%
  summarise(GroupImportance = mean(Overall)) %>%
  arrange(desc(GroupImportance))

print(grouped_imp_r)

#Plot variable importance

ggplot(grouped_imp_r, aes(x = reorder(Group, GroupImportance), y = GroupImportance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Grouped Variable Importance for RBF SVM", x = "Variable", y = "Importance")

#Top important variables 

important_vars_r <- grouped_imp_r %>%
  filter(GroupImportance > 10) %>%
  pull(Group)

cat("Selected variables:\n")
print(important_vars_r)

# Get feature names that match the selected groups
selected_features_r <- imp_r %>%
  filter(gsub("(_.*)", "", Variable) %in% important_vars_r) %>%
  pull(Variable)

print(selected_features_r)

#Subset

X_train_selected_r  <- X_train_smote[, selected_features_r, drop = FALSE]
X_test_selected_r <- X_test[, selected_features_r, drop = FALSE]

common_vars_r <- intersect(colnames(X_train_selected_r), colnames(X_test_selected_r))
X_train_selected_r <- X_train_selected_r[, common_vars_r]
X_test_selected_r <- X_test_selected_r[, common_vars_r]


# RBF SVM
svm_rbf_selected <- train(
  x = X_train_selected_r, y = y_train_smote_r,
  method = "svmRadial",
  trControl = ctrl,
  preProcess = c("center", "scale"),
  tuneGrid = expand.grid(
    sigma = c(0.001, 0.01, 0.1),
    C = c(0.5, 1, 2)
  )
)


#Evaluate

svm_linear_metrics_sel <- evaluate_svm_model(svm_linear_selected, X_train_selected, y_train_smote, X_test_selected, y_test, "SVM (Selected Linear)")
svm_poly_metrics_sel   <- evaluate_svm_model(svm_poly_selected,   X_train_selected_p, y_train_smote, X_test_selected_p, y_test, "SVM (Selected Poly)")
svm_rbf_metrics_sel    <- evaluate_svm_model(svm_rbf_selected,    X_train_selected_r, y_train_smote, X_test_selected_r, y_test, "SVM (Selected RBF)")

svm_selected_results <- rbind(svm_linear_metrics_sel, svm_poly_metrics_sel, svm_rbf_metrics_sel)
print(svm_selected_results)


#Model performance

ggplot(svm_selected_results, aes(x = Model, y = F1, fill = Set)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "Selected Feature SVM Models (F1 Score)", y = "F1 Score") +
  coord_flip()


