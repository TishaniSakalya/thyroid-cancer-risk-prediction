library(dplyr)
library(caret)
library(ROSE)
library(naivebayes)
library(fastDummies)
library(MLmetrics)

set.seed(42)

# Load data
df <- read.csv("thyroid_cancer_risk_data.csv")
df <- df[-16]  #remove risk
df <- df[-1]   #remove ID

# Convert categorical variables
binary_cols <- c("Gender", "Family_History", "Radiation_Exposure", 
                 "Iodine_Deficiency", "Smoking", "Obesity", "Diabetes", "Diagnosis")

df <- df %>%
  mutate(Gender = recode(Gender, "Male" = 0, "Female" = 1),
         Diagnosis = recode(Diagnosis, "Benign" = 0, "Malignant" = 1)) %>%
  mutate(across(c("Family_History", "Radiation_Exposure", "Iodine_Deficiency", 
                  "Smoking", "Obesity", "Diabetes"), ~ recode(., "No" = 0, "Yes" = 1))) %>%
  mutate(across(all_of(binary_cols), as.numeric))

# One-hot encoding
df <- df %>%
  mutate(across(c("Country", "Ethnicity"), as.factor)) %>%
  fastDummies::dummy_cols(select_columns = c("Country", "Ethnicity"), 
                          remove_selected_columns = TRUE, remove_first_dummy = TRUE)

# Split response and predictors
df$Diagnosis <- factor(df$Diagnosis, labels = c("Benign", "Malignant"))
y <- df$Diagnosis
X <- df %>% select(-Diagnosis)

# Train-test split
train_idx <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_idx, ]
y_train <- y[train_idx]
X_test <- X[-train_idx, ]
y_test <- y[-train_idx]

# Combine for ROSE
train_data <- cbind(X_train, Diagnosis = y_train)

# Fix column names for formula safety
predictors <- setdiff(names(train_data), "Diagnosis")
predictors <- make.names(predictors)
names(train_data) <- make.names(names(train_data))

# Build formula
formula_str <- paste("Diagnosis ~", paste(predictors, collapse = " + "))
rose_formula <- as.formula(formula_str)

# Apply ROSE
set.seed(42)
rose_data <- ROSE(rose_formula, data = train_data, seed = 1)$data

# Separate after resampling
X_train_rose <- rose_data %>% select(-Diagnosis)
y_train_rose <- rose_data$Diagnosis

# Standardize numeric columns
num_cols <- c("Age", "TSH_Level", "T4_Level", "T3_Level", "Nodule_Size")
num_cols <- intersect(c("Age", "TSH_Level", "T4_Level", "T3_Level", "Nodule_Size"), names(X_train_rose))

scaler <- preProcess(X_train_rose[, num_cols], method = c("center", "scale"))
X_train_rose[, num_cols] <- predict(scaler, X_train_rose[, num_cols])
X_test[, num_cols] <- predict(scaler, X_test[, num_cols])

# Fill any missing dummy columns in X_test with 0s
missing_cols <- setdiff(colnames(X_train_rose), colnames(X_test))
X_test[, missing_cols] <- 0

# Reorder to match training column order
X_test <- X_test[, colnames(X_train_rose)]

View(X_train_rose)
View(X_test)



# Naive Bayes (base)
nb_model <- naive_bayes(x = X_train_rose, y = y_train_rose)

# Predictions
pred_train <- predict(nb_model, X_train_rose)
pred_test <- predict(nb_model, X_test)

# Evaluation function
get_metrics <- function(true, pred, model_name, set_type) {
  acc <- Accuracy(pred, true)
  prec <- Precision(pred, true, positive = "Malignant")
  rec <- Recall(pred, true, positive = "Malignant")
  f1 <- F1_Score(pred, true, positive = "Malignant")
  data.frame(Model = model_name, Set = set_type, Accuracy = acc, Precision = prec, Recall = rec, F1 = f1)
}

# Evaluate
nb_rose_metrics <- rbind(
  get_metrics(y_train_rose, pred_train, "Naive Bayes (ROSE)", "Train"),
  get_metrics(y_test, pred_test, "Naive Bayes (ROSE)", "Test")
)
print(nb_rose_metrics)

# Bind response and predictors again for caret
train_rose_full <- cbind(X_train_rose, Diagnosis = y_train_rose)

# Tune grid with wider range for adjust parameter
nb_grid <- expand.grid(
  laplace = c(0, 0.5, 1, 2),  # Include more options for laplace
  usekernel = c(TRUE, FALSE),
  adjust = seq(0.1, 1, by = 0.1)  # Adjust values from 0.1 to 1 with 0.1 intervals
)

# TrainControl setup
ctrl <- trainControl(
  method = "cv", 
  number = 5, 
  classProbs = TRUE, 
  summaryFunction = twoClassSummary,
  verboseIter = TRUE
)

# Retrain the model with the updated tuning grid
set.seed(123)
nb_caret_model <- train(
  Diagnosis ~ ., 
  data = train_rose_full,  # Use the combined dataset with resampled data
  method = "naive_bayes",
  trControl = ctrl,
  tuneGrid = nb_grid,
  metric = "ROC",  # Using ROC for performance evaluation
  preProcess = NULL
)

# Best model after tuning
print(nb_caret_model$bestTune)

# Plot the tuning results to understand the impact of hyperparameters
plot(nb_caret_model)

# Predictions with the tuned model
nb_caret_pred_train <- predict(nb_caret_model, newdata = X_train_rose)
nb_caret_pred_test <- predict(nb_caret_model, newdata = X_test)

# Evaluate the performance
nb_caret_metrics <- rbind(
  get_metrics(y_train_rose, nb_caret_pred_train, "Naive Bayes (Tuned)", "Train"),
  
  
  get_metrics(y_test, nb_caret_pred_test, "Naive Bayes (Tuned)", "Test")
)

# Print updated metrics
print(nb_caret_metrics)


#Variable importance



# Extract the importance table
imp_df <- nb_var_imp$importance
print(imp_df)
imp_df$Feature <- rownames(imp_df)

colnames(imp_df)
imp_df$Overall <- rowMeans(imp_df[, c("Benign", "Malignant")])

# Extract variable name without dummy suffixes
imp_df$Group <- gsub("(_.*)", "", imp_df$Feature)

# Now group and summarize
library(dplyr)
grouped_imp <- imp_df %>%
  group_by(Group) %>%
  summarise(GroupImportance = mean(Overall)) %>%
  arrange(desc(GroupImportance))

print(grouped_imp)

library(ggplot2)

# Plot grouped importance
ggplot(grouped_imp, aes(x = reorder(Group, GroupImportance), y = GroupImportance)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Grouped Variable Importance",
       x = "Variable Group",
       y = "Mean Importance (across classes)") +
  theme_minimal()

# Model fitting with only the most important variables

# Filter for important groups
important_vars <- grouped_imp %>%
  filter(GroupImportance > 10) %>%
  pull(Group)

print(important_vars)  # See which ones made the cut

# Get feature names that match the selected groups
selected_features <- imp_df %>%
  filter(gsub("(_.*)", "", Feature) %in% important_vars) %>%
  pull(Feature)

# Filter columns in training/testing sets
X_train_sel <- X_train_rose[, selected_features, drop = FALSE]
X_test_sel <- X_test[, selected_features, drop = FALSE]

# Combine response for caret training
train_data_sel <- cbind(X_train_sel, Diagnosis = y_train_rose)

# Reuse same training control and grid
set.seed(123)
nb_sel_model <- train(
  Diagnosis ~ ., 
  data = train_data_sel,
  method = "naive_bayes",
  trControl = ctrl,
  tuneGrid = nb_grid,
  metric = "ROC",
  preProcess = NULL
)

# Best model after tuning
print(nb_sel_model$bestTune)

# Plot the tuning results to understand the impact of hyperparameters
plot(nb_sel_model)

# Predictions with the tuned model
nb_sel_pred_train <- predict(nb_sel_model, newdata = X_train_sel)
nb_sel_pred_test <- predict(nb_sel_model, newdata = X_test_sel)

# Evaluate the performance
nb_sel_metrics <- rbind(
  get_metrics(y_train, nb_sel_pred_train, "Naive Bayes (Selected)", "Train"),
  
  
  get_metrics(y_test, nb_sel_pred_test, "Naive Bayes (Selected)", "Test")
)

# Print updated metrics
print(nb_sel_metrics)
