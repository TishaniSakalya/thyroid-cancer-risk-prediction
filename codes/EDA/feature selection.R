# --------------------------------------------
# Load Required Libraries
# --------------------------------------------
library(dplyr)
library(caret)
library(glmnet)
library(ggplot2)
library(pROC)

# --------------------------------------------
# Load and Preprocess Dataset
# --------------------------------------------
data_path <- "C:/Users/User/Desktop/Colombo uni/3year - 2nd sem/ST 3082/Final project/archive (1)/thyroid_cancer_risk_data.csv"
thyroid_data <- read.csv(data_path)

# Remove unnecessary columns
thyroid_data <- thyroid_data %>% select(-Patient_ID, -Thyroid_Cancer_Risk)

# Define outcome variable (Diagnosis or Metastasis)
# We'll first do it for Diagnosis
thyroid_data$Diagnosis <- as.factor(thyroid_data$Diagnosis)

# Save labels
diagnosis_labels <- thyroid_data$Diagnosis

# Define categorical columns
categorical_cols <- c("Gender", "Family_History", "Radiation_Exposure", "Iodine_Deficiency",
                      "Smoking", "Obesity", "Diabetes", "Country", "Ethnicity")

# Convert to factors
thyroid_data[categorical_cols] <- lapply(thyroid_data[categorical_cols], as.factor)

# One-hot encode categorical variables
dummy_model <- dummyVars(Diagnosis ~ ., data = thyroid_data)
thyroid_encoded <- predict(dummy_model, newdata = thyroid_data)

# Standardize features
thyroid_scaled <- scale(thyroid_encoded)

# --------------------------------------------
# Feature Selection using Lasso (Logistic Regression)
# --------------------------------------------
# Prepare response variable
y <- as.numeric(diagnosis_labels) - 1  # Convert to 0/1

# Perform Lasso with cross-validation
set.seed(123)
cv_model <- cv.glmnet(thyroid_scaled, y, family = "binomial", alpha = 1, nfolds = 10)

# Plot cross-validation curve
plot(cv_model)

# Optimal lambda
best_lambda <- cv_model$lambda.1se #Tune the Lasso Parameters
cat("Best lambda:", best_lambda, "\n")

# Fit final model
lasso_model <- glmnet(thyroid_scaled, y, family = "binomial", alpha = 1, lambda = best_lambda)

# Extract selected features
selected_features <- rownames(coef(lasso_model))[coef(lasso_model)[, 1] != 0]
selected_features <- selected_features[selected_features != "(Intercept)"]
cat("Selected Features:\n", selected_features, "\n")

# --------------------------------------------
# Evaluate Model
# --------------------------------------------
# Predict probabilities
prob_predictions <- predict(lasso_model, newx = thyroid_scaled, type = "response")

# ROC and AUC
roc_obj <- roc(y, as.numeric(prob_predictions))
auc_val <- auc(roc_obj)
cat("AUC:", auc_val, "\n")
plot(roc_obj, main = "ROC Curve for Diagnosis Prediction")
#If the classes are imbalanced, AUC might not be enough.
library(PRROC)
pr <- pr.curve(scores.class0 = prob_predictions[y == 1],
               scores.class1 = prob_predictions[y == 0],
               curve = TRUE)
plot(pr)


