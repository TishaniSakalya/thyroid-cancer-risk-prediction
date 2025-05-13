# --------------------------------------------
# Load Required Libraries
# --------------------------------------------
library(dplyr)
library(ggplot2)
library(cluster)
library(factoextra)
library(caret)
library(mclust)

# --------------------------------------------
# Load and Preprocess Dataset
# --------------------------------------------
data_path <- "C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Final project/thyroid_cancer_risk_data.csv"
thyroid_data <- read.csv(data_path)

# Drop unneeded columns
thyroid_data <- thyroid_data %>% select(-Patient_ID, -Thyroid_Cancer_Risk)

# Sample 10,000 for memory efficiency
set.seed(123)
thyroid_sample <- sample_n(thyroid_data, 10000)

# Save labels for later evaluation
diagnosis_labels <- thyroid_sample$Diagnosis

# Remove Diagnosis column for clustering
thyroid_features <- thyroid_sample %>% select(-Diagnosis)

# Define categorical columns
categorical_cols <- c("Gender", "Family_History", "Radiation_Exposure", "Iodine_Deficiency",
                      "Smoking", "Obesity", "Diabetes", "Country", "Ethnicity")

# Convert to factors
thyroid_features[categorical_cols] <- lapply(thyroid_features[categorical_cols], as.factor)

# One-hot encode categorical variables
dummy_model <- dummyVars(" ~ .", data = thyroid_features)
thyroid_encoded <- predict(dummy_model, newdata = thyroid_features)

# Standardize data
thyroid_scaled <- scale(thyroid_encoded)

# --------------------------------------------
# Elbow Method to Choose Optimal k
# --------------------------------------------
wss <- sapply(1:10, function(k) {
  set.seed(123)
  kmeans(thyroid_scaled, centers = k, nstart = 10)$tot.withinss
})

# Plot Elbow Curve
elbow_df <- data.frame(Clusters = 1:10, WSS = wss)
ggplot(elbow_df, aes(x = Clusters, y = WSS)) +
  geom_line(color = "steelblue") +
  geom_point(color = "darkred", size = 2) +
  labs(title = "Elbow Method for Optimal Number of Clusters",
       x = "Number of Clusters (k)", y = "Total Within-Cluster Sum of Squares") +
  theme_minimal()

# --------------------------------------------
# K-Means Clustering (k = 3)
# --------------------------------------------
set.seed(123)
k_optimal <- 3
kmeans_model <- kmeans(thyroid_scaled, centers = k_optimal, nstart = 25)

# Add cluster & diagnosis labels
thyroid_clustered <- as.data.frame(thyroid_scaled)
thyroid_clustered$Cluster <- as.factor(kmeans_model$cluster)
thyroid_clustered$Diagnosis <- diagnosis_labels

# Cluster plot using PCA
fviz_cluster(kmeans_model, data = thyroid_scaled,
             geom = "point", ellipse.type = "norm", 
             palette = "jco", ggtheme = theme_minimal(),
             main = paste("K-means Clustering (k =", k_optimal, ") with PCA"))

# --------------------------------------------
# Cluster Evaluation
# --------------------------------------------
# Confusion Matrix (unsupervised vs true diagnosis)
cluster_eval <- table(Cluster = kmeans_model$cluster, Diagnosis = diagnosis_labels)
print(cluster_eval)

# --------------------------------------------
# Cluster Interpretation: Numeric & Categorical
# --------------------------------------------

# Attach cluster labels to original (unscaled) data
thyroid_sample$Cluster <- as.factor(kmeans_model$cluster)

# Numeric Summary
numeric_summary <- thyroid_sample %>%
  group_by(Cluster) %>%
  summarise(across(where(is.numeric), mean, na.rm = TRUE), .groups = "drop")
print(numeric_summary)

# Categorical Mode Summary
mode_fun <- function(x) names(which.max(table(x)))
categorical_summary <- thyroid_sample %>%
  group_by(Cluster) %>%
  summarise(across(all_of(categorical_cols), mode_fun, .names = "mode_{.col}"), .groups = "drop")
print(categorical_summary)

# --------------------------------------------
# Clustering Accuracy Estimation (Optional)
# --------------------------------------------

# Adjusted Rand Index
ari <- adjustedRandIndex(kmeans_model$cluster, as.numeric(factor(diagnosis_labels)))
cat("Adjusted Rand Index (ARI):", round(ari, 3), "\n")

# Naive Accuracy Estimate (Most common label per cluster)
cluster_to_label <- apply(cluster_eval, 1, function(x) names(which.max(x)))
mapped_preds <- sapply(kmeans_model$cluster, function(cl) cluster_to_label[as.character(cl)])
accuracy <- mean(mapped_preds == diagnosis_labels)
cat("Naive Clustering Accuracy:", round(accuracy * 100, 2), "%\n")

# Silhouette Analysis
library(cluster)

# Compute silhouette values
sil <- silhouette(kmeans_model$cluster, dist(thyroid_scaled))

# Average silhouette width
avg_sil_width <- mean(sil[, "sil_width"])
cat("Average Silhouette Width:", round(avg_sil_width, 3), "\n")

# Plot silhouette
fviz_silhouette(sil, palette = "jco", ggtheme = theme_minimal())

