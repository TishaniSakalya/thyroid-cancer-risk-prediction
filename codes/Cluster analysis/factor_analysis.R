# Load required libraries
library(FactoMineR)
library(factoextra)
library(plotly)
library(ggplot2)
library(gridExtra)
library(arulesCBA)
library(dplyr)
library(tidyr)
library(DataExplorer)
library(cluster)

# Set seed for reproducibility
set.seed(42)

# Read the full dataset
data <- read.csv("C:/Users/sakal/Documents/ISMF/Sem 6/ST 3082-Statistical Learning I/Final project/thyroid_cancer_risk_data.csv")

# Split data into training and testing sets (80/20 split)
sample_indices <- sample(seq_len(nrow(data)), size = 0.8 * nrow(data))
train_data <- data[sample_indices, ]
test_data <- data[-sample_indices, ]

# Print the size of each set
cat("Training set size:", nrow(train_data), "\n")
cat("Testing set size:", nrow(test_data), "\n")

# Save the split datasets
write.csv(train_data, file = "train_data.csv", row.names = FALSE)
write.csv(test_data, file = "test_data.csv", row.names = FALSE)

# Reload training data for FAMD
data <- read.csv("train_data.csv")

# Convert relevant variables to factors
factor_vars <- c("Gender", "Country", "Ethnicity", "Family_History", "Radiation_Exposure",
                 "Iodine_Deficiency", "Smoking", "Obesity", "Diabetes", "Diagnosis")

data[factor_vars] <- lapply(data[factor_vars], as.factor)

# Store diagnosis labels before removing from data
Diagnosis_labels <- data$Diagnosis 

# Sample 10,000 rows for analysis
data <- data[sample(1:nrow(data), 10000), ]
rownames(data) <- NULL  # Reset row names

# Remove irrelevant columns for FAMD
data_famd <- data[, !(names(data) %in% c("Patient_ID", "Thyroid_Cancer_Risk", "Diagnosis"))]

# Perform FAMD
famd_result <- FAMD(data_famd, graph = FALSE)

# View eigenvalues (optional)
famd_result$eig

# Perform hierarchical clustering on FAMD results
res.hcpc <- HCPC(famd_result, nb.clust = -1, graph = FALSE)

table(res.hcpc$data.clust$clust)

# Convert cluster labels from HCPC to numeric
cluster_labels <- as.numeric(res.hcpc$data.clust$clust)

# Use FAMD coordinates (already numeric) for distance calculation
dist_matrix <- dist(famd_result$ind$coord)

# Compute silhouette
sil <- silhouette(cluster_labels, dist_matrix)

# Average silhouette width
avg_sil_width <- mean(sil[, "sil_width"])
cat("Average silhouette width:", round(avg_sil_width, 3), "\n")

# Optional: visualize
fviz_silhouette(sil)


# Plot the clusters
fviz_cluster(res.hcpc, geom = "point")

# Add diagnosis labels back to clustered data
res.hcpc$data.clust$Diagnosis <- data[as.numeric(rownames(res.hcpc$data.clust)), "Diagnosis"]

# Rename clustered data for convenience
data_clust <- res.hcpc$data.clust

# Check cluster sizes
table(data_clust$clust)

# Split data by cluster
clustered_data_list <- split(data_clust, data_clust$clust)

# Generate EDA report for each cluster
for (i in seq_along(clustered_data_list)) {
  cluster_name <- names(clustered_data_list)[i]
  cat("Generating report for Cluster", cluster_name, "\n")
  
  create_report(clustered_data_list[[i]],
                output_file = paste0("Cluster_", cluster_name, "_EDA_Report.html"),
                output_dir = "cluster_reports_3")
}

# Analyze proportions of categorical variables within each cluster
cat_vars <- c("Gender", "Country", "Ethnicity", "Family_History", "Radiation_Exposure",
              "Iodine_Deficiency", "Smoking", "Obesity", "Diabetes", "Diagnosis")

for (var in cat_vars) {
  cat("\n===== Proportion of", var, "within each Cluster =====\n\n")
  
  prop_table <- data_clust %>%
    group_by(clust, .data[[var]]) %>%
    summarise(n = n(), .groups = "drop") %>%
    group_by(clust) %>%
    mutate(Proportion = round(n / sum(n), 3)) %>%
    arrange(clust, desc(Proportion))
  
  print(prop_table)
}
