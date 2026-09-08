# =====================================================================
# OBESITY RISK PREDICTION - MACHINE LEARNING PIPELINE
# =====================================================================

# 1. Clean environment
rm(list = ls())

# Automatic installation and loading of required packages
required_packages <- c("readr", "dplyr", "scorecard", "scales", "ggplot2", 
                       "Information", "rpart", "rpart.plot", "randomForest", 
                       "ROCR", "MASS", "rstudioapi")

new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]

if(length(new_packages)) {
  install.packages(new_packages, dependencies = TRUE)
}

library(readr)
library(dplyr)
library(scorecard)
library(scales)
library(ggplot2)
library(Information)
library(rpart)
library(rpart.plot)
library(randomForest)
library(ROCR)
library(MASS)
library(rstudioapi)

# =====================================================================
# STEP 2: Set Working Directory & Load Data
# =====================================================================
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}

dataset <- read.csv("ObesityDataSet_raw_and_data_sinthetic.csv", header = TRUE)

dataset <- dataset %>%
  mutate(Target = ifelse(grepl("Obesity", NObeyesdad), 1, 0)) %>%
  dplyr::select(-Weight, -NObeyesdad)

dataset <- dataset %>%
  mutate(across(where(is.character), as.factor))

# =====================================================================
# STEP 3: Automated Binning of Continuous Variables (Quantile-based)
# =====================================================================
continuous_vars <- c("Age", "Height", "FCVC", "NCP", "CH2O", "FAF", "TUE")

for (var in continuous_vars) {
  breaks <- unique(quantile(dataset[[var]], probs = seq(0, 1, 0.25), names = FALSE))
  
  if (length(breaks) > 1) {
    new_col_name <- paste0(var, "_Bin")
    dataset[[new_col_name]] <- cut(dataset[[var]], 
                                   breaks = breaks, 
                                   include.lowest = TRUE, 
                                   dig.lab = 2)
    dataset[[var]] <- NULL 
  } else {
    message(paste("Variable", var, "has too few unique values for quartile binning."))
  }
}

# =====================================================================
# STEP 4: Information Value (IV) Analysis
# =====================================================================
dataset_iv <- dataset
dataset_iv$Target <- as.numeric(as.character(dataset_iv$Target))

IV <- create_infotables(data = dataset_iv, y = "Target")
IV$Summary$IV <- round(IV$Summary$IV, 4)

cat("\n--- Information Value (Predictive Power) ---\n")
print(IV$Summary)

# =====================================================================
# STEP 5: Train / Test Split
# =====================================================================
set.seed(42)
test_prop <- 0.25
test_index <- (runif(nrow(dataset_iv)) < test_prop)

dataset_iv.test  <- dataset_iv[test_index, ]
dataset_iv.train <- dataset_iv[!test_index, ]

dataset_iv.train$Target <- as.factor(dataset_iv.train$Target)
dataset_iv.test$Target  <- as.factor(dataset_iv.test$Target)

# =====================================================================
# STEP 6: Model Training (Decision Trees & Random Forest)
# =====================================================================
tree <- rpart(Target ~ ., data = dataset_iv.train, method = "class")

tree2 <- rpart(Target ~ ., data = dataset_iv.train, method = "class",
               control = list(cp = 0.005, minbucket = 15, maxdepth = 6))

set.seed(42)
rf <- randomForest(Target ~ ., data = dataset_iv.train,
                   ntree = 500, mtry = 3, importance = TRUE)

# =====================================================================
# STEP 7: Evaluation Functions & Metrics Calculation
# =====================================================================
EvaluateCM <- function(cm) {
  cm2 <- matrix(0, nrow = 2, ncol = 2, dimnames = list(predicted = c("0","1"), actual = c("0","1")))
  cm2[rownames(cm), colnames(cm)] <- cm
  
  TP <- cm2["1","1"]; TN <- cm2["0","0"]; FP <- cm2["1","0"]; FN <- cm2["0","1"]
  
  data.frame(
    TP = TP, FP = FP, TN = TN, FN = FN,
    accuracy    = (TP + TN) / sum(cm2),
    sensitivity = TP / (TP + FN),
    specificity = TN / (TN + FP),
    precision   = TP / (TP + FP),
    F1          = (2 * TP) / (2 * TP + FP + FN)
  )
}

KappaFromCM <- function(cm) {
  cm2 <- matrix(0, nrow = 2, ncol = 2, dimnames = list(predicted = c("0","1"), actual = c("0","1")))
  cm2[rownames(cm), colnames(cm)] <- cm
  N <- sum(cm2)
  po <- sum(diag(cm2)) / N
  rowm <- rowSums(cm2) / N
  colm <- colSums(cm2) / N
  pe <- sum(rowm * colm)
  return((po - pe) / (1 - pe))
}

prob_tree  <- predict(tree,  newdata = dataset_iv.test, type = "prob")[, "1"]
pred_tree  <- predict(tree,  newdata = dataset_iv.test, type = "class")

prob_tree2 <- predict(tree2, newdata = dataset_iv.test, type = "prob")[, "1"]
pred_tree2 <- predict(tree2, newdata = dataset_iv.test, type = "class")

prob_rf    <- predict(rf,    newdata = dataset_iv.test, type = "prob")[, "1"]
pred_rf    <- predict(rf,    newdata = dataset_iv.test, type = "response")

CM_tree  <- table(predicted = pred_tree,  actual = dataset_iv.test$Target)
CM_tree2 <- table(predicted = pred_tree2, actual = dataset_iv.test$Target)
CM_rf    <- table(predicted = pred_rf,    actual = dataset_iv.test$Target)

m_tree  <- EvaluateCM(CM_tree)
m_tree2 <- EvaluateCM(CM_tree2)
m_rf    <- EvaluateCM(CM_rf)

k_tree  <- KappaFromCM(CM_tree)
k_tree2 <- KappaFromCM(CM_tree2)
k_rf    <- KappaFromCM(CM_rf)

# =====================================================================
# STEP 8: ROCR Performance (ROC, AUC, Lift, PR)
# =====================================================================
actual_numeric <- as.numeric(dataset_iv.test$Target == "1")

pred_tree_obj  <- ROCR::prediction(prob_tree,  actual_numeric)
pred_tree2_obj <- ROCR::prediction(prob_tree2, actual_numeric)
pred_rf_obj    <- ROCR::prediction(prob_rf,    actual_numeric)

roc_tree  <- ROCR::performance(pred_tree_obj,  "tpr", "fpr")
roc_tree2 <- ROCR::performance(pred_tree2_obj, "tpr", "fpr")
roc_rf    <- ROCR::performance(pred_rf_obj,    "tpr", "fpr")

auc_tree  <- ROCR::performance(pred_tree_obj,  "auc")@y.values[[1]]
auc_tree2 <- ROCR::performance(pred_tree2_obj, "auc")@y.values[[1]]
auc_rf    <- ROCR::performance(pred_rf_obj,    "auc")@y.values[[1]]

lift_tree  <- ROCR::performance(pred_tree_obj,  "lift", "rpp")
lift_tree2 <- ROCR::performance(pred_tree2_obj, "lift", "rpp")
lift_rf    <- ROCR::performance(pred_rf_obj,    "lift", "rpp")

pr_tree  <- ROCR::performance(pred_tree_obj,  "prec", "rec")
pr_tree2 <- ROCR::performance(pred_tree2_obj, "prec", "rec")
pr_rf    <- ROCR::performance(pred_rf_obj,    "prec", "rec")

# =====================================================================
# STEP 9: PLOTS (Displayed directly in the RStudio Plots pane)
# =====================================================================

# 1. Decision Trees Visualization
rpart.plot(tree, under = FALSE, tweak = 1.3, fallen.leaves = TRUE)
rpart.plot(tree2, under = FALSE, tweak = 1.3, fallen.leaves = TRUE)

# 2. Random Forest Variable Importance
varImpPlot(rf)

# 3. ROC Curves Comparison
plot(roc_tree, col="black", lwd=3,
     xlab="False positive rate", ylab="True positive rate")
plot(roc_tree2, col="red", lwd=3, add=TRUE)
plot(roc_rf, col="green4", lwd=3, add=TRUE)
abline(a=0, b=1, lty=2)
legend("bottomright",
       legend=c("TREE", "TREE2", "RandomForest"),
       col=c("black","red","green4"), lwd=3, bty="n")

# 4. Lift Curves Comparison
plot(lift_tree, col="black", lwd=3,
     xlab="Rate of positive predictions", ylab="Lift value")
plot(lift_lift2 <- lift_tree2, col="red", lwd=3, add=TRUE) # (lub po prostu lift_tree2)
plot(lift_tree2, col="red", lwd=3, add=TRUE)
plot(lift_rf, col="green4", lwd=3, add=TRUE)
legend("topright",
       legend=c("TREE", "TREE2", "RandomForest"),
       col=c("black","red","green4"), lwd=3, bty="n")

# 5. Precision-Recall Curves Comparison
plot(pr_tree, col="black", lwd=3,
     xlab="Recall", ylab="Precision")
plot(pr_tree2, col="red", lwd=3, add=TRUE)
plot(pr_rf, col="green4", lwd=3, add=TRUE)
legend("bottomleft",
       legend=c("TREE", "TREE2", "RandomForest"),
       col=c("black","red","green4"), lwd=3, bty="n")

cat("\nWartości AUC dla poszczególnych modeli wynoszą:\n")
cat(paste0(" • Random Forest – AUC = ", round(auc_rf, 4), "\n"))
cat(paste0(" • Tree 2 (Pruned) – AUC = ", round(auc_tree2, 4), "\n"))
cat(paste0(" • Tree 1 (Default) – AUC = ", round(auc_tree, 4), "\n\n"))

# =====================================================================
# STEP 10: FINAL METRICS SUMMARY TABLE
# =====================================================================
results_table <- rbind(
  data.frame(Model = "TREE",  m_tree,  Kappa = k_tree,  AUC = auc_tree),
  data.frame(Model = "TREE2", m_tree2, Kappa = k_tree2, AUC = auc_tree2),
  data.frame(Model = "RF",    m_rf,    Kappa = k_rf,    AUC = auc_rf)
)

num_cols <- setdiff(names(results_table), "Model")
results_table[num_cols] <- lapply(results_table[num_cols], function(x) round(x, 4))

cat("\n--- PERFORMANCE METRICS SUMMARY (TEST SET) ---\n")
print(results_table)

