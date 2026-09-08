# Obesity Risk Prediction & Behavioral Analysis (Machine Learning in R)

## Overview
This repository contains a comprehensive machine learning classification pipeline designed to predict obesity risk based on dietary habits, physical condition, and genetic/demographic factors. The project utilizes a hybrid dataset of 2,111 records (incorporating synthetic data balancing via SMOTE). 

The workflow compares three models—Standard Decision Tree, Pruned/Controlled Decision Tree, and Random Forest—evaluating their performance using rigorous metrics including Information Value (IV), ROC/AUC, Lift, and Precision-Recall curves.

## Key Findings & Business Insights
* **Dominant Predictor:** `family_history_with_overweight` emerged as the strongest predictor by far, boasting an Information Value (IV) of **1.3194**.
* **Behavioral Factors:** Snacking habits (`CAEC`) and meal frequency (`NCP`) play a critical secondary role in determining weight categories.
* **Best Performing Model:** **Random Forest** significantly outperformed single decision trees, achieving an **Accuracy of 90.2%** and an outstanding **AUC of 0.9724** on the test set, effectively balancing high sensitivity with a low false-positive rate.

## Methodology & Tech Stack
* **Language:** R
* **Data Processing & Feature Engineering:** `dplyr`, `scorecard` (Information Value calculation, quantile binning)
* **Machine Learning:** `rpart` (Decision Trees), `randomForest` (Ensemble modeling)
* **Evaluation & Metrics:** `ROCR` (ROC, AUC, Lift, Precision-Recall), custom Confusion Matrix evaluation (Sensitivity, Specificity, F1-score, Cohen's Kappa)

## Project Structure
* `model_pipeline.R` - Complete script covering data preprocessing, IV calculation, model training, and performance evaluation.

## Model Comparison Summary (Test Set)
| Model | Accuracy | Sensitivity | Specificity | Precision | F1-Score | Kappa | AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TREE** (Default) | 0.8036 | 0.8295 | 0.7815 | 0.7643 | 0.7955 | 0.6071 | 0.8682 |
| **TREE2** (Controlled) | 0.7839 | 0.9109 | 0.6755 | 0.7057 | 0.7953 | 0.5742 | 0.8452 |
| **Random Forest** | **0.9018** | **0.9109** | **0.8940** | **0.8801** | **0.8952** | **0.8029** | **0.9724** |
