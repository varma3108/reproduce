# 🎓 Predicting Student Performance: A Machine Learning Replication

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green)](https://xgboost.ai/)
[![pandas](https://img.shields.io/badge/pandas-Data%20Analysis-yellow)](https://pandas.pydata.org/)

---

This repository contains the code for an **independent replication and validation** of a machine learning project aimed at **predicting student performance** at the *University of South Florida (USF)*.  
The goal is to identify students at risk of failing courses, enabling potential early interventions, by applying various predictive models to a large-scale academic dataset.

This work focuses on reproducing the complete experimental pipeline — from **data preprocessing and class balancing** to **model training and evaluation** — to confirm the robustness and reproducibility of the original study's findings.

---

## 🧠 Project Overview

Predicting academic success is a critical application of educational data mining.  
This project leverages a dataset of **over 2.6 million academic records** from USF to build models that can forecast student outcomes (e.g., pass/fail or specific letter grades).

All models were retrained using identical data splits, hyperparameters, and preprocessing techniques, and the reproduced metrics were found to be consistent with the original study.

### 🔍 Key Methodologies

- **Data:** 2.6M+ USF student records, including high school GPA, SAT/ACT scores, majors, and course history.  
- **Feature Engineering:** Previous coursework and grades encoded into multi-hot vectors.  
- **Class Imbalance Handling:**  
  1. **Undersampling:** Balancing dataset to a 50:50 pass/fail ratio.  
  2. **Weighted Undersampling:** Adjusted sampling ratio based on class sizes.  
  3. **SMOTE (Synthetic Minority Oversampling Technique):** Generated synthetic fail-class samples.

---

## 📊 Models and Key Findings

This project re-implemented and validated six machine learning algorithms.  
The results confirmed the findings of the original project.

| Model | Key Reproduced Finding |
| :--- | :--- |
| **Decision Tree** | Achieved **83.7% accuracy** and a **0.30 F1-score** for the failing class. Interpretable but weak for minority class. |
| **Gradient Boosting** | Best trade-off between accuracy and recall — **81.7% accuracy** and **0.42 F1 (Fail)** with undersampling. |
| **AdaBoost** | Strong only with balanced data (**74.6% acc / 0.76 Fail F1**). Without balancing: **88% acc / 0 Fail F1**. |
| **XGBoost** | Robust model achieving **71% binary accuracy** and **52% multi-class accuracy**. SHAP confirmed feature importance. |
| **Multi-Layer Perceptron (MLP)** | Underperformed (**71% accuracy**) due to sparse input and overfitting. |
| **Support Vector Classifier (SVC)** | Computationally impractical for this scale — excessive training time and memory use. |

### ✅ Conclusion

This replication validated the original project’s results, demonstrating understanding of the end-to-end ML pipeline.  
Findings show that **ensemble methods (Gradient Boosting, AdaBoost)**, combined with **class balancing**, deliver the most reliable performance for educational data.

---

## 🚀 Getting Started

### 🔧 Prerequisites

Ensure Python **3.7+** is installed along with the required libraries:

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn shap
