# Student Dropout Prediction Using Machine Learning

## Overview

This project predicts student dropout risk using supervised machine learning models across three stages of the student lifecycle. It compares **XGBoost** and **Neural Networks (Keras)** to evaluate how predictive performance changes as more data becomes available over time.

The goal is to identify the most effective stage for early intervention to improve student retention.

---

## Problem Statement

Student retention is critical for educational institutions due to its impact on revenue, reputation, and student success. Early identification of at-risk students enables timely interventions that can reduce dropout rates.

This project builds predictive models to classify whether a student will:

* **Complete the course (1)**
* **Drop out (0)**

---

## Dataset Stages

The analysis is based on three progressive datasets:

* **Stage 1: Applicant Data**

  * Course information, demographics, and enrollment details

* **Stage 2: Engagement Data**

  * Attendance, absence records, and behavioral indicators

* **Stage 3: Academic Performance Data**

  * Module performance, pass/fail records, and academic progression

---

## Models Used

Two supervised learning models were implemented:

* **XGBoost Classifier**

  * Efficient for structured tabular data
  * Captures non-linear relationships

* **Neural Network (Keras)**

  * Multi-layer dense architecture
  * ReLU activation, dropout regularization, binary cross-entropy loss

---

## Data Preprocessing

* Missing values handled using imputation
* Categorical variables encoded using one-hot encoding
* Ordinal features mapped to numerical values
* Age engineered from date of birth
* Target variable converted to binary classification

---

## Evaluation Metrics

Models were evaluated using:

* Accuracy
* Precision
* Recall
* AUC
* Confusion Matrix

Due to class imbalance (~85% completion rate), **AUC and recall were prioritized**.

---

## Key Results

### Stage 1 (Applicant Data)

* ~89% accuracy across models
* Limited predictive power due to early-stage features
* XGBoost slightly outperformed Neural Network in AUC

### Stage 2 (Engagement Data)

* Improved performance (~90% accuracy)
* Significant improvement in recall
* Engagement features (attendance/absence) were strong predictors
* Best balance for early intervention

### Stage 3 (Academic Performance Data)

* Near-perfect performance (AUC ≈ 0.999)
* Strong influence of academic performance variables
* Not suitable for early intervention (data available too late)

---

## Key Insights

* Predictive performance improves significantly with data availability
* Stage 2 provides the best trade-off between early intervention and accuracy
* XGBoost is more stable and slightly stronger in early stages
* Hyperparameter tuning had minimal impact compared to feature quality
* Data quality is more important than model complexity

---

## Recommendations

* Use **Stage 2 models** for real-world intervention systems
* Prioritize **recall** over accuracy for dropout detection
* Prefer **XGBoost** for production deployment
* Focus on improving data collection rather than model complexity
* Use Stage 3 mainly for validation, not intervention

---

## Conclusion

This project demonstrates that machine learning can effectively predict student dropout risk, with performance improving as more data becomes available. However, engagement data (Stage 2) provides the optimal balance between early detection and actionable intervention.

The study highlights that **feature availability and data quality are more important than model complexity or hyperparameter tuning** in educational prediction tasks.

---

## Tech Stack

* Python
* XGBoost
* TensorFlow / Keras
* Pandas, NumPy
* Scikit-learn
* Matplotlib / Seaborn

---

## Note

The dataset used in this project is not publicly available due to institutional data privacy restrictions.
