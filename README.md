# Student Dropout Prediction Using Machine Learning

## Project Overview

This project develops and compares supervised machine learning models to predict student dropout using progressively richer datasets collected throughout the student lifecycle.

The objective is to identify the stage at which student dropout can be predicted most effectively while still allowing timely intervention. Two classification models—**XGBoost** and a **Neural Network (Keras)**—are trained and evaluated across three stages of available student information.

> **Note:** The original dataset is proprietary and subject to a confidentiality agreement. Therefore, the dataset and model outputs are not included in this public repository.

---

## Objectives

- Predict student dropout using supervised machine learning
- Compare XGBoost and Neural Network performance
- Evaluate model performance across three stages of the student journey
- Identify the most effective stage for early intervention
- Analyse feature importance and model performance

---

## Dataset Stages

The project evaluates predictive performance using three progressively richer datasets:

### Stage 1 – Applicant Information
- Student demographics
- Course information
- Enrolment details

### Stage 2 – Student Engagement
- Attendance records
- Absence information
- Student engagement indicators

### Stage 3 – Academic Performance
- Module results
- Pass/fail records
- Academic progression information

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Keras Tuner
- Matplotlib

---

## Project Workflow

1. Data preprocessing
   - Missing value handling
   - Feature engineering
   - One-hot encoding
   - Ordinal encoding
   - Data standardisation

2. Model development
   - XGBoost classifier
   - Neural Network (Keras)

3. Hyperparameter tuning
   - GridSearchCV
   - Keras Tuner

4. Model evaluation
   - Accuracy
   - Precision
   - Recall
   - ROC AUC
   - Confusion Matrix

5. Model comparison across all three stages

---

## Key Findings

- Predictive performance improves as additional student information becomes available.
- Stage 2 provides the best balance between predictive accuracy and the opportunity for early intervention.
- XGBoost consistently performs slightly better than the Neural Network during the earlier stages.
- Hyperparameter tuning produces only marginal improvements compared with the impact of feature quality.
- Feature engineering and data availability contribute more to model performance than increased model complexity.

---

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Repository Contents

```
.
├── student_dropout_prediction.py
├── README.md
└── requirements.txt
```

---

## Confidentiality

The dataset used in this project is proprietary and cannot be shared publicly.

Only the source code and methodology are included in this repository. All datasets and generated outputs have been removed to comply with confidentiality requirements.

---

## Author

**Aseel Seef Azmy**

- LinkedIn: [https://www.linkedin.com/in/aseel-seef-azmy/](https://www.linkedin.com/in/aseel-seef-azmy/)
