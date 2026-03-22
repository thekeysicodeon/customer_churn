# Customer Churn Prediction

This project analyzes telecom customer data to understand why customers leave a service and to build predictive models for churn using classical machine learning and Artificial Neural Networks.

## Data Preparation
I cleaned and prepared the dataset by handling data type issues, encoding categorical variables, and applying SMOTE to address class imbalance. Through exploratory analysis, I identified that customers with low tenure, month-to-month contracts, and higher monthly charges are most likely to churn.

## Models Built
This project implements and compares five models:
- **Custom ANN** (NumPy) — built from scratch with manual backpropagation, He initialization, ReLU activations, and mini-batch SGD with momentum
- **Keras ANN** — feedforward network with Dropout regularization and Adam optimizer
- **Logistic Regression** — interpretable linear baseline with threshold tuning
- **Random Forest** — ensemble model with GridSearchCV tuning
- **SVM** — RBF kernel classifier

## Results
| Model | Accuracy | ROC-AUC |
|---|---|---|
| Custom ANN | 74.24% | 80.08% |
| Keras ANN | 76.22% | 83.38% |
| Logistic Regression | 73.81% | 84.02% |
| Random Forest | 77.93% | 82.26% |
| SVM | 75.73% | 81.64% |

Logistic Regression achieves the highest ROC-AUC (84.02%) and recall on churners, making it the most effective model for identifying at-risk customers in a retention context.

## Dataset
[Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Run
```bash
pip install tensorflow-macos keras-tuner imbalanced-learn scikit-learn pandas numpy matplotlib seaborn jupyter
jupyter notebook notebooks/churn_ann.ipynb
```
