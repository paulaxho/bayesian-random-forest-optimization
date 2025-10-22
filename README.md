# Random Forest Classification with Bayesian Optimization and Feature Selection

This project demonstrates the use of **Bayesian hyperparameter optimization** to tune a **Random Forest classifier**, followed by **automatic feature selection** and **comparative model evaluation**.  

The notebook is designed for reproducible experiments in **Google Colab**, where training and test data are loaded from Google Drive.

---

## 🚀 Overview

| Step | Description |
|------|--------------|
| **1. Data Loading** | Reads multiple training and test CSV files from Google Drive. |
| **2. Preprocessing** | Handles missing values with `SimpleImputer` and standardizes features using `StandardScaler`. |
| **3. Bayesian Optimization** | Uses `BayesSearchCV` from `scikit-optimize` to find optimal Random Forest parameters. |
| **4. Feature Selection** | Applies `SelectFromModel` to retain the top 250 most important features. |
| **5. Evaluation** | Compares model accuracy and classification reports across various data splits. |
| **6. Confidence Weighting** | Incorporates a "confidence" score as sample weights during training. |
| **7. Visualization** | Includes accuracy comparison plots, hyperparameter effect plots, feature importance charts, and confusion matrices. |
| **8. Predictions Export** | Saves final test predictions as `predictions.csv`. |

---

## 🧩 Techniques Used

- **Random Forest Classifier**
- **Bayesian Optimization** (`skopt.BayesSearchCV`)
- **Feature Selection** (`SelectFromModel`)
- **Sample Weighting** using “confidence” scores
- **Model Evaluation** using MSE, MAE, Accuracy, Confusion Matrix
- **Data Visualization** using `matplotlib`

---
