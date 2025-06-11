# Customer Churn Prediction using Machine Learning

## Overview
This project predicts customer churn for an e-commerce platform using advanced machine learning techniques. The workflow includes data preprocessing, feature engineering, handling class imbalance, model training, evaluation, and ensemble learning.

## Dataset
- **Source:** `/kaggle/input/e-commerce-customer-churn/data_ecommerce_customer_churn.csv`
- **Features:**
  - Customer demographics
  - Order preferences
  - Tenure, satisfaction, device registration, etc.
  - Target: `Churn` (binary)

## Steps
1. **Data Loading & Preprocessing**
   - Load data with pandas
   - Encode categorical features (`PreferedOrderCat`, `MaritalStatus`)
   - Handle missing values (mean imputation for numerics)
   - Feature scaling (StandardScaler)
   - Feature engineering (e.g., `Tenure_Satisfaction`)

2. **Class Imbalance Handling**
   - Use SMOTEENN to balance classes

3. **Feature Selection**
   - SelectKBest with ANOVA F-value

4. **Model Training & Evaluation**
   - Models: Random Forest, XGBoost, Logistic Regression, Gradient Boosting
   - Train/test split (80/20)
   - Hyperparameter tuning (GridSearchCV, optional)
   - Evaluation metrics: Accuracy, Classification Report, ROC Curve, Confusion Matrix

5. **Ensemble Learning**
   - StackingClassifier with Random Forest, XGBoost, Logistic Regression as base learners and Gradient Boosting as meta-learner

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- xgboost
- matplotlib
- seaborn

Install dependencies:
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn
```

## Usage
1. Place the dataset at the specified path or update the path in the notebook.
2. Open `code.ipynb` in VS Code or Jupyter.
3. Run all cells sequentially to:
   - Preprocess data
   - Train and evaluate models
   - Visualize results (ROC, confusion matrix)

## Results
- The notebook prints training and test accuracy, classification reports, and displays ROC curves and confusion matrices for each model and the stacking ensemble.

## Notes
- The code is modular: you can tune models, add/remove features, or swap algorithms easily.
- For best results, ensure the dataset is clean and paths are correct.

## Author
- Sazibul Islam Siam

## License
This project is for educational purposes.
