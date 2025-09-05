Retail Sales Prediction

This project is part of the DS Internship – Modeling Assignment.
The goal is to help Client ABC, a large U.S.-based retailer, predict sales for new store locations using Linear Regression.


Problem Statement


ABC management is planning to expand by opening more stores. To support this decision, the objective is:
Build a Linear Regression model to predict store sales based on historical store data.
Identify key drivers of sales.
Evaluate model performance using metrics such as MAPE, R², MAE, RMSE.
Analyze performance across store types and population classes.
Dataset: 691 stores with sales and demographic characteristics.
Target variable: Average Monthly Sales (2019).


 Dataset Overview
Rows: 691
Columns: 37


Key features:

Store characteristics: Centre Type, Pop class, Climate, Store Sq Ft, etc.
Demographic variables: Population, Income levels, Age groups.
Sales: Sales (target variable).



Approach

Data Loading & Cleaning
Handled missing values with mean imputation.
Encoded categorical variables using One-Hot Encoding.
Model Development
Split dataset into train (80%) and test (20%).
Built a Linear Regression model.

Evaluated performance on both sets.



Model Evaluation
Training R²: 0.65
Testing R²: 0.18 (indicating overfitting).
MAPE, MAE, RMSE calculated for train & test sets.




Residual Analysis

Visualized residuals to check model fit.
Observed high variance in predictions for rural stores.
Error Analysis by Groups
MAPE by Pop Class:
Suburban: ~17.7% (best performance).
Urban: ~19.0%.
Rural: ~33.3% (worst performance).
MAPE by Pop Class × Centre Type:
Suburban Malls & Strips: ~13% error (best).
Rural Strips: ~44% error (worst).



Key Insights

Model explains variance well on training data but struggles on test data → overfitting.

Suburban stores are predicted better than Rural stores.

Store type (Mall/Strip/Outlet) and demographics significantly influence sales.

Some “ChangeDate” features dominate coefficients → suggest feature engineering.



Recommendations
Use feature selection or regularization (Ridge/Lasso) to reduce overfitting.
Collect more data for Rural stores.


Explore advanced ML models (Random Forest, Gradient Boosting) for improved performance.

Repository Structure
Retail-Sales-Prediction/
│── data/                     # Dataset (not included due to NDA)
│── modeling.py               # Main Python script
│── README.md                 # Project documentation
│── results/                  # Output plots & evaluation results




Final Conclusion

The Linear Regression model provides initial insights into sales prediction but suffers from overfitting. Suburban locations are easier to predict compared to Rural ones. Future work should involve regularization, feature engineering, and advanced ML models.
