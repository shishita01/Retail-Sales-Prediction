import pandas as pd
import zipfile
from zipfile import ZipFile
import docx
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



# zip_path = r"C:\Users\ishit\OneDrive\Desktop\txt format\Documents\VS code\DSassignment\Modeling - Module 4.zip"


# with ZipFile(zip_path, 'r') as z:
#     with z.open("Modeling - Assignments/Problem Statement/DS Internship - Modeling - ProblemStatement.docx") as f:
#         doc = docx.Document(f)
        
#         for para in doc.paragraphs:
#             print(para.text)


# zip_path = r"C:\Users\ishit\OneDrive\Desktop\txt format\Documents\VS code\DSassignment\Modeling - Module 4.zip"
# extract_to = r"C:\Users\ishit\OneDrive\Desktop\txt format\Documents\VS code\DSassignment"

# with zipfile.ZipFile(zip_path, 'r') as z:
#     z.extract("Modeling - Assignments/Data/DS Internship - Modeling - Data.xlsx", extract_to)



# excel_path = r"C:\Users\ishit\OneDrive\Desktop\txt format\Documents\VS code\DSassignment\Modeling - Assignments\Data\DS Internship - Modeling - Data.xlsx"

# df = pd.read_excel(excel_path)

# print(df.head())



# with zipfile.ZipFile(zip_path, 'r') as z:
#     print(z.namelist())



excel_path = r"C:\Users\ishit\OneDrive\Desktop\txt format\Documents\VS code\DSassignment\Modeling - Assignments\Data\DS Internship - Modeling - Data.xlsx"
df = pd.read_excel(excel_path)







# excel file
df = pd.read_excel("DSassignment/Modeling - Assignments/Data/DS Internship - Modeling - Data.xlsx")

print(df.shape)
print(df.head())



# column names, data types, and missing values
df.info()

# Basic statistics 
print(df.describe())

# Checking missing values
print(df.isna().sum())




# Define features (X) and target (y)
X = df.drop(columns=["Sales"])
y = df["Sales"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)


# Converting categorical columns into numeric (dummy variables)
X_encoded = pd.get_dummies(X, drop_first=True)

print("Before encoding:", X.shape)
print("After encoding:", X_encoded.shape)
print(X_encoded.head())






# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)



# Fill missing values 
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)



# linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# predictions
print("Training R²:", model.score(X_train, y_train))
print("Testing R²:", model.score(X_test, y_test))




# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Training metrics
print("Training MAE:", mean_absolute_error(y_train, y_train_pred))
print("Training MSE:", mean_squared_error(y_train, y_train_pred))
print("Training RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("Training R²:", r2_score(y_train, y_train_pred))

# Testing metrics
print("Testing MAE:", mean_absolute_error(y_test, y_test_pred))
print("Testing MSE:", mean_squared_error(y_test, y_test_pred))
print("Testing RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("Testing R²:", r2_score(y_test, y_test_pred))




# Coefficients
coefficients = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": model.coef_
})

# Sort by absolute value of coefficients
coefficients["Abs_Coefficient"] = coefficients["Coefficient"].abs()
coefficients = coefficients.sort_values(by="Abs_Coefficient", ascending=False)





# Add absolute coefficient for sorting
coefficients_filtered = coefficients[~coefficients["Feature"].str.contains("ChangeDate")]


coefficients_filtered = coefficients_filtered.sort_values(by="Abs_Coefficient", ascending=False)
print(coefficients_filtered.head(15))






# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Residuals
train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred


# Residual plots
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_train_pred, y=train_residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title("Training Set Residuals")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test_pred, y=test_residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title("Testing Set Residuals")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.tight_layout()



# Distribution of residuals
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.histplot(train_residuals, kde=True, bins=30, color="blue")
plt.title("Training Residual Distribution")

plt.subplot(1,2,2)
sns.histplot(test_residuals, kde=True, bins=30, color="orange")
plt.title("Testing Residual Distribution")

plt.tight_layout()





# Reset index for proper alignment
y_test = y_test.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

# Combine actual and predicted values for the test set
test_results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_test_pred
})


# If "Pop class" was part of the original features, we need to get it from the original df
test_results["Pop class"] = df.loc[X_test.index, "Pop class"].values

print(test_results.head())





def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


group_mape = test_results.groupby("Pop class").apply(
    lambda g: mean_absolute_percentage_error(g["Actual"], g["Predicted"])
)

print("MAPE by Pop Class:")
print(group_mape)



# If "Centre Type" was part of the original features, we need to get it from the original df
if "Centre Type" in df.columns:
    test_results["Centre Type"] = df.loc[X_test.index, "Centre Type"].values

# Group by Centre Type
centre_mape = test_results.groupby("Centre Type").apply(
    lambda g: mean_absolute_percentage_error(g["Actual"], g["Predicted"])
)

print("\nMAPE by Centre Type:")
print(centre_mape)





if "Pop class" in test_results.columns and "Centre Type" in test_results.columns:
    cross_mape = test_results.groupby(["Pop class", "Centre Type"]).apply(
        lambda g: mean_absolute_percentage_error(g["Actual"], g["Predicted"])
    )

    print("\nMAPE by Pop class × Centre Type:")
    print(cross_mape)





# Visualizations
cross_mape_df = cross_mape.reset_index()
cross_mape_df.columns = ["Pop class", "Centre Type", "MAPE"]

plt.figure(figsize=(10,6))
sns.barplot(data=cross_mape_df, x="Pop class", y="MAPE", hue="Centre Type")
plt.title("MAPE by Pop class × Centre Type")
plt.ylabel("MAPE (%)")






plt.show()




# ---------------- Final Conclusion ---------------- #

print("\n📌 Final Conclusion:")

print("""
1. The Linear Regression model explains about 65% of the variance in training data (R² = 0.65),
   but only ~18% on test data, which means the model is overfitting and not generalizing well.

2. Suburban stores have the lowest error (MAPE ~17.7%), while Rural stores have the highest error (~33.3%).
   This suggests the model struggles to predict Rural store sales.

3. Among store types:
   - Suburban Malls and Strips performed the best (lowest error ~13%).
   - Rural Strips performed the worst (highest error ~44%).

4. Feature analysis shows that some "ChangeDate" variables have very large coefficients,
   meaning the model is heavily influenced by store change history.

✅ Recommendation:
- Try feature selection or regularization (Ridge/Lasso) to reduce overfitting.
- Collect more data for rural stores or engineer better features for them.
- Consider advanced models (Random Forest, Gradient Boosting) for future improvement.

Overall, the model provides useful insights into store performance,
but there is room for improvement in accuracy, especially for rural locations.
""")






