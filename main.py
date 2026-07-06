import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def get_data_from_kaggle():
    from kagglehub.datasets import dataset_download
    downloaded_path = dataset_download(
        "blastchar/telco-customer-churn",
        path="WA_Fn-UseC_-Telco-Customer-Churn.csv",
        output_dir=".",
    )
    os.rename(downloaded_path, "./telco.csv")


# ==================== 0. Check if the dataset exists, if not download it from Kaggle ====================
if not os.path.exists(r"./telco.csv"):
    print("download")
    get_data_from_kaggle()

assert os.path.exists(r"./telco.csv"), "Dataset not found. Please download it from Kaggle."


# ==================== 1. Load and inspect the data ====================
data = pd.read_csv(r"./telco.csv")
data = data.convert_dtypes()

# print(data.head())
# data.info()

# ---------- 1.1 change `No internet service` to `No`.
data.replace({ "No internet service": "No", "No phone service": "No"}, inplace=True)

# ---------- 1.2 Convert Yes/No and Male/Female to boolean(integer).
for col in data.columns:
    st = set(data[col].unique())
    if st == {"Yes", "No"}: data[col] = data[col].map({"Yes": 1, "No": 0})
    elif st == {"Male", "Female"}: data[col] = data[col].map({"Female": 0, "Male": 1})


# data.info()


# ==================== 2. Define the variables ====================
# ---------- 2.1 One-hot encode the categorical variables.
data = pd.get_dummies(data, columns=["InternetService"], drop_first=True)
bool_cols = data.select_dtypes(include=["boolean"]).columns
data[bool_cols] = data[bool_cols].astype("Int64")

# ---------- 2.2 Define the features and target variable.
feature_cols = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "InternetService_Fiber optic", "InternetService_No"
]
features = data[feature_cols]
target = data["MonthlyCharges"]

# features.info()
# target.info()


# ==================== 3. Spilit the data ====================
X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.8, random_state=818)


# ==================== 4. Fit a multivariable linear regression ====================
model = LinearRegression()
model.fit(X_train, y_train)


# ==================== 5. Report the parameters ====================
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


# ==================== 6. Evaluate on the test set ====================
mae = mean_absolute_error(y_test, model.predict(X_test))
mse = mean_squared_error(y_test, model.predict(X_test))
r2 = r2_score(y_test, model.predict(X_test))
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")


# ==================== 7. Compare to a baseline ====================
# ---------- 7.1 Data preparation
addon_cols = ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
data_new = pd.DataFrame({"num_addons": data[addon_cols].sum(axis=1), "MonthlyCharges": data["MonthlyCharges"]})

# ---------- 7.2 Split the data and train a baseline model
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(data_new[["num_addons"]], data_new["MonthlyCharges"], train_size=0.8, random_state=818)
base_model = LinearRegression()
base_model.fit(X_train_new, y_train_new)

# ---------- 7.3 Evaluate the baseline model
mae_base = mean_absolute_error(y_test_new, base_model.predict(X_test_new))
mse_base = mean_squared_error(y_test_new, base_model.predict(X_test_new))
r2_base = r2_score(y_test_new, base_model.predict(X_test_new))
print(f"Baseline Mean Absolute Error: {mae_base:.2f}")
print(f"Baseline Mean Squared Error: {mse_base:.2f}")
print(f"Baseline R-squared: {r2_base:.2f}")
