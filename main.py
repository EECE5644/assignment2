import os
import pandas as pd

def get_data_from_kaggle():
    from kagglehub.datasets import dataset_download
    downloaded_path = dataset_download(
        "blastchar/telco-customer-churn",
        path="WA_Fn-UseC_-Telco-Customer-Churn.csv",
        output_dir=".",
    )
    os.rename(downloaded_path, "./telco.csv")


# 0. Check if the dataset exists, if not download it from Kaggle
if not os.path.exists(r"./telco.csv"):
    print("download")
    get_data_from_kaggle()

assert os.path.exists(r"./telco.csv"), "Dataset not found. Please download it from Kaggle."


# 1. Load and inspect the data
data = pd.read_csv(r"./telco.csv")
data = data.convert_dtypes()

print(data.head())
# data.info()

# 1.1 change `No internet service` to `No`.
data.replace({ "No internet service": "No", "No phone service": "No"}, inplace=True)

# 1.2 Convert Yes/No and Male/Female to boolean(integer).
for col in data.columns:
    st = set(data[col].unique())
    if st == {"Yes", "No"}: data[col] = data[col].map({"Yes": 1, "No": 0})
    elif st == {"Male", "Female"}: data[col] = data[col].map({"Female": 0, "Male": 1})


data.info()
