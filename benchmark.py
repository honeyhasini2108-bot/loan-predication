import time
import requests
import sqlite3
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("\n================ MODEL COMPARISON =================")

# Load dataset
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

df.drop("Loan_ID", axis=1, inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

for col in df.select_dtypes(include="object"):
    df[col].fillna(df[col].mode()[0], inplace=True)

df["Loan_Status"] = df["Loan_Status"].map({"Y":1, "N":0})
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))

print(f"Logistic Regression Accuracy: {round(lr_acc*100,2)}%")
print(f"Random Forest Accuracy: {round(rf_acc*100,2)}%")

if rf_acc > lr_acc:
    print("Random Forest performs better.\n")

print("================ API SPEED TEST =================")

sample_data = {
    "Gender":1,
    "Married":1,
    "Dependents":0,
    "Education":1,
    "Self_Employed":0,
    "ApplicantIncome":5000,
    "CoapplicantIncome":2000,
    "LoanAmount":150,
    "Loan_Amount_Term":360,
    "Credit_History":1,
    "Property_Area":2
}

def call_api():
    requests.post("http://127.0.0.1:8000/predict", json=sample_data)

# Single thread test
start = time.time()
for _ in range(100):
    call_api()
end = time.time()
print(f"Single Thread - 100 requests time: {round(end-start,2)} sec")

# Multi-thread test
start = time.time()
with ThreadPoolExecutor(max_workers=10) as executor:
    executor.map(lambda x: call_api(), range(100))
end = time.time()
print(f"Multi Thread - 100 requests time: {round(end-start,2)} sec")

print("\n================ DATABASE INSERT TEST =================")

conn = sqlite3.connect("loan_data.db")
cursor = conn.cursor()

start = time.time()
for i in range(500):
    cursor.execute("""
        INSERT INTO applications (income, loan, term, status, created_at)
        VALUES (?, ?, ?, ?, datetime('now'))
    """, (5000, 150, 360, "Approved"))
conn.commit()
end = time.time()

print(f"500 records inserted in: {round(end-start,2)} sec")

conn.close()

print("\n=========== BENCHMARK COMPLETE ==========")

