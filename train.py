import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
file_path = r"C:\Users\Sania\Downloads\dataset1\train_u6lujuX_CVtuZ9i.csv"
df = pd.read_csv(file_path)
print("Dataset Loaded Successfully")
print(df.head())
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
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
print("Model Training Completed")
y_pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred)*100,2), "%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
joblib.dump(model, "loan_model.pkl")
joblib.dump(X.columns.tolist(), "loan_columns.pkl")
print("Model & Columns Saved Successfully!")
