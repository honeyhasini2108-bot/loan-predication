import pandas as pd
df1 = pd.read_csv(r"C:\Users\Sania\Downloads\dataset1\test_Y3wMUE5_7gLdaTN.csv",
                 usecols=["Loan_ID", "Gender", "ApplicantIncome"])
df2 = pd.read_csv(r"C:\Users\Sania\Downloads\dataset1\train_u6lujuX_CVtuZ9i.csv",
                 usecols=["Loan_ID", "Credit_History", "LoanAmount"])
df3 = pd.read_csv(r"C:\Users\Sania\Downloads\Dataset 2.csv",
                 usecols=["Loan_ID", "Property_Area"])
merged = pd.merge(df1, df2, on="Loan_ID", how="outer")
merged = pd.merge(merged, df3, on="Loan_ID", how="outer")
merged.to_csv(r"C:\Users\Sania\Downloads\merged.csv", index=False)
print("Merge successful! file saved in Downloads.")
