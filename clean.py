import pandas as pd
df = pd.read_csv(r"C:\Users\Sania\Downloads\merged.csv")
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
df.drop_duplicates(inplace=True)
df.to_csv(r"C:\Users\Sania\Downloads\cleaned.csv", index=False)
print("Data cleaning completed!")
