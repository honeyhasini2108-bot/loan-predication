from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from datetime import datetime

# ================= FASTAPI SETUP =================
app = FastAPI(title="Loan Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= LOAD MODEL =================
model = joblib.load("loan_model.pkl")
columns = joblib.load("loan_columns.pkl")

executor = ThreadPoolExecutor(max_workers=4)

# ================= DATABASE SETUP =================
conn = sqlite3.connect("loan_data.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    income REAL,
    loan REAL,
    term INTEGER,
    status TEXT,
    created_at TEXT
)
""")
conn.commit()

# ================= INPUT SCHEMA =================
class LoanInput(BaseModel):
    Gender: int
    Married: int
    Dependents: int
    Education: int
    Self_Employed: int
    ApplicantIncome: int
    CoapplicantIncome: int
    LoanAmount: int
    Loan_Amount_Term: int
    Credit_History: int
    Property_Area: int

# ================= HOME ROUTE =================
@app.get("/")
def home():
    return {"message": "Loan Prediction API Running 🚀"}

# ================= PREDICTION LOGIC =================
def predict_logic(data_dict):

    df = pd.DataFrame([data_dict])
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=columns, fill_value=0)

    prediction = model.predict(df_encoded)[0]

    hints = []

    if data_dict["Credit_History"] == 0:
        hints.append("Poor Credit History")

    if data_dict["ApplicantIncome"] < 2500:
        hints.append("Low Applicant Income")

    if data_dict["LoanAmount"] > 250:
        hints.append("High Loan Amount")

    if data_dict["Loan_Amount_Term"] > 360:
        hints.append("Very Long Loan Term")

    if prediction == 1:
        status = "Approved"
        reason = "Strong financial profile and good credit history"
    else:
        status = "Not Approved"
        if not hints:
            hints.append("Risk factors detected based on model evaluation")
        reason = ", ".join(hints)

    # ================= SAVE TO DATABASE =================
    cursor.execute("""
        INSERT INTO applications (income, loan, term, status, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        data_dict["ApplicantIncome"],
        data_dict["LoanAmount"],
        data_dict["Loan_Amount_Term"],
        status,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()

    return {
        "status": status,
        "reason": reason
    }

# ================= PREDICT ENDPOINT =================
@app.post("/predict")
async def predict(data: LoanInput):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            predict_logic,
            data.dict()
        )
        return result
    except Exception as e:
        return {"error": str(e)}

# ================= BANKER SUMMARY ENDPOINT =================
@app.get("/banker-summary")
def banker_summary():

    cursor.execute("SELECT COUNT(*) FROM applications")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM applications WHERE status='Approved'")
    approved = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM applications WHERE status='Not Approved'")
    rejected = cursor.fetchone()[0]

    approval_rate = 0
    if total > 0:
        approval_rate = round((approved / total) * 100, 2)

    cursor.execute("""
        SELECT id, income, loan, term, status 
        FROM applications 
        ORDER BY id DESC 
        LIMIT 5
    """)
    recent = cursor.fetchall()

    return {
        "total": total,
        "approved": approved,
        "rejected": rejected,
        "approval_rate": approval_rate,
        "recent": recent
    }
