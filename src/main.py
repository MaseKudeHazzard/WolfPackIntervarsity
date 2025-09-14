from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import shap
import sqlite3
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Optional

# Initialize FastAPI app
app = FastAPI(title="Micro-Loan Underwriting Assistant", description="API for processing micro-loan applications using alternative data and gamification.")

# Load machine learning model, scaler, and background data
try:
    model = joblib.load("logistic_regression_model.pkl")
    scaler = joblib.load("scaler.pkl")
    background_data = joblib.load("background_data.pkl")
    explainer = shap.LinearExplainer(model, background_data)
except Exception as e:
    raise RuntimeError(f"Failed to load model, scaler, background data, or explainer: {str(e)}")

# Database setup
def init_db():
    conn = sqlite3.connect("loans.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            transaction_frequency FLOAT,
            avg_transaction_amount FLOAT,
            utility_payment_consistency FLOAT,
            airtime_topup_frequency FLOAT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS loans (
            loan_id TEXT PRIMARY KEY,
            user_id TEXT,
            amount FLOAT,
            decision TEXT,
            score FLOAT,
            application_date TEXT,
            due_date TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS repayments (
            repayment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            loan_id TEXT,
            payment_date TEXT,
            amount FLOAT,
            status TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (loan_id) REFERENCES loans(loan_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_gamification (
            user_id TEXT PRIMARY KEY,
            repayment_streak INTEGER DEFAULT 0,
            points_earned INTEGER DEFAULT 0,
            badges_earned TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)
    conn.commit()
    conn.close()

@app.on_event("startup")
async def startup_event():
    init_db()

# Pydantic models
class LoanApplication(BaseModel):
    user_id: str = Field(..., min_length=1, description="Unique user identifier")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount")
    transaction_frequency: float = Field(..., ge=0, description="Frequency of transactions per month")
    avg_transaction_amount: float = Field(..., ge=0, description="Average transaction amount")
    utility_payment_consistency: float = Field(..., ge=0, le=1, description="Consistency of utility payments (0-1)")
    airtime_topup_frequency: float = Field(..., ge=0, description="Frequency of airtime top-ups per month")

class Repayment(BaseModel):
    user_id: str = Field(..., min_length=1, description="Unique user identifier")
    loan_id: str = Field(..., min_length=1, description="Unique loan identifier")
    payment_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Payment date in YYYY-MM-DD format")
    amount: float = Field(..., gt=0, description="Repayment amount")

# Helper functions
def is_first_application(user_id: str) -> bool:
    conn = sqlite3.connect("loans.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM loans WHERE user_id = ?", (user_id,))
    count = cursor.fetchone()[0]
    conn.close()
    return count == 0

def next_due_date() -> str:
    return (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

def fetch_user_data(user_id: str) -> Dict:
    conn = sqlite3.connect("loans.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    user_data = cursor.fetchone()
    conn.close()
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "transaction_frequency": user_data[1],
        "avg_transaction_amount": user_data[2],
        "utility_payment_consistency": user_data[3],
        "airtime_topup_frequency": user_data[4]
    }

# Endpoints
@app.post("/loan/apply", summary="Apply for a loan using alternative data")
async def apply_loan(application: LoanApplication):
    # Prepare features for model
    features = [
        application.transaction_frequency,
        application.avg_transaction_amount,
        application.utility_payment_consistency,
        application.airtime_topup_frequency
    ]
    
    try:
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Score with model
        score = model.predict_proba(features_scaled)[0][1] * 100
        decision = "approve" if score > 70 else "deny"
        
        # SHAP explanation
        shap_values = explainer(features_scaled)[0].values
        explanation = {
            "transaction_frequency": float(shap_values[0]),
            "avg_transaction_amount": float(shap_values[1]),
            "utility_payment_consistency": float(shap_values[2]),
            "airtime_topup_frequency": float(shap_values[3])
        }
        
        # Update gamification
        points = 50
        badges = ["First Application"] if is_first_application(application.user_id) else []
        loan_id = f"L{application.user_id}_{int(datetime.now().timestamp())}"
        due_date = next_due_date()
        
        # Save to database
        conn = sqlite3.connect("loans.db")
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT OR IGNORE INTO users (user_id, transaction_frequency, avg_transaction_amount, utility_payment_consistency, airtime_topup_frequency) VALUES (?, ?, ?, ?, ?)",
                (application.user_id, application.transaction_frequency, application.avg_transaction_amount, application.utility_payment_consistency, application.airtime_topup_frequency)
            )
            cursor.execute(
                "INSERT INTO loans (loan_id, user_id, amount, decision, score, application_date, due_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (loan_id, application.user_id, application.loan_amount, decision, score, datetime.now().strftime("%Y-%m-%d"), due_date)
            )
            cursor.execute(
                "INSERT OR IGNORE INTO user_gamification (user_id, repayment_streak, points_earned, badges_earned) VALUES (?, 0, 0, '')",
                (application.user_id,)
            )
            cursor.execute(
                "UPDATE user_gamification SET points_earned = points_earned + ?, badges_earned = ? WHERE user_id = ?",
                (points, ",".join(badges), application.user_id)
            )
            conn.commit()
        except sqlite3.Error as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            conn.close()
        
        return {
            "user_id": application.user_id,
            "loan_id": loan_id,
            "decision": decision,
            "score": score,
            "explanation": explanation,
            "points_earned": points,
            "badges_earned": badges,
            "message": f"Loan {decision}! Repay by {due_date} to earn 50 points."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing loan application: {str(e)}")

@app.get("/user/progress/{user_id}", summary="Retrieve user progress and gamification metrics")
async def get_user_progress(user_id: str):
    conn = sqlite3.connect("loans.db")
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        user_data = cursor.fetchone()
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        cursor.execute("SELECT * FROM user_gamification WHERE user_id = ?", (user_id,))
        gamification_data = cursor.fetchone()
        if not gamification_data:
            gamification_data = (user_id, 0, 0, "")
        
        cursor.execute("SELECT payment_date, status, amount FROM repayments WHERE user_id = ?", (user_id,))
        progress_map = [{"date": row[0], "status": row[1], "amount": row[2]} for row in cursor.fetchall()]
        
        return {
            "user_id": user_id,
            "alternative_data": {
                "transaction_frequency": user_data[1],
                "avg_transaction_amount": user_data[2],
                "utility_payment_consistency": user_data[3],
                "airtime_topup_frequency": user_data[4]
            },
            "gamification": {
                "repayment_streak": gamification_data[1],
                "points_earned": gamification_data[2],
                "badges_earned": gamification_data[3].split(",") if gamification_data[3] else [],
                "progress_map": progress_map
            }
        }
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        conn.close()

@app.post("/repayment/record", summary="Record a loan repayment and update gamification")
async def record_repayment(repayment: Repayment):
    conn = sqlite3.connect("loans.db")
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT amount, due_date FROM loans WHERE loan_id = ?", (repayment.loan_id,))
        loan = cursor.fetchone()
        if not loan or abs(loan[0] - repayment.amount) > 0.01:
            raise HTTPException(status_code=400, detail="Invalid loan or amount")
        
        due_date = datetime.strptime(loan[1], "%Y-%m-%d")
        payment_date = datetime.strptime(repayment.payment_date, "%Y-%m-%d")
        status = "on-time" if payment_date <= due_date + timedelta(days=1) else "late"
        
        cursor.execute("SELECT repayment_streak, points_earned, badges_earned FROM user_gamification WHERE user_id = ?", (repayment.user_id,))
        gamification = cursor.fetchone()
        if not gamification:
            cursor.execute(
                "INSERT INTO user_gamification (user_id, repayment_streak, points_earned, badges_earned) VALUES (?, 0, 0, '')",
                (repayment.user_id,)
            )
            gamification = (0, 0, "")
        
        new_streak = gamification[0] + 1 if status == "on-time" else 0
        points = gamification[1] + 50 if status == "on-time" else gamification[1]
        badges = gamification[2].split(",") if gamification[2] else []
        if new_streak == 3 and "Consistent Payer" not in badges:
            badges.append("Consistent Payer")
            points += 100
        if new_streak == 5 and "Reliable Borrower" not in badges:
            badges.append("Reliable Borrower")
            points += 200
        
        # Recalculate score
        user_data = fetch_user_data(repayment.user_id)
        user_data["repayment_streak"] = new_streak
        features = [user_data["transaction_frequency"], user_data["avg_transaction_amount"], 
                    user_data["utility_payment_consistency"], user_data["airtime_topup_frequency"]]
        features_scaled = scaler.transform([features])
        new_score = model.predict_proba(features_scaled)[0][1] * 100
        
        cursor.execute(
            "INSERT INTO repayments (user_id, loan_id, payment_date, amount, status) VALUES (?, ?, ?, ?, ?)",
            (repayment.user_id, repayment.loan_id, repayment.payment_date, repayment.amount, status)
        )
        cursor.execute(
            "UPDATE user_gamification SET repayment_streak = ?, points_earned = ?, badges_earned = ? WHERE user_id = ?",
            (new_streak, points, ",".join(badges), repayment.user_id)
        )
        conn.commit()
        
        return {
            "user_id": repayment.user_id,
            "loan_id": repayment.loan_id,
            "status": status,
            "new_repayment_streak": new_streak,
            "points_earned": 50 if status == "on-time" else 0,
            "badges_earned": ["Consistent Payer"] if new_streak == 3 else ["Reliable Borrower"] if new_streak == 5 else [],
            "new_score": new_score,
            "message": "Repayment recorded! You earned 50 points and a badge." if status == "on-time" else "Repayment recorded."
        }
    except sqlite3.Error as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing repayment: {str(e)}")
    finally:
        conn.close()
