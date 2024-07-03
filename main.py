from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from prediction_model.predict import make_prediction

app = FastAPI(title= "Loan Prediction API", description= "API for Loan Prediction", version= "1.0")

origins = ["*"]

app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class LoanData(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str

@app.get("/")
def index():
    return {"message": "Welcome to Loan Prediction Project"}

@app.post("/predict")
def predict(loan_data: LoanData):
    data = loan_data.model_dump()
    prediction = make_prediction([data])["prediction"][0]
    if prediction == 'Y':
        prediction = 'Approved'
    else:
        prediction = 'Rejected'
    return {"status": prediction}

if __name__ == "__main__":
    uvicorn.run(app)