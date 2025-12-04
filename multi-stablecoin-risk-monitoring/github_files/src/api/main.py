"""
Multi-Stablecoin AI Risk Monitoring System
FastAPI Application

Author: Aditya Sakhale
Institution: NYU School of Professional Studies
Date: November 2025
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import custom modules
from src.models.ensemble import EnsembleModel
from src.feature_engineering.features import FeatureEngineer
from src.llm.explainability import LLMExplainer

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Stablecoin AI Risk Monitoring API",
    description="Real-time risk assessment for USDT, USDC, DAI, and BUSD",
    version="1.0.0",
    contact={
        "name": "Aditya Sakhale",
        "email": "as12345@nyu.edu"
    }
)

# Initialize models
ensemble_model = EnsembleModel()
feature_engineer = FeatureEngineer()
llm_explainer = LLMExplainer()


# Request/Response Models
class TransactionRequest(BaseModel):
    """Single transaction prediction request"""
    stablecoin: str  # USDT, USDC, DAI, BUSD
    transaction_hash: Optional[str] = None
    sender_address: str
    receiver_address: str
    value: float
    timestamp: Optional[datetime] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "stablecoin": "USDT",
                "sender_address": "0x123...",
                "receiver_address": "0x456...",
                "value": 1000000.0
            }
        }


class BatchRequest(BaseModel):
    """Batch prediction request"""
    transactions: List[TransactionRequest]


class RiskResponse(BaseModel):
    """Risk prediction response"""
    risk_score: float
    risk_level: str  # LOW, MEDIUM, HIGH
    confidence: float
    model_contributions: dict
    timestamp: datetime


class ExplainRequest(BaseModel):
    """Explanation request"""
    stablecoin: str
    risk_score: float
    features: dict
    include_regulatory_mapping: bool = True


class ExplainResponse(BaseModel):
    """LLM explanation response"""
    explanation: str
    risk_factors: List[str]
    regulatory_references: Optional[List[str]]
    response_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    models_loaded: bool
    timestamp: datetime


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Multi-Stablecoin AI Risk Monitoring API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=ensemble_model.is_loaded(),
        timestamp=datetime.now()
    )


@app.post("/predict", response_model=RiskResponse, tags=["Prediction"])
async def predict_risk(request: TransactionRequest):
    """
    Predict risk score for a single transaction.
    
    Returns:
        - risk_score: Float between 0 and 1
        - risk_level: LOW (0-0.33), MEDIUM (0.34-0.66), HIGH (0.67-1.0)
        - confidence: Model confidence score
        - model_contributions: Individual model scores
    """
    try:
        # Validate stablecoin
        valid_stablecoins = ["USDT", "USDC", "DAI", "BUSD"]
        if request.stablecoin.upper() not in valid_stablecoins:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stablecoin. Must be one of: {valid_stablecoins}"
            )
        
        # Engineer features
        features = feature_engineer.compute_features(
            stablecoin=request.stablecoin,
            sender=request.sender_address,
            receiver=request.receiver_address,
            value=request.value,
            timestamp=request.timestamp or datetime.now()
        )
        
        # Get ensemble prediction
        prediction = ensemble_model.predict(features)
        
        # Determine risk level
        if prediction["risk_score"] < 0.33:
            risk_level = "LOW"
        elif prediction["risk_score"] < 0.67:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        return RiskResponse(
            risk_score=prediction["risk_score"],
            risk_level=risk_level,
            confidence=prediction["confidence"],
            model_contributions=prediction["model_contributions"],
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=List[RiskResponse], tags=["Prediction"])
async def batch_predict(request: BatchRequest):
    """
    Predict risk scores for multiple transactions.
    
    Optimized for batch processing with vectorized operations.
    """
    try:
        results = []
        for txn in request.transactions:
            result = await predict_risk(txn)
            results.append(result)
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
async def explain_risk(request: ExplainRequest):
    """
    Get LLM-generated explanation for a risk score.
    
    Uses Llama 3.1 70B via Groq API to generate natural language
    explanations with regulatory mapping (SR 11-7, BCBS 248).
    """
    try:
        start_time = datetime.now()
        
        # Generate explanation
        explanation = llm_explainer.generate_explanation(
            stablecoin=request.stablecoin,
            risk_score=request.risk_score,
            features=request.features,
            include_regulatory=request.include_regulatory_mapping
        )
        
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ExplainResponse(
            explanation=explanation["text"],
            risk_factors=explanation["risk_factors"],
            regulatory_references=explanation.get("regulatory_references"),
            response_time_ms=response_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    print("Loading ensemble models...")
    ensemble_model.load_models()
    print("Models loaded successfully!")


# Run with: uvicorn src.api.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
