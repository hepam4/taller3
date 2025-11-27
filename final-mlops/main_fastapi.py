"""
FastAPI Application para predicción de salarios Data Science
Taller 3 MLOps - Universidad EIA
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import json
import boto3
from datetime import datetime
import os
from typing import List

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

app = FastAPI(
    title="Data Science Salaries Prediction API",
    description="API para predecir salarios en Data Science",
    version="1.0.0"
)

# Variables globales para modelos
best_model = None
preprocessor = None
best_model_info = None
feature_names = None

# AWS S3 Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "taller3-predictions")

try:
    s3_client = boto3.client("s3", region_name=AWS_REGION)
except Exception as e:
    print(f"Warning: No se pudo conectar a S3: {e}")
    s3_client = None


# ============================================================================
# CARGAR MODELOS AL INICIAR
# ============================================================================

@app.on_event("startup")
async def load_models():
    """Cargar modelo y preprocessor al iniciar"""
    global best_model, preprocessor, best_model_info, feature_names
    
    try:
        # Cargar mejor modelo
        best_model = joblib.load("../models/best_model.pkl")
        print("✓ Mejor modelo cargado")
        
        # Cargar preprocessor
        preprocessor = joblib.load("../models/preprocessor.pkl")
        print("✓ Preprocessor cargado")
        
        # Cargar info del modelo
        with open("../models/best_model_info.json", "r") as f:
            best_model_info = json.load(f)
        print(f"✓ Modelo: {best_model_info['model_name']}")
        
        # Obtener nombres de features
        feature_names = joblib.load("../models/feature_names.pkl") if os.path.exists("../models/feature_names.pkl") else None
        
    except FileNotFoundError as e:
        print(f"Error: No se encontraron los archivos del modelo: {e}")
        raise


# ============================================================================
# DEFINIR MODELOS DE ENTRADA
# ============================================================================

class SalaryPredictionRequest(BaseModel):
    """Modelo de entrada para predicción individual"""
    work_year: int
    experience_level: str  # EN, MI, SE, EX
    employment_type: str   # PT, FT, CT, FR
    job_title: str
    remote_ratio: int      # 0, 50, 100
    company_size: str      # S, M, L
    company_location: str  # Código país (e.g., US, UK)

    class Config:
        schema_extra = {
            "example": {
                "work_year": 2023,
                "experience_level": "SE",
                "employment_type": "FT",
                "job_title": "Data Scientist",
                "remote_ratio": 100,
                "company_size": "L",
                "company_location": "US"
            }
        }


class SalaryPredictionResponse(BaseModel):
    """Modelo de salida para predicción"""
    salary_prediction: float
    model_name: str
    model_r2: float
    prediction_id: str


class HealthResponse(BaseModel):
    """Modelo de respuesta para health check"""
    status: str
    model_loaded: bool
    model_name: str


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check del API"""
    return {
        "status": "ok",
        "model_loaded": best_model is not None,
        "model_name": best_model_info['model_name'] if best_model_info else "unknown"
    }


@app.post("/predict", response_model=SalaryPredictionResponse)
async def predict_salary(request: SalaryPredictionRequest):
    """
    Predicción individual de salario
    
    Input:
    - work_year: Año de trabajo (e.g., 2023)
    - experience_level: EN (Entry), MI (Mid), SE (Senior), EX (Executive)
    - employment_type: PT (Part-time), FT (Full-time), CT (Contract), FR (Freelance)
    - job_title: Título del puesto (e.g., Data Scientist, ML Engineer)
    - remote_ratio: % remoto (0, 50, o 100)
    - company_size: S (Small), M (Medium), L (Large)
    - company_location: Código país (e.g., US, UK, CA)
    
    Returns:
    - Predicción del salario en USD
    """
    
    try:
        # Validar modelo cargado
        if best_model is None:
            raise HTTPException(status_code=500, detail="Modelo no cargado")
        
        # Crear DataFrame con los datos de entrada
        input_data = pd.DataFrame([{
            'work_year': request.work_year,
            'remote_ratio': request.remote_ratio,
            'experience_level': request.experience_level,
            'employment_type': request.employment_type,
            'job_title': request.job_title,
            'company_size': request.company_size,
            'company_location': request.company_location
        }])
        
        # Preprocesar datos
        input_processed = preprocessor.transform(input_data)
        
        # Realizar predicción
        prediction = best_model.predict(input_processed)[0]
        
        # Generar ID único para la predicción
        prediction_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Guardar predicción en S3 (opcional)
        if s3_client:
            try:
                save_prediction_to_s3(
                    prediction_id=prediction_id,
                    request=request,
                    prediction=prediction
                )
            except Exception as e:
                print(f"Warning: No se guardó en S3: {e}")
        
        return {
            "salary_prediction": float(prediction),
            "model_name": best_model_info['model_name'],
            "model_r2": best_model_info['r2'],
            "prediction_id": prediction_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {str(e)}")


@app.get("/model-info")
async def model_info():
    """Obtener información del modelo"""
    if best_model_info is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    
    return {
        "model_name": best_model_info['model_name'],
        "rmse": best_model_info['rmse'],
        "mae": best_model_info['mae'],
        "r2": best_model_info['r2'],
        "description": f"Modelo {best_model_info['model_name']} para predicción de salarios en Data Science"
    }


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def save_prediction_to_s3(prediction_id: str, request: SalaryPredictionRequest, prediction: float):
    """Guardar predicción en S3"""
    
    try:
        # Crear JSON con la predicción
        prediction_data = {
            "prediction_id": prediction_id,
            "timestamp": datetime.now().isoformat(),
            "input": {
                "work_year": request.work_year,
                "experience_level": request.experience_level,
                "employment_type": request.employment_type,
                "job_title": request.job_title,
                "remote_ratio": request.remote_ratio,
                "company_size": request.company_size,
                "company_location": request.company_location
            },
            "prediction": {
                "salary_usd": float(prediction),
                "model": best_model_info['model_name'],
                "r2": best_model_info['r2']
            }
        }
        
        # Guardar en S3
        s3_key = f"predictions/{datetime.now().strftime('%Y/%m/%d')}/{prediction_id}.json"
        
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=json.dumps(prediction_data, indent=2),
            ContentType="application/json"
        )
        
        print(f"✓ Predicción guardada en S3: s3://{S3_BUCKET}/{s3_key}")
        
    except Exception as e:
        print(f"Error guardando en S3: {e}")
        raise


# ============================================================================
# RUTAS ADICIONALES
# ============================================================================

@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "Data Science Salaries Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model-info"
        }
    }


@app.get("/docs")
async def docs():
    """Documentación automática"""
    return {"message": "Ver documentación en http://localhost:8000/docs"}


# ============================================================================
# INICIAR APLICACIÓN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
