# app/config.py

import os
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Invoice Processing System"

    # File Upload Configuration
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: set = {"pdf", "jpg", "jpeg", "png", "zip"}

    # OCR Configuration
    OCR_MODEL_ID: str = "jinhybr/OCR-LayoutLMv3-Invoice"
    OCR_TIMEOUT: int = 300  # 5 minutes

    # Processing Configuration
    MULTI_PAGE_THRESHOLD: float = 0.95  # 95% confidence for multi-page detection
    INVOICE_NUMBER_ACCURACY: float = 0.95  # 95% accuracy for invoice number extraction
    TOTAL_MATH_ACCURACY: float = 1.0  # 100% accuracy for total calculations

    # Output Configuration
    OUTPUT_FORMAT: str = "csv"  # or "excel"

    # Hugging Face Configuration
    HF_API_TOKEN: str = Field(..., env="HF_API_TOKEN")

    # Google Cloud Vision Configuration (Backup OCR)
    GCV_CREDENTIALS: str = Field(..., env="GOOGLE_APPLICATION_CREDENTIALS")

    # Database Configuration (for potential future use)
    DATABASE_URL: str = Field(..., env="DATABASE_URL")

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

def get_settings() -> Settings:
    return settings
    