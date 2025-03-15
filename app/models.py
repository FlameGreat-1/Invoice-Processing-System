from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import date
from decimal import Decimal
import re

class Address(BaseModel):
    street: str
    city: str
    state: Optional[str]
    country: str
    postal_code: str

class Vendor(BaseModel):
    name: str
    address: Address

class InvoiceItem(BaseModel):
    description: str
    quantity: int
    unit_price: Decimal
    total: Decimal

class Invoice(BaseModel):
    filename: str
    invoice_number: str = Field(..., regex=r'^INV-\d{3}$')
    vendor: Vendor
    invoice_date: date
    grand_total: Decimal
    taxes: Decimal
    final_total: Decimal
    items: List[InvoiceItem]
    pages: int = Field(ge=1, le=2)

    @validator('final_total')
    def validate_final_total(cls, v, values):
        if 'grand_total' in values and 'taxes' in values:
            expected_total = values['grand_total'] + values['taxes']
            if abs(v - expected_total) > Decimal('0.01'):
                raise ValueError(f"Final total {v} does not match grand total {values['grand_total']} plus taxes {values['taxes']}")
        return v

    @validator('invoice_date')
    def validate_invoice_date(cls, v):
        if v > date.today():
            raise ValueError("Invoice date cannot be in the future")
        return v

class ProcessingResult(BaseModel):
    success: bool
    message: str
    invoices: List[Invoice] = []
    errors: List[str] = []

class FileUpload(BaseModel):
    filename: str
    content_type: str
    file_size: int

    @validator('content_type')
    def validate_content_type(cls, v):
        allowed_types = {'application/pdf', 'image/jpeg', 'image/png', 'application/zip'}
        if v not in allowed_types:
            raise ValueError(f"Unsupported file type: {v}")
        return v

    @validator('file_size')
    def validate_file_size(cls, v):
        max_size = 100 * 1024 * 1024  # 100MB
        if v > max_size:
            raise ValueError(f"File size exceeds maximum allowed size of 100MB")
        return v

class ExportFormat(BaseModel):
    format: str = Field(..., regex='^(csv|excel)$')

class ProcessingStatus(BaseModel):
    status: str
    progress: float = Field(ge=0, le=100)
    message: Optional[str]
    