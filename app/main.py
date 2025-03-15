from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import tempfile
import os
import uuid
from app.config import settings
from app.utils.file_handler import FileHandler
from app.utils.ocr_engine import ocr_engine
from app.utils.data_extractor import data_extractor
from app.utils.validator import invoice_validator, flag_anomalies
from app.utils.exporter import export_invoices
from app.models import Invoice, ProcessingStatus
import logging

app = FastAPI(title=settings.PROJECT_NAME, version="1.0.0")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

file_handler = FileHandler()

class ProcessingRequest(BaseModel):
    task_id: str

class ProcessingResponse(BaseModel):
    task_id: str
    status: ProcessingStatus

processing_tasks = {}

@app.post("/upload/", response_model=ProcessingRequest)
async def upload_files(files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None):
    task_id = str(uuid.uuid4())
    processing_tasks[task_id] = ProcessingStatus(status="Queued", progress=0, message="Task queued")
    
    background_tasks.add_task(process_files, task_id, files)
    
    return ProcessingRequest(task_id=task_id)

async def process_files(task_id: str, files: List[UploadFile]):
    try:
        processing_tasks[task_id] = ProcessingStatus(status="Processing", progress=0, message="Starting processing")
        
        temp_dir = tempfile.mkdtemp()
        processed_files = []

        for idx, file in enumerate(files):
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            processed_files.extend(file_handler.process_upload(file_path))
            progress = (idx + 1) / len(files) * 30
            processing_tasks[task_id] = ProcessingStatus(status="Processing", progress=progress, message=f"Processed {idx + 1} of {len(files)} files")

        ocr_results = await ocr_engine.process_images(processed_files)
        processing_tasks[task_id] = ProcessingStatus(status="Processing", progress=60, message="OCR completed")

        extracted_data = [data_extractor.extract_data(result) for result in ocr_results.values()]
        processing_tasks[task_id] = ProcessingStatus(status="Processing", progress=80, message="Data extraction completed")

        validated_data = [invoice for invoice, is_valid, _ in invoice_validator.validate_invoice_batch(extracted_data) if is_valid]
        flagged_invoices = flag_anomalies(validated_data)
        
        csv_output = export_invoices(validated_data, 'csv')
        excel_output = export_invoices(validated_data, 'excel')
        
        # Save outputs to temporary files
        csv_path = os.path.join(temp_dir, f"{task_id}_invoices.csv")
        excel_path = os.path.join(temp_dir, f"{task_id}_invoices.xlsx")
        
        with open(csv_path, 'wb') as f:
            f.write(csv_output.getvalue())
        with open(excel_path, 'wb') as f:
            f.write(excel_output.getvalue())
        
        processing_tasks[task_id] = ProcessingStatus(status="Completed", progress=100, message="Processing completed")
        
    except Exception as e:
        logger.error(f"Error processing files for task {task_id}: {str(e)}")
        processing_tasks[task_id] = ProcessingStatus(status="Failed", progress=100, message=f"Error: {str(e)}")

@app.get("/status/{task_id}", response_model=ProcessingResponse)
async def get_processing_status(task_id: str):
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return ProcessingResponse(task_id=task_id, status=processing_tasks[task_id])

@app.get("/download/{task_id}")
async def download_results(task_id: str, format: str = "csv"):
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if processing_tasks[task_id].status != "Completed":
        raise HTTPException(status_code=400, detail="Processing not completed")
    
    temp_dir = tempfile.gettempdir()
    if format.lower() == "csv":
        file_path = os.path.join(temp_dir, f"{task_id}_invoices.csv")
        media_type = "text/csv"
    elif format.lower() == "excel":
        file_path = os.path.join(temp_dir, f"{task_id}_invoices.xlsx")
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        raise HTTPException(status_code=400, detail="Invalid format specified")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(file_path, media_type=media_type, filename=os.path.basename(file_path))

@app.on_event("startup")
async def startup_event():
    # Perform any necessary startup tasks
    logger.info("Application is starting up")

@app.on_event("shutdown")
async def shutdown_event():
    # Perform any necessary cleanup tasks
    logger.info("Application is shutting down")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    