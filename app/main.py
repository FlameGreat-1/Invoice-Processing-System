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
from app.celery_app import process_file_task, process_multiple_files_task
from celery.result import AsyncResult
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
async def upload_files(files: List[UploadFile] = File(...)):
    task_id = str(uuid.uuid4())
    processing_tasks[task_id] = ProcessingStatus(status="Queued", progress=0, message="Task queued")
    
    temp_dir = tempfile.mkdtemp()
    file_paths = []

    for file in files:
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        file_paths.append(file_path)

    if len(files) == 1:
        # Single file processing
        celery_task = process_file_task.delay(task_id, file_paths[0])
    else:
        # Multiple files processing
        celery_task = process_multiple_files_task.delay(task_id, file_paths)
    
    processing_tasks[task_id] = ProcessingStatus(status="Processing", progress=0, message="Processing started")
    
    return ProcessingRequest(task_id=task_id)

@app.get("/status/{task_id}", response_model=ProcessingResponse)
async def get_processing_status(task_id: str):
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    celery_task = AsyncResult(task_id)
    if celery_task.state == 'PENDING':
        status = "Queued"
    elif celery_task.state == 'STARTED':
        status = "Processing"
    elif celery_task.state == 'SUCCESS':
        status = "Completed"
    else:
        status = "Failed"
    
    progress = celery_task.info.get('progress', 0) if celery_task.info else 0
    message = celery_task.info.get('message', '') if celery_task.info else ''
    
    return ProcessingResponse(task_id=task_id, status=ProcessingStatus(status=status, progress=progress, message=message))

@app.get("/download/{task_id}")
async def download_results(task_id: str, format: str = "csv"):
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    celery_task = AsyncResult(task_id)
    if celery_task.state != 'SUCCESS':
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
    logger.info("Application is starting up")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application is shutting down")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
