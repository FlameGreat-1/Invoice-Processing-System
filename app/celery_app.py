from celery import Celery
from app.config import settings
from app.utils.file_handler import FileHandler
from app.utils.ocr_engine import ocr_engine
from app.utils.data_extractor import data_extractor
from app.utils.validator import invoice_validator, flag_anomalies
from app.utils.exporter import export_invoices
import os
import tempfile
from typing import List
import asyncio

celery_app = Celery('invoice_processing', broker=settings.CELERY_BROKER_URL)

file_handler = FileHandler()

@celery_app.task(bind=True)
def process_file_task(self, task_id: str, file_path: str):
    try:
        self.update_state(state='STARTED', meta={'progress': 0, 'message': 'Starting processing'})
        
        processed_files = file_handler.process_upload(file_path)
        self.update_state(state='STARTED', meta={'progress': 30, 'message': 'File processed'})

        loop = asyncio.get_event_loop()
        ocr_results = loop.run_until_complete(ocr_engine.process_images(processed_files))
        self.update_state(state='STARTED', meta={'progress': 60, 'message': 'OCR completed'})

        extracted_data = [data_extractor.extract_data(result) for result in ocr_results.values()]
        self.update_state(state='STARTED', meta={'progress': 80, 'message': 'Data extraction completed'})

        validated_data = [invoice for invoice, is_valid, _ in invoice_validator.validate_invoice_batch(extracted_data) if is_valid]
        flagged_invoices = flag_anomalies(validated_data)
        
        csv_output = export_invoices(validated_data, 'csv')
        excel_output = export_invoices(validated_data, 'excel')
        
        temp_dir = tempfile.gettempdir()
        csv_path = os.path.join(temp_dir, f"{task_id}_invoices.csv")
        excel_path = os.path.join(temp_dir, f"{task_id}_invoices.xlsx")
        
        with open(csv_path, 'wb') as f:
            f.write(csv_output.getvalue())
        with open(excel_path, 'wb') as f:
            f.write(excel_output.getvalue())
        
        self.update_state(state='SUCCESS', meta={'progress': 100, 'message': 'Processing completed'})
        
    except Exception as e:
        self.update_state(state='FAILURE', meta={'progress': 100, 'message': f'Error: {str(e)}'})
        raise

@celery_app.task(bind=True)
def process_multiple_files_task(self, task_id: str, file_paths: List[str]):
    try:
        self.update_state(state='STARTED', meta={'progress': 0, 'message': 'Starting processing'})
        
        processed_files = []
        for idx, file_path in enumerate(file_paths):
            processed_files.extend(file_handler.process_upload(file_path))
            progress = (idx + 1) / len(file_paths) * 30
            self.update_state(state='STARTED', meta={'progress': progress, 'message': f'Processed {idx + 1} of {len(file_paths)} files'})

        loop = asyncio.get_event_loop()
        ocr_results = loop.run_until_complete(ocr_engine.process_images(processed_files))
        self.update_state(state='STARTED', meta={'progress': 60, 'message': 'OCR completed'})

        extracted_data = [data_extractor.extract_data(result) for result in ocr_results.values()]
        self.update_state(state='STARTED', meta={'progress': 80, 'message': 'Data extraction completed'})

        validated_data = [invoice for invoice, is_valid, _ in invoice_validator.validate_invoice_batch(extracted_data) if is_valid]
        flagged_invoices = flag_anomalies(validated_data)
        
        csv_output = export_invoices(validated_data, 'csv')
        excel_output = export_invoices(validated_data, 'excel')
        
        temp_dir = tempfile.gettempdir()
        csv_path = os.path.join(temp_dir, f"{task_id}_invoices.csv")
        excel_path = os.path.join(temp_dir, f"{task_id}_invoices.xlsx")
        
        with open(csv_path, 'wb') as f:
            f.write(csv_output.getvalue())
        with open(excel_path, 'wb') as f:
            f.write(excel_output.getvalue())
        
        self.update_state(state='SUCCESS', meta={'progress': 100, 'message': 'Processing completed'})
        
    except Exception as e:
        self.update_state(state='FAILURE', meta={'progress': 100, 'message': f'Error: {str(e)}'})
        raise

celery_app.conf.task_routes = {
    'app.celery_app.process_file_task': {'queue': 'single_file'},
    'app.celery_app.process_multiple_files_task': {'queue': 'multiple_files'},
}
