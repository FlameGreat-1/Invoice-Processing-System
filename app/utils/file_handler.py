import os
import zipfile
import magic
from fastapi import UploadFile, HTTPException
from typing import List, Dict
from PIL import Image
import fitz  # PyMuPDF
import io
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from app.config import settings
from app.models import FileUpload

class FileHandler:
    def __init__(self, upload_dir: str = "/tmp/invoice_uploads"):
        self.upload_dir = upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=settings.MAX_WORKERS)

    async def save_upload(self, file: UploadFile) -> FileUpload:
        content_type = magic.from_buffer(await file.read(1024), mime=True)
        await file.seek(0)
        
        if content_type not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {content_type}")
        
        file_size = 0
        file_path = os.path.join(self.upload_dir, f"{uuid.uuid4()}_{file.filename}")
        
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(8192):
                file_size += len(chunk)
                if file_size > settings.MAX_UPLOAD_SIZE:
                    os.remove(file_path)
                    raise HTTPException(status_code=400, detail="File size exceeds the maximum allowed size")
                buffer.write(chunk)
        
        return FileUpload(filename=file_path, content_type=content_type, file_size=file_size)

    async def process_uploads(self, file_uploads: List[FileUpload]) -> List[Dict[str, any]]:
        tasks = [self.process_upload(file_upload) for file_upload in file_uploads]
        results = await asyncio.gather(*tasks)
        return [item for sublist in results for item in sublist]  # Flatten the list of lists

    async def process_upload(self, file_upload: FileUpload) -> List[Dict[str, any]]:
        if file_upload.content_type == 'application/zip':
            return await self._process_zip(file_upload.filename)
        elif file_upload.content_type == 'application/pdf':
            return await self._process_pdf(file_upload.filename)
        else:  # Image files
            return [await self._process_image(file_upload.filename)]

    async def _process_zip(self, zip_path: str) -> List[Dict[str, any]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._process_zip_sync, zip_path)

    def _process_zip_sync(self, zip_path: str) -> List[Dict[str, any]]:
        extracted_files = []
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if not file_info.filename.endswith('/'):  # Not a directory
                    with zip_ref.open(file_info) as file:
                        content = file.read()
                        content_type = magic.from_buffer(content, mime=True)
                        if content_type in settings.ALLOWED_EXTENSIONS:
                            if content_type == 'application/pdf':
                                extracted_files.extend(self._process_pdf_content(file_info.filename, content))
                            else:
                                extracted_files.append(self._process_image_content(file_info.filename, content))
        return extracted_files

    async def _process_pdf(self, pdf_path: str) -> List[Dict[str, any]]:
        loop = asyncio.get_event_loop()
        with open(pdf_path, 'rb') as file:
            content = await loop.run_in_executor(self.executor, file.read)
        return await loop.run_in_executor(self.executor, self._process_pdf_content, os.path.basename(pdf_path), content)

    def _process_pdf_content(self, filename: str, content: bytes) -> List[Dict[str, any]]:
        doc = fitz.open(stream=content, filetype="pdf")
        pages = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            pages.append({
                'filename': f"{filename}_page_{page_num+1}.png",
                'content': img_bytes,
                'page_number': page_num + 1,
                'total_pages': len(doc)
            })
        doc.close()
        return [{
            'filename': filename,
            'content': content,
            'pages': pages,
            'is_multipage': len(pages) > 1
        }]

    async def _process_image(self, image_path: str) -> Dict[str, any]:
        loop = asyncio.get_event_loop()
        with open(image_path, 'rb') as file:
            content = await loop.run_in_executor(self.executor, file.read)
        return await loop.run_in_executor(self.executor, self._process_image_content, os.path.basename(image_path), content)

    def _process_image_content(self, filename: str, content: bytes) -> Dict[str, any]:
        return {
            'filename': filename,
            'content': content,
            'pages': [{
                'filename': filename,
                'content': content,
                'page_number': 1,
                'total_pages': 1
            }],
            'is_multipage': False
        }

    async def clean_up(self, file_path: str):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self._clean_up_sync, file_path)

    def _clean_up_sync(self, file_path: str):
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Error deleting file {file_path}: {e}")

file_handler = FileHandler()
