import os
import zipfile
import magic
from fastapi import UploadFile, HTTPException
from typing import List, Tuple
from PIL import Image
import fitz  # PyMuPDF
import io
import uuid
from app.config import settings
from app.models import FileUpload

class FileHandler:
    def __init__(self, upload_dir: str = "/tmp/invoice_uploads"):
        self.upload_dir = upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)

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

    def process_upload(self, file_upload: FileUpload) -> List[Tuple[str, bytes]]:
        if file_upload.content_type == 'application/zip':
            return self._process_zip(file_upload.filename)
        elif file_upload.content_type == 'application/pdf':
            return self._process_pdf(file_upload.filename)
        else:  # Image files
            return [(file_upload.filename, open(file_upload.filename, 'rb').read())]

    def _process_zip(self, zip_path: str) -> List[Tuple[str, bytes]]:
        extracted_files = []
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if not file_info.filename.endswith('/'):  # Not a directory
                    with zip_ref.open(file_info) as file:
                        content = file.read()
                        content_type = magic.from_buffer(content, mime=True)
                        if content_type in settings.ALLOWED_EXTENSIONS:
                            extracted_files.append((file_info.filename, content))
        return extracted_files

    def _process_pdf(self, pdf_path: str) -> List[Tuple[str, bytes]]:
        pdf_pages = []
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            pdf_pages.append((f"{os.path.basename(pdf_path)}_page_{page_num+1}.png", img_bytes))
        doc.close()
        return pdf_pages

    def clean_up(self, file_path: str):
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Error deleting file {file_path}: {e}")

    @staticmethod
    def is_multi_page(pages: List[Tuple[str, bytes]]) -> bool:
        if len(pages) <= 1:
            return False
        
        # Compare first two pages to check if they're likely part of the same invoice
        img1 = Image.open(io.BytesIO(pages[0][1]))
        img2 = Image.open(io.BytesIO(pages[1][1]))
        
        # Simple heuristic: check if images have similar dimensions and color histograms
        size_similarity = abs(img1.size[0] - img2.size[0]) / max(img1.size[0], img2.size[0])
        hist1 = img1.histogram()
        hist2 = img2.histogram()
        hist_similarity = sum(min(h1, h2) for h1, h2 in zip(hist1, hist2)) / sum(hist1)
        
        return size_similarity < 0.1 and hist_similarity > 0.8

