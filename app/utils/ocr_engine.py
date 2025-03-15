import asyncio
from typing import List, Tuple, Dict
import aiohttp
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3Tokenizer, LayoutLMv3ForSequenceClassification, LayoutLMv3Processor
import torch
from PIL import Image
import io
import numpy as np
from google.cloud import vision
from google.cloud.vision import types
from app.config import settings
from app.models import ProcessingStatus
import logging
import cv2
import pytesseract
import fitz  # PyMuPDF for PDF handling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCREngine:
    def __init__(self):
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(settings.OCR_MODEL_ID)
        self.feature_extractor = LayoutLMv3FeatureExtractor.from_pretrained(settings.OCR_MODEL_ID)
        self.tokenizer = LayoutLMv3Tokenizer.from_pretrained(settings.OCR_MODEL_ID)
        self.processor = LayoutLMv3Processor.from_pretrained(settings.OCR_MODEL_ID)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.gcv_client = vision.ImageAnnotatorClient()
        self.cache = {}  # Simple cache for storing processed results

    async def process_documents(self, documents: List[Tuple[str, bytes]]) -> Dict[str, Dict]:
        results = {}
        tasks = []
        semaphore = asyncio.Semaphore(5)  # Limit concurrent processing

        async def process_single_document(doc_name: str, doc_bytes: bytes):
            async with semaphore:
                try:
                    # Check cache first
                    cache_key = hash(doc_bytes)
                    if cache_key in self.cache:
                        results[doc_name] = self.cache[cache_key]
                        return

                    # Determine if it's a multi-page document
                    is_multipage = self._is_multipage_document(doc_bytes)

                    if is_multipage:
                        ocr_result = await self._process_multipage_with_layoutlm(doc_name, doc_bytes)
                    else:
                        ocr_result = await self._process_single_page(doc_name, doc_bytes)

                    results[doc_name] = ocr_result
                    self.cache[cache_key] = ocr_result  # Cache the result
                except Exception as e:
                    logger.error(f"Error processing {doc_name}: {str(e)}")
                    results[doc_name] = {"error": str(e)}

        for doc_name, doc_bytes in documents:
            task = asyncio.create_task(process_single_document(doc_name, doc_bytes))
            tasks.append(task)

        await asyncio.gather(*tasks)
        return results

    def _is_multipage_document(self, doc_bytes: bytes) -> bool:
        try:
            pdf_doc = fitz.open(stream=doc_bytes, filetype="pdf")
            return len(pdf_doc) > 1
        except:
            return False  # Not a PDF or single-page document

    async def _process_multipage_with_layoutlm(self, doc_name: str, doc_bytes: bytes) -> Dict:
        pdf_doc = fitz.open(stream=doc_bytes, filetype="pdf")
        pages = []
        for page_num in range(len(pdf_doc)):
            page = pdf_doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append(img)

        # Process all pages together
        encoding = self.processor(pages, return_tensors="pt", padding=True, truncation=True)
        for k, v in encoding.items():
            encoding[k] = v.to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoding)

        # Extract text and bounding boxes for all pages
        words, boxes = [], []
        for page, page_encoding in zip(pages, encoding['input_ids']):
            page_words, page_boxes = self._extract_text_and_boxes_layoutlm(page, page_encoding)
            words.extend(page_words)
            boxes.extend(page_boxes)

        return {
            "words": words,
            "boxes": boxes,
            "is_multipage": True,
            "num_pages": len(pdf_doc)
        }

    async def _process_single_page(self, image_name: str, image_bytes: bytes) -> Dict:
        try:
            ocr_result = await self._process_with_layoutlm(image_name, image_bytes)
            if not ocr_result:
                logger.warning(f"LayoutLM failed for {image_name}, falling back to Google Cloud Vision")
                ocr_result = await self._process_with_gcv(image_name, image_bytes)
            if not ocr_result:
                logger.warning(f"GCV failed for {image_name}, falling back to Tesseract")
                ocr_result = await self._process_with_tesseract(image_name, image_bytes)
            return ocr_result
        except Exception as e:
            logger.error(f"Error in single page processing for {image_name}: {str(e)}")
            return {"error": str(e)}

    async def _process_with_layoutlm(self, image_name: str, image_bytes: bytes) -> Dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        encoding = self.processor(image, return_tensors="pt")
        for k, v in encoding.items():
            encoding[k] = v.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        words, boxes = self._extract_text_and_boxes_layoutlm(image, encoding['input_ids'][0])
        
        return {
            "words": words,
            "boxes": boxes.tolist(),
            "is_multipage": False,
            "num_pages": 1
        }

    def _extract_text_and_boxes_layoutlm(self, image: Image, encoding) -> Tuple[List[str], List[List[int]]]:
        words = self.tokenizer.convert_ids_to_tokens(encoding)
        words = [word for word in words if word not in [self.tokenizer.pad_token, self.tokenizer.cls_token, self.tokenizer.sep_token]]
        
        boxes = self.feature_extractor(image)['bbox'][0]
        boxes = [box for box, word in zip(boxes, words) if word not in [self.tokenizer.pad_token, self.tokenizer.cls_token, self.tokenizer.sep_token]]

        return words, boxes

    async def _process_with_gcv(self, image_name: str, image_bytes: bytes) -> Dict:
        image = types.Image(content=image_bytes)
        response = await asyncio.to_thread(self.gcv_client.document_text_detection, image)
        document = response.full_text_annotation

        words = []
        boxes = []
        for page in document.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        words.append(''.join([symbol.text for symbol in word.symbols]))
                        vertices = [(vertex.x, vertex.y) for vertex in word.bounding_box.vertices]
                        boxes.append(vertices)

        return {
            "words": words,
            "boxes": boxes,
            "is_multipage": False,
            "num_pages": 1
        }

    async def _process_with_tesseract(self, image_name: str, image_bytes: bytes) -> Dict:
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)

        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        custom_config = r'--oem 3 --psm 6'
        details = pytesseract.image_to_data(threshold, output_type=pytesseract.Output.DICT, config=custom_config, lang='eng')

        words = []
        boxes = []
        n_boxes = len(details['text'])
        for i in range(n_boxes):
            if int(details['conf'][i]) > 60:
                (x, y, w, h) = (details['left'][i], details['top'][i], details['width'][i], details['height'][i])
                words.append(details['text'][i])
                boxes.append([x, y, x+w, y+h])

        return {
            "words": words,
            "boxes": boxes,
            "is_multipage": False,
            "num_pages": 1
        }

    async def update_processing_status(self, total_documents: int, processed_documents: int) -> ProcessingStatus:
        progress = (processed_documents / total_documents) * 100
        return ProcessingStatus(
            status="Processing" if processed_documents < total_documents else "Complete",
            progress=progress,
            message=f"Processed {processed_documents} out of {total_documents} documents"
        )

ocr_engine = OCREngine()
