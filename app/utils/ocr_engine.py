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
from concurrent.futures import ThreadPoolExecutor

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
        self.executor = ThreadPoolExecutor(max_workers=settings.MAX_WORKERS)

    async def process_documents(self, documents: List[Dict[str, any]]) -> Dict[str, Dict]:
        results = {}
        batches = self._create_batches(documents, batch_size=settings.BATCH_SIZE)
        
        for batch in batches:
            batch_results = await asyncio.gather(*[self._process_document(doc) for doc in batch])
            results.update(batch_results)
        
        return results

    def _create_batches(self, documents: List[Dict[str, any]], batch_size: int) -> List[List[Dict[str, any]]]:
        return [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

    async def _process_document(self, document: Dict[str, any]) -> Tuple[str, Dict]:
        try:
            cache_key = hash(document['content'])
            if cache_key in self.cache:
                return document['filename'], self.cache[cache_key]

            if document['is_multipage']:
                ocr_result = await self._process_multipage_with_layoutlm(document)
            else:
                ocr_result = await self._process_single_page(document)

            self.cache[cache_key] = ocr_result
            return document['filename'], ocr_result
        except Exception as e:
            logger.error(f"Error processing {document['filename']}: {str(e)}")
            return document['filename'], {"error": str(e)}

    async def _process_multipage_with_layoutlm(self, document: Dict[str, any]) -> Dict:
        pages = [Image.open(io.BytesIO(page['content'])) for page in document['pages']]

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
            "num_pages": len(pages)
        }

    async def _process_single_page(self, document: Dict[str, any]) -> Dict:
        image_bytes = document['content']
        image_name = document['filename']
        
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
            "boxes": boxes,
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
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._process_with_tesseract_sync, image_name, image_bytes)

    def _process_with_tesseract_sync(self, image_name: str, image_bytes: bytes) -> Dict:
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
