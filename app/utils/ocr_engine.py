import asyncio
from typing import List, Tuple, Dict
import aiohttp
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3Tokenizer, LayoutLMv3ForSequenceClassification
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCREngine:
    def __init__(self):
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(settings.OCR_MODEL_ID)
        self.feature_extractor = LayoutLMv3FeatureExtractor.from_pretrained(settings.OCR_MODEL_ID)
        self.tokenizer = LayoutLMv3Tokenizer.from_pretrained(settings.OCR_MODEL_ID)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.gcv_client = vision.ImageAnnotatorClient()

    async def process_images(self, images: List[Tuple[str, bytes]]) -> Dict[str, Dict]:
        results = {}
        tasks = []
        semaphore = asyncio.Semaphore(5)  # Limit concurrent processing

        async def process_single_image(image_name: str, image_bytes: bytes):
            async with semaphore:
                try:
                    ocr_result = await self._process_with_layoutlm(image_name, image_bytes)
                    if not ocr_result:
                        logger.warning(f"LayoutLM failed for {image_name}, falling back to Google Cloud Vision")
                        ocr_result = await self._process_with_gcv(image_name, image_bytes)
                    if not ocr_result:
                        logger.warning(f"GCV failed for {image_name}, falling back to Tesseract")
                        ocr_result = await self._process_with_tesseract(image_name, image_bytes)
                    results[image_name] = ocr_result
                except Exception as e:
                    logger.error(f"Error processing {image_name}: {str(e)}")
                    results[image_name] = {"error": str(e)}

        for image_name, image_bytes in images:
            task = asyncio.create_task(process_single_image(image_name, image_bytes))
            tasks.append(task)

        await asyncio.gather(*tasks)
        return results

    async def _process_with_layoutlm(self, image_name: str, image_bytes: bytes) -> Dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        encoding = self.feature_extractor(image, return_tensors="pt")
        for k, v in encoding.items():
            encoding[k] = v.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        logits = outputs.logits
        predicted_class = logits.argmax(-1).item()
        
        words, boxes = self._extract_text_and_boxes_layoutlm(image)
        
        return {
            "predicted_class": predicted_class,
            "words": words,
            "boxes": boxes.tolist()
        }

    def _extract_text_and_boxes_layoutlm(self, image: Image) -> Tuple[List[str], np.ndarray]:
        # Convert PIL Image to numpy array
        image_np = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Apply thresholding to preprocess the image
        threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Perform text detection
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

        return words, np.array(boxes)

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
            "boxes": boxes
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
            "boxes": boxes
        }

    async def update_processing_status(self, total_images: int, processed_images: int) -> ProcessingStatus:
        progress = (processed_images / total_images) * 100
        return ProcessingStatus(
            status="Processing" if processed_images < total_images else "Complete",
            progress=progress,
            message=f"Processed {processed_images} out of {total_images} images"
        )

ocr_engine = OCREngine()
