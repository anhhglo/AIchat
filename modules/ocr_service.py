# modules/ocr_service.py
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image, ImageEnhance
import requests
from io import BytesIO
from config import OCR_MODEL_ID
import os
from typing import Optional
import tempfile
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
import gc
import re


class OCRService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(OCRService, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        print(f"[OCRService] Loading model: {OCR_MODEL_ID}...")
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self.device == "cuda" else torch.float32

            self.model = AutoModelForImageTextToText.from_pretrained(
                OCR_MODEL_ID,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=dtype,
                trust_remote_code=True,
                token=os.getenv("HF_TOKEN"),
                low_cpu_mem_usage=True,
            )

            if self.device == "cpu":
                self.model.to("cpu")

            self.processor = AutoProcessor.from_pretrained(
                OCR_MODEL_ID,
                use_fast=True,
                trust_remote_code=True,
                token=os.getenv("HF_TOKEN"),
            )
            print(f"[OCRService] ✅ Model loaded on {self.device}")
            self._initialized = True
        except Exception as e:
            print(f"[OCRService] ❌ Error loading model: {e}")
            raise

    def _load_image_pil(self, image_source) -> Optional[Image.Image]:
        """Load image from file path or URL."""
        try:
            if isinstance(image_source, Image.Image):
                return image_source.convert("RGB")
            if str(image_source).startswith(("http://", "https://")):
                response = requests.get(image_source, timeout=10)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert("RGB")
            else:
                return Image.open(image_source).convert("RGB")
        except Exception as e:
            print(f"[OCRService] ⚠️ Error loading image {image_source}: {e}")
            return None

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR accuracy."""
        try:
            image = image.convert("L")  # Grayscale
            image = ImageEnhance.Contrast(image).enhance(2.0)
            image = ImageEnhance.Sharpness(image).enhance(2.0)

            # Resize small images
            if image.width < 800:
                scale = 1024 / image.width
                new_size = (int(image.width * scale), int(image.height * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            return image.convert("RGB")  # Back to RGB for model
        except Exception as e:
            print(f"[OCRService] ⚠️ Preprocessing failed: {e}")
            return image

    def _aggressive_clean(self, text: str) -> str:
        """
        Clean OCR output: remove repetitions, junk characters,
        and hallucination loops typical of GOT-OCR models.
        """
        if not text:
            return ""

        # 1. Detect and truncate hallucination loops
        triggers = ["I'm ", "I am ", "We are ", "You are ", "It is ", "There are "]
        for trigger in triggers:
            if text.lower().count(trigger.lower()) > 4:
                print(f"[OCRService] Detected hallucination loop: '{trigger}'. Truncating.")
                parts = re.split(re.escape(trigger), text, flags=re.IGNORECASE)
                text = trigger.join(parts[:4])
                break

        lines = text.split("\n")
        cleaned = []

        for line in lines:
            line = line.strip()
            if len(line) < 3:
                continue

            # Skip lines with too few alphabet characters
            alpha_ratio = sum(c.isalpha() or c.isspace() for c in line) / len(line)
            if alpha_ratio < 0.6:
                continue

            # Skip repeated character lines (aaaaa, hhhhh)
            if re.search(r"(.)\1{4,}", line):
                continue

            # Skip long consonant-only sequences (junk)
            if re.search(r"[B-DF-HJ-NP-TV-Z\s]{10,}", line, re.IGNORECASE):
                continue

            cleaned.append(line)

        return "\n".join(cleaned)

    def _extract_text_from_pil(self, image: Image.Image) -> str:
        """Run OCR on a PIL Image."""
        try:
            processed = self._preprocess_image(image)
            inputs = self.processor(processed, return_tensors="pt").to(self.model.device)

            if self.device == "cuda":
                inputs = {
                    k: v.to(torch.float16) if torch.is_floating_point(v) else v
                    for k, v in inputs.items()
                }

            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=1024,
                    repetition_penalty=1.5,
                )

            text = self.processor.decode(
                generated[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            clean_text = self._aggressive_clean(text)

            if self.device == "cuda":
                del inputs, generated
                torch.cuda.empty_cache()

            return clean_text
        except Exception as e:
            print(f"[OCRService] ❌ OCR extraction error: {e}")
            return ""

    def extract_text(self, image_source) -> str:
        """
        Main entry point: extract text from an image file, URL, PIL Image, or PDF.
        """
        if not self._initialized:
            return ""

        # PIL Image passed directly
        if hasattr(image_source, "convert"):
            return self._extract_text_from_pil(image_source)

        source_lower = str(image_source).lower()

        # Image files
        if source_lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff")):
            print(f"[OCRService] Processing image: {image_source}")
            image = self._load_image_pil(image_source)
            return self._extract_text_from_pil(image) if image else ""

        # PDF files
        elif source_lower.endswith(".pdf"):
            print(f"[OCRService] Processing PDF: {image_source}")
            all_text = ""
            try:
                if source_lower.startswith("http"):
                    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
                        r = requests.get(image_source, timeout=30)
                        tmp.write(r.content)
                        tmp.flush()
                        images = convert_from_path(tmp.name, dpi=300)
                else:
                    images = convert_from_path(image_source, dpi=300)

                for i, img in enumerate(images):
                    print(f"[OCRService] OCR page {i+1}/{len(images)}")
                    all_text += f"\n--- Page {i+1} ---\n{self._extract_text_from_pil(img)}\n"
                    images[i] = None  # Free memory
                gc.collect()
                return all_text
            except PDFInfoNotInstalledError:
                print("[OCRService] ❌ poppler-utils not installed!")
                return ""
            except Exception as e:
                print(f"[OCRService] ❌ PDF OCR error: {e}")
                return ""

        return ""


# Singleton instance
ocr_service = OCRService()
