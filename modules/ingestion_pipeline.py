# modules/ingestion_pipeline.py
"""
Ingestion Pipeline: extract text → chunk → embed → store in Pinecone.
Supports PDF (digital + scanned OCR) and image files.
"""

import os
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from modules.ocr_service import ocr_service
from modules.vector_store import vector_store
from config import CHUNK_SIZE, CHUNK_OVERLAP


class IngestionPipeline:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        print("[IngestionPipeline] ✅ Initialized")

    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using PyMuPDF (fast, for digital PDFs)."""
        print(f"[IngestionPipeline] Extracting text with PyMuPDF: {file_path}")
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text() or ""
            doc.close()
            return text
        except Exception as e:
            print(f"[IngestionPipeline] ⚠️ PyMuPDF extraction failed: {e}")
            return ""

    def _ocr_pdf(self, file_path: str) -> str:
        """OCR a scanned PDF page-by-page."""
        print(f"[IngestionPipeline] PDF appears scanned. Running OCR...")
        try:
            try:
                page_count = len(fitz.open(file_path))
            except Exception:
                page_count = 10

            dpi = 300 if page_count <= 10 else 200
            print(f"[IngestionPipeline] {page_count} pages, DPI={dpi}")

            images = convert_from_path(file_path, dpi=dpi)
            all_text = ""
            for i, page_image in enumerate(images):
                print(f"[IngestionPipeline] OCR page {i+1}/{len(images)}")
                page_text = ocr_service.extract_text(page_image)
                all_text += f"\n\n--- Page {i+1} ---\n\n{page_text}\n"
            return all_text
        except PDFInfoNotInstalledError:
            print("[IngestionPipeline] ❌ poppler-utils not installed!")
            raise
        except Exception as e:
            print(f"[IngestionPipeline] ❌ PDF OCR error: {e}")
            raise

    def ingest_file(self, file_path: str, file_source: str, thread_id: str = "default") -> bool:
        """
        Full ingestion pipeline: extract → chunk → embed → store in Pinecone.
        Tags all chunks with thread_id for user-specific filtering.
        """
        print(f"[IngestionPipeline] Ingesting: {file_path} (user: {thread_id})")
        extracted_text = ""
        file_ext = os.path.splitext(file_path)[1].lower()

        try:
            # Image files → OCR
            if file_ext in (".png", ".jpg", ".jpeg", ".bmp"):
                print("[IngestionPipeline] Running OCR for image file...")
                extracted_text = ocr_service.extract_text(file_path)

            # PDF files → PyMuPDF first, fallback to OCR
            elif file_ext == ".pdf":
                extracted_text = self._extract_text_from_pdf(file_path)
                if not extracted_text or len(extracted_text.strip()) < 100:
                    print("[IngestionPipeline] Digital text insufficient, switching to OCR")
                    extracted_text = self._ocr_pdf(file_path)
            else:
                print(f"[IngestionPipeline] ⚠️ Unsupported file type: {file_ext}")
                return False

            if not extracted_text or len(extracted_text.strip()) < 10:
                print("[IngestionPipeline] ❌ No text extracted from file")
                return False

            # Chunk
            print(f"[IngestionPipeline] Extracted {len(extracted_text)} chars. Chunking...")
            chunks = self.text_splitter.split_text(extracted_text)
            source_basename = os.path.basename(file_source)

            documents = [
                Document(
                    page_content=chunk,
                    metadata={
                        "source": source_basename,
                        "user_id": thread_id,
                        "chunk_number": i,
                    },
                )
                for i, chunk in enumerate(chunks)
            ]

            if not documents:
                print("[IngestionPipeline] ❌ No documents after chunking")
                return False

        except Exception as e:
            print(f"[IngestionPipeline] ❌ Extraction/chunking error: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Store in Pinecone
        try:
            vector_store.add_documents(documents)
            print(f"[IngestionPipeline] ✅ Ingested: {source_basename} ({len(documents)} chunks) → Pinecone")
            return True
        except Exception as e:
            print(f"[IngestionPipeline] ❌ Vector store error: {e}")
            return False


# Singleton instance
ingestion_pipeline = IngestionPipeline()
