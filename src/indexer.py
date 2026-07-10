import os
from dotenv import load_dotenv
load_dotenv(override=True)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = os.getenv(
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python"
)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["PADDLE_DISABLE_ONEDNN"] = "1"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_call_stack_level"] = "0"

import re
import time
import uuid
from typing import List, Dict
from loguru import logger
import chromadb

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_store")
RESEARCH_COLLECTION = "research_papers"


def clean_collection_name(filename: str) -> str:
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', filename.replace('.', '_'))
    name = re.sub(r'_+', '_', name)
    name = name.strip('_-')
    if not name or not name[0].isalnum():
        name = 'col_' + name
    if not name[-1].isalnum():
        name = name.rstrip('_-') + '_x'
    return name[:200]


# Lazy globals
ocr = None
_ocr_init_error: str | None = None

def _get_ocr():
    global ocr, _ocr_init_error
    if ocr is not None or _ocr_init_error is not None:
        return ocr
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang="en")
        return ocr
    except Exception as e:
        _ocr_init_error = str(e)
        logger.error(f"[indexer] OCR init failed: {e}")
        return None

def detect_file_type(file_path: str) -> str:
    import fitz
    ext = file_path.lower().split('.')[-1]
    if ext in ['jpg', 'jpeg', 'png']:
        return "image"
    elif ext == 'txt':
        return "text"
    elif ext == 'pdf':
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text = page.get_text()
                if text.strip():
                    return "pdf_normal"
            return "pdf_scanned"
        except Exception as e:
            logger.error(f"[indexer] Error detecting PDF type: {e}")
            return "pdf_scanned"
    return "unknown"

class DocumentIndexer:
    def __init__(self):
        self.chroma = chromadb.PersistentClient(path=CHROMA_PATH)
        self.embed_fn = None
        self._embed_init_error: str | None = None

    def _ensure_embed_fn(self):
        if self.embed_fn is not None or self._embed_init_error is not None:
            return
        try:
            from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
            self.embed_fn = ONNXMiniLM_L6_V2()
        except Exception as e:
            self._embed_init_error = str(e)
            logger.error(f"[indexer] Embedding init failed: {e}")

    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap

        logger.info(f"[indexer] Split into {len(chunks)} chunks")
        return chunks

    def _chunk_document(self, text: str) -> List[str]:
        chunks = []
        sections = re.split(r'(?=\n#{1,3} )', "\n" + text)
        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(section.split()) <= CHUNK_SIZE:
                chunks.append(section)
                continue
            paragraphs = section.split('\n\n')
            current = ""
            for p in paragraphs:
                p = p.strip()
                if not p:
                    continue
                candidate = (current + "\n\n" + p).strip()
                if len(candidate.split()) <= CHUNK_SIZE:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                        overlap = " ".join(current.split()[-CHUNK_OVERLAP:])
                        current = (overlap + "\n\n" + p).strip()
                    else:
                        sub = self.chunk_text(p, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
                        if sub:
                            chunks.extend(sub[:-1])
                            current = sub[-1]
            if current:
                if len(current.split()) > CHUNK_SIZE:
                    chunks.extend(self.chunk_text(current, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP))
                else:
                    chunks.append(current)
        return chunks or self.chunk_text(text)

    def _call_gemini_with_retry(self, prompt_fn, max_retries=2):
        for attempt in range(max_retries + 1):
            try:
                return prompt_fn()
            except Exception as e:
                if "429" in str(e) and attempt < max_retries:
                    wait_time = 5 * (attempt + 1)
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                else:
                    raise

    def _generate_doc_summary(self, full_text: str) -> str:
        try:
            from src.vertex_client import get_gemini_model
            logger.info(f"[indexer] Generating summary for {len(full_text)} chars...")
            model = get_gemini_model("gemini-2.5-flash-lite")
            response = self._call_gemini_with_retry(
                lambda: model.generate_content(
                    f"Summarize this document in 2-4 lines only:\n\n{full_text[:8000]}"
                )
            )
            summary = response.text.strip()
            logger.info(f"[indexer] Generated summary: {summary[:100]}")
            return summary
        except Exception as e:
            logger.error(f"[indexer] Summary generation failed: {e}")
            return ""

    def _classify_topic(self, full_text: str) -> str:
        try:
            from src.vertex_client import get_gemini_model
            model = get_gemini_model("gemini-2.5-flash-lite")
            response = self._call_gemini_with_retry(
                lambda: model.generate_content(
                    "Classify this document into ONE short topic category "
                    "(2-4 words max, e.g. 'NLP - Transformers', 'Computer Vision', "
                    "'Reinforcement Learning', 'Finance Report', 'Legal Document'). "
                    f"Respond with ONLY the category, nothing else.\n\n{full_text[:5000]}"
                )
            )
            topic = response.text.strip()
            logger.info(f"[indexer] Classified topic: {topic}")
            return topic
        except Exception as e:
            logger.error(f"[indexer] Topic classification failed: {e}")
            return ""

    def index_text(self, text: str, collection_name: str) -> Dict:
        """Index a piece of text into ChromaDB."""
        self._ensure_embed_fn()
        if self._embed_init_error:
            raise RuntimeError(f"Embedding initialization failed: {self._embed_init_error}")
        
        logger.info(f"[indexer] Indexing into collection: {collection_name}")

        try:
            self.chroma.delete_collection(collection_name)
        except:
            pass

        collection = self.chroma.create_collection(
            name=collection_name,
            embedding_function=self.embed_fn
        )

        chunks = self.chunk_text(text)

        collection.add(
            documents=chunks,
            ids=[str(uuid.uuid4()) for _ in chunks]
        )

        logger.info(f"[indexer] Indexed {len(chunks)} chunks into '{collection_name}'")
        return {
            "collection_name": collection_name,
            "chunks": len(chunks),
            "total_chars": len(text)
        }

    def process_image(self, file_path: str, collection_name: str) -> int:
        import base64
        from PIL import Image
        from groq import Groq

        self._ensure_embed_fn()
        if self._embed_init_error:
            raise RuntimeError(f"Embedding initialization failed: {self._embed_init_error}")

        logger.info(f"[indexer] Processing image {file_path}")

        # 1. Use PIL (Pillow) to open the image file
        with Image.open(file_path) as img:
            img.verify()  # Just ensuring it's a valid image

        # 2. Use PaddleOCR to extract text
        ocr_instance = _get_ocr()
        if ocr_instance is None:
            raise RuntimeError(f"OCR initialization failed: {_ocr_init_error}")
        
        result = ocr_instance.ocr(file_path, cls=False)
        texts = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2 and line[1]:
                    texts.append(str(line[1][0]))
        ocr_text = " ".join(texts)

        # 3. Use Groq Vision API
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        prompt = (
            "Describe what is in this image in detail. "
            "If it contains charts, tables or diagrams explain what data they show. "
            "If it contains text, summarize what the text says."
        )

        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_string}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1024
            )
            llava_description = response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"[indexer] Groq Vision API call failed, skipping visual description: {e}")
            llava_description = ""

        # 4. Combine both outputs
        combined = f"OCR Text:\n{ocr_text}\n\nVisual Description:\n{llava_description}"
        logger.info("[indexer] Calling summary generation...")
        doc_summary = self._generate_doc_summary(combined)
        topic = self._classify_topic(combined)

        # 5. Markdown-aware chunking (splits on headings, falls back to paragraph/word chunking)
        chunks = self._chunk_document(combined)

        # 6. Store in ChromaDB
        combined_chunks = [f"Document context: {doc_summary}\n\n{c}" for c in chunks]

        try:
            self.chroma.delete_collection(collection_name)
        except:
            pass

        collection = self.chroma.create_collection(
            name=collection_name,
            embedding_function=self.embed_fn
        )

        filename = os.path.basename(file_path)
        metadatas = [
            {"chunk_type": "image", "source": filename, "doc_summary": doc_summary, "topic": topic}
            for _ in combined_chunks
        ]

        collection.add(
            documents=combined_chunks,
            metadatas=metadatas,
            ids=[str(uuid.uuid4()) for _ in combined_chunks]
        )

        logger.info(f"[indexer] Stored {len(chunks)} chunks for image {filename}")
        
        # 7. Return number of chunks stored
        return len(chunks)

    def process_normal_pdf(self, file_path: str, collection_name: str) -> int:
        import fitz
        import tempfile

        self._ensure_embed_fn()
        if self._embed_init_error:
            raise RuntimeError(f"Embedding initialization failed: {self._embed_init_error}")

        logger.info(f"[indexer] Processing normal PDF {file_path}")
        doc = fitz.open(file_path)
        filename = os.path.basename(file_path)
        
        combined_text = ""
        page_offsets = []
        
        # 1. Extract text and build combined string with offsets
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                start_idx = len(combined_text)
                combined_text += text + "\n\n"
                end_idx = len(combined_text)
                page_offsets.append((start_idx, end_idx, page_num))
                
        doc_summary = ""
        topic = ""
        if combined_text.strip():
            logger.info("[indexer] Calling summary generation...")
            doc_summary = self._generate_doc_summary(combined_text)
            topic = self._classify_topic(combined_text)

        # 2. Markdown-aware chunking (splits on headings, falls back to paragraph/word chunking)
        chunks = self._chunk_document(combined_text) if combined_text.strip() else []

        text_chunks_data = []
        for c in chunks:
            idx = combined_text.find(c[:100])
            if idx == -1:
                idx = combined_text.find(c[:50])
            if idx == -1:
                idx = combined_text.find(c[:20])
            page_num = 1
            for start_idx, end_idx, p_num in page_offsets:
                if start_idx <= idx <= end_idx:
                    page_num = p_num
                    break
            text_chunks_data.append({"text": c, "page": page_num})
            
        # 3. Extract and process images
        image_chunks_data = []
        for page_num, page in enumerate(doc, start=1):
            image_list = page.get_images(full=True)
            for _, img in enumerate(image_list):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n >= 5:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                    
                fd, temp_img_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                pix.save(temp_img_path)
                
                # Call process_image using a dummy collection so it doesn't wipe our final one
                temp_col_name = f"temp_{uuid.uuid4().hex[:8]}"
                self.process_image(temp_img_path, temp_col_name)
                
                temp_col = self.chroma.get_collection(temp_col_name)
                data = temp_col.get()
                if data and data['documents']:
                    for doc_text in data['documents']:
                        image_chunks_data.append(doc_text)
                        
                try:
                    self.chroma.delete_collection(temp_col_name)
                except:
                    pass
                try:
                    os.remove(temp_img_path)
                except:
                    pass

        # 4. Save everything to ChromaDB
        try:
            self.chroma.delete_collection(collection_name)
        except:
            pass

        collection = self.chroma.create_collection(
            name=collection_name,
            embedding_function=self.embed_fn
        )

        all_docs = []
        all_metas = []

        for tc in text_chunks_data:
            all_docs.append(f"Document context: {doc_summary}\n\n{tc['text']}")
            all_metas.append({"chunk_type": "text", "source": filename, "page": tc["page"], "doc_summary": doc_summary, "topic": topic})

        for ic in image_chunks_data:
            all_docs.append(f"Document context: {doc_summary}\n\n{ic}")
            all_metas.append({"chunk_type": "image", "source": filename, "doc_summary": doc_summary, "topic": topic})
            
        if all_docs:
            collection.add(
                documents=all_docs,
                metadatas=all_metas,
                ids=[str(uuid.uuid4()) for _ in all_docs]
            )

        total_stored = len(all_docs)
        logger.info(f"[indexer] Stored {total_stored} chunks for PDF {filename}")
        return total_stored

    def process_scanned_pdf(self, file_path: str, collection_name: str) -> int:
        import tempfile
        import base64
        import cv2
        import numpy as np
        from pdf2image import convert_from_path
        from groq import Groq

        self._ensure_embed_fn()
        if self._embed_init_error:
            raise RuntimeError(f"Embedding initialization failed: {self._embed_init_error}")

        logger.info(f"[indexer] Processing scanned PDF {file_path}")
        filename = os.path.basename(file_path)
        
        # 1. Convert to images
        try:
            pages = convert_from_path(file_path)
        except Exception as e:
            logger.error(f"[indexer] pdf2image failed (likely missing poppler): {e}. Fallback to PyMuPDF rendering.")
            import fitz
            from PIL import Image
            doc = fitz.open(file_path)
            pages = []
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                if pix.n >= 5:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                pages.append(img)
                
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        combined_text = ""
        page_offsets = []
        temp_files_to_delete = []
        
        for page_num, page_img in enumerate(pages, start=1):
            fd, temp_img_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            page_img.save(temp_img_path, format="PNG")
            temp_files_to_delete.append(temp_img_path)
            
            # 2. OpenCV Preprocessing
            cv_img = cv2.imread(temp_img_path)
            # Convert to grayscale
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            
            # Deskew (fix tilted/rotated scan)
            bitwise_not = cv2.bitwise_not(gray)
            coords = np.column_stack(np.where(bitwise_not > 0))
            if len(coords) > 0:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
                (h, w) = gray.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                deskewed = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            else:
                deskewed = gray
                
            # Denoise using cv2.fastNlMeansDenoising
            denoised = cv2.fastNlMeansDenoising(deskewed, None, h=10, searchWindowSize=21, templateWindowSize=7)
            
            # Threshold using cv2.threshold (Otsu's method)
            _, thresholded = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            fd2, cleaned_img_path = tempfile.mkstemp(suffix=".png")
            os.close(fd2)
            cv2.imwrite(cleaned_img_path, thresholded)
            temp_files_to_delete.append(cleaned_img_path)
            
            # 3. PaddleOCR
            ocr_instance = _get_ocr()
            if ocr_instance is None:
                raise RuntimeError(f"OCR initialization failed: {_ocr_init_error}")
            
            result = ocr_instance.ocr(cleaned_img_path, cls=False)
            texts = []
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2 and line[1]:
                        texts.append(str(line[1][0]))
            ocr_text = " ".join(texts)
                
            # 4. Groq Vision API
            with open(cleaned_img_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                
            prompt = (
                "Does this page contain any charts, diagrams, "
                "photos or visual content beyond regular text? "
                "If yes, describe what you see in detail. "
                "If no, just reply NO."
            )
            
            try:
                response = client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{encoded_string}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=1024
                )
                vision_resp = response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"[indexer] Groq Vision API error: {e}")
                vision_resp = "NO"
                
            visual_description = ""
            if not (vision_resp.upper().strip() == "NO" or vision_resp.upper().strip().startswith("NO.")):
                visual_description = vision_resp
                
            # 5. Combine OCR + Visual
            page_content = f"--- Page {page_num} ---\n\nOCR Text:\n{ocr_text}\n"
            if visual_description:
                page_content += f"\nVisual Description:\n{visual_description}\n"
                
            start_idx = len(combined_text)
            combined_text += page_content + "\n\n"
            end_idx = len(combined_text)
            page_offsets.append((start_idx, end_idx, page_num))
            
        # 6. Markdown-aware chunking (splits on headings, falls back to paragraph/word chunking)
        doc_summary = ""
        topic = ""
        chunks = []
        if combined_text.strip():
            logger.info("[indexer] Calling summary generation...")
            doc_summary = self._generate_doc_summary(combined_text)
            topic = self._classify_topic(combined_text)
            chunks = self._chunk_document(combined_text)
                        
        text_chunks_data = []
        for c in chunks:
            idx = combined_text.find(c[:100])
            if idx == -1:
                idx = combined_text.find(c[:50])
            if idx == -1:
                idx = combined_text.find(c[:20])
            page_num = 1
            for start_idx, end_idx, p_num in page_offsets:
                if start_idx <= idx <= end_idx:
                    page_num = p_num
                    break
            text_chunks_data.append({"text": c, "page": page_num})
            
        # 7. Store in ChromaDB
        try:
            self.chroma.delete_collection(collection_name)
        except:
            pass

        collection = self.chroma.create_collection(
            name=collection_name,
            embedding_function=self.embed_fn
        )

        all_docs = []
        all_metas = []

        for tc in text_chunks_data:
            all_docs.append(f"Document context: {doc_summary}\n\n{tc['text']}")
            all_metas.append({"chunk_type": "ocr", "source": filename, "page": tc["page"], "doc_summary": doc_summary, "topic": topic})
            
        if all_docs:
            collection.add(
                documents=all_docs,
                metadatas=all_metas,
                ids=[str(uuid.uuid4()) for _ in all_docs]
            )

        # 8. Delete all temporary files
        for fpath in temp_files_to_delete:
            try:
                os.remove(fpath)
            except:
                pass

        total_stored = len(all_docs)
        logger.info(f"[indexer] Stored {total_stored} chunks for scanned PDF {filename}")
        
        # 9. Return total chunks stored
        return total_stored

    def index_file(self, file_path: str, collection_name: str = None) -> Dict:
        filename_only = os.path.basename(file_path)
        collection_name = clean_collection_name(filename_only)
        logger.info(f"[indexer] Filename: {filename_only}, Collection: {collection_name}")

        file_type = detect_file_type(file_path)
        logger.info(f"[indexer] Detected: {file_type} → routing to correct pipeline")

        total_chars = 0
        chunks = 0

        if file_type == "text":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            total_chars = len(text)
            res = self.index_text(text, collection_name)
            chunks = res["chunks"]
        elif file_type == "image":
            chunks = self.process_image(file_path, collection_name)
        elif file_type == "pdf_normal":
            chunks = self.process_normal_pdf(file_path, collection_name)
        elif file_type == "pdf_scanned":
            chunks = self.process_scanned_pdf(file_path, collection_name)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        return {
            "collection_name": collection_name,
            "file_type": file_type,
            "chunks": chunks,
            "total_chars": total_chars
        }

    def list_collections(self) -> List[str]:
        return [c.name for c in self.chroma.list_collections()]

    def delete_collection(self, collection_name: str) -> None:
        self.chroma.delete_collection(collection_name)

    def index_file_to_research(self, file_path: str, filename: str) -> dict:
        """Index a file into the shared research_papers collection.

        Uses the same extraction pipeline as index_file() but stores all chunks
        into a single shared collection with 'source' and 'collection_name' metadata
        so individual papers can be queried or deleted by filter.
        """
        try:
            import shutil
            import tempfile as _tempfile

            self._ensure_embed_fn()
            if self._embed_init_error:
                raise RuntimeError(f"Embedding initialization failed: {self._embed_init_error}")

            # Copy to a temp dir under the original filename so os.path.basename()
            # inside the existing process_* methods records the correct 'source'.
            tmp_dir = _tempfile.mkdtemp()
            named_tmp = os.path.join(tmp_dir, filename)
            staging_coll = clean_collection_name(os.path.basename(filename))

            try:
                shutil.copy2(file_path, named_tmp)
                result = self.index_file(named_tmp)

                staging_col = self.chroma.get_collection(name=staging_coll)
                data = staging_col.get(include=["documents", "metadatas", "embeddings"])
                docs = data.get("documents") or []
                metas = data.get("metadatas") or [{} for _ in docs]
            finally:
                try:
                    self.chroma.delete_collection(staging_coll)
                except Exception:
                    pass
                shutil.rmtree(tmp_dir, ignore_errors=True)

            if not docs:
                return {
                    "chunks": 0,
                    "source": filename,
                    "collection": RESEARCH_COLLECTION,
                    "file_type": result.get("file_type", "unknown"),
                }

            # Add collection_name to each chunk's metadata so delete-by-paper works
            filename_only = os.path.basename(filename)
            cleaned_name = clean_collection_name(filename_only)
            logger.info(f"[indexer] Collection name: {cleaned_name}")
            for meta in metas:
                meta["collection_name"] = cleaned_name

            from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
            embed_fn = ONNXMiniLM_L6_V2()
            research_col = self.chroma.get_or_create_collection(
                name=RESEARCH_COLLECTION,
                embedding_function=embed_fn,
            )

            research_col.add(
                documents=docs,
                metadatas=metas,
                ids=[str(uuid.uuid4()) for _ in docs],
            )
            logger.info(f"[indexer] Added {len(docs)} chunks for '{filename}' to '{RESEARCH_COLLECTION}'")

            return {
                "chunks": len(docs),
                "source": filename,
                "collection": RESEARCH_COLLECTION,
                "file_type": result.get("file_type", "unknown"),
            }
        except TypeError as e:
            import traceback
            logger.error(f"[indexer] TypeError in index_file_to_research: {e}")
            logger.error(traceback.format_exc())
            raise

    def delete_paper_chunks(self, collection_name: str) -> int:
        """Delete all chunks for a paper from research_papers via metadata filter.

        Also falls back to deleting a legacy standalone collection of the same name.
        Returns the number of chunks removed from research_papers.
        """
        deleted = 0
        try:
            research_col = self.chroma.get_collection(name=RESEARCH_COLLECTION)
            before = research_col.count()
            research_col.delete(where={"collection_name": collection_name})
            after = research_col.count()
            deleted = before - after
            logger.info(
                f"[indexer] Deleted {deleted} chunks for '{collection_name}' from {RESEARCH_COLLECTION}"
            )
        except Exception:
            pass
        try:
            self.chroma.delete_collection(collection_name)
            logger.info(f"[indexer] Deleted standalone collection '{collection_name}'")
        except Exception:
            pass
        return deleted
