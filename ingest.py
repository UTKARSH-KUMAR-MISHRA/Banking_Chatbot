import os
import hashlib
import json
import gc
import time
from typing import List

import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embeddings import GeminiEmbeddings
import google.genai as genai
from langchain_chroma import Chroma

# Optional OCR dependencies (used only if available)
try:
    from pdf2image import convert_from_path
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False


DATA_FOLDER = "data"
PERSIST_DIR = "chroma_db"
HASHES_FILE = os.path.join(PERSIST_DIR, "chunk_hashes.json")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_existing_hashes() -> set:
    if os.path.isfile(HASHES_FILE):
        try:
            with open(HASHES_FILE, "r", encoding="utf-8") as f:
                arr = json.load(f)
            return set(arr)
        except Exception:
            return set()
    return set()


def save_hashes(hashes: set):
    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(HASHES_FILE, "w", encoding="utf-8") as f:
        json.dump(list(hashes), f, ensure_ascii=False)


def ocr_page_from_pdf(path: str, page_number: int) -> str:
    if not OCR_AVAILABLE:
        return ""
    try:
        images = convert_from_path(path, first_page=page_number, last_page=page_number)
        if not images:
            return ""
        text = pytesseract.image_to_string(images[0])
        return text
    except Exception:
        return ""


def get_api_key() -> str:
    # 1. Try to get from streamlit secrets if available
    try:
        import streamlit as st
        api_key = st.secrets.get("GENAI_API_KEY")
        if api_key:
            return api_key
    except Exception:
        pass
    
    # 2. Try to load from .streamlit/secrets.toml manually
    secrets_path = os.path.join(".streamlit", "secrets.toml")
    if os.path.exists(secrets_path):
        try:
            with open(secrets_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() == "GENAI_API_KEY":
                            return v.strip().strip('"').strip("'")
        except Exception:
            pass

    # 3. Environment variable
    return os.getenv("GENAI_API_KEY")


def run_ingestion(progress_callback=None) -> str:
    """
    Builds/updates the Chroma vector store from PDFs in DATA_FOLDER.
    Processes file-by-file to keep memory consumption low.
    """
    def report(msg: str):
        print(msg)
        if progress_callback:
            progress_callback(msg)

    api_key = get_api_key()
    if not api_key:
        raise ValueError("GENAI_API_KEY not found in Streamlit secrets, .streamlit/secrets.toml, or environment variables.")
    
    client = genai.Client(api_key=api_key)
    embedding_function = GeminiEmbeddings(client=client)

    # If DB exists, check compatibility. If incompatible (e.g. dimension mismatch), delete it.
    if os.path.isdir(PERSIST_DIR):
        try:
            db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_function)
            db.similarity_search("test", k=1)
            report("Existing database is compatible. Incrementing updates...")
        except Exception as e:
            report(f"Existing database incompatible or corrupted ({e}). Rebuilding database...")
            import shutil
            shutil.rmtree(PERSIST_DIR, ignore_errors=True)
            if os.path.exists(HASHES_FILE):
                try:
                    os.remove(HASHES_FILE)
                except:
                    pass

    # Re-initialize DB context after possible deletion
    db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_function)

    if not os.path.isdir(DATA_FOLDER):
        report("Data folder not found.")
        return "No data folder found."

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    existing_hashes = load_existing_hashes()
    hashes = set(existing_hashes)
    total_added = 0

    pdf_files = [f for f in sorted(os.listdir(DATA_FOLDER)) if f.lower().endswith(".pdf")]
    
    for idx, file in enumerate(pdf_files):
        path = os.path.join(DATA_FOLDER, file)
        report(f"[{idx+1}/{len(pdf_files)}] Processing {file}...")
        
        try:
            reader = pypdf.PdfReader(path)
            num_pages = len(reader.pages)
            report(f"  Total pages: {num_pages}.")
            limit = min(num_pages, 25)
            pages = []
            for i in range(limit):
                page_text = reader.pages[i].extract_text() or ""
                doc = Document(page_content=page_text, metadata={"source": path, "page": i + 1})
                pages.append(doc)
            report(f"  Loaded first {len(pages)} pages.")
        except Exception as e:
            report(f"  Error loading {file}: {e}")
            continue

        file_docs = []
        for i, doc in enumerate(pages):
            meta = dict(doc.metadata or {})
            meta["source_file"] = file
            meta["page"] = i + 1
            text = (doc.page_content or "").strip()
            if len(text) < 50:
                report(f"    Page {i+1} text is short ({len(text)} chars). Trying OCR...")
                start_ocr = time.time()
                ocr_text = ocr_page_from_pdf(path, i + 1)
                duration = time.time() - start_ocr
                report(f"    OCR finished in {duration:.2f}s.")
                if ocr_text and len(ocr_text) > len(text):
                    text = ocr_text
            doc.page_content = text
            doc.metadata = meta
            file_docs.append(doc)

        # Split into chunks
        chunks = text_splitter.split_documents(file_docs)
        
        # Deduplicate
        new_chunks = []
        for c in chunks:
            h = sha256_text(c.page_content)
            if h in hashes:
                continue
            hashes.add(h)
            new_chunks.append(c)

        if new_chunks:
            report(f"  Embedding and adding {len(new_chunks)} chunks...")
            
            # Batch embedding requests to respect rate limits if needed (e.g. 50 chunks at a time)
            chunk_batch_size = 50
            for k in range(0, len(new_chunks), chunk_batch_size):
                batch = new_chunks[k:k+chunk_batch_size]
                db.add_documents(batch)
                
                # Sleep to stay within 100 RPM API quota limit
                report("  Sleeping 30s to respect API quota...")
                time.sleep(30.0)
            
            total_added += len(new_chunks)
            save_hashes(hashes)
            report(f"  Successfully added {len(new_chunks)} chunks.")
        else:
            report("  No new chunks to add.")

        # Progressive memory cleanup
        del pages
        del file_docs
        del chunks
        del new_chunks
        gc.collect()

    report(f"Ingestion completed. Total new chunks added: {total_added}")
    return f"Ingestion completed. Added {total_added} chunks."


def main():
    run_ingestion()


if __name__ == "__main__":
    main()