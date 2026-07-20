import os
import hashlib
import json
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
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


def collect_documents() -> List:
    documents: List = []
    if not os.path.isdir(DATA_FOLDER):
        print(f"Data folder '{DATA_FOLDER}' not found.")
        return documents

    for file in sorted(os.listdir(DATA_FOLDER)):
        if not file.lower().endswith(".pdf"):
            continue
        path = os.path.join(DATA_FOLDER, file)
        loader = PyPDFLoader(path)
        pages = loader.load()
        # Attach filename and page metadata; try OCR for short pages
        for i, doc in enumerate(pages):
            meta = dict(doc.metadata or {})
            meta["source_file"] = file
            meta["page"] = i + 1
            text = (doc.page_content or "").strip()
            if len(text) < 50:
                ocr_text = ocr_page_from_pdf(path, i + 1)
                if ocr_text and len(ocr_text) > len(text):
                    text = ocr_text
            # mutate the loader Document (avoid importing langchain.schema.Document)
            doc.page_content = text
            doc.metadata = meta
            documents.append(doc)

    return documents


def run_ingestion(progress_callback=None) -> str:
    """
    Builds/updates the Chroma vector store from PDFs in DATA_FOLDER.

    progress_callback: optional callable(str) used to report status — e.g.
    pass st.write or a Streamlit spinner's label updater when calling this
    from app.py, so the person sees what's happening on first boot.

    Returns a short human-readable status message.
    """
    def report(msg: str):
        print(msg)
        if progress_callback:
            progress_callback(msg)

    documents = collect_documents()
    if not documents:
        report("No documents found to ingest.")
        return "No PDFs found in data/ — knowledge base not built."

    # Improved chunking: smaller chunks for better retrieval granularity
    report("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    # Deduplicate chunks using a simple hash store
    existing_hashes = load_existing_hashes()
    new_chunks = []
    hashes = set(existing_hashes)
    for c in chunks:
        h = sha256_text(c.page_content)
        if h in hashes:
            continue
        hashes.add(h)
        new_chunks.append(c)

    if not new_chunks:
        report("No new chunks to add. Database is up-to-date.")
        return "Knowledge base already up-to-date."

    report(f"Embedding {len(new_chunks)} chunks (first run can take a minute)...")
    embedding_function = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"local_files_only": False},
    )

    # If DB exists, try to append; otherwise create from documents
    if os.path.isdir(PERSIST_DIR):
        db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_function)
        try:
            db.add_documents(new_chunks)
            status = f"Appended {len(new_chunks)} new chunks to existing DB."
        except Exception:
            # Fallback: rebuild full DB from all chunks
            all_chunks = chunks
            db = Chroma.from_documents(all_chunks, embedding_function, persist_directory=PERSIST_DIR)
            status = f"Rebuilt DB with {len(all_chunks)} chunks."
    else:
        db = Chroma.from_documents(new_chunks, embedding_function, persist_directory=PERSIST_DIR)
        status = f"Created DB with {len(new_chunks)} chunks."

    # Persist chunk hashes for future dedup
    save_hashes(hashes)
    report(status + " Database taiyaar hai!")
    return status


def main():
    run_ingestion()


if __name__ == "__main__":
    main()