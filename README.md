# Banking Mitra

Banking Mitra is a Streamlit-based banking chatbot designed to answer customer queries using PDF-backed bank documents and multimodal inputs.

## Key features

- Streamlit interface with a polished banking theme
- Google Generative AI Gemini integration for smart responses
- PDF knowledge retrieval from `/data` via `ingest.py`
- Voice note support with `st.audio_input` and audio upload fallback
- Camera/photo support with `st.camera_input` and image upload fallback
- Bilingual answer support in Hindi and English
- Session-managed chat history and document retrieval insights

## Files included

- `app.py` – main Streamlit chatbot application
- `ingest.py` – document ingestion script for creating the Chroma vector database
- `requirements.txt` – Python dependencies list
- `.gitignore` – recommended ignores for the repo
- `data/` – optional PDF documents to use as knowledge sources
- `chroma_db/` – local vector store database (ignored in repo)

## Setup

1. Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install the dependencies:

```powershell
pip install -r requirements.txt
```

3. Add your bank PDF documents to the `data/` folder.

4. Build the knowledge base by running:

```powershell
python ingest.py
```

5. Set your Google Generative AI key:

- Add `GENAI_API_KEY` in Streamlit secrets or
- Set environment variable `GENAI_API_KEY`

6. Run the app:

```powershell
streamlit run app.py
```

7. Open the app in your browser at:

```text
http://localhost:8501
```

## Usage notes

- Use `Reset Conversation` to clear chat history.
- If camera or microphone capture is blocked, use the file upload fallback instead.
- For best results, keep the app running on `localhost` and use a secure browser session.

## GitHub repository upload

1. Initialize git locally:

```powershell
git init -b main
```

2. Stage and commit the project:

```powershell
git add .
git commit -m "Initial commit: Banking Mitra Streamlit chatbot"
```

3. Create a repository on GitHub and add the remote, for example:

```powershell
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

## Notes

- Keep sensitive PDFs outside the tracked repository if they contain private banking data.
- If you want to include sample documents, add them to `data/` before pushing or add a separate sample data repository.

---

This project is ideal for showcasing a multimodal banking chatbot with document search, voice/photo input, and AI-powered support for Hindi and English users.