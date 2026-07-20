import os
import time
from typing import List
from langchain_core.embeddings import Embeddings
import google.genai as genai

class GeminiEmbeddings(Embeddings):
    def __init__(self, client: genai.Client, model_name: str = "gemini-embedding-2"):
        self.client = client
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        safe_texts = [t if t.strip() else " " for t in texts]
        contents = [
            genai.types.Content(parts=[genai.types.Part.from_text(text=t)])
            for t in safe_texts
        ]
        
        max_retries = 5
        base_delay = 20.0
        for attempt in range(max_retries):
            try:
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=contents
                )
                return [emb.values for emb in response.embeddings]
            except Exception as e:
                is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                if is_rate_limit and attempt < max_retries - 1:
                    sleep_time = base_delay * (2 ** attempt)
                    print(f"Rate limit hit during batch embedding. Retrying in {sleep_time:.1f}s (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(sleep_time)
                    continue
                
                # If we exhausted retries or hit another error, fallback to one-by-one embedding
                if attempt == max_retries - 1 or not is_rate_limit:
                    print(f"Batch embedding failed: {e}. Falling back to one-by-one embedding...")
                    return [self.embed_query(t) for t in safe_texts]
                raise e

    def embed_query(self, text: str) -> List[float]:
        safe_text = text if text.strip() else " "
        max_retries = 5
        base_delay = 15.0
        for attempt in range(max_retries):
            try:
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=safe_text
                )
                return response.embeddings[0].values
            except Exception as e:
                is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                if is_rate_limit and attempt < max_retries - 1:
                    sleep_time = base_delay * (2 ** attempt)
                    print(f"Rate limit hit in embed_query. Waiting {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                    continue
                raise e
