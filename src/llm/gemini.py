from __future__ import annotations
from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI

@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    api_key = "AIzaSyB4TYXQYZOeoR8UlYCFe8duXLYZ3kv5st4" 

    return ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.2,
        version="v1",
        
    )