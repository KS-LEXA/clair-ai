from __future__ import annotations
<<<<<<< HEAD
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
=======

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(Path(__file__).parents[2] / ".env")


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.2,
    )
>>>>>>> origin/dev
