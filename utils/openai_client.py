import os
from openai import OpenAI

def get_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY fehlt (.env setzen).")
    
    # Einfache OpenAI-Client-Konfiguration ohne problematische Parameter
    return OpenAI(api_key=key)
