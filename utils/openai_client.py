import os
from openai import OpenAI

def get_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY fehlt (.env setzen).")
    headers = {
        "User-Agent": "fitness-form-assistant/0.4",
        "X-Stainless-Client-User-Agent": '{"name":"fitness-form-assistant","version":"0.4"}'
    }
    return OpenAI(api_key=key, default_headers=headers)
