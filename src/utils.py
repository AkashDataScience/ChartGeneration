# === Standard Library ===
import os
import re
import json
import base64
import mimetypes
from pathlib import Path

# === Third-Party ===
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel
from google.genai import types
from PIL import Image  # (kept if you need it elsewhere)
from dotenv import load_dotenv
from google import genai
from html import escape

# === Env & Clients ===
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=gemini_api_key) if gemini_api_key else genai.Client()

from openai import OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

import time

FALLBACK_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gpt-4o-mini",
    "gpt-4o",
    "gemini-3.1-pro"
]

def with_retries(func):
    def wrapper(*args, **kwargs):
        mutable_args = list(args)
        
        # Determine the current model
        current_model = None
        if mutable_args:
            current_model = mutable_args[0]
        elif 'model' in kwargs:
            current_model = kwargs['model']
        elif 'model_name' in kwargs:
            current_model = kwargs['model_name']
            
        retries = 5  # Give it a few attempts to cascade down the models
        backoff = 2
        
        for attempt in range(retries):
            try:
                return func(*mutable_args, **kwargs)
            except Exception as e:
                error_str = str(e)
                if ("503" in error_str or "429" in error_str) and attempt < retries - 1:
                    print(f"\n[API Warning] Received {error_str[:60]}... on {current_model}")
                    
                    try:
                        # Find the next model in the fallback list
                        current_idx = FALLBACK_MODELS.index(current_model)
                        next_model = FALLBACK_MODELS[current_idx + 1]
                        print(f"-> Falling back to {next_model} (Attempt {attempt+1}/{retries})")
                        
                        current_model = next_model
                        if mutable_args:
                            mutable_args[0] = current_model
                        elif 'model' in kwargs:
                            kwargs['model'] = current_model
                        elif 'model_name' in kwargs:
                            kwargs['model_name'] = current_model
                            
                        # Brief pause to avoid spamming the API
                        time.sleep(1)
                    except (ValueError, IndexError):
                        # Not in list or exhausted the list -> standard backoff
                        print(f"-> No further fallback models available. Retrying in {backoff} seconds... ({attempt+1}/{retries})")
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 30)
                else:
                    raise e
    return wrapper

@with_retries
def get_response(model: str, prompt: str, response_schema: type[BaseModel] | None = None):
    """Call a Gemini or OpenAI model and return its text or structured output."""
    if model.startswith("gpt-"):
        if response_schema:
            response = openai_client.beta.chat.completions.parse(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format=response_schema
            )
            return response.choices[0].message.parsed, response.usage.prompt_tokens, response.usage.completion_tokens
        else:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens

    config = types.GenerateContentConfig()
    if response_schema:
        config.response_mime_type = "application/json"
        config.response_schema = response_schema
        
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config
    )
    
    in_tok = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
    out_tok = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
    
    if response_schema:
        return response.parsed, in_tok, out_tok
    return response.text, in_tok, out_tok
    
# === Data Loading ===
def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load CSV and derive date parts commonly used in charts."""
    df = pd.read_csv(csv_path)
    # Be tolerant if 'date' exists
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["quarter"] = df["date"].dt.quarter
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
    return df

# === Helpers ===
def make_schema_text(df: pd.DataFrame) -> str:
    """Return a human-readable schema from a DataFrame."""
    return "\n".join(f"- {c}: {dt}" for c, dt in df.dtypes.items())



def encode_image_b64(path: str) -> tuple[str, str]:
    """Return (media_type, base64_str) for an image file path."""
    mime, _ = mimetypes.guess_type(path)
    media_type = mime or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return media_type, b64


import base64

import pandas as pd
from typing import Any

def print_html(content: Any, title: str | None = None, is_image: bool = False):
    """
    Print content to the terminal.
    """
    if title:
        print(f"\n=== {title} ===")
    
    if is_image and isinstance(content, str):
        print(f"[Image Output Saved to: {content}]")
    elif isinstance(content, pd.DataFrame):
        print(content.to_string(index=False))
    elif isinstance(content, pd.Series):
        print(content.to_frame().to_string())
    else:
        print(content)
    print("-" * 40)

    

    
@with_retries
def image_gemini_call(model_name: str, prompt: str, image_path: str, response_schema: type[BaseModel] | None = None):
    if model_name.startswith("gpt-"):
        media_type, b64_image = encode_image_b64(image_path)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64_image}"}}
            ]
        }]
        if response_schema:
            response = openai_client.beta.chat.completions.parse(
                model=model_name,
                messages=messages,
                response_format=response_schema
            )
            return response.choices[0].message.parsed, response.usage.prompt_tokens, response.usage.completion_tokens
        else:
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens

    from PIL import Image
    image = Image.open(image_path)
    config = types.GenerateContentConfig()
    if response_schema:
        config.response_mime_type = "application/json"
        config.response_schema = response_schema
        
    response = client.models.generate_content(
        model=model_name,
        contents=[image, prompt],
        config=config
    )
    
    in_tok = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
    out_tok = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
    
    if response_schema:
        return response.parsed, in_tok, out_tok
    return response.text, in_tok, out_tok