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


def get_response(model: str, prompt: str, response_schema: type[BaseModel] | None = None):
    """Call a Gemini model and return its text or structured output."""
    config = types.GenerateContentConfig()
    if response_schema:
        config.response_mime_type = "application/json"
        config.response_schema = response_schema
        
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config
    )
    if response_schema:
        return response.parsed
    return response.text
    
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

    

    
def image_gemini_call(model_name: str, prompt: str, image_path: str, response_schema: type[BaseModel] | None = None):
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
    if response_schema:
        return response.parsed
    return response.text