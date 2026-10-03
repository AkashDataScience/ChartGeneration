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

from openai import AsyncOpenAI
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Groq uses the same OpenAI client, just a different endpoint and key
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = AsyncOpenAI(
    api_key=groq_api_key if groq_api_key else "dummy_key",
    base_url="https://api.groq.com/openai/v1"
)

import time
import asyncio
import json
from src.tools.agent_tools import TOOLS_SCHEMA, TOOL_DISPATCH

FALLBACK_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gpt-4o-mini",
    "gpt-4o",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b"
]

def with_retries(func):
    async def wrapper(*args, **kwargs):
        mutable_args = list(args)
        
        # Determine the current model
        current_model = None
        if mutable_args:
            current_model = mutable_args[0]
        elif 'model' in kwargs:
            current_model = kwargs['model']
        elif 'model_name' in kwargs:
            current_model = kwargs['model_name']
            
        retries = 10  # Give it a few attempts to cascade down the models
        backoff = 2
        
        for attempt in range(retries):
            try:
                return await func(*mutable_args, **kwargs)
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
                        await asyncio.sleep(1)
                    except (ValueError, IndexError):
                        # Not in list or exhausted the list -> standard backoff
                        print(f"-> No further fallback models available. Retrying in {backoff} seconds... ({attempt+1}/{retries})")
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 30)
                else:
                    raise e
    return wrapper

@with_retries
async def run_agent_loop(model: str, messages: list, response_schema: type[BaseModel] | None = None):
    """Runs a recursive agent loop using OpenAI's tool calling API."""

    in_tok_total = 0
    out_tok_total = 0
    
    if model.startswith("gpt-") or model.startswith("llama") or model.startswith("mixtral") or model.startswith("openai/") or model.startswith("qwen/"):
        # ==========================================
        # OpenAI/Groq Tool Calling Loop
        # ==========================================
        active_client = groq_client if not model.startswith("gpt-") or model.startswith("openai/") else openai_client

        while True:
            # 1. Call LLM with tools
            response = await active_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto"
            )
            msg = response.choices[0].message
            in_tok_total += response.usage.prompt_tokens
            out_tok_total += response.usage.completion_tokens
            
            # 2. Check if LLM wants to use a tool
            if msg.tool_calls:
                # Append the assistant's message exactly as it was returned
                messages.append(msg)
                
                # Execute all tool calls
                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    # Better formatted logging
                    print(f"\n  \033[94m[Agent Action]\033[0m Calling tool: \033[1m{func_name}\033[0m")
                    for k, v in func_args.items():
                        print(f"  \033[90m{k}:\033[0m")
                        for line in str(v).split('\n'):
                            print(f"    \033[36m{line}\033[0m")
                    
                    if func_name in TOOL_DISPATCH:
                        try:
                            result = TOOL_DISPATCH[func_name](**func_args)
                        except Exception as e:
                            result = f"Tool execution failed: {str(e)}"
                    else:
                        result = f"Error: Tool {func_name} not found."
                    
                    print(f"  \033[92m[Agent Action]\033[0m Result generated.")
                    
                    # Append the result back to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": str(result)
                    })
                
                # Loop repeats! LLM sees the tool results and decides what to do next.
                continue
                
            else:
                # 3. No more tool calls! Return final structured output
                if response_schema:
                    # To guarantee the Pydantic schema, do one final parse call
                    # (You could also use strict Structured Outputs above, but keeping it split handles complex tool paths better)
                    parse_response = await active_client.beta.chat.completions.parse(
                        model=model,
                        messages=messages,
                        response_format=response_schema
                    )
                    in_tok_total += parse_response.usage.prompt_tokens
                    out_tok_total += parse_response.usage.completion_tokens
                    return parse_response.choices[0].message.parsed, in_tok_total, out_tok_total
                
                return msg.content, in_tok_total, out_tok_total
            
    else:
        # ==========================================
        # Gemini Tool Calling Loop
        # ==========================================
        gemini_contents = []
        for m in messages:
            if m.get("role") == "user":
                gemini_contents.append(types.Content(role="user", parts=[types.Part.from_text(text=m.get("content", ""))]))
                
        # Map the OpenAI TOOLS_SCHEMA into Gemini's format to avoid SDK automatic tool calling bugs
        import copy
        gemini_tools = copy.deepcopy(TOOLS_SCHEMA)
        for t in gemini_tools:
            if "strict" in t["function"]:
                del t["function"]["strict"]
            if "additionalProperties" in t["function"]["parameters"]:
                del t["function"]["parameters"]["additionalProperties"]
                
        config = types.GenerateContentConfig(
            tools=[{"function_declarations": [t["function"] for t in gemini_tools]}]
        )
        
        while True:
            response = await client.aio.models.generate_content(
                model=model,
                contents=gemini_contents,
                config=config
            )
            
            in_tok_total += response.usage_metadata.prompt_token_count if response.usage_metadata else 0
            out_tok_total += response.usage_metadata.candidates_token_count if response.usage_metadata else 0
            
            if response.function_calls:
                # ---------------------------------------------------------
                # WORKAROUND: Bypassing GenAI SDK 'thought_signature' Bug
                # The SDK drops the thought_signature field, breaking the proxy 
                # if we send it back in history. Instead, we modify the original 
                # user prompt with the tool results and restart the context!
                # ---------------------------------------------------------
                
                tool_results_text = "\n\n[SYSTEM NOTE: The following tools were automatically executed on your behalf:\n"
                for fc in response.function_calls:
                    # Strip any potential proxy namespace prefixes
                    func_name = fc.name.split(":")[-1] if ":" in fc.name else fc.name
                    func_args = fc.args
                    # Better formatted logging
                    print(f"\n  \033[94m[Agent Action - Gemini]\033[0m Calling tool: \033[1m{func_name}\033[0m")
                    
                    # Convert proto MapComposite to standard dict if needed, else iterate directly
                    try:
                        args_dict = dict(func_args)
                    except:
                        args_dict = func_args
                        
                    for k, v in args_dict.items():
                        print(f"  \033[90m{k}:\033[0m")
                        for line in str(v).split('\n'):
                            print(f"    \033[36m{line}\033[0m")
                    
                    if func_name in TOOL_DISPATCH:
                        try:
                            result = TOOL_DISPATCH[func_name](**func_args)
                        except Exception as e:
                            result = f"Tool execution failed: {str(e)}"
                    else:
                        result = f"Error: Tool {func_name} not found."
                        
                    print(f"  \033[92m[Agent Action - Gemini]\033[0m Result generated.")
                    tool_results_text += f"\n--- Result from {func_name} ---\n{str(result)}\n"
                
                tool_results_text += "Please continue fulfilling the original instruction using this new information.]\n"
                
                # Append the results directly to the user's original message
                messages[-1]["content"] += tool_results_text
                
                # Rebuild gemini_contents from scratch to wipe out the broken model history
                gemini_contents = []
                for m in messages:
                    if m.get("role") == "user":
                        gemini_contents.append(types.Content(role="user", parts=[types.Part.from_text(text=m.get("content", ""))]))
                
                continue
                
            else:
                # No more tool calls! Return final structured output
                if response_schema:
                    config_schema = types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema
                    )
                    # Force Gemini to map its final thoughts into the structured JSON schema
                    gemini_contents.append(response.candidates[0].content)
                    gemini_contents.append(types.Content(role="user", parts=[types.Part.from_text(text="Please output your final response exactly matching the requested JSON schema.")]))
                    
                    parse_response = await client.aio.models.generate_content(
                        model=model,
                        contents=gemini_contents,
                        config=config_schema
                    )
                    in_tok_total += parse_response.usage_metadata.prompt_token_count if parse_response.usage_metadata else 0
                    out_tok_total += parse_response.usage_metadata.candidates_token_count if parse_response.usage_metadata else 0
                    return parse_response.parsed, in_tok_total, out_tok_total
                
                return response.text, in_tok_total, out_tok_total

@with_retries
async def get_response(model: str, prompt: str, response_schema: type[BaseModel] | None = None):
    """Call a Gemini or OpenAI model and return its text or structured output."""
    if model.startswith("gpt-"):
        if response_schema:
            response = await openai_client.beta.chat.completions.parse(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format=response_schema
            )
            return response.choices[0].message.parsed, response.usage.prompt_tokens, response.usage.completion_tokens
        else:
            response = await openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens

    config = types.GenerateContentConfig()
    if response_schema:
        config.response_mime_type = "application/json"
        config.response_schema = response_schema
        
    response = await client.aio.models.generate_content(
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
async def image_gemini_call(model_name: str, prompt: str, image_path: str, response_schema: type[BaseModel] | None = None):
    if model_name.startswith("gpt-") or model_name.startswith("llama") or model_name.startswith("mixtral") or model_name.startswith("openai/") or model_name.startswith("qwen/"):
        active_client = groq_client if not model_name.startswith("gpt-") or model_name.startswith("openai/") else openai_client
        media_type, b64_image = encode_image_b64(image_path)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64_image}"}}
            ]
        }]
        try:
            if response_schema:
                response = await active_client.beta.chat.completions.parse(
                    model=model_name,
                    messages=messages,
                    response_format=response_schema
                )
                return response.choices[0].message.parsed, response.usage.prompt_tokens, response.usage.completion_tokens
            else:
                response = await active_client.chat.completions.create(
                    model=model_name,
                    messages=messages
                )
                return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens
        except Exception as e:
            if "must be a string" in str(e) or "invalid_request_error" in str(e):
                print(f"[API Warning] Model {model_name} does not support vision. Falling back to text-only prompt.")
                messages[0]["content"] = prompt + "\n(Note: The image could not be analyzed because this model does not support vision. Please reflect on the code instead.)"
                if response_schema:
                    response = await active_client.beta.chat.completions.parse(
                        model=model_name,
                        messages=messages,
                        response_format=response_schema
                    )
                    return response.choices[0].message.parsed, response.usage.prompt_tokens, response.usage.completion_tokens
                else:
                    response = await active_client.chat.completions.create(
                        model=model_name,
                        messages=messages
                    )
                    return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens
            else:
                raise e

    from PIL import Image
    image = Image.open(image_path)
    config = types.GenerateContentConfig()
    if response_schema:
        config.response_mime_type = "application/json"
        config.response_schema = response_schema
        
    response = await client.aio.models.generate_content(
        model=model_name,
        contents=[image, prompt],
        config=config
    )
    
    in_tok = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
    out_tok = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
    
    if response_schema:
        return response.parsed, in_tok, out_tok
    return response.text, in_tok, out_tok