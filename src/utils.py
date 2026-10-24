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
import sys
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

FALLBACK_MODELS = [
    "gemini-4-argon",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gpt-4o-mini",
    "gpt-4o",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b"
]

# Track models that hit quota/credit exhaustion so subsequent calls skip straight to a working model
_EXHAUSTED_MODELS: set[str] = set()
_LAST_WORKING_MODEL: str | None = None


def sanitize_python_code(code: str) -> str:
    """
    Clean LLM-generated Python code:
      1. Strip markdown code fences (```python ... ```).
      2. Replace literal '\\n' and '\\t' sequences that appear OUTSIDE string literals/comments
         (fixes `SyntaxError: unexpected character after line continuation character`).
      3. Replace strict `format='%H:%M:%S'` time parsing with `format='mixed'` so `HH:MM`
         strings do not become all-NaT and cause empty pivot/heatmap errors.
    """
    if not code:
        return ""
    cleaned = code.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:python|py)?\s*|\s*```$", "", cleaned, flags=re.DOTALL).strip()

    # Walk character by character and unescape \n and \t only when outside quotes and comments
    out = []
    i = 0
    n = len(cleaned)
    in_str = None  # None, "'", '"', "'''", '"""'
    in_comment = False

    while i < n:
        ch = cleaned[i]

        if in_comment:
            out.append(ch)
            if ch == "\n":
                in_comment = False
            i += 1
            continue

        if in_str is not None:
            if ch == "\\" and i + 1 < n:
                out.append(ch)
                out.append(cleaned[i + 1])
                i += 2
                continue
            if cleaned.startswith(in_str, i):
                out.append(in_str)
                i += len(in_str)
                in_str = None
                continue
            out.append(ch)
            i += 1
            continue

        # Outside strings and comments
        if ch == "#":
            in_comment = True
            out.append(ch)
            i += 1
            continue

        if cleaned.startswith("'''", i) or cleaned.startswith('"""', i):
            in_str = cleaned[i:i + 3]
            out.append(in_str)
            i += 3
            continue

        if ch in ("'", '"'):
            in_str = ch
            out.append(ch)
            i += 1
            continue

        # Literal \n or \t outside of any string literal is invalid Python syntax
        if ch == "\\" and i + 1 < n:
            nxt = cleaned[i + 1]
            if nxt == "n":
                out.append("\n")
                i += 2
                continue
            elif nxt == "t":
                out.append("    ")
                i += 2
                continue

        out.append(ch)
        i += 1

    result = "".join(out)
    # Prevent `%H:%M:%S` from wiping out `HH:MM` time strings into NaT
    result = re.sub(r"format\s*=\s*['\"]%H:%M:%S['\"]", "format='mixed'", result)
    # Fix common matplotlib hallucination: `bold=True` -> `fontweight='bold'`
    result = re.sub(r"\bbold\s*=\s*True\b", "fontweight='bold'", result)
    return result


def with_retries(func):
    async def wrapper(*args, **kwargs):
        global _LAST_WORKING_MODEL
        mutable_args = list(args)
        
        def _get_model():
            if mutable_args:
                return mutable_args[0]
            if 'model' in kwargs:
                return kwargs['model']
            if 'model_name' in kwargs:
                return kwargs['model_name']
            return None

        def _set_model(new_m: str):
            if mutable_args:
                mutable_args[0] = new_m
            elif 'model' in kwargs:
                kwargs['model'] = new_m
            elif 'model_name' in kwargs:
                kwargs['model_name'] = new_m

        current_model = _get_model()

        # If the requested model was already marked quota-exhausted in this session, jump to working model
        if current_model in _EXHAUSTED_MODELS:
            if _LAST_WORKING_MODEL and _LAST_WORKING_MODEL not in _EXHAUSTED_MODELS:
                current_model = _LAST_WORKING_MODEL
                _set_model(current_model)
            else:
                for candidate in FALLBACK_MODELS:
                    if candidate not in _EXHAUSTED_MODELS:
                        current_model = candidate
                        _set_model(current_model)
                        break

        retries = 10  # Give it a few attempts to cascade down the models
        backoff = 2
        
        for attempt in range(retries):
            try:
                res = await func(*mutable_args, **kwargs)
                if current_model:
                    _LAST_WORKING_MODEL = current_model
                return res
            except Exception as e:
                def get_error_str(err):
                    if hasattr(err, 'exceptions'):
                        return " | ".join(get_error_str(sub_e) for sub_e in err.exceptions)
                    return str(err)
                error_str = get_error_str(e)
                
                retryable_tokens = ("503", "429", "404", "413", "Request too large", "tool_use_failed", "json_validate_failed", "rate_limit")
                if any(tok in error_str for tok in retryable_tokens) and attempt < retries - 1:
                    print(f"\n[API Warning] Received {error_str[:80]}... on {current_model}")
                    if any(q in error_str for q in ("429", "404", "no credits", "RESOURCE_EXHAUSTED")):
                        _EXHAUSTED_MODELS.add(current_model)
                    
                    try:
                        # Find the next non-exhausted model in the fallback list
                        current_idx = FALLBACK_MODELS.index(current_model)
                        next_model = None
                        for cand in FALLBACK_MODELS[current_idx + 1:]:
                            if cand not in _EXHAUSTED_MODELS:
                                next_model = cand
                                break
                        if next_model is None:
                            next_model = FALLBACK_MODELS[current_idx + 1]
                        print(f"-> Falling back to {next_model} (Attempt {attempt+1}/{retries})")
                        
                        current_model = next_model
                        _set_model(current_model)
                            
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

def cast_tool_arguments(func_name: str, args_dict: dict, tools: list) -> dict:
    """Dynamically cast arguments based on the tool's input_schema to prevent LLM type hallucinations."""
    target_tool = next((t for t in tools if t.name == func_name), None)
    if target_tool and hasattr(target_tool, "input_schema") and 'properties' in target_tool.input_schema:
        for param_name, param_info in target_tool.input_schema['properties'].items():
            if param_name in args_dict:
                val = args_dict[param_name]
                expected_type = param_info.get('type')
                try:
                    if expected_type == 'integer':
                        args_dict[param_name] = int(val)
                    elif expected_type == 'number':
                        args_dict[param_name] = float(val)
                    elif expected_type == 'array' and isinstance(val, str):
                        import json
                        args_dict[param_name] = json.loads(val)
                except (ValueError, SyntaxError, TypeError):
                    pass
    return args_dict


def _try_salvage_failed_tool_call(err: Exception) -> tuple[str, dict] | None:
    """
    Attempt to recover tool name and arguments when an OpenAI/Groq model emits slightly
    malformed JSON in a tool call (e.g. invalid escape like \\p instead of \\np, or trailing '"}"}').
    """
    failed_gen = None
    if hasattr(err, "body") and isinstance(err.body, dict):
        err_dict = err.body.get("error") or {}
        if isinstance(err_dict, dict):
            failed_gen = err_dict.get("failed_generation")

    if not failed_gen:
        m = re.search(r"'failed_generation':\s*'(.+?)'\s*\}", str(err), re.DOTALL)
        if m:
            try:
                failed_gen = bytes(m.group(1), "utf-8").decode("unicode_escape")
            except Exception:
                failed_gen = m.group(1)

    if not failed_gen or not isinstance(failed_gen, str):
        return None

    candidate = failed_gen.strip()
    # Fix common trailing quote/brace hallucination: "}"} -> "}}
    candidate = re.sub(r'"\s*\}\s*"\s*\}\s*$', '"}}', candidate)
    # Fix invalid JSON backslash escapes (e.g. \pay_month where model dropped 'n' from \npay_month)
    candidate_fixed_nl = re.sub(r'\\(?![/\"\\bfnrt]|u[0-9a-fA-F]{4})', r'\\n', candidate)

    for attempt_str in (candidate, candidate_fixed_nl):
        try:
            data = json.loads(attempt_str)
            if isinstance(data, dict) and "name" in data and "arguments" in data:
                args = data["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)
                if isinstance(args, dict):
                    return data["name"], args
        except Exception:
            continue
    return None


async def _parse_structured_openai_or_groq(
    active_client,
    model: str,
    messages: list,
    response_schema: type[BaseModel]
) -> tuple[BaseModel, int, int]:
    """
    Parse final structured output. For OpenAI models, tries beta.chat.completions.parse first;
    for Groq models (or on fallback), uses a compact single-turn json_object call with validation repair.
    """
    if active_client is openai_client:
        try:
            parse_response = await active_client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_schema
            )
            in_tok = parse_response.usage.prompt_tokens if parse_response.usage else 0
            out_tok = parse_response.usage.completion_tokens if parse_response.usage else 0
            return parse_response.choices[0].message.parsed, in_tok, out_tok
        except Exception as parse_err:
            print(f"  [Schema Parse Fallback] Strict parse failed on {model} ({str(parse_err)[:80]}). Falling back to JSON mode...")

    schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
    base_user_prompt = ""
    tool_summaries = []
    last_assistant_text = ""

    for m in messages:
        if isinstance(m, dict):
            role = m.get("role", "user")
            content = str(m.get("content", ""))
            if role == "user" and not base_user_prompt:
                base_user_prompt = content[:18000]
            elif role == "tool":
                tool_summaries.append(f"[{m.get('name', 'tool')}]:\n{content[:1800]}")
            elif role == "user" and "[SYSTEM NOTE:" in content:
                tool_summaries.append(content[:1800])
            elif role == "assistant" and content:
                last_assistant_text = content[:3000]
        else:
            content = getattr(m, "content", None)
            if content:
                last_assistant_text = str(content)[:3000]

    combined_context = base_user_prompt
    if tool_summaries:
        combined_context += "\n\n=== COLLECTED TOOL OUTPUTS ===\n" + "\n\n".join(tool_summaries[-5:])
    if last_assistant_text:
        combined_context += f"\n\n=== YOUR DRAFT ANALYSIS ===\n{last_assistant_text}"

    flat_messages = [
        {
            "role": "user",
            "content": (
                f"{combined_context}\n\n"
                "Now output your final response strictly as a single valid JSON object conforming "
                "to the following JSON Schema (include all top-level keys, do not wrap in markdown fences):\n"
                f"{schema_json}"
            )
        }
    ]
    fallback_resp = await active_client.chat.completions.create(
        model=model,
        messages=flat_messages,
        response_format={"type": "json_object"}
    )
    raw_text = (fallback_resp.choices[0].message.content or "{}").strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.DOTALL).strip()
    in_tok = fallback_resp.usage.prompt_tokens if fallback_resp.usage else 0
    out_tok = fallback_resp.usage.completion_tokens if fallback_resp.usage else 0

    try:
        parsed_obj = response_schema.model_validate_json(raw_text)
        return parsed_obj, in_tok, out_tok
    except Exception as val_err:
        print(f"  [Schema Repair] Fixing validation issue on {model}: {str(val_err)[:100]}...")
        repair_messages = flat_messages + [
            {"role": "assistant", "content": raw_text},
            {
                "role": "user",
                "content": (
                    f"Your JSON failed validation with:\n{val_err}\n\n"
                    "Please return the corrected JSON object with all required fields populated."
                )
            }
        ]
        repair_resp = await active_client.chat.completions.create(
            model=model,
            messages=repair_messages,
            response_format={"type": "json_object"}
        )
        repair_text = (repair_resp.choices[0].message.content or "{}").strip()
        if repair_text.startswith("```"):
            repair_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", repair_text, flags=re.DOTALL).strip()
        in_tok += repair_resp.usage.prompt_tokens if repair_resp.usage else 0
        out_tok += repair_resp.usage.completion_tokens if repair_resp.usage else 0
        parsed_obj = response_schema.model_validate_json(repair_text)
        return parsed_obj, in_tok, out_tok

@with_retries
async def run_agent_loop(model: str, mcp_prompt_name: str = None, mcp_prompt_args: dict = None, messages: list = None, response_schema: type[BaseModel] | None = None, requested_resources: list[str] = None):
    """Runs a recursive agent loop using OpenAI's tool calling API."""

    in_tok_total = 0
    out_tok_total = 0
    
    # 1. Setup the MCP Server Parameters
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["src/mcp_server.py"],
        env=None
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            if mcp_prompt_name:
                mcp_prompt = await session.get_prompt(mcp_prompt_name, arguments=mcp_prompt_args or {})
                prompt_text = mcp_prompt.messages[0].content.text
                messages = [{"role": "user", "content": prompt_text}]

            if requested_resources:
                resource_context = "\n[Available Resources from User]:\n"
                for res in requested_resources:
                    try:
                        data = await session.read_resource(f"file://{res}")
                        resource_context += f"Resource URI: file://{res}\nData:\n{data.contents[0].text}\n"
                    except Exception as e:
                        resource_context += f"Failed to load resource {res}: {e}\n"
                if messages:
                    messages[0]["content"] += resource_context
                else:
                    messages = [{"role": "user", "content": resource_context}]

            # Fetch tools dynamically from the MCP server
            mcp_tools_response = await session.list_tools()
            
            # Convert MCP tools to OpenAI Schema
            TOOLS_SCHEMA = []
            for tool in mcp_tools_response.tools:
                TOOLS_SCHEMA.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema
                    }
                })
                
            if model.startswith("gpt-") or model.startswith("llama") or model.startswith("mixtral") or model.startswith("openai/") or model.startswith("qwen/"):
                # ==========================================
                # OpenAI/Groq Tool Calling Loop
                # ==========================================
                active_client = groq_client if not model.startswith("gpt-") or model.startswith("openai/") else openai_client
                tool_parse_retries = 0
        
                while True:
                    # 1. Call LLM with tools (with automatic recovery from malformed tool JSON)
                    try:
                        response = await active_client.chat.completions.create(
                            model=model,
                            messages=messages,
                            tools=TOOLS_SCHEMA,
                            tool_choice="auto"
                        )
                    except Exception as loop_err:
                        err_text = str(loop_err)
                        if ("413" in err_text or "Request too large" in err_text) and response_schema:
                            print(f"\n  [Context Limit Warning] Reached request size limit on {model}; proceeding to final synthesis...")
                            parsed_obj, p_in, p_out = await _parse_structured_openai_or_groq(
                                active_client, model, messages, response_schema
                            )
                            in_tok_total += p_in
                            out_tok_total += p_out
                            return parsed_obj, in_tok_total, out_tok_total
                        if "tool_use_failed" in err_text or "Failed to parse tool call arguments" in err_text:
                            salvaged = _try_salvage_failed_tool_call(loop_err)
                            if salvaged is not None:
                                func_name, func_args = salvaged
                                func_args = cast_tool_arguments(func_name, func_args, mcp_tools_response.tools)
                                print(f"\n  \033[93m[Agent Action - Salvaged Tool Call]\033[0m Calling tool: \033[1m{func_name}\033[0m")
                                for k, v in func_args.items():
                                    print(f"  \033[90m{k}:\033[0m")
                                    for line in str(v).split('\n'):
                                        print(f"    \033[36m{line}\033[0m")
                                try:
                                    mcp_result = await session.call_tool(func_name, arguments=func_args)
                                    result = mcp_result.content[0].text
                                except Exception as tool_e:
                                    result = f"Error: Tool {func_name} failed: {str(tool_e)}"
                                print(f"  \033[92m[Agent Action]\033[0m Result generated.")
                                messages.append({
                                    "role": "user",
                                    "content": (
                                        f"[SYSTEM NOTE: Executed tool `{func_name}` on your behalf. "
                                        f"Result:\n{result}\nPlease continue.]"
                                    )
                                })
                                continue
                            elif tool_parse_retries < 3:
                                tool_parse_retries += 1
                                print(
                                    f"\n  [Tool Call Warning] Malformed JSON in tool call on {model}. "
                                    f"Requesting simpler tool call ({tool_parse_retries}/3)..."
                                )
                                messages.append({
                                    "role": "user",
                                    "content": (
                                        "[SYSTEM WARNING: Your previous tool call failed JSON parsing (`tool_use_failed`). "
                                        "Please retry the tool call with a short 2-3 line snippet, no `#` comments, "
                                        "and valid JSON escaping.]"
                                    )
                                })
                                continue
                        raise loop_err

                    msg = response.choices[0].message
                    in_tok_total += response.usage.prompt_tokens if response.usage else 0
                    out_tok_total += response.usage.completion_tokens if response.usage else 0
                    
                    # 2. Check if LLM wants to use a tool
                    if msg.tool_calls:
                        # Append the assistant's message exactly as it was returned
                        messages.append(msg)
                        
                        # Execute all tool calls
                        for tool_call in msg.tool_calls:
                            func_name = tool_call.function.name
                            func_args = json.loads(tool_call.function.arguments)
                            
                            # Cast arguments to prevent LLM hallucinations
                            func_args = cast_tool_arguments(func_name, func_args, mcp_tools_response.tools)

                            # Better formatted logging
                            print(f"\n  \033[94m[Agent Action]\033[0m Calling tool: \033[1m{func_name}\033[0m")
                            for k, v in func_args.items():
                                print(f"  \033[90m{k}:\033[0m")
                                for line in str(v).split('\n'):
                                    print(f"    \033[36m{line}\033[0m")
                            
                            try:
                                # Forward tool call to MCP Server instead of running locally!
                                mcp_result = await session.call_tool(func_name, arguments=func_args)
                                result = mcp_result.content[0].text
                            except Exception as e:
                                result = f"Error: Tool {func_name} failed: {str(e)}"
                            
                            print(f"  \033[92m[Agent Action]\033[0m Result generated.")
                            
                            # Append the result back to messages (truncated to avoid 413 Request too large)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": func_name,
                                "content": str(result)[:3500]
                            })
                        
                        # Loop repeats! LLM sees the tool results and decides what to do next.
                        continue
                        
                    else:
                        # 3. No more tool calls! Return final structured output
                        if response_schema:
                            parsed_obj, p_in, p_out = await _parse_structured_openai_or_groq(
                                active_client, model, messages, response_schema
                            )
                            in_tok_total += p_in
                            out_tok_total += p_out
                            return parsed_obj, in_tok_total, out_tok_total
                        
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
                            
                            # Convert proto MapComposite to standard dict if needed, else iterate directly
                            try:
                                args_dict = dict(func_args)
                            except:
                                args_dict = func_args

                            # Cast arguments to prevent LLM hallucinations
                            args_dict = cast_tool_arguments(func_name, args_dict, mcp_tools_response.tools)

                            # Better formatted logging
                            print(f"\n  \033[94m[Agent Action - Gemini]\033[0m Calling tool: \033[1m{func_name}\033[0m")
                            
                            for k, v in args_dict.items():
                                print(f"  \033[90m{k}:\033[0m")
                                for line in str(v).split('\n'):
                                    print(f"    \033[36m{line}\033[0m")
                            
                            try:
                                # Forward tool call to MCP Server instead of running locally!
                                mcp_result = await session.call_tool(func_name, arguments=args_dict)
                                result = mcp_result.content[0].text
                            except Exception as e:
                                result = f"Error: Tool {func_name} failed: {str(e)}"
                                
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
async def image_gemini_call(model_name: str, mcp_prompt_name: str = None, mcp_prompt_args: dict = None, prompt: str = None, image_path: str = None, response_schema: type[BaseModel] | None = None, requested_resources: list[str] = None):
    if requested_resources or mcp_prompt_name:
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["src/mcp_server.py"],
            env=None
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                if mcp_prompt_name:
                    mcp_prompt = await session.get_prompt(mcp_prompt_name, arguments=mcp_prompt_args or {})
                    prompt = mcp_prompt.messages[0].content.text

                if requested_resources:
                    resource_context = "\n[Available Resources from User]:\n"
                    for res in requested_resources:
                        try:
                            data = await session.read_resource(f"file://{res}")
                            resource_context += f"Resource URI: file://{res}\nData:\n{data.contents[0].text}\n"
                        except Exception as e:
                            resource_context += f"Failed to load resource {res}: {e}\n"
                    prompt = (prompt or "") + resource_context

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
                return await _parse_structured_openai_or_groq(
                    active_client, model_name, messages, response_schema
                )
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
                    return await _parse_structured_openai_or_groq(
                        active_client, model_name, messages, response_schema
                    )
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