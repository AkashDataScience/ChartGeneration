import json
import re
import os
import src.utils as utils
from pydantic import BaseModel

class ChartCodeResponse(BaseModel):
    thought_process: str
    python_code: str

class ReflectionResponse(BaseModel):
    critique: str
    feedback: str
    refined_code: str

def generate_chart_code(instruction: str, model: str, out_path_v1: str, df, previous_code: str = "") -> tuple[str, str]:
    """Generate Python code to make a plot with matplotlib using tag-based wrapping."""

    prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "generation_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()
        
    previous_code_section = ""
    if previous_code:
        previous_code_section = f"Here is the code you generated previously. Modify it to fulfill the new instruction:\n```python\n{previous_code}\n```"

    prompt = prompt_template.format(
        instruction=instruction, 
        out_path_v1=out_path_v1, 
        schema=utils.make_schema_text(df),
        previous_code_section=previous_code_section
    )

    response = utils.get_response(model, prompt, response_schema=ChartCodeResponse)
    return response.thought_process, response.python_code


def reflect_on_image_and_regenerate(
    chart_path: str,
    instruction: str,
    model_name: str,
    out_path_v2: str,
    code_v1: str,
    df  
) -> tuple[str, str, str]:
    """
    Critique the chart IMAGE and the original code against the instruction, 
    then return refined matplotlib code.
    Returns (critique, feedback, refined_code).
    Works with Gemini vision-capable models (e.g. gemini-2.5-flash).
    """

    prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "reflection_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    prompt = prompt_template.format(
        code_v1=code_v1,
        out_path_v2=out_path_v2,
        instruction=instruction,
        schema=utils.make_schema_text(df)
    )

    # Send the chart image + prompt to the reflection model and get structured response
    structured_response = utils.image_gemini_call(
        model_name, 
        prompt, 
        chart_path, 
        response_schema=ReflectionResponse
    )

    return structured_response.critique, structured_response.feedback, structured_response.refined_code


def fix_chart_code(instruction: str, bad_code: str, error_message: str, model: str, df) -> str:
    """Ask the model to fix Python code that produced an error."""
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "error_fix_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()
        
    prompt = prompt_template.format(
        instruction=instruction, 
        bad_code=bad_code, 
        error_message=error_message,
        schema=utils.make_schema_text(df)
    )

    response = utils.get_response(model, prompt, response_schema=ChartCodeResponse)
    return response.python_code
