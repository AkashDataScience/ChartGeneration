import json
import re
import os
import src.utils as utils
from pydantic import BaseModel, Field

class ChartCodeResponse(BaseModel):
    reasoning_type: str = Field(description="The type of reasoning used (e.g., 'data_exploration', 'code_generation', 'error_correction')")
    step_by_step_plan: list[str] = Field(description="A sequential breakdown of the code generation strategy.")
    internal_self_check: str = Field(description="A validation check evaluating whether the planned steps meet all hard constraints (e.g., using plt.close(), correct data aggregation).")
    thought_process: str = Field(description="General explanation of the code logic.")
    python_code: str = Field(description="The final valid Python code.")

class ReflectionResponse(BaseModel):
    reasoning_type: str = Field(description="The type of reasoning used (e.g., 'visual_evaluation', 'code_critique')")
    step_by_step_plan: list[str] = Field(description="A sequential breakdown of the required fixes based on the critique.")
    internal_self_check: str = Field(description="A validation check ensuring the critique accurately reflects the chart image and original code.")
    critique: str
    feedback: str
    refined_code: str

async def generate_chart_code(instruction: str, model: str, out_path_v1: str, previous_code: str = "", requested_resources: list[str] = None) -> tuple[str, str, int, int]:
    """Generate Python code to make a plot with matplotlib using tag-based wrapping."""

    previous_code_section = ""
    if previous_code:
        previous_code_section = f"Here is the code you generated previously. Modify it to fulfill the new instruction:\n```python\n{previous_code}\n```"

    mcp_prompt_args = {
        "instruction": instruction,
        "out_path_v1": out_path_v1,
        "previous_code_section": previous_code_section
    }

    parsed, in_tok, out_tok = await utils.run_agent_loop(
        model=model, 
        mcp_prompt_name="generation_prompt",
        mcp_prompt_args=mcp_prompt_args,
        response_schema=ChartCodeResponse, 
        requested_resources=requested_resources
    )
    
    full_thought = f"Reasoning Type: {parsed.reasoning_type}\n"
    full_thought += f"Plan:\n" + "\n".join([f"  - {s}" for s in parsed.step_by_step_plan]) + "\n"
    full_thought += f"Self Check: {parsed.internal_self_check}\n"
    full_thought += f"Thought Process: {parsed.thought_process}"

    return full_thought, parsed.python_code, in_tok, out_tok


async def reflect_on_image_and_regenerate(
    chart_path: str,
    instruction: str,
    model_name: str,
    out_path_v2: str,
    code_v1: str,
    requested_resources: list[str] = None
) -> tuple[str, str, str, int, int]:
    """
    Critique the chart IMAGE and the original code against the instruction, 
    then return refined matplotlib code.
    Returns (critique, feedback, refined_code).
    Works with Gemini vision-capable models (e.g. gemini-2.5-flash).
    """

    mcp_prompt_args = {
        "code_v1": code_v1,
        "out_path_v2": out_path_v2,
        "instruction": instruction
    }

    # Send the chart image + prompt to the reflection model and get structured response
    parsed, in_tok, out_tok = await utils.image_gemini_call(
        model_name=model_name,
        mcp_prompt_name="reflection_prompt",
        mcp_prompt_args=mcp_prompt_args,
        image_path=chart_path, 
        response_schema=ReflectionResponse,
        requested_resources=requested_resources
    )

    full_critique = f"Reasoning Type: {parsed.reasoning_type}\n"
    full_critique += f"Plan:\n" + "\n".join([f"  - {s}" for s in parsed.step_by_step_plan]) + "\n"
    full_critique += f"Self Check: {parsed.internal_self_check}\n"
    full_critique += f"Critique: {parsed.critique}"

    return full_critique, parsed.feedback, parsed.refined_code, in_tok, out_tok


async def fix_chart_code(instruction: str, bad_code: str, error_message: str, model: str) -> tuple[str, int, int]:
    """Ask the model to fix Python code that produced an error."""
    
    mcp_prompt_args = {
        "instruction": instruction, 
        "bad_code": bad_code, 
        "error_message": error_message
    }

    parsed, in_tok, out_tok = await utils.run_agent_loop(
        model=model, 
        mcp_prompt_name="error_fix_prompt",
        mcp_prompt_args=mcp_prompt_args,
        response_schema=ChartCodeResponse
    )
    return parsed.python_code, in_tok, out_tok
