import json
import re
import os
import src.utils as utils
from pydantic import BaseModel, Field, model_validator

class CandidatePlan(BaseModel):
    approach_name: str = Field(default="Default Plan", description="A short name for this approach (e.g., 'Bar Chart', 'Line Chart', 'Aggregated Table').")
    steps: list[str] = Field(default_factory=list, description="The steps required to execute this approach.")
    pros: list[str] = Field(default_factory=list, description="Advantages of this approach.")
    cons: list[str] = Field(default_factory=list, description="Disadvantages or risks of this approach.")

class CodeDecomposition(BaseModel):
    step_name: str = Field(default="Execution Step", description="Name of the execution step (e.g., 'Data Loading', 'Aggregation', 'Plotting').")
    code_snippet: str = Field(default="", description="The isolated Python code for this specific step.")

class ChartCodeResponse(BaseModel):
    reasoning_type: str = Field(default="code_generation", description="The type of reasoning used (e.g., 'data_exploration', 'code_generation', 'error_correction')")
    candidate_plans: list[CandidatePlan] = Field(default_factory=list, description="Provide at least 2 or 3 completely different approaches to solve the user's request.")
    chosen_plan_name: str = Field(default="Default Plan", description="The approach_name of the best plan from your candidates.")
    internal_self_check: str = Field(default="Verified constraints.", description="A validation check evaluating whether the chosen plan meets all hard constraints (e.g., using plt.close(), correct data aggregation).")
    thought_process: str = Field(default="", description="Explanation of why the chosen plan is better than the alternatives.")
    decomposed_code_steps: list[CodeDecomposition] = Field(default_factory=list, description="The final code broken down into distinct, sequential execution blocks.")

    @model_validator(mode="before")
    @classmethod
    def _coerce_code_steps(cls, data):
        if isinstance(data, dict) and not data.get("decomposed_code_steps"):
            for alt_key in ("python_code", "code", "final_code", "script"):
                if isinstance(data.get(alt_key), str) and data[alt_key].strip():
                    data["decomposed_code_steps"] = [
                        {"step_name": "Full Visualization Script", "code_snippet": data[alt_key]}
                    ]
                    break
        return data

    @model_validator(mode="after")
    def _validate_non_empty_code(self):
        combined = "\n".join(s.code_snippet for s in self.decomposed_code_steps if s.code_snippet).strip()
        if not combined or "savefig" not in combined:
            raise ValueError(
                "decomposed_code_steps must contain complete Python plotting code including plt.savefig(...) and plt.close()."
            )
        return self

class ReflectionResponse(BaseModel):
    reasoning_type: str = Field(default="visual_evaluation", description="The type of reasoning used (e.g., 'visual_evaluation', 'code_critique')")
    step_by_step_plan: list[str] = Field(default_factory=list, description="A sequential breakdown of the required fixes based on the critique.")
    internal_self_check: str = Field(default="Verified constraints.", description="A validation check ensuring the critique accurately reflects the chart image and original code.")
    critique: str = Field(default="")
    feedback: str = Field(default="")
    refined_code: str = Field(default="")

    @model_validator(mode="after")
    def _validate_refined_code(self):
        if not self.refined_code.strip() or "savefig" not in self.refined_code:
            raise ValueError(
                "refined_code must contain the complete Python matplotlib script including plt.savefig(...) and plt.close()."
            )
        return self

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
    for plan in parsed.candidate_plans:
        full_thought += f"\nCandidate: {plan.approach_name}\n"
        full_thought += f"  Pros: {', '.join(plan.pros)}\n"
        full_thought += f"  Cons: {', '.join(plan.cons)}\n"
    full_thought += f"\nChosen Plan: {parsed.chosen_plan_name}\n"
    full_thought += f"Self Check: {parsed.internal_self_check}\n"
    full_thought += f"Thought Process: {parsed.thought_process}"

    final_python_code = "\n\n".join([
        f"# --- STEP: {step.step_name} ---\n{utils.sanitize_python_code(step.code_snippet)}" 
        for step in parsed.decomposed_code_steps
    ])

    return full_thought, utils.sanitize_python_code(final_python_code), in_tok, out_tok


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

    return full_critique, parsed.feedback, utils.sanitize_python_code(parsed.refined_code), in_tok, out_tok


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
    
    final_python_code = "\n\n".join([
        f"# --- STEP: {step.step_name} ---\n{utils.sanitize_python_code(step.code_snippet)}" 
        for step in parsed.decomposed_code_steps
    ])
    
    return utils.sanitize_python_code(final_python_code), in_tok, out_tok
