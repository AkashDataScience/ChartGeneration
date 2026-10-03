import src.utils as utils
from src.agents.chart_agent import generate_chart_code, reflect_on_image_and_regenerate, fix_chart_code
from src.tools.code_executor import execute_chart_code

async def execute_with_retries(code: str, df, instruction: str, model: str, max_retries: int = 3) -> tuple[str, int, int]:
    current_code = code
    attempts = 0
    total_in = 0
    total_out = 0
    while attempts < max_retries:
        success, error_msg = execute_chart_code(current_code, df)
        if success:
            return current_code, total_in, total_out
        
        utils.print_html(f"Execution failed with error. Retrying ({attempts+1}/{max_retries})...\nError Snippet: {error_msg[-200:]}")
        current_code, in_tok, out_tok = await fix_chart_code(instruction, current_code, error_msg, model, df)
        total_in += in_tok
        total_out += out_tok
        attempts += 1
    
    # Try one last time; if it fails, throw an exception
    success, error_msg = execute_chart_code(current_code, df)
    if not success:
        raise RuntimeError(f"Failed to execute code after {max_retries} retries.\nLast error: {error_msg}")
    return current_code, total_in, total_out

async def run_workflow(
    dataset_path: str,
    user_instructions: str,
    generation_model: str,
    reflection_model: str,   
    image_basename: str = "chart",
    previous_code: str = "",
):
    """
    End-to-end pipeline:
      1) load dataset
      2) generate V1 code
      3) execute V1 → produce chart_v1.png
      4) reflect on V1 (image + original code) → feedback + refined code
      5) execute V2 → produce chart_v2.png

    Returns a dict with all artifacts (codes, feedback, image paths).
    """
    # 0) Load dataset
    df = utils.load_and_prepare_data(dataset_path)
    utils.print_html(df.sample(n=5), title="Random Sample of Dataset")

    # Paths to store charts
    out_v1 = f"{image_basename}_v1.png"
    out_v2 = f"{image_basename}_v2.png"

    # Token Trackers
    total_input_tokens = 0
    total_output_tokens = 0

    # 1) Generate code (V1)
    utils.print_html("Step 1: Generating chart code (V1)… 📈")
    thought_v1, code_v1, in_tok, out_tok = await generate_chart_code(
        instruction=user_instructions,
        model=generation_model,
        out_path_v1=out_v1,
        df=df,
        previous_code=previous_code,
    )
    total_input_tokens += in_tok
    total_output_tokens += out_tok
    utils.print_html(thought_v1, title="Thought Process (V1)")
    utils.print_html(code_v1, title="LLM output with first draft code (V1)")

    # 2) Execute V1 with ReAct Loop
    utils.print_html("Step 2: Executing chart code (V1) with ReAct Error Correction… 💻")
    code_v1, in_tok, out_tok = await execute_with_retries(code_v1, df, user_instructions, generation_model)
    total_input_tokens += in_tok
    total_output_tokens += out_tok
    utils.print_html(code_v1, title="Final successful code (V1)")
    utils.print_html(out_v1, is_image=True, title="Generated Chart (V1)")

    # 3) Reflect on V1 (image + original code) to get feedback and refined code (V2)
    utils.print_html("Step 3: Reflecting on V1 (image + code) and generating improvements… 🔁")
    critique, feedback, code_v2, in_tok, out_tok = await reflect_on_image_and_regenerate(
        chart_path=out_v1,
        instruction=user_instructions,
        model_name=reflection_model,
        out_path_v2=out_v2,
        code_v1=code_v1,  
        df=df,
    )
    total_input_tokens += in_tok
    total_output_tokens += out_tok
    utils.print_html(critique, title="Reflection Critique on V1")
    utils.print_html(feedback, title="Reflection feedback on V1")
    utils.print_html(code_v2, title="LLM output with revised code (V2)")

    # 4) Execute V2 with ReAct Loop
    utils.print_html("Step 4: Executing refined chart code (V2) with ReAct Error Correction… 🖼️")
    code_v2, in_tok, out_tok = await execute_with_retries(code_v2, df, user_instructions, reflection_model)
    total_input_tokens += in_tok
    total_output_tokens += out_tok
    utils.print_html(code_v2, title="Final successful code (V2)")
    utils.print_html(out_v2, is_image=True, title="Regenerated Chart (V2)")
    
    utils.print_html(f"Total Input Tokens: {total_input_tokens:,}\nTotal Output Tokens: {total_output_tokens:,}", title="Token Usage Report")

    return {
        "thought_v1": thought_v1,
        "code_v1": code_v1,
        "chart_v1": out_v1,
        "critique": critique,
        "feedback": feedback,
        "code_v2": code_v2,
        "chart_v2": out_v2,
    }
