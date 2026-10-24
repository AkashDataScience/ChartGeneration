import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import src.utils as utils
from src.agents.chart_agent import (
    generate_chart_code,
    reflect_on_image_and_regenerate,
    fix_chart_code,
)
from src.agents.analysis_agent import (
    plan_data_analysis,
    run_statistical_investigation,
    synthesize_markdown_report,
)
from src.tools.code_executor import execute_chart_code


async def execute_with_retries(
    code: str,
    instruction: str,
    model: str,
    max_retries: int = 3,
    expected_output_path: str | None = None
) -> tuple[str, int, int]:
    current_code = code
    attempts = 0
    total_in = 0
    total_out = 0
    while attempts < max_retries:
        success, error_msg = execute_chart_code(current_code, expected_output_path=expected_output_path)
        if success:
            return current_code, total_in, total_out

        utils.print_html(
            f"Execution failed with error. Retrying ({attempts+1}/{max_retries})...\n"
            f"Error Snippet: {error_msg[-250:]}"
        )
        current_code, in_tok, out_tok = await fix_chart_code(
            instruction, current_code, error_msg, model
        )
        total_in += in_tok
        total_out += out_tok
        attempts += 1

    # Try one last time; if it fails, throw an exception
    success, error_msg = execute_chart_code(current_code, expected_output_path=expected_output_path)
    if not success:
        raise RuntimeError(
            f"Failed to execute code after {max_retries} retries.\nLast error: {error_msg}"
        )
    return current_code, total_in, total_out


async def run_workflow(
    user_instructions: str,
    generation_model: str,
    reflection_model: str,
    image_basename: str = "chart",
    previous_code: str = "",
    requested_resources: list[str] = None,
    output_dir: str = "."
):
    """
    Single-chart generation and reflection pipeline:
      1) generate V1 code
      2) execute V1 -> produce <image_basename>_v1.png
      3) reflect on V1 (image + original code) -> feedback + refined code
      4) execute V2 -> produce <image_basename>_v2.png

    Returns a dict with all artifacts (codes, feedback, image paths, token counts).
    """
    requested_resources = requested_resources or []

    # Ensure output directory exists and use POSIX paths to avoid backslash escape issues in generated code
    if output_dir and output_dir != ".":
        os.makedirs(output_dir, exist_ok=True)
        out_v1 = Path(output_dir, f"{image_basename}_v1.png").as_posix()
        out_v2 = Path(output_dir, f"{image_basename}_v2.png").as_posix()
    else:
        out_v1 = f"{image_basename}_v1.png"
        out_v2 = f"{image_basename}_v2.png"

    # Token Trackers
    total_input_tokens = 0
    total_output_tokens = 0

    # 1) Generate code (V1)
    utils.print_html(f"Step 1: Generating chart code (V1) for '{image_basename}'… 📈")
    thought_v1, code_v1, in_tok, out_tok = await generate_chart_code(
        instruction=user_instructions,
        model=generation_model,
        out_path_v1=out_v1,
        previous_code=previous_code,
        requested_resources=requested_resources
    )
    total_input_tokens += in_tok
    total_output_tokens += out_tok
    utils.print_html(thought_v1, title=f"Thought Process (V1 - {image_basename})")
    utils.print_html(code_v1, title=f"LLM output with first draft code (V1 - {image_basename})")

    # 2) Execute V1 with ReAct Loop
    utils.print_html(f"Step 2: Executing chart code (V1 - {image_basename}) with ReAct Error Correction… 💻")
    code_v1, in_tok, out_tok = await execute_with_retries(
        code_v1, user_instructions, generation_model, expected_output_path=out_v1
    )
    total_input_tokens += in_tok
    total_output_tokens += out_tok
    utils.print_html(code_v1, title=f"Final successful code (V1 - {image_basename})")
    utils.print_html(out_v1, is_image=True, title=f"Generated Chart (V1 - {image_basename})")

    # 3) Reflect on V1 (image + original code) to get feedback and refined code (V2)
    critique = ""
    feedback = ""
    code_v2 = code_v1
    try:
        utils.print_html(f"Step 3: Reflecting on V1 (image + code) for '{image_basename}'… 🔁")
        critique, feedback, code_v2, in_tok, out_tok = await reflect_on_image_and_regenerate(
            chart_path=out_v1,
            instruction=user_instructions,
            model_name=reflection_model,
            out_path_v2=out_v2,
            code_v1=code_v1,
            requested_resources=requested_resources
        )
        total_input_tokens += in_tok
        total_output_tokens += out_tok
        utils.print_html(critique, title=f"Reflection Critique on V1 ({image_basename})")
        utils.print_html(feedback, title=f"Reflection Feedback on V1 ({image_basename})")
        utils.print_html(code_v2, title=f"LLM output with revised code (V2 - {image_basename})")

        # 4) Execute V2 with ReAct Loop
        utils.print_html(f"Step 4: Executing refined chart code (V2 - {image_basename}) with ReAct Error Correction… 🖼️")
        code_v2, in_tok, out_tok = await execute_with_retries(
            code_v2, user_instructions, reflection_model, expected_output_path=out_v2
        )
        total_input_tokens += in_tok
        total_output_tokens += out_tok
        utils.print_html(code_v2, title=f"Final successful code (V2 - {image_basename})")
        utils.print_html(out_v2, is_image=True, title=f"Regenerated Chart (V2 - {image_basename})")
    except Exception as v2_err:
        utils.print_html(
            f"V2 refinement encountered an error ({str(v2_err)[:150]}); retaining verified V1 chart as final output.",
            title=f"Fallback to Verified V1 Chart ({image_basename})"
        )
        if os.path.exists(out_v1) and not os.path.exists(out_v2):
            shutil.copyfile(out_v1, out_v2)
        code_v2 = code_v1

    utils.print_html(
        f"Total Input Tokens: {total_input_tokens:,}\nTotal Output Tokens: {total_output_tokens:,}",
        title=f"Token Usage Report ({image_basename})"
    )

    return {
        "thought_v1": thought_v1,
        "code_v1": code_v1,
        "chart_v1": out_v1,
        "critique": critique,
        "feedback": feedback,
        "code_v2": code_v2,
        "chart_v2": out_v2,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }


def _sanitize_slug(name: str, fallback: str = "chart") -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower()).strip("_")
    return slug or fallback


async def run_full_analysis_workflow(
    dataset_path: str,
    user_goal: str = "",
    generation_model: str = "gemini-4-argon",
    reflection_model: str = "gemini-4-argon",
    max_charts: int = 3,
    reports_root: str = "reports"
) -> dict:
    """
    Autonomous End-to-End Data Analysis Pipeline:
      Stage 1: Automated Data Profiling, Quality Audit & Analysis Planning
      Stage 2: Quantitative & Statistical Exploration (Hypothesis Testing)
      Stage 3: Multi-Chart Visual Exploration with Visual Reflection (V1 -> V2)
      Stage 4: Executive Report Synthesis (saves report.md + terminal summary)
    """
    dataset_posix = Path(dataset_path).as_posix()
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at path: {dataset_path}")

    dataset_stem = _sanitize_slug(Path(dataset_path).stem, fallback="dataset")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(reports_root, f"{dataset_stem}_{timestamp}")
    charts_dir = report_dir / "charts"
    os.makedirs(charts_dir, exist_ok=True)

    total_input_tokens = 0
    total_output_tokens = 0

    # =========================================================================
    # STAGE 1: Automated Data Profiling, Quality Check & Analysis Planning
    # =========================================================================
    utils.print_html(
        f"Dataset: {dataset_posix}\nGoal: {user_goal or 'Comprehensive Exploratory & Business Analysis'}\n"
        f"Output Directory: {report_dir.as_posix()}",
        title="STAGE 1: Automated Data Profiling & Analysis Planning 🔍"
    )
    analysis_plan, in_tok, out_tok = await plan_data_analysis(
        dataset_path=dataset_posix,
        user_goal=user_goal,
        model=generation_model,
        max_charts=max_charts
    )
    total_input_tokens += in_tok
    total_output_tokens += out_tok

    plan_summary_lines = [
        f"Dataset Summary:\n  {analysis_plan.dataset_summary}\n",
        "Data Quality & Cleaning Audit:"
    ]
    for issue in analysis_plan.data_quality_issues:
        plan_summary_lines.append(
            f"  - [{issue.column} | {issue.issue_type}] {issue.description}\n"
            f"    -> Action: {issue.recommended_cleaning_step}"
        )
    plan_summary_lines.append(f"\nCleaning Rules Summary:\n  {analysis_plan.cleaning_instructions_summary}\n")
    plan_summary_lines.append("Analytical Questions / Hypotheses:")
    for idx, q in enumerate(analysis_plan.analytical_questions, 1):
        plan_summary_lines.append(f"  {idx}. {q}")
    plan_summary_lines.append("\nPlanned Visualizations:")
    for idx, ct in enumerate(analysis_plan.chart_tasks[:max_charts], 1):
        plan_summary_lines.append(f"  {idx}. [{ct.chart_id}] {ct.title} — {ct.rationale}")

    utils.print_html("\n".join(plan_summary_lines), title="Stage 1 Output: Data Quality Audit & Analysis Plan")

    # =========================================================================
    # STAGE 2: Quantitative & Statistical Exploration
    # =========================================================================
    utils.print_html(
        "Testing analytical hypotheses quantitatively using MCP statistical tools… 📊",
        title="STAGE 2: Statistical Deep-Dive & Hypothesis Testing"
    )
    stat_findings, in_tok, out_tok = await run_statistical_investigation(
        dataset_path=dataset_posix,
        user_goal=user_goal,
        analysis_plan=analysis_plan,
        model=generation_model
    )
    total_input_tokens += in_tok
    total_output_tokens += out_tok

    stat_lines = []
    for idx, f in enumerate(stat_findings.findings, 1):
        stat_lines.append(f"Hypothesis {idx}: {f.question}")
        stat_lines.append(f"Methodology: {f.methodology}")
        stat_lines.append(f"Quantitative Evidence: {f.quantitative_evidence}")
        stat_lines.append(f"Table:\n{f.markdown_table}")
        stat_lines.append(f"Conclusion: {f.conclusion}\n" + ("-" * 30))
    stat_lines.append("Key Takeaways:")
    for kt in stat_findings.key_takeaways:
        stat_lines.append(f"  * {kt}")

    utils.print_html("\n".join(stat_lines), title="Stage 2 Output: Quantitative Findings")

    # =========================================================================
    # STAGE 3: Multi-Chart Visual Exploration with Visual Reflection
    # =========================================================================
    utils.print_html(
        f"Generating and reflecting on {min(len(analysis_plan.chart_tasks), max_charts)} charts… 🎨",
        title="STAGE 3: Multi-Chart Generation & Visual Reflection"
    )
    chart_results = []
    for idx, task in enumerate(analysis_plan.chart_tasks[:max_charts], 1):
        safe_id = _sanitize_slug(task.chart_id, fallback=f"chart_{idx}")
        enriched_instruction = (
            f"Dataset path: '{dataset_posix}'.\n"
            f"Chart Title: {task.title}\n"
            f"Task Instruction: {task.instruction}\n"
            f"Mandatory Data Cleaning / Preprocessing Notes: {analysis_plan.cleaning_instructions_summary}"
        )

        utils.print_html(
            f"Chart {idx}/{min(len(analysis_plan.chart_tasks), max_charts)}: {task.title} ({safe_id})",
            title=f"Running Chart Workflow [{safe_id}]"
        )
        try:
            wf_res = await run_workflow(
                user_instructions=enriched_instruction,
                generation_model=generation_model,
                reflection_model=reflection_model,
                image_basename=safe_id,
                previous_code="",
                requested_resources=[dataset_posix],
                output_dir=charts_dir.as_posix()
            )
            total_input_tokens += wf_res["total_input_tokens"]
            total_output_tokens += wf_res["total_output_tokens"]

            chart_results.append({
                "chart_id": safe_id,
                "title": task.title,
                "instruction": enriched_instruction,
                "rationale": task.rationale,
                "chart_v1": wf_res["chart_v1"],
                "chart_v2": wf_res["chart_v2"],
                "relative_chart_path": f"charts/{safe_id}_v2.png",
                "code_v1": wf_res["code_v1"],
                "code_v2": wf_res["code_v2"],
                "critique": wf_res["critique"],
                "feedback": wf_res["feedback"],
            })
        except Exception as chart_err:
            utils.print_html(
                f"Warning: Chart '{safe_id}' failed after retries and will be skipped so the report can still be generated.\n"
                f"Reason: {str(chart_err)[:300]}",
                title=f"Chart Skipped [{safe_id}] ⚠️"
            )

    # =========================================================================
    # STAGE 4: Executive Report Synthesis
    # =========================================================================
    utils.print_html(
        "Synthesizing Data Profile, Quantitative Findings, and Reflected Visualizations into report.md… 📝",
        title="STAGE 4: Executive Report Synthesis"
    )
    report_response, in_tok, out_tok = await synthesize_markdown_report(
        dataset_path=dataset_posix,
        user_goal=user_goal,
        analysis_plan=analysis_plan,
        stat_findings=stat_findings,
        chart_results=chart_results,
        model=generation_model
    )
    total_input_tokens += in_tok
    total_output_tokens += out_tok

    report_path = report_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_response.full_markdown_report)

    utils.print_html(report_response.executive_summary, title="EXECUTIVE SUMMARY")
    utils.print_html(
        f"Full Markdown Report saved to: {report_path.as_posix()}\n"
        f"Charts Directory: {charts_dir.as_posix()}\n"
        f"Total Pipeline Input Tokens: {total_input_tokens:,}\n"
        f"Total Pipeline Output Tokens: {total_output_tokens:,}",
        title="End-to-End Analysis Complete ✅"
    )

    return {
        "dataset_path": dataset_posix,
        "user_goal": user_goal,
        "report_dir": report_dir.as_posix(),
        "report_path": report_path.as_posix(),
        "charts_dir": charts_dir.as_posix(),
        "analysis_plan": analysis_plan,
        "stat_findings": stat_findings,
        "chart_results": chart_results,
        "executive_summary": report_response.executive_summary,
        "full_markdown_report": report_response.full_markdown_report,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }

