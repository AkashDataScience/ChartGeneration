import asyncio
import os
import re
from pathlib import Path

import src.utils as utils
from src.core.workflow import run_full_analysis_workflow, run_workflow
from src.agents.analysis_agent import handle_followup_request


def parse_dataset_and_goal(raw_input: str, default_dataset: str = "data/coffee_sales.csv") -> tuple[str, str]:
    """
    Extract dataset path (from @mention or direct file path) and optional user goal from user input.
    """
    text = raw_input.strip()
    if not text:
        return default_dataset, ""

    # 1. Check for @path/to/file.csv mention
    tokens = text.split()
    for token in tokens:
        if token.startswith("@"):
            candidate = token[1:].strip(",;\"'")
            goal = text.replace(token, "").strip()
            return candidate, goal

    # 2. Check if first token or whole string is an existing file path
    if os.path.exists(text) and os.path.isfile(text):
        return text, ""

    first_token = tokens[0].strip(",;\"'")
    if os.path.exists(first_token) and os.path.isfile(first_token):
        goal = " ".join(tokens[1:]).strip()
        return first_token, goal

    # 3. Fallback: use default dataset and treat the entire input as the analysis goal
    return default_dataset, text


def build_followup_context(analysis_result: dict) -> str:
    """Build a concise summary of the completed analysis for follow-up Q&A and chart modifications."""
    plan = analysis_result["analysis_plan"]
    findings = analysis_result["stat_findings"]
    charts = analysis_result["chart_results"]

    chart_list_str = "\n".join(
        f"- chart_id='{c['chart_id']}': {c['title']} (saved at {c['chart_v2']})"
        for c in charts
    )
    takeaways_str = "\n".join(f"- {t}" for t in findings.key_takeaways)

    return (
        f"Dataset Summary: {plan.dataset_summary}\n"
        f"Cleaning Rules: {plan.cleaning_instructions_summary}\n"
        f"Key Takeaways:\n{takeaways_str}\n"
        f"Generated Charts:\n{chart_list_str}\n"
        f"Executive Summary:\n{analysis_result['executive_summary']}"
    )


async def main():
    generation_model = "gemini-4-argon"
    reflection_model = "gemini-4-argon"
    default_dataset = "data/coffee_sales.csv"

    print("=" * 65)
    print("  Welcome to the Autonomous End-to-End Data Analysis Agent!")
    print("  Type 'exit' or 'quit' at any prompt to stop.")
    print("=" * 65)

    initial_input = input(
        f"\nEnter dataset path and optional goal (press Enter for '{default_dataset}'):\n> "
    )
    if initial_input.strip().lower() in ["exit", "quit"]:
        return

    dataset_path, user_goal = parse_dataset_and_goal(initial_input, default_dataset=default_dataset)

    if not user_goal:
        optional_goal = input(
            "Optional: Enter a specific analysis focus/goal (or press Enter for full autonomous EDA):\n> "
        ).strip()
        if optional_goal.lower() in ["exit", "quit"]:
            return
        user_goal = optional_goal

    print(f"\nStarting End-to-End Data Analysis on '{dataset_path}'...")
    analysis_state = await run_full_analysis_workflow(
        dataset_path=dataset_path,
        user_goal=user_goal,
        generation_model=generation_model,
        reflection_model=reflection_model,
        max_charts=3
    )

    followup_counter = 1
    last_code = (
        analysis_state["chart_results"][-1]["code_v2"]
        if analysis_state["chart_results"]
        else ""
    )

    while True:
        chart_ids = [c["chart_id"] for c in analysis_state["chart_results"]]
        print("\n" + "=" * 65)
        print(f"Active Dataset: {analysis_state['dataset_path']}")
        print(f"Report Path:    {analysis_state['report_path']}")
        print(f"Chart IDs:      {', '.join(chart_ids)}")
        print("Options:")
        print("  - Ask any follow-up data question")
        print("  - Request a new chart or modify an existing chart_id")
        print("  - Type 'analyze @path/to/dataset.csv [goal]' to analyze a new dataset")
        print("  - Type 'exit' to quit")
        print("=" * 65)

        user_input = input("\nFollow-up question or chart instruction > ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            break

        # Check if user wants a fresh end-to-end analysis on a new dataset
        if user_input.lower().startswith("analyze "):
            rest = user_input[len("analyze "):].strip()
            new_ds, new_goal = parse_dataset_and_goal(rest, default_dataset=analysis_state["dataset_path"])
            print(f"\nStarting new End-to-End Data Analysis on '{new_ds}'...")
            analysis_state = await run_full_analysis_workflow(
                dataset_path=new_ds,
                user_goal=new_goal,
                generation_model=generation_model,
                reflection_model=reflection_model,
                max_charts=3
            )
            last_code = (
                analysis_state["chart_results"][-1]["code_v2"]
                if analysis_state["chart_results"]
                else ""
            )
            continue

        # Handle follow-up question or chart request
        context_str = build_followup_context(analysis_state)
        followup_res, in_tok, out_tok = await handle_followup_request(
            dataset_path=analysis_state["dataset_path"],
            question=user_input,
            analysis_context=context_str,
            model=generation_model
        )

        if not followup_res.is_chart_request:
            utils.print_html(followup_res.answer, title="Follow-Up Analysis Answer 💡")
        else:
            # Determine if modifying a specific chart or creating a new one
            target_id = followup_res.target_chart_id.strip()
            matched_chart = next(
                (c for c in analysis_state["chart_results"] if c["chart_id"] == target_id),
                None
            )
            prev_code = matched_chart["code_v2"] if matched_chart else ""
            if not prev_code and any(w in user_input.lower() for w in ["modify", "update", "change", "convert"]):
                prev_code = last_code

            basename = target_id if matched_chart else f"followup_chart_{followup_counter}"
            followup_counter += 1

            enriched_instruction = (
                f"Dataset path: '{analysis_state['dataset_path']}'.\n"
                f"User Instruction: {followup_res.chart_instruction or user_input}\n"
                f"Cleaning Notes: {analysis_state['analysis_plan'].cleaning_instructions_summary}"
            )

            wf_res = await run_workflow(
                user_instructions=enriched_instruction,
                generation_model=generation_model,
                reflection_model=reflection_model,
                image_basename=basename,
                previous_code=prev_code,
                requested_resources=[analysis_state["dataset_path"]],
                output_dir=analysis_state["charts_dir"]
            )
            last_code = wf_res.get("code_v2") or wf_res.get("code_v1")
            if matched_chart:
                matched_chart["code_v2"] = last_code
                matched_chart["chart_v2"] = wf_res["chart_v2"]
            else:
                analysis_state["chart_results"].append({
                    "chart_id": basename,
                    "title": f"Follow-up Chart ({basename})",
                    "instruction": enriched_instruction,
                    "rationale": user_input,
                    "chart_v1": wf_res["chart_v1"],
                    "chart_v2": wf_res["chart_v2"],
                    "relative_chart_path": f"charts/{basename}_v2.png",
                    "code_v1": wf_res["code_v1"],
                    "code_v2": last_code,
                    "critique": wf_res["critique"],
                    "feedback": wf_res["feedback"],
                })
            print(f"\nChart saved to: {wf_res['chart_v2']}")


if __name__ == "__main__":
    asyncio.run(main())
