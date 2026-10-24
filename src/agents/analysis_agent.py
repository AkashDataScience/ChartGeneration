import json
import src.utils as utils
from pydantic import BaseModel, Field, model_validator


class DataQualityIssue(BaseModel):
    column: str = Field(default="dataset", description="Column name (or 'dataset' if table-wide like duplicate rows).")
    issue_type: str = Field(default="general", description="Type of issue (e.g., 'missing_values', 'date_parsing', 'outliers', 'duplicates').")
    description: str = Field(default="", description="Detailed findings from profiling (exact counts/percentages and root cause).")
    recommended_cleaning_step: str = Field(default="", description="How downstream analysis and plotting scripts should handle this.")


class ChartTask(BaseModel):
    chart_id: str = Field(default="chart_1", description="Short snake_case identifier for file naming (e.g., 'monthly_revenue_trend').")
    title: str = Field(default="Data Visualization", description="Human-readable chart title.")
    instruction: str = Field(
        default="",
        description=(
            "Detailed matplotlib plotting instruction referencing the exact dataset path, "
            "columns, aggregations, and any required cleaning/date-parsing rules."
        )
    )
    rationale: str = Field(default="", description="Why this visualization is important for understanding the dataset.")


class AnalysisPlanResponse(BaseModel):
    dataset_summary: str = Field(default="", description="Overview of dataset dimensions, domain, date range, and structure.")
    data_quality_issues: list[DataQualityIssue] = Field(default_factory=list, description="List of data quality, missing value, outlier, or type findings.")
    cleaning_instructions_summary: str = Field(
        default="",
        description="Concise summary of mandatory Pandas preprocessing steps to include in all chart generation prompts."
    )
    analytical_questions: list[str] = Field(default_factory=list, description="3 to 4 substantive analytical questions/hypotheses to test.")
    chart_tasks: list[ChartTask] = Field(default_factory=list, description="Suite of complementary visualizations to generate.")


class HypothesisFinding(BaseModel):
    question: str = Field(default="", description="The analytical question or hypothesis tested.")
    methodology: str = Field(default="", description="How the hypothesis was tested using Pandas/statistical queries.")
    quantitative_evidence: str = Field(default="", description="Exact numbers, percentages, averages, or test statistics computed.")
    markdown_table: str = Field(default="", description="A clean GitHub-flavored Markdown table summarizing the key figures.")
    conclusion: str = Field(default="", description="Analytical interpretation and takeaway.")


class StatisticalFindingsResponse(BaseModel):
    findings: list[HypothesisFinding] = Field(default_factory=list, description="Quantitative findings for each analytical question.")
    key_takeaways: list[str] = Field(default_factory=list, description="Top high-level quantitative takeaways across the dataset.")

    @model_validator(mode="after")
    def _populate_missing_takeaways(self):
        if not self.key_takeaways and self.findings:
            self.key_takeaways = [f.conclusion for f in self.findings if f.conclusion]
        return self


class ReportSynthesisResponse(BaseModel):
    executive_summary: str = Field(
        default="",
        description="Concise terminal-friendly executive summary highlighting key metrics and recommendations."
    )
    full_markdown_report: str = Field(
        default="",
        description="Complete publication-ready Markdown report including tables, embedded chart links, and insights."
    )

    @model_validator(mode="after")
    def _ensure_both_fields(self):
        if not self.executive_summary and self.full_markdown_report:
            self.executive_summary = self.full_markdown_report[:1200]
        elif not self.full_markdown_report and self.executive_summary:
            self.full_markdown_report = f"# Data Analysis Report\n\n{self.executive_summary}"
        return self


class FollowUpResponse(BaseModel):
    is_chart_request: bool = Field(
        default=False,
        description="True if the user is asking to create or modify a chart/plot; False if asking a data question."
    )
    target_chart_id: str = Field(
        default="",
        description="If modifying an existing chart from the report, its chart_id; otherwise empty string."
    )
    chart_instruction: str = Field(
        default="",
        description="If is_chart_request is True, the complete standalone instruction for generating/modifying the chart."
    )
    answer: str = Field(
        default="",
        description="If is_chart_request is False, the detailed data-backed answer; if True, a brief confirmation."
    )


async def plan_data_analysis(
    dataset_path: str,
    user_goal: str,
    model: str,
    max_charts: int = 3
) -> tuple[AnalysisPlanResponse, int, int]:
    """Stage 1: Profile dataset, audit data quality, and create a structured analysis & visualization plan."""
    mcp_prompt_args = {
        "dataset_path": dataset_path,
        "user_goal": user_goal or "Perform a comprehensive exploratory and business data analysis.",
        "max_charts": str(max_charts)
    }

    parsed, in_tok, out_tok = await utils.run_agent_loop(
        model=model,
        mcp_prompt_name="data_profiling_and_plan_prompt",
        mcp_prompt_args=mcp_prompt_args,
        response_schema=AnalysisPlanResponse,
        requested_resources=[dataset_path]
    )
    return parsed, in_tok, out_tok


async def run_statistical_investigation(
    dataset_path: str,
    user_goal: str,
    analysis_plan: AnalysisPlanResponse,
    model: str
) -> tuple[StatisticalFindingsResponse, int, int]:
    """Stage 2: Execute statistical queries to test every analytical question in the plan."""
    mcp_prompt_args = {
        "dataset_path": dataset_path,
        "user_goal": user_goal or "Perform a comprehensive exploratory and business data analysis.",
        "analysis_plan_json": analysis_plan.model_dump_json(indent=2)
    }

    parsed, in_tok, out_tok = await utils.run_agent_loop(
        model=model,
        mcp_prompt_name="statistical_analysis_prompt",
        mcp_prompt_args=mcp_prompt_args,
        response_schema=StatisticalFindingsResponse,
        requested_resources=[dataset_path]
    )
    return parsed, in_tok, out_tok


async def synthesize_markdown_report(
    dataset_path: str,
    user_goal: str,
    analysis_plan: AnalysisPlanResponse,
    stat_findings: StatisticalFindingsResponse,
    chart_results: list[dict],
    model: str
) -> tuple[ReportSynthesisResponse, int, int]:
    """Stage 4: Synthesize all profiling, statistical findings, and visual insights into a Markdown report."""
    charts_summary = []
    for item in chart_results:
        charts_summary.append({
            "chart_id": item["chart_id"],
            "title": item["title"],
            "rationale": item["rationale"],
            "relative_chart_path": item["relative_chart_path"],
            "reflection_critique": item.get("critique", ""),
            "reflection_feedback": item.get("feedback", "")
        })

    mcp_prompt_args = {
        "dataset_path": dataset_path,
        "user_goal": user_goal or "Perform a comprehensive exploratory and business data analysis.",
        "analysis_plan_json": analysis_plan.model_dump_json(indent=2),
        "stat_findings_json": stat_findings.model_dump_json(indent=2),
        "charts_summary_json": json.dumps(charts_summary, indent=2)
    }

    parsed, in_tok, out_tok = await utils.run_agent_loop(
        model=model,
        mcp_prompt_name="report_synthesis_prompt",
        mcp_prompt_args=mcp_prompt_args,
        response_schema=ReportSynthesisResponse
    )
    return parsed, in_tok, out_tok


async def handle_followup_request(
    dataset_path: str,
    question: str,
    analysis_context: str,
    model: str
) -> tuple[FollowUpResponse, int, int]:
    """Stage 5: Answer follow-up analytical questions or route chart generation/refinement requests."""
    mcp_prompt_args = {
        "dataset_path": dataset_path,
        "question": question,
        "analysis_context": analysis_context
    }

    parsed, in_tok, out_tok = await utils.run_agent_loop(
        model=model,
        mcp_prompt_name="followup_qa_prompt",
        mcp_prompt_args=mcp_prompt_args,
        response_schema=FollowUpResponse,
        requested_resources=[dataset_path]
    )
    return parsed, in_tok, out_tok
