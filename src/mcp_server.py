from mcp.server.mcpserver import MCPServer
import pandas as pd
import numpy as np
import math
import io
import sys

# Create the MCP Server instance
mcp = MCPServer("ChartAgent Tools Server")

@mcp.tool()
def get_dataset_columns(file_path: str) -> str:
    """Read a CSV dataset and return its columns."""
    try:
        df = pd.read_csv(file_path)
        return f"Columns in {file_path}: {list(df.columns)}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_dataset_sample(file_path: str, n: int = 5) -> str:
    """Get a random sample of rows from a dataset."""
    try:
        df = pd.read_csv(file_path)
        sample = df.sample(min(n, len(df)))
        return f"Sample from {file_path}:\n{sample.to_string()}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def profile_dataset(file_path: str) -> str:
    """
    Perform comprehensive automated data profiling on a CSV dataset.
    Returns shape, column data types, missing values, duplicates, numeric summary statistics,
    IQR outlier counts, categorical top values, and detected date ranges.
    """
    try:
        df = pd.read_csv(file_path)
        n_rows, n_cols = df.shape
        mem_kb = df.memory_usage(deep=True).sum() / 1024.0
        dup_count = int(df.duplicated().sum())

        lines = [
            f"=== DATA PROFILE: {file_path} ===",
            f"Shape: {n_rows:,} rows x {n_cols} columns",
            f"Memory Usage: {mem_kb:.1f} KB",
            f"Duplicate Rows: {dup_count:,} ({dup_count / max(n_rows, 1) * 100:.2f}%)",
            "\n--- COLUMNS, TYPES & MISSING VALUES ---"
        ]

        for col in df.columns:
            dtype = str(df[col].dtype)
            nulls = int(df[col].isna().sum())
            null_pct = (nulls / max(n_rows, 1)) * 100
            uniq = int(df[col].nunique(dropna=True))
            lines.append(
                f"- {col} (dtype: {dtype}): {nulls:,} missing ({null_pct:.1f}%), {uniq:,} unique values"
            )

        # Numeric columns summary + outlier detection
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            lines.append("\n--- NUMERIC COLUMNS SUMMARY & OUTLIERS (IQR Method) ---")
            desc = df[num_cols].describe().T
            for col in num_cols:
                s = df[col].dropna()
                if len(s) == 0:
                    continue
                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = q3 - q1
                outliers = int(((s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))).sum())
                skew = float(s.skew()) if len(s) > 2 else 0.0
                row = desc.loc[col]
                lines.append(
                    f"- {col}: mean={row['mean']:.3f}, std={row['std']:.3f}, "
                    f"min={row['min']:.3f}, 25%={q1:.3f}, median={row['50%']:.3f}, "
                    f"75%={q3:.3f}, max={row['max']:.3f}, skew={skew:.2f}, "
                    f"IQR_outliers={outliers:,}"
                )

        # Categorical / Object columns summary + date range detection
        cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        if cat_cols:
            lines.append("\n--- CATEGORICAL / TEXT COLUMNS & DATE DETECTION ---")
            for col in cat_cols:
                s = df[col].dropna()
                if len(s) == 0:
                    lines.append(f"- {col}: all values missing")
                    continue
                # Check if column looks like a date
                if any(k in col.lower() for k in ["date", "dt", "day", "timestamp"]):
                    parsed_dt = pd.to_datetime(s, errors="coerce")
                    valid_dt = parsed_dt.dropna()
                    if len(valid_dt) > 0.8 * len(s):
                        lines.append(
                            f"- {col} [DETECTED DATE]: range {valid_dt.min().strftime('%Y-%m-%d')} "
                            f"to {valid_dt.max().strftime('%Y-%m-%d')} "
                            f"({int(parsed_dt.isna().sum())} unparseable)"
                        )
                        continue

                # Check if column looks like a time string (e.g. HH:MM or HH:MM:SS)
                if "time" in col.lower():
                    sample_str = str(s.iloc[0]).strip()
                    colon_count = sample_str.count(":")
                    fmt_hint = "%H:%M" if colon_count == 1 else "%H:%M:%S"
                    lines.append(
                        f"- {col} [DETECTED TIME (sample='{sample_str}', format='{fmt_hint}')]: "
                        f"To extract hour safely, use `df['{col}'].astype(str).str.split(':').str[0].astype(int)` "
                        f"or `pd.to_datetime(df['{col}'], format='{fmt_hint}', errors='coerce').dt.hour` "
                        f"(do NOT use format='%H:%M:%S' if format is '%H:%M'!)"
                    )
                    continue

                top_vals = s.value_counts().head(5)
                top_str = ", ".join([f"'{k}' ({v:,})" for k, v in top_vals.items()])
                lines.append(f"- {col}: top values -> {top_str}")

        lines.append("\n--- SAMPLE ROWS (First 5) ---")
        lines.append(df.head(5).to_string())
        return "\n".join(lines)
    except Exception as e:
        return f"Error profiling dataset {file_path}: {str(e)}"

@mcp.tool()
def run_statistical_query(file_path: str, python_code: str) -> str:
    """
    Execute concise analytical Python code against a CSV dataset.
    `df` (already loaded via pd.read_csv(file_path)), `pd`, `np`, and `math` are pre-loaded in scope.
    Keep `python_code` short (2-5 lines), do NOT include `#` comments, and use `print(...)` to output results.
    """
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    try:
        df = pd.read_csv(file_path)
        local_env = {"df": df, "pd": pd, "np": np, "math": math, "__builtins__": __builtins__}
        exec(python_code, local_env)
        output = redirected_output.getvalue()
        return output if output.strip() else "Query executed successfully (use print() to display results)."
    except Exception as e:
        return f"Error running statistical query: {str(e)}"
    finally:
        sys.stdout = old_stdout

@mcp.tool()
def run_python_script(code: str) -> str:
    """Execute Python code and return its stdout output."""
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    try:
        exec(code, {})
        output = redirected_output.getvalue()
        return output if output else "Code executed successfully with no output."
    except Exception as e:
        return f"Error executing code: {str(e)}"
    finally:
        sys.stdout = old_stdout

@mcp.tool()
def verify_pandas_logic(file_path: str, pandas_expression: str) -> str:
    """
    Verify a pandas data transformation (e.g., aggregation, filtering).
    Provide the path to the CSV and an expression assuming the dataframe is named 'df'.
    Example: df.groupby('coffee_name')['price'].sum()
    """
    try:
        df = pd.read_csv(file_path)
        # Safely evaluate the pandas expression
        result = eval(pandas_expression, {"pd": pd, "df": df})
        
        output = f"Expression verified successfully!\n"
        output += f"Type: {type(result)}\n"
        
        if isinstance(result, (pd.DataFrame, pd.Series)):
            output += f"Shape: {result.shape}\n"
            output += f"Head:\n{result.head().to_string()}\n"
        else:
            output += f"Value: {result}\n"
            
        return output
    except Exception as e:
        return f"Error verifying pandas logic: {str(e)}"

@mcp.resource("file://{filepath}")
def read_local_file(filepath: str) -> str:
    """Read a local file and return its contents or a sample if it's large."""
    try:
        if filepath.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(filepath)
            return f"--- {filepath} (Columns: {list(df.columns)}) ---\n{df.sample(min(5, len(df))).to_string()}"
        with open(filepath, "r", encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {filepath}: {str(e)}"

@mcp.prompt()
def generation_prompt(instruction: str, out_path_v1: str, previous_code_section: str = "") -> str:
    return f"""You are a data visualization expert capable of multi-step reasoning.

The code should create a visualization based on the files requested in the context.

{previous_code_section}

User instruction: {instruction}

=== YOUR WORKFLOW & MULTI-TURN CONVERSATION LOOP ===
1. Separation of Tools: If you are unsure about the dataset structure, USE YOUR TOOLS first (e.g., get_dataset_columns, get_dataset_sample) to verify the data before writing code.
2. Iterative Planning: When you receive tool results, incorporate that new information to revise your plan. If a requested file or column is missing, use your tools to search for alternatives or handle the error gracefully.
3. Intermediate Verification: Use the `verify_pandas_logic` tool to test your Pandas aggregations or filters BEFORE generating the final visualization code. Ensure the data shape and values are exactly what you expect to plot.
4. Multi-Path Planning (Tree-of-Thoughts): Generate at least 2 different candidate approaches for how to process and visualize the data. Evaluate their pros and cons.
5. Path Selection: Choose the best path based on the user's exact instructions and data constraints.
6. Internal Self-Check: Before finalizing, verify:
   - Are you loading the requested files manually?
   - Are you closing the plots with plt.close() instead of plt.show()?
   - Are you selecting numeric columns explicitly during aggregations?
7. Error Handling: If a tool returns an error, adjust your plan and try again.
8. Explicit Path Decomposition: Do not write the final Python script as one massive block. Break it down into isolated `decomposed_code_steps` (e.g., Data Loading, Filtering/Aggregation, Plotting).

Requirements for the code:
1. Load the requested files manually using pandas (e.g., pd.read_csv('path')).
2. Use pandas, numpy, and matplotlib ONLY for plotting (do NOT import seaborn, and NEVER run pip install or subprocess). For heatmaps, use `ax.imshow(matrix.values, aspect='auto', cmap='viridis')` with `fig.colorbar(...)`.
3. Add clear title, axis labels, and legend if needed.
4. Save the figure as '{out_path_v1}' with dpi=300.
5. Do not call plt.show().
6. Close all plots with plt.close().
7. Add all necessary import python statements.
8. CRITICAL: when aggregating (sum, mean, count, etc.), always select the numeric column explicitly, e.g. df.groupby(['year', 'coffee_name'])['price'].sum().
9. CRITICAL TIME PARSING: When extracting hour from a time column like '06:14' (HH:MM), use `df['time'].astype(str).str.split(':').str[0].astype(int)` instead of `pd.to_datetime(..., format='%H:%M:%S')` which turns HH:MM values into NaT/empty arrays!
10. Inside each `code_snippet`, use normal multi-line Python code (do NOT double-escape newlines as literal `\\n` between statements).

Here is an example of the high-quality, aesthetic code you should aim to write:

EXAMPLE 1 (Bar Chart):
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/coffee_sales.csv')
df_agg = df[df['year'] == 2024].groupby('coffee_name')['price'].sum()
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(df_agg.index, df_agg.values, color='#4C72B0', edgecolor='none')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title("Coffee Sales 2024", fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('{out_path_v1}', dpi=300)
plt.close()
Once you have completed your tools and planning, output your final result.
"""


@mcp.prompt()
def reflection_prompt(instruction: str, out_path_v2: str, code_v1: str) -> str:
    return f"""You are a data visualization expert capable of multi-step visual reasoning.
Your task: critique the attached chart and the original code against the given instruction,
then return improved matplotlib code.

Original code (for context):
{code_v1}

=== YOUR WORKFLOW & MULTI-TURN CONVERSATION LOOP ===
1. Separation of Tools: If you need to verify dataset columns to fix a bug, USE YOUR TOOLS first (e.g., get_dataset_columns).
2. Iterative Planning: Incorporate tool results to dynamically revise your plan. If a requested file/column is missing, use tools to search for alternatives or handle the error gracefully.
3. Explicit Reasoning: Think step-by-step about what is visually wrong with the chart or logically wrong with the code. Label your reasoning_type (e.g., visual_evaluation, code_critique).
4. Structured Planning: Create a step-by-step plan detailing the fixes needed.
5. Internal Self-Check: Verify that your critique accurately reflects both the image provided and the constraints requested.
6. Error Fallbacks: If the image is unreadable or the code cannot be corrected with available context, output a robust try/except block or explain the limitation clearly in your feedback.

OUTPUT FORMAT (STRICT):
Once you have completed your tools and planning, output your final result.

HARD CONSTRAINTS:
- Use pandas/matplotlib only (no seaborn).
- Load any requested files manually using pandas (e.g., pd.read_csv('path')).
- Save to '{out_path_v2}' with dpi=300.
- Always call plt.close() at the end (no plt.show()).
- Include all necessary import statements.
- CRITICAL TYPE RULE: Always aggregate properly.

Instruction:
{instruction}
"""


@mcp.prompt()
def error_fix_prompt(instruction: str, bad_code: str, error_message: str) -> str:
    return f"""You are an expert Python data visualization debugger capable of multi-step diagnostic reasoning.
You previously wrote code to fulfill this instruction:
{instruction}

However, when running the code, it produced the following error:
{error_message}

Here is the bad code you wrote:
{bad_code}

=== YOUR WORKFLOW ===
1. Separation of Tools: If the error relates to a missing column or bad data type, use your tools (get_dataset_columns, get_dataset_sample) to inspect the data before guessing the fix.
2. Explicit Reasoning: Think step-by-step to diagnose the traceback. Label your reasoning_type as error_correction.
3. Structured Planning: Formulate a step-by-step plan to fix the error.
4. Internal Self-Check: Verify your proposed fix directly addresses the traceback without violating hard constraints (no plt.show(), must save figure, must include imports).
5. Error Fallbacks: If you are unsure, write a robust try/except block or check the data structure again.
6. Explicit Path Decomposition: Break the corrected code down into isolated `decomposed_code_steps` (e.g., Data Loading, Filtering, Plotting).
7. HARD CONSTRAINTS:
   - Use pandas, numpy, and matplotlib ONLY (do NOT use seaborn and NEVER run pip install or subprocess).
   - If an error says `zero-size array to reduction operation fmin` or empty DataFrame after date/time parsing, fix the time parsing (e.g. use `df['time'].astype(str).str.split(':').str[0].astype(int)` for `HH:MM` strings instead of `format='%H:%M:%S'`).
   - Do NOT put literal `\\n` line-continuation characters between statements on a single line.

Please fix the error and return the corrected Python code once your plan is ready.
"""


@mcp.prompt()
def data_profiling_and_plan_prompt(dataset_path: str, user_goal: str, max_charts: str = "3") -> str:
    return f"""You are a Lead Data Scientist conducting Stage 1 (Automated Data Profiling, Quality Audit, and Analysis Planning) of an end-to-end data analysis on `{dataset_path}`.

User Goal / Focus: {user_goal}

=== YOUR WORKFLOW ===
1. Call the `profile_dataset` tool on `{dataset_path}` to inspect row/column counts, data types, missing values, duplicates, numeric distributions, IQR outliers, categorical values, and date ranges.
2. Optionally call `run_statistical_query` or `verify_pandas_logic` if you need to inspect specific anomalies, missing-value relationships (e.g., whether nulls in one column correspond to a specific category in another), or date/time formats.
3. Document all Data Quality & Cleaning Findings:
   - Identify missing values, why they occur, and how downstream code should treat them.
   - Identify required type conversions (e.g., `pd.to_datetime`, extracting hour/month/year/day_of_week).
   - Identify any duplicate rows or outliers and whether they are valid records or anomalies.
4. Formulate 3 to 4 substantive analytical questions / hypotheses that uncover deep patterns, trends, segments, and drivers in `{dataset_path}` aligned with the user's goal.
5. Design exactly {max_charts} complementary visualization tasks (`chart_tasks`) to visually illustrate the most important dimensions of the dataset (e.g., temporal trends, category/product breakdown, distribution/price/behavior analysis).
   - Each `ChartTask` must have a clean snake_case `chart_id` (e.g., `monthly_revenue_trend`), a descriptive `title`, a detailed `instruction` (explicitly mentioning `{dataset_path}` and any required date parsing or cleaning steps), and a `rationale`.
"""


@mcp.prompt()
def statistical_analysis_prompt(dataset_path: str, user_goal: str, analysis_plan_json: str) -> str:
    return f"""You are a Senior Quantitative Data Analyst conducting Stage 2 (Statistical Exploration & Hypothesis Testing) on `{dataset_path}`.

User Goal / Focus: {user_goal}

Here is the Stage 1 Data Profile & Analysis Plan:
{analysis_plan_json}

=== YOUR WORKFLOW ===
1. For EVERY analytical question in the plan, use `run_statistical_query` (or `verify_pandas_logic`) on `{dataset_path}` to compute exact quantitative metrics, aggregations, percentages, growth rates, correlations, or distributions.
2. IMPORTANT TOOL RULES FOR `run_statistical_query`:
   - `df`, `pd`, and `np` are ALREADY loaded—do NOT write `import pandas as pd` or `pd.read_csv(...)`.
   - Do NOT include `#` comments inside `python_code` and keep each snippet concise (2–5 lines) to avoid JSON escaping errors.
3. Do NOT guess or fabricate numbers—always execute Python code via `run_statistical_query` using `print()` to get exact values from `{dataset_path}`.
4. Once all hypotheses have been verified with real numbers, return your structured findings, including formatted Markdown tables (`markdown_table`) and actionable `key_takeaways`.
"""


@mcp.prompt()
def report_synthesis_prompt(
    dataset_path: str,
    user_goal: str,
    analysis_plan_json: str,
    stat_findings_json: str,
    charts_summary_json: str
) -> str:
    return f"""You are a Principal Data Scientist authoring the final End-to-End Data Analysis Report for `{dataset_path}`.

User Goal / Focus: {user_goal}

=== STAGE 1: DATA PROFILE & QUALITY AUDIT ===
{analysis_plan_json}

=== STAGE 2: QUANTITATIVE & STATISTICAL FINDINGS ===
{stat_findings_json}

=== STAGE 3: GENERATED & REFLECTED VISUALIZATIONS ===
{charts_summary_json}

=== YOUR TASK ===
CRITICAL DATA INTEGRITY RULE:
- You MUST use ONLY the exact numbers, months, percentages, revenue totals, and Markdown tables provided in Stage 2 (QUANTITATIVE & STATISTICAL FINDINGS) above.
- NEVER invent, guess, or substitute different months (e.g. do NOT claim June/December are highest if Stage 2 shows October 2024 and February 2025 are highest; do NOT claim cash is 35% if Stage 2 shows card is 97.55% and cash is 2.45%).
- Copy the exact Markdown tables and exact figures from Stage 2 into your report.

Synthesize all findings into:
1. `executive_summary`: A concise, high-impact summary suitable for printing directly in the terminal (highlighting dataset scale, top 3–5 quantitative insights with exact numbers, and top recommendations).
2. `full_markdown_report`: A comprehensive, publication-grade Markdown document (`report.md`) structured with:
   - **Title & Executive Summary**
   - **1. Dataset Overview & Data Quality Audit** (shape, date coverage, missing values, duplicates, outliers, and cleaning rules applied)
   - **2. Key Quantitative Findings & Hypothesis Testing** (incorporating the exact metrics and Markdown tables from Stage 2)
   - **3. Visual Exploration & Chart Insights** (embed each generated chart using its relative markdown link `![<Title>](<relative_chart_path>)` provided in the Stage 3 summary, followed by visual interpretation and key insights)
   - **4. Strategic Conclusions & Actionable Recommendations**
"""


@mcp.prompt()
def followup_qa_prompt(dataset_path: str, question: str, analysis_context: str) -> str:
    return f"""You are an interactive Data Analysis Agent assisting the user with follow-up exploration on `{dataset_path}`.

=== PRIOR ANALYSIS CONTEXT ===
{analysis_context}

=== USER FOLLOW-UP REQUEST ===
{question}

=== YOUR WORKFLOW ===
1. Determine whether the user is asking an analytical/statistical question OR requesting a new/modified chart.
   - If the user explicitly asks to plot, chart, graph, visualize, or modify a chart/figure, set `is_chart_request` to `True`, provide the detailed plotting instruction in `chart_instruction` (referencing `{dataset_path}`), and optionally specify `target_chart_id` if they are modifying an existing chart from the report.
   - Otherwise, set `is_chart_request` to `False`, use `run_statistical_query` or `verify_pandas_logic` on `{dataset_path}` to compute the exact answer from the data, and provide a thorough, well-formatted response in `answer`.
"""


if __name__ == "__main__":
    # Start the server on stdio (Standard Input/Output)
    mcp.run()
