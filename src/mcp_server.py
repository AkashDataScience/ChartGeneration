from mcp.server.mcpserver import MCPServer
import pandas as pd
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
2. Use matplotlib for plotting.
3. Add clear title, axis labels, and legend if needed.
4. Save the figure as '{out_path_v1}' with dpi=300.
5. Do not call plt.show().
6. Close all plots with plt.close().
7. Add all necessary import python statements
8. CRITICAL: when aggregating (sum, mean, count, etc.), always select the numeric column explicitly, e.g. df.groupby(['year', 'coffee_name'])['price'].sum().

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

Please fix the error and return the corrected Python code once your plan is ready.
"""

if __name__ == "__main__":
    # Start the server on stdio (Standard Input/Output)
    mcp.run()
