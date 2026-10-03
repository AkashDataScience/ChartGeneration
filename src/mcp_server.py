from mcp.server.fastmcp import FastMCP
import pandas as pd
import io
import sys

# Create the MCP Server instance
mcp = FastMCP("ChartAgent Tools Server")

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
    return f"""You are a data visualization expert.

The code should create a visualization based on the files requested in the context.

{previous_code_section}

User instruction: {instruction}

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
"""


@mcp.prompt()
def reflection_prompt(instruction: str, out_path_v2: str, code_v1: str) -> str:
    return f"""You are a data visualization expert.
Your task: critique the attached chart and the original code against the given instruction,
then return improved matplotlib code.

Original code (for context):
{code_v1}

OUTPUT FORMAT (STRICT):
Return the required structured output JSON containing the critique, feedback, and refined Python code.

3) Import all necessary libraries in the code. Don't assume any imports from the original code.

HARD CONSTRAINTS:
- Use pandas/matplotlib only (no seaborn).
- Load any requested files manually using pandas (e.g., pd.read_csv('path')).
- Save to '{out_path_v2}' with dpi=300.
- Always call plt.close() at the end (no plt.show()).
- Include all necessary import statements.

CRITICAL TYPE RULE: Always aggregate properly.

Instruction:
{instruction}
"""


@mcp.prompt()
def error_fix_prompt(instruction: str, bad_code: str, error_message: str) -> str:
    return f"""You are an expert Python data visualization debugger.
You previously wrote code to fulfill this instruction:
{instruction}

However, when running the code, it produced the following error:
{error_message}

Here is the bad code you wrote:
{bad_code}

Please fix the error and return the corrected Python code.
The code should use matplotlib, manually load any requested datasets based on context, and save the figure as required in the instruction. Do NOT call plt.show().
"""

if __name__ == "__main__":
    # Start the server on stdio (Standard Input/Output)
    mcp.run()
