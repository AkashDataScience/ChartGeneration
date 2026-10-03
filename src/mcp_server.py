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

if __name__ == "__main__":
    # Start the server on stdio (Standard Input/Output)
    mcp.run()
