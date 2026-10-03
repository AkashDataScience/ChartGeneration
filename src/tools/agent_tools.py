import pandas as pd
import io
import contextlib

def get_dataset_columns(file_path: str) -> str:
    """Returns the column names and their data types for a given CSV file."""
    try:
        df = pd.read_csv(file_path)
        schema = []
        for col, dtype in df.dtypes.items():
            schema.append(f"- {col}: {dtype}")
        
        # Also return a small sample
        sample = df.head(3).to_markdown()
        return f"Columns:\n{chr(10).join(schema)}\n\nSample Data:\n{sample}"
    except Exception as e:
        return f"Error reading dataset: {str(e)}"

def run_python_script(code: str) -> str:
    """Executes a block of Python code and returns the console output (stdout)."""
    # Redirect stdout to capture print statements
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            # Execute the code in a clean dictionary
            exec(code, {"__builtins__": __builtins__})
        return stdout.getvalue() or "Code executed successfully with no output."
    except Exception as e:
        return f"Error executing code:\n{str(e)}"

# The schema mapping for OpenAI Tool Calling
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_dataset_columns",
            "description": "Returns the column names and their data types for a given CSV file, along with a small sample of the data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the CSV file (e.g. data/coffee_sales.csv)"
                    }
                },
                "required": ["file_path"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_python_script",
            "description": "Executes a block of Python code and returns the console output (stdout). Use this to explore data, test filters, or run calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The python code to execute. Should include necessary imports like pandas."
                    }
                },
                "required": ["code"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]

# Dispatcher to easily run the tools when the LLM requests them
TOOL_DISPATCH = {
    "get_dataset_columns": get_dataset_columns,
    "run_python_script": run_python_script
}
