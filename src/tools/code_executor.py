import re

def execute_chart_code(code: str, df) -> str:
    """Extracts python code from <execute_python> tags and runs it."""
    match = re.search(r"<execute_python>([\s\S]*?)</execute_python>", code)
    if match:
        extracted_code = match.group(1).strip()
        exec_globals = {"df": df}
        exec(extracted_code, exec_globals)
        return extracted_code
    return ""
