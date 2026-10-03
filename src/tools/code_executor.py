import traceback

def execute_chart_code(code: str) -> tuple[bool, str]:
    """Runs the provided python code directly. Returns (success, error_msg)."""
    exec_globals = {}
    try:
        exec(code, exec_globals)
        return True, ""
    except Exception as e:
        error_msg = traceback.format_exc()
        return False, error_msg
