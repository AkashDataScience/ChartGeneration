import os
import traceback
import src.utils as utils


def execute_chart_code(code: str, expected_output_path: str | None = None) -> tuple[bool, str]:
    """Runs the provided python code directly and verifies the output chart file exists."""
    exec_globals = {}
    try:
        sanitized = utils.sanitize_python_code(code)
        if not sanitized.strip():
            return (
                False,
                "Generated Python code is empty. You MUST include all plotting steps and "
                f"plt.savefig('{expected_output_path or 'chart.png'}', dpi=300) inside decomposed_code_steps."
            )

        exec(sanitized, exec_globals)

        if expected_output_path and not os.path.exists(expected_output_path):
            return (
                False,
                f"Code executed without throwing an exception, but the output file '{expected_output_path}' "
                f"was NOT created! Make sure your code includes the matplotlib plotting step and calls "
                f"plt.savefig('{expected_output_path}', dpi=300) followed by plt.close()."
            )
        return True, ""
    except Exception:
        error_msg = traceback.format_exc()
        return False, error_msg
