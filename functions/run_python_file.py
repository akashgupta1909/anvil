import os
import subprocess
import sys

TIMEOUT_SECONDS = 30


def run_python_file(
    working_directory: str, file_path: str, args: list[str] | None = None
) -> str:
    """Run a Python file inside `working_directory` and return its output.

    Always returns a string; errors are returned as strings prefixed with "Error:".
    """
    try:
        working_dir_abs = os.path.abspath(working_directory)
        target_file = os.path.normpath(os.path.join(working_dir_abs, file_path))

        if os.path.commonpath([working_dir_abs, target_file]) != working_dir_abs:
            return f'Error: Cannot execute "{file_path}" as it is outside the permitted working directory'

        if not os.path.isfile(target_file):
            return f'Error: "{file_path}" does not exist or is not a regular file'

        if not target_file.endswith(".py"):
            return f'Error: "{file_path}" is not a Python file'

        command = [sys.executable, target_file]
        if args:
            command.extend(args)

        result = subprocess.run(
            command,
            cwd=working_dir_abs,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )

        parts = []
        if result.returncode != 0:
            parts.append(f"Process exited with code {result.returncode}")
        if not result.stdout and not result.stderr:
            parts.append("No output produced")
        else:
            if result.stdout:
                parts.append(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                parts.append(f"STDERR:\n{result.stderr}")
        return "\n".join(parts)
    except Exception as e:
        return f"Error: executing Python file: {e}"


def main() -> None:
    print(run_python_file("calculator", "main.py", ["3 + 5"]))


if __name__ == "__main__":
    main()
