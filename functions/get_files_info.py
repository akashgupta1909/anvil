import os

schema_get_files_info = {
    "type": "function",
    "function": {
        "name": "get_files_info",
        "description": "Lists files in a specified directory relative to the working directory, providing file size and directory status",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory path to list files from, relative to the working directory (default is the working directory itself)",
                },
            },
        },
    },
}


def get_files_info(working_directory: str, directory: str = ".") -> str:
    """List the contents of `directory`, which must be inside `working_directory`.

    Always returns a string; errors are returned as strings prefixed with "Error:".
    """
    try:
        working_dir_abs = os.path.abspath(working_directory)
        target_dir = os.path.normpath(os.path.join(working_dir_abs, directory))

        if os.path.commonpath([working_dir_abs, target_dir]) != working_dir_abs:
            return f'Error: Cannot list "{directory}" as it is outside the permitted working directory'

        if not os.path.isdir(target_dir):
            return f'Error: "{directory}" is not a directory'

        lines = []
        for name in os.listdir(target_dir):
            item_path = os.path.join(target_dir, name)
            lines.append(
                f"- {name}: file_size={os.path.getsize(item_path)} bytes, "
                f"is_dir={os.path.isdir(item_path)}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def main() -> None:
    print(get_files_info("calculator", "pkg"))


if __name__ == "__main__":
    main()
