from functions.run_python_file import run_python_file

CASES = [
    ("calculator", "main.py", None),
    ("calculator", "main.py", ["3 + 5"]),
    ("calculator", "tests.py", None),
    ("calculator", "../main.py", None),
    ("calculator", "nonexistent.py", None),
    ("calculator", "lorem.txt", None),
]


def main() -> None:
    for working_directory, file_path, args in CASES:
        print(f'run_python_file("{working_directory}", "{file_path}", {args}):')
        print(run_python_file(working_directory, file_path, args))
        print()


if __name__ == "__main__":
    main()
