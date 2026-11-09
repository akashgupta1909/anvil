from functions.get_files_info import get_files_info

CASES = [
    ("calculator", ".", "Result for current directory:"),
    ("calculator", "pkg", "Result for 'pkg' directory:"),
    ("calculator", "/bin", "Result for '/bin' directory:"),
    ("calculator", "../", "Result for '../' directory:"),
]


def main() -> None:
    for working_directory, directory, header in CASES:
        print(f'get_files_info("{working_directory}", "{directory}"):')
        print(header)
        result = get_files_info(working_directory, directory)
        print("\n".join(f"  {line}" for line in result.splitlines()))
        print()


if __name__ == "__main__":
    main()
