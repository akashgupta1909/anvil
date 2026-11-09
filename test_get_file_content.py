from functions.get_file_content import get_file_content


def main() -> None:
    result = get_file_content("calculator", "lorem.txt")
    print(f"lorem.txt length: {len(result)}")
    print(f"lorem.txt truncated: {'truncated' in result}")
    print()

    for file_path in ["main.py", "pkg/calculator.py", "/bin/cat", "pkg/does_not_exist.py"]:
        print(f'get_file_content("calculator", "{file_path}"):')
        print(get_file_content("calculator", file_path))
        print()


if __name__ == "__main__":
    main()
