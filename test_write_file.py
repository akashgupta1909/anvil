from functions.write_file import write_file

CASES = [
    ("calculator", "lorem.txt", "wait, this isn't lorem ipsum"),
    ("calculator", "pkg/morelorem.txt", "lorem ipsum dolor sit amet"),
    ("calculator", "/tmp/temp.txt", "this should not be allowed"),
]


def main() -> None:
    for working_directory, file_path, content in CASES:
        print(f'write_file("{working_directory}", "{file_path}", "{content}"):')
        print(write_file(working_directory, file_path, content))
        print()


if __name__ == "__main__":
    main()
