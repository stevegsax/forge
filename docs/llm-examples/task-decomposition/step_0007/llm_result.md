# Acceptance Criteria Results

Three parallel results, one per leaf node.

---

## Result 1: node-001

```json
{
  "node_id": "node-001",
  "acceptance_criteria": [
    "file_printer.py is importable without errors: `import file_printer` succeeds",
    "print_files() reads all regular files in a given directory and prints each file's contents to stdout",
    "Non-text (binary) files are skipped with a message to stderr: 'Skipping binary file: <filename>'",
    "When the directory contains no files, print_files() produces no output and does not raise an exception",
    "Each file's contents are preceded by a header line: '=== <filename> ==='"
  ]
}
```

## Result 2: node-002

```json
{
  "node_id": "node-002",
  "acceptance_criteria": [
    "Running `python file_printer.py` prints files from the current directory",
    "Running `python file_printer.py /some/path` prints files from the specified directory",
    "Running `python file_printer.py --help` prints usage information without errors",
    "The main() function is callable independently of the if __name__ == '__main__' guard"
  ]
}
```

## Result 3: node-003

```json
{
  "node_id": "node-003",
  "acceptance_criteria": [
    "test_file_printer.py contains at least 4 test functions covering: text files, binary files, empty directory, and permission errors",
    "All tests pass when run with `pytest test_file_printer.py`",
    "Tests use pytest's tmp_path fixture for filesystem isolation (no side effects on the real filesystem)",
    "ruff check passes on test_file_printer.py with no lint errors"
  ]
}
```
