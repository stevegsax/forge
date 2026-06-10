# Acceptance Criteria Prompts

Three parallel LLM calls, one per leaf node.

---

## Call 1: node-001 (Create file_printer module)

### System Message

You are a criteria generator. For a given leaf task, produce specific,
testable acceptance criteria that define "done."

For software leaf tasks, acceptance criteria should include:

- What tests should pass?
- What lint/type checks apply?
- What does correct output look like?
- Is the module importable without errors?
- Are edge cases handled (empty directories, binary files, permissions)?

Each criterion must be specific and testable — not vague ("works correctly")
but precise ("returns empty string when directory contains no files").

Respond with JSON:

```json
{
  "node_id": "the node being evaluated",
  "acceptance_criteria": [
    "Criterion 1: specific, testable condition"
  ]
}
```

### User Message

Goal: Create a Python module named file_printer that reads all files from the current working directory and prints their contents to stdout. The module should handle non-text files gracefully, include a command-line entry point, and have unit tests.

Node to evaluate:
  ID: node-001
  Title: Create file_printer module
  Description: Create the core file_printer.py module with a function that iterates over files in the current working directory, reads each file's contents, and prints them to stdout. Handle non-text files gracefully by catching UnicodeDecodeError and printing a skip message. Include a file header (filename separator) before each file's contents for readability.
  Execution type: llm_call

---

## Call 2: node-002 (Add CLI entry point)

### System Message

(Same as Call 1)

### User Message

Goal: (same as above)

Node to evaluate:
  ID: node-002
  Title: Add CLI entry point
  Description: Add a command-line entry point to file_printer.py with an if __name__ == '__main__' block that calls the print_files() function. Optionally accept a directory path argument (defaulting to the current directory) via argparse.
  Execution type: llm_call

---

## Call 3: node-003 (Write unit tests)

### System Message

(Same as Call 1)

### User Message

Goal: (same as above)

Node to evaluate:
  ID: node-003
  Title: Write unit tests
  Description: Create test_file_printer.py with unit tests for the file_printer module. Test cases: (1) reading a directory with text files prints their contents, (2) binary files are skipped gracefully, (3) empty directory produces no output, (4) the function handles permission errors without crashing. Use pytest and tmp_path fixture for isolated test directories.
  Execution type: llm_call
