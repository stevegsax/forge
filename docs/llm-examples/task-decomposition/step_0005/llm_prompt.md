# Atomicity Check Prompts

Three parallel LLM calls, one per leaf node. Each uses the same
template with different node data.

---

## Call 1: node-001 (Create file_printer module)

### System Message

You are an atomicity judge. Determine whether a task can be completed
in a single LLM call (document completion).

A software leaf task is atomic if:

- It can be completed in a single LLM call (document completion)
- It modifies at most 2-3 files
- It has a clear, testable outcome
- It does not require multiple sequential decisions

If the task requires creating a module AND writing tests for it, it is
NOT atomic — split it.

Respond with JSON:

```json
{
  "node_id": "the node being checked",
  "is_atomic": true,
  "rationale": "Why this task is or is not atomic",
  "suggested_split": []
}
```

### User Message

Node to check:
  ID: node-001
  Title: Create file_printer module
  Description: Create the core file_printer.py module with a function that iterates over files in the current working directory, reads each file's contents, and prints them to stdout. Handle non-text files gracefully by catching UnicodeDecodeError and printing a skip message. Include a file header (filename separator) before each file's contents for readability.
  Execution type: llm_call
  Estimated complexity: simple

---

## Call 2: node-002 (Add CLI entry point)

### System Message

(Same as Call 1)

### User Message

Node to check:
  ID: node-002
  Title: Add CLI entry point
  Description: Add a command-line entry point to file_printer.py with an if __name__ == '__main__' block that calls the print_files() function. Optionally accept a directory path argument (defaulting to the current directory) via argparse.
  Execution type: llm_call
  Estimated complexity: simple

---

## Call 3: node-003 (Write unit tests)

### System Message

(Same as Call 1)

### User Message

Node to check:
  ID: node-003
  Title: Write unit tests
  Description: Create test_file_printer.py with unit tests for the file_printer module. Test cases: (1) reading a directory with text files prints their contents, (2) binary files are skipped gracefully, (3) empty directory produces no output, (4) the function handles permission errors without crashing. Use pytest and tmp_path fixture for isolated test directories.
  Execution type: llm_call
  Estimated complexity: simple
