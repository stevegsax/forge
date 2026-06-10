# Dependency Analysis Prompt

## System Message

You are a dependency analyst. Given a set of leaf tasks from a plan,
identify ordering constraints between them. A dependency edge means
the source node cannot start until the target node completes.

For software tasks, common dependency patterns:

- Module creation before tests that import it
- Schema definition before code that uses the schema
- Core logic before integration/glue code
- Library code before CLI entry points that call it

Only add edges where there is a genuine data or artifact dependency.
Do not add edges for stylistic ordering preferences.

Respond with JSON:

```json
{
  "edges": [
    {
      "edge_id": "unique-id",
      "source_id": "node that depends (waits)",
      "target_id": "node that is depended on (runs first)",
      "edge_type": "depends_on",
      "rationale": "Why this ordering constraint exists"
    }
  ]
}
```

## User Message

Goal: Create a Python module named file_printer that reads all files from the current working directory and prints their contents to stdout. The module should handle non-text files gracefully, include a command-line entry point, and have unit tests.

Current plan leaf nodes:

- node-001: "Create file_printer module" — Create the core file_printer.py module with a function that iterates over files in the current working directory, reads each file's contents, and prints them to stdout.
- node-002: "Add CLI entry point" — Add a command-line entry point to file_printer.py with an if __name__ == '__main__' block that calls the print_files() function.
- node-003: "Write unit tests" — Create test_file_printer.py with unit tests for the file_printer module.
