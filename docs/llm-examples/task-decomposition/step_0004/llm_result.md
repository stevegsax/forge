# First Pass Decomposition Result

```json
{
  "nodes": [
    {
      "node_id": "node-001",
      "title": "Create file_printer module",
      "description": "Create the core file_printer.py module with a function that iterates over files in the current working directory, reads each file's contents, and prints them to stdout. Handle non-text files gracefully by catching UnicodeDecodeError and printing a skip message. Include a file header (filename separator) before each file's contents for readability.",
      "execution_type": "llm_call",
      "workflow_type": "software",
      "estimated_complexity": "simple",
      "is_leaf": true,
      "context": {
        "files_to_create": ["file_printer.py"],
        "key_functions": ["print_files()"]
      }
    },
    {
      "node_id": "node-002",
      "title": "Add CLI entry point",
      "description": "Add a command-line entry point to file_printer.py with an if __name__ == '__main__' block that calls the print_files() function. Optionally accept a directory path argument (defaulting to the current directory) via argparse.",
      "execution_type": "llm_call",
      "workflow_type": "software",
      "estimated_complexity": "simple",
      "is_leaf": true,
      "context": {
        "files_to_modify": ["file_printer.py"],
        "key_functions": ["main()"]
      }
    },
    {
      "node_id": "node-003",
      "title": "Write unit tests",
      "description": "Create test_file_printer.py with unit tests for the file_printer module. Test cases: (1) reading a directory with text files prints their contents, (2) binary files are skipped gracefully, (3) empty directory produces no output, (4) the function handles permission errors without crashing. Use pytest and tmp_path fixture for isolated test directories.",
      "execution_type": "llm_call",
      "workflow_type": "software",
      "estimated_complexity": "simple",
      "is_leaf": true,
      "context": {
        "files_to_create": ["test_file_printer.py"],
        "test_framework": "pytest"
      }
    }
  ],
  "edges": [
    {
      "edge_id": "edge-pc-001",
      "source_id": "node-001",
      "target_id": "node-root",
      "edge_type": "parent_child",
      "rationale": "Core module is a top-level child of the root plan node"
    },
    {
      "edge_id": "edge-pc-002",
      "source_id": "node-002",
      "target_id": "node-root",
      "edge_type": "parent_child",
      "rationale": "CLI entry point is a top-level child of the root plan node"
    },
    {
      "edge_id": "edge-pc-003",
      "source_id": "node-003",
      "target_id": "node-root",
      "edge_type": "parent_child",
      "rationale": "Unit tests are a top-level child of the root plan node"
    }
  ]
}
```
