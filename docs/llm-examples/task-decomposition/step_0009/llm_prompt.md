# Adversarial Review Prompts

Three parallel LLM calls, one per judge persona. All receive the same
plan but different persona instructions.

---

## Call 1: Expert Skeptic

### System Message

You are Expert Skeptic. Your role is to identify edge cases, failure
modes, and missing error handling in task plans. You look for what
will go wrong, not what will go right.

## Plan Under Review

```json
{
  "plan_id": "plan-7a3f",
  "version": 5,
  "goal_statement": "Create a Python module named file_printer that reads all files from the current working directory and prints their contents to stdout. The module should handle non-text files gracefully, include a command-line entry point, and have unit tests.",
  "workflow_type": "software",
  "nodes": {
    "node-root": {
      "node_id": "node-root",
      "title": "Root",
      "description": "Plan root node",
      "execution_type": "llm_call",
      "workflow_type": "software",
      "acceptance_criteria": [],
      "estimated_complexity": "simple",
      "context": {},
      "children": ["node-001", "node-002", "node-003"],
      "is_leaf": false
    },
    "node-001": {
      "node_id": "node-001",
      "title": "Create file_printer module",
      "description": "Create the core file_printer.py module with a function that iterates over files in the current working directory, reads each file's contents, and prints them to stdout. Handle non-text files gracefully by catching UnicodeDecodeError and printing a skip message. Include a file header (filename separator) before each file's contents for readability.",
      "execution_type": "llm_call",
      "workflow_type": "software",
      "acceptance_criteria": [
        "file_printer.py is importable without errors: `import file_printer` succeeds",
        "print_files() reads all regular files in a given directory and prints each file's contents to stdout",
        "Non-text (binary) files are skipped with a message to stderr: 'Skipping binary file: <filename>'",
        "When the directory contains no files, print_files() produces no output and does not raise an exception",
        "Each file's contents are preceded by a header line: '=== <filename> ==='"
      ],
      "estimated_complexity": "simple",
      "context": {"files_to_create": ["file_printer.py"], "key_functions": ["print_files()"]},
      "children": [],
      "is_leaf": true
    },
    "node-002": {
      "node_id": "node-002",
      "title": "Add CLI entry point",
      "description": "Add a command-line entry point to file_printer.py with an if __name__ == '__main__' block that calls the print_files() function. Optionally accept a directory path argument (defaulting to the current directory) via argparse.",
      "execution_type": "llm_call",
      "workflow_type": "software",
      "acceptance_criteria": [
        "Running `python file_printer.py` prints files from the current directory",
        "Running `python file_printer.py /some/path` prints files from the specified directory",
        "Running `python file_printer.py --help` prints usage information without errors",
        "The main() function is callable independently of the if __name__ == '__main__' guard"
      ],
      "estimated_complexity": "simple",
      "context": {"files_to_modify": ["file_printer.py"], "key_functions": ["main()"]},
      "children": [],
      "is_leaf": true
    },
    "node-003": {
      "node_id": "node-003",
      "title": "Write unit tests",
      "description": "Create test_file_printer.py with unit tests for the file_printer module. Test cases: (1) reading a directory with text files prints their contents, (2) binary files are skipped gracefully, (3) empty directory produces no output, (4) the function handles permission errors without crashing. Use pytest and tmp_path fixture for isolated test directories.",
      "execution_type": "llm_call",
      "workflow_type": "software",
      "acceptance_criteria": [
        "test_file_printer.py contains at least 4 test functions covering: text files, binary files, empty directory, and permission errors",
        "All tests pass when run with `pytest test_file_printer.py`",
        "Tests use pytest's tmp_path fixture for filesystem isolation (no side effects on the real filesystem)",
        "ruff check passes on test_file_printer.py with no lint errors"
      ],
      "estimated_complexity": "simple",
      "context": {"files_to_create": ["test_file_printer.py"], "test_framework": "pytest"},
      "children": [],
      "is_leaf": true
    }
  },
  "edges": [
    {"edge_id": "edge-pc-001", "source_id": "node-001", "target_id": "node-root", "edge_type": "parent_child", "rationale": "Core module is a top-level child of the root plan node"},
    {"edge_id": "edge-pc-002", "source_id": "node-002", "target_id": "node-root", "edge_type": "parent_child", "rationale": "CLI entry point is a top-level child of the root plan node"},
    {"edge_id": "edge-pc-003", "source_id": "node-003", "target_id": "node-root", "edge_type": "parent_child", "rationale": "Unit tests are a top-level child of the root plan node"},
    {"edge_id": "edge-dep-001", "source_id": "node-002", "target_id": "node-001", "edge_type": "depends_on", "rationale": "The CLI entry point imports and calls print_files() from the core module"},
    {"edge_id": "edge-dep-002", "source_id": "node-003", "target_id": "node-001", "edge_type": "depends_on", "rationale": "The unit tests import and exercise print_files() from the core module"}
  ]
}
```

## Evaluation Criteria

Score each dimension 1-5:

1. COMPLETENESS: Does the plan cover the entire goal?
2. GRANULARITY: Is each leaf task truly atomic (one LLM call / one human action / one function)?
3. FEASIBILITY: Can each leaf task actually be completed as described?
4. DEPENDENCY_CORRECTNESS: Are ordering constraints correct and complete?
5. ACCEPTANCE_CRITERIA_QUALITY: Are the "done" conditions specific and testable?

## Required Response Structure

### Arguments AGAINST This Plan

List every weakness, gap, risk, and failure mode you can identify.
Be thorough and adversarial. Assume the plan WILL fail and explain why.

### Arguments FOR This Plan

Now argue that the plan is adequate despite the weaknesses above.
Which weaknesses are acceptable? Which are mitigated by other aspects of the plan?

### Verdict

APPROVE or REJECT.
If REJECT, list the specific changes required (not suggestions -- requirements).

### Scores

(JSON with verdict, scores, required_changes, arguments_against, arguments_for)

---

## Call 2: Completeness Auditor

### System Message

You are Completeness Auditor. Your role is to identify coverage gaps,
missing steps, and overlooked requirements in task plans. You ensure
nothing is left out.

(Remainder of prompt identical to Call 1 — same plan, same evaluation
criteria, same response structure.)

---

## Call 3: Dependency Critic

### System Message

You are Dependency Critic. Your role is to identify ordering errors,
hidden dependencies, and missed parallelism opportunities in task
plans. You ensure the execution order is both correct and efficient.

(Remainder of prompt identical to Call 1 — same plan, same evaluation
criteria, same response structure.)
