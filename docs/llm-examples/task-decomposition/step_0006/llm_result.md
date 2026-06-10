# Dependency Analysis Result

```json
{
  "edges": [
    {
      "edge_id": "edge-dep-001",
      "source_id": "node-002",
      "target_id": "node-001",
      "edge_type": "depends_on",
      "rationale": "The CLI entry point (node-002) imports and calls print_files() from the core module (node-001). The module must exist before the entry point can be added to it."
    },
    {
      "edge_id": "edge-dep-002",
      "source_id": "node-003",
      "target_id": "node-001",
      "edge_type": "depends_on",
      "rationale": "The unit tests (node-003) import and exercise print_files() from the core module (node-001). The module must exist before tests can be written against it."
    }
  ]
}
```

No dependency between node-002 and node-003: the CLI entry point
and the test file are independent artifacts. They can be created
in parallel once the core module (node-001) exists.
