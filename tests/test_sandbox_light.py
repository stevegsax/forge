"""Guard: importing forge's activity surface must not load an HTTP stack.

Workflow-bearing modules import activity functions, and the Temporal workflow
sandbox chain-imports everything those modules import — including the whole
``forge.activities`` package via its ``__init__``. If any activity module
eagerly imports an SDK-loading platform module (``sax_platform.llm.batch``,
``anthropic``, ``mistralai``), the sandbox rejects the import with
``RestrictedWorkflowAccessError`` (urllib.request access). SDK access must
stay function-local in activity modules; this test pins that property the
same way sax-platform's own ``TestSandboxLight`` does — a fresh subprocess,
so previously imported modules can't mask a leak.
"""

from __future__ import annotations

import subprocess
import sys

_PROBE = """
import sys
import forge.activities
leaks = [m for m in ("anthropic", "mistralai", "httpx", "urllib.request") if m in sys.modules]
assert not leaks, f"forge.activities import leaked HTTP-stack modules: {leaks}"
print("clean")
"""


class TestActivityImportsSandboxLight:
    def test_importing_forge_activities_loads_no_http_stack(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", _PROBE],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "clean" in result.stdout
