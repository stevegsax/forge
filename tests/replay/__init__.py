"""Replay coverage for the T4.1 timer-loop batch transport.

Committed workflow histories (``histories/``) plus a temporalio Replayer test
(``tests/test_replay.py``) pin today's workflow event sequences, so a future
change to workflow logic that would break an in-flight (up-to-25h) batch wait
non-deterministically fails CI instead of a production replay.

Regenerate the histories with: ``uv run python -m tests.replay.regenerate``.
"""
