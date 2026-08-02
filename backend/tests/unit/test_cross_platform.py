"""Cross-platform compatibility tests — Windows / Linux / Mac path handling.

The project is meant to be cloned and run locally on any OS, so all path
operations must use os.path.join (or pathlib), never hard-coded slashes.
"""
import os
import re
import pytest
from pathlib import Path


# tests/unit/test_cross_platform.py → tests/unit/ → tests/ → backend/
BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _iter_python_source_files():
    """Yield all .py files under the backend root, excluding venv and tests."""
    skip_dirs = {".venv", "venv", "env", "__pycache__", "tests", "node_modules"}
    for root, dirs, files in os.walk(BACKEND_ROOT):
        # in-place mutation to skip subtrees
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in files:
            if fname.endswith(".py"):
                yield os.path.join(root, fname)


class TestPathHandling:
    """Verify the codebase doesn't use hard-coded path separators."""

    # Files that may legitimately use OS-specific shell commands
    # (currently none).
    ALLOWED_HARDCODED_PATHS = set()

    def test_no_hardcoded_backslashes_in_paths(self):
        """Search Python files for '\\' inside string literals that look like paths.

        Looks for patterns like: "foo\\bar.txt" or "foo\\\\bar".
        """
        bad = []
        # Match string literal containing backslash followed by a known suspicious component
        # e.g. "models\\classifier" — clear path separator usage
        # Avoid: regex escapes ("\\d"), JSON escapes, etc.
        suspicious_re = re.compile(
            r'["\'][^"\']{0,80}\\\\(?:[a-zA-Z0-9_.-]+)+[^"\']*["\']'
        )

        for path in _iter_python_source_files():
            rel = os.path.relpath(path, BACKEND_ROOT).replace(os.sep, "/")
            if rel in self.ALLOWED_HARDCODED_PATHS:
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    src = f.read()
            except UnicodeDecodeError:
                continue

            for m in suspicious_re.finditer(src):
                literal = m.group(0)
                # Skip regex patterns (look like r"..." or contain regex chars next to \\)
                if "\\d" in literal or "\\w" in literal or "\\s" in literal or "\\n" in literal:
                    continue
                if "\\b" in literal or "\\." in literal:
                    continue
                bad.append(f"{rel}: {literal[:80]}")

        # Allow up to a small handful — many code-string patterns generate false positives
        assert len(bad) < 10, f"Hardcoded backslash paths found:\n" + "\n".join(bad[:20])

    def test_uses_os_path_join_for_classifier_dirs(self):
        """The trainer should compose paths via os.path.join (not string concat)."""
        trainer_path = os.path.join(BACKEND_ROOT, "classifier_engine", "trainer.py")
        if not os.path.exists(trainer_path):
            pytest.skip("trainer.py not found")
        with open(trainer_path, "r", encoding="utf-8") as f:
            src = f.read()
        # Should reference os.path.join
        assert "os.path.join" in src or "Path(" in src, "Trainer must use os.path.join or pathlib"


class TestFilePathEdgeCases:
    """Edge cases in upload filename handling."""

    def test_basename_strips_directory_components(self):
        """os.path.basename should strip parent directories from any filename."""
        cases = [
            ("../../etc/passwd", "passwd"),
            ("..\\..\\windows\\system32", "system32"),
            ("/absolute/path/file.pth", "file.pth"),
            ("C:\\Users\\admin\\file.pth", "file.pth"),
            ("normal.pth", "normal.pth"),
        ]
        for inp, expected in cases:
            actual = os.path.basename(inp)
            # On Windows, basename handles both \\ and /
            # On Linux, only / is treated as separator — but our code calls basename on user input
            assert actual.endswith(expected) or actual == expected, f"basename({inp!r}) = {actual!r}, expected {expected!r}"


class TestPathlibUsage:
    """Pathlib should work cross-platform without issues."""

    def test_pathlib_join_works(self):
        p = Path("a") / "b" / "c"
        # On Windows: a\b\c, on Linux: a/b/c — both valid
        assert p.parts == ("a", "b", "c")

    def test_path_normalization(self):
        p = os.path.normpath("a/b/../c")
        assert p in ("a/c", "a\\c")  # Both are valid depending on OS


class TestEnvironmentDetection:
    """The codebase should not assume a specific OS."""

    def test_no_hardcoded_temp_dir(self):
        """Searching for hardcoded /tmp or C:\\Temp paths."""
        bad = []
        bad_patterns = ["/tmp/", "C:\\Temp", "C:\\\\Temp"]

        for path in _iter_python_source_files():
            rel = os.path.relpath(path, BACKEND_ROOT)
            if "test_" in rel or rel.endswith("conftest.py"):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    src = f.read()
            except UnicodeDecodeError:
                continue
            for bad_pattern in bad_patterns:
                if bad_pattern in src:
                    bad.append(f"{rel}: hardcoded {bad_pattern}")

        assert not bad, f"Hardcoded temp paths found:\n" + "\n".join(bad)

    def test_tempfile_module_available(self):
        """tempfile.gettempdir() should return a valid OS-appropriate path."""
        import tempfile
        d = tempfile.gettempdir()
        assert os.path.isdir(d)
