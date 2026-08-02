"""Run the full test suite with line + branch coverage analysis.

Produces:
  tests/coverage.json     - machine-readable coverage report
  tests/coverage_html/    - human-readable HTML coverage browser
  tests/test_results.json - per-test pass/fail outcomes
  tests/report.html       - pytest HTML test report

Usage:
    cd backend
    python tests/run_coverage.py            # full coverage
    python tests/run_coverage.py --quick    # skip slow integration tests
"""
import argparse
import os
import subprocess
import sys


BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_pytest_with_coverage(quick: bool = False):
    """Run pytest with coverage. Returns the subprocess exit code."""
    cmd = [
        sys.executable, "-m", "pytest", "tests/",
        # Coverage flags
        "--cov=routes",
        "--cov=utils",
        "--cov=evaluation",
        "--cov=classifier_engine",
        "--cov=sql_scripts",
        "--cov=gavel_pipeline",
        "--cov-branch",
        "--cov-report=json:tests/coverage.json",
        "--cov-report=html:tests/coverage_html",
        "--cov-report=term-missing:skip-covered",
        # Test reporters
        "--json-report",
        "--json-report-file=tests/test_results.json",
        "--html=tests/report.html",
        "--self-contained-html",
        "-q",
    ]
    if quick:
        cmd.extend(["-m", "not slow"])

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.call(cmd, cwd=BACKEND_ROOT, env=env)


def print_coverage_summary():
    """Print a per-module summary of coverage from coverage.json."""
    import json
    cov_path = os.path.join(BACKEND_ROOT, "tests", "coverage.json")
    if not os.path.exists(cov_path):
        print("[!] coverage.json not found — coverage may have failed")
        return

    with open(cov_path) as f:
        cov = json.load(f)

    backend_files = {p: d for p, d in cov["files"].items()
                     if not p.startswith("tests")
                     and ".venv" not in p
                     and "__pycache__" not in p}

    # Group by top-level module directory
    by_module = {}
    for path, data in backend_files.items():
        norm = path.replace("\\", "/")
        top = norm.split("/")[0]
        by_module.setdefault(top, []).append(data["summary"])

    print()
    print("=" * 75)
    print(f"{'Module':<25} {'Stmts':>7} {'Miss':>7} {'Branch':>8} {'BrPart':>8} {'Line %':>8}")
    print("-" * 75)

    grand_stmts = grand_miss = grand_branches = grand_partial = 0
    for module, summaries in sorted(by_module.items()):
        s = sum(x["num_statements"] for x in summaries)
        m = sum(x["missing_lines"] for x in summaries)
        b = sum(x.get("num_branches", 0) for x in summaries)
        bp = sum(x.get("num_partial_branches", 0) for x in summaries)
        pct = ((s - m) / s * 100) if s else 0
        print(f"{module:<25} {s:>7d} {m:>7d} {b:>8d} {bp:>8d} {pct:>7.1f}%")
        grand_stmts += s
        grand_miss += m
        grand_branches += b
        grand_partial += bp

    print("-" * 75)
    grand_pct = ((grand_stmts - grand_miss) / grand_stmts * 100) if grand_stmts else 0
    print(f"{'TOTAL':<25} {grand_stmts:>7d} {grand_miss:>7d} {grand_branches:>8d} {grand_partial:>8d} {grand_pct:>7.1f}%")
    print("=" * 75)
    print()
    print(f"Detailed reports:")
    print(f"  HTML coverage:   {os.path.join(BACKEND_ROOT, 'tests', 'coverage_html', 'index.html')}")
    print(f"  Test results:    {os.path.join(BACKEND_ROOT, 'tests', 'report.html')}")
    print(f"  Coverage JSON:   {cov_path}")


def main():
    parser = argparse.ArgumentParser(description="Run GAVEL test suite with coverage")
    parser.add_argument("--quick", action="store_true", help="Skip slow integration tests")
    parser.add_argument("--no-summary", action="store_true", help="Skip the coverage summary print")
    args = parser.parse_args()

    print(f"Running tests with coverage (quick={args.quick})...")
    rc = run_pytest_with_coverage(quick=args.quick)

    if not args.no_summary:
        print_coverage_summary()

    sys.exit(rc)


if __name__ == "__main__":
    main()
