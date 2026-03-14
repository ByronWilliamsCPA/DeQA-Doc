"""Generate all figures for the DeQA-Doc Technical Report Series.

Runs each paper's figure generation script in sequence, reporting success/failure.

Usage:
    python research/papers/generate_all.py
    python research/papers/generate_all.py --paper 1 3 6   # specific papers only
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

PAPERS_DIR = Path(__file__).resolve().parent
PAPER_DIRS = sorted(PAPERS_DIR.glob("[0-9]*_*/"))


def load_and_run(paper_dir: Path) -> tuple[str, bool, str]:
    """Import and execute a paper's generate_figures.py, returning (name, success, message)."""
    name = paper_dir.name
    script = paper_dir / "figures" / "generate_figures.py"

    if not script.exists():
        return name, False, "generate_figures.py not found"

    try:
        spec = importlib.util.spec_from_file_location(f"{name}.generate_figures", script)
        if spec is None or spec.loader is None:
            return name, False, "Failed to create module spec"

        module = importlib.util.module_from_spec(spec)
        # Ensure shared infrastructure is importable
        papers_path = str(PAPERS_DIR)
        if papers_path not in sys.path:
            sys.path.insert(0, papers_path)

        spec.loader.exec_module(module)

        # Run main() if it exists, otherwise the module-level code already ran
        if hasattr(module, "main"):
            module.main()

        return name, True, "OK"
    except Exception as e:
        return name, False, f"{type(e).__name__}: {e}"


def main() -> None:
    """Run figure generation for all or selected papers."""
    parser = argparse.ArgumentParser(description="Generate all paper figures")
    parser.add_argument(
        "--paper",
        type=int,
        nargs="+",
        help="Paper numbers to generate (e.g., 1 3 6). Default: all.",
    )
    args = parser.parse_args()

    # Filter to selected papers if specified
    if args.paper:
        selected = {f"{n:02d}" for n in args.paper}
        dirs = [d for d in PAPER_DIRS if d.name[:2] in selected]
    else:
        dirs = list(PAPER_DIRS)

    if not dirs:
        print("No matching paper directories found.")
        sys.exit(1)

    print(f"Generating figures for {len(dirs)} paper(s)...\n")

    results: list[tuple[str, bool, str]] = []
    for paper_dir in dirs:
        print(f"  [{paper_dir.name}] ", end="", flush=True)
        t0 = time.time()
        name, success, msg = load_and_run(paper_dir)
        elapsed = time.time() - t0
        status = "OK" if success else f"FAILED: {msg}"
        print(f"{status} ({elapsed:.1f}s)")
        results.append((name, success, msg))

    # Summary
    passed = sum(1 for _, s, _ in results if s)
    failed = sum(1 for _, s, _ in results if not s)
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(results)}")

    if failed:
        print("\nFailed papers:")
        for name, success, msg in results:
            if not success:
                print(f"  - {name}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
