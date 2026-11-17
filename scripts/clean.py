import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

PATTERNS = [
    "build",
    "dist",
    ".pytest_cache",
    ".mypy_cache",
    "htmlcov",
    ".coverage",
    "src/*.egg-info",
    "runs",
]


def remove(path):
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def main():
    include_venv = "--all" in sys.argv
    for pattern in PATTERNS:
        for path in ROOT.glob(pattern):
            remove(path)
    for path in ROOT.rglob("__pycache__"):
        remove(path)
    if include_venv:
        remove(ROOT / ".venv")
    print("clean finished")


if __name__ == "__main__":
    main()
