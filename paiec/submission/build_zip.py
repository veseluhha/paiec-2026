import argparse
import datetime
import fnmatch
import hashlib
import sys
import zipfile
from pathlib import Path

from .check_zip import CheckError, check

SKIP = ("fit.py", "fit_*.py", "test_*.py", "*_test.py", "conftest.py", "*.pyc", "*.zip", "*.md", "*.ipynb", ".*")
SKIP_DIRS = {"__pycache__", "tests", ".ipynb_checkpoints"}


def collect(src):
    return sorted((p for p in src.rglob("*") if p.is_file()
                   and not SKIP_DIRS & set(p.relative_to(src).parts[:-1])
                   and not any(fnmatch.fnmatch(p.name, s) for s in SKIP)),
                  key=lambda p: p.relative_to(src).as_posix())


def write_zip(src, files, dst):
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            info = zipfile.ZipInfo(p.relative_to(src).as_posix(), (1980, 1, 1, 0, 0, 0))
            info.compress_type, info.create_system, info.external_attr = zipfile.ZIP_DEFLATED, 3, 0o644 << 16
            zf.writestr(info, p.read_bytes())


def main():
    ap = argparse.ArgumentParser(description="build a deterministic submission zip from a fitted directory and validate it")
    ap.add_argument("src", type=Path)
    ap.add_argument("--out", type=Path, default=Path("dist"))
    ap.add_argument("--name")
    ap.add_argument("--static-only", action="store_true")
    a = ap.parse_args()
    src = a.src.resolve()
    files = collect(src)
    if not (src / "model.py").exists():
        sys.exit(f"error: no model.py in {src}")
    a.out.mkdir(parents=True, exist_ok=True)
    tmp = a.out / f".{a.name or src.name}.zip.tmp"
    write_zip(src, files, tmp)
    sha = hashlib.sha256(tmp.read_bytes()).hexdigest()
    dst = a.out / f"{a.name or src.name}_{datetime.date.today():%Y%m%d}_{sha[:8]}.zip"
    tmp.replace(dst)
    try:
        report = check(dst, a.static_only)
    except CheckError as e:
        dst.unlink()
        sys.exit(f"check_zip failed, zip removed: {e}")
    print(f"{dst}\nsha256 {sha}\nfiles: {', '.join(p.relative_to(src).as_posix() for p in files)}")
    for where, sec in report.get("timings", {}).items():
        print(f"  {where}: {sec:.4f} s")
    for w in report["warnings"]:
        print(f"  warning: {w}")


if __name__ == "__main__":
    main()
