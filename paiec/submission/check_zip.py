#!/usr/bin/env python3
import argparse
import ast
import copy
import json
import math
import os
import posixpath
import random
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

MAX_MODELS = 5
SMOKE_ENV = "PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"
MARK = "@@check_zip@@"
ARTIFACT_SUFFIXES = {
    ".bin", ".ckpt", ".csv", ".joblib", ".json", ".model", ".npy", ".npz", ".parquet",
    ".pickle", ".pkl", ".pt", ".pth", ".safetensors", ".txt", ".ubj", ".yaml", ".yml",
}
LOADERS = {
    "open", "load", "loadtxt", "genfromtxt", "load_model", "load_file", "from_file", "read_csv",
    "read_parquet", "read_pickle", "read_json", "read_excel", "read_text", "read_bytes",
}
HF_KEYWORDS = ("pretrained_model_name_or_path", "model_name_or_path", "model_name", "model", "model_id", "repo_id", "path")
PATH_KEYWORDS = ("path", "file", "filename", "filepath", "fname")
HF_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?")
SECRET = re.compile(r"(hf|sk|ghp|gho|github_pat)[_-]", re.I)
REQUIREMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*(\[[^\]]+\])?(\s*(===|==|~=|!=|<=|>=|<|>|;).*)?")
SUBJECT_KEYS = ("normalized_name", "provider", "release_date", "access_date", "harness", "harness_version",
                "reasoning_effort", "subject_features_extra")


class CheckError(Exception):
    pass


def fail(msg):
    raise CheckError(msg)


def check_members(zf):
    raw = [n for n in zf.namelist() if not n.endswith("/")]
    names = [n.replace("\\", "/") for n in raw]
    warnings = []
    for n in names:
        p = PurePosixPath(n)
        if p.is_absolute() or ".." in p.parts or re.match(r"[A-Za-z]:", n):
            fail(f"unsafe path in zip: {n!r}")
    if len(set(names)) != len(names):
        fail("zip contains duplicate members with the same path")
    base = {n: PurePosixPath(n).name for n in names}
    if "subjects.txt" in base.values():
        fail("subjects.txt is not allowed: the platform selects subjects")
    if any(n.lower().endswith(".zip") for n in names):
        fail("nested zip is not allowed: put submission files at the top level")
    if "model.py" not in names:
        if "model.py" in base.values():
            fail("model.py is inside a folder: zip the contents of the submission folder, not the folder itself")
        fail("no model.py at the zip root")
    nested = sorted(n for n in names if n != "model.py" and base[n] == "model.py")
    if nested:
        warnings.append(f"nested model.py is ignored by the platform: {nested}")
    for name in ("models.txt", "requirements.txt"):
        bad = sorted(n for n in names if n != name and base[n] == name)
        if bad:
            fail(f"{name} must be at the zip root, found {bad[0]}")
    return names, warnings


def parse_models(zf, names):
    if "models.txt" not in names:
        return []
    lines = (s.strip() for s in zf.read("models.txt").decode("utf-8", "replace").splitlines())
    models = [s for s in lines if s and not s.startswith("#")]
    if len(models) > MAX_MODELS:
        fail(f"models.txt lists {len(models)} models, at most {MAX_MODELS} allowed")
    seen = set()
    for m in models:
        if m in seen:
            fail(f"duplicate in models.txt: {m!r}")
        seen.add(m)
        if not HF_ID.fullmatch(m) or ".." in m or any(SECRET.match(part) for part in m.split("/")):
            fail(f"models.txt must contain only hf repo ids like org/name, no urls, paths or tokens: {m!r}")
    return models


def check_requirements(zf, names):
    if "requirements.txt" not in names:
        return []
    text = zf.read("requirements.txt").decode("utf-8", "replace")
    packages = []
    for i, raw in enumerate(text.splitlines(), 1):
        line = re.sub(r"\s+#.*$", "", raw.strip()).strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-", "./", "../", "/")) or "://" in line or not REQUIREMENT.fullmatch(line):
            fail(f"requirements.txt, line {i}: {line!r} — only named pip packages, no options, urls or paths")
        packages.append(line)
    return packages


def dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


class Source:
    def __init__(self, path, root):
        self.rel = path.relative_to(root)
        self.tree = ast.parse(path.read_text(errors="replace"), filename=str(self.rel))
        self.consts, self.here = {}, set()
        for node in self.tree.body:
            target = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else getattr(node, "target", None)
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not isinstance(target, ast.Name) or node.value is None:
                continue
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                self.consts[target.id] = node.value.value
            elif self.is_here(node.value):
                self.here.add(target.id)

    def is_here(self, node):
        if isinstance(node, ast.Name):
            return node.id == "__file__" or node.id in self.here
        if isinstance(node, ast.Attribute):
            return node.attr in ("parent", "parents") and self.is_here(node.value)
        if isinstance(node, ast.Subscript):
            return self.is_here(node.value)
        if isinstance(node, ast.Call):
            name = dotted(node.func)
            if name in ("Path", "pathlib.Path", "os.path.dirname", "os.path.abspath", "os.path.realpath") and node.args:
                return self.is_here(node.args[0])
            return isinstance(node.func, ast.Attribute) and node.func.attr in ("resolve", "absolute") and self.is_here(node.func.value)
        return False

    def literal(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value or None
        if isinstance(node, ast.Name):
            return self.consts.get(node.id)
        if isinstance(node, ast.Call):
            name = dotted(node.func)
            if name in ("Path", "pathlib.Path", "PurePath") and len(node.args) == 1:
                return self.literal(node.args[0])
            if name.endswith("path.join"):
                parts = [self.literal(a) for a in node.args if not self.is_here(a)]
                return posixpath.join(*parts) if parts and None not in parts else None
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Div)):
            right = self.literal(node.right)
            if isinstance(node.op, ast.Div) and self.is_here(node.left):
                return right
            left = self.literal(node.left)
            if left is None or right is None:
                return None
            return left + right if isinstance(node.op, ast.Add) else posixpath.join(left, right)
        return None

    def imports(self):
        pkg = list(self.rel.parent.parts)
        out = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                out.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level - 1 <= len(pkg):
                base = pkg[:len(pkg) - node.level + 1] if node.level else []
                mod = ".".join(base + (node.module.split(".") if node.module else []))
                out.update({mod} - {""})
                out.update(f"{mod}.{a.name}".lstrip(".") for a in node.names if a.name != "*")
        return {".".join(m.split(".")[:i + 1]) for m in out for i in range(m.count(".") + 1)}

    def calls(self):
        return (n for n in ast.walk(self.tree) if isinstance(n, ast.Call))


def runtime_sources(root):
    modules, sources = {}, {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root)
        try:
            src = Source(path, root)
        except SyntaxError as e:
            if rel.as_posix() in ("model.py", "labeling.py"):
                fail(f"syntax error in {rel}:{e.lineno}: {e.msg}")
            continue
        sources[rel] = src
        parts = rel.with_suffix("").parts
        modules[".".join(parts[:-1] if parts[-1] == "__init__" else parts)] = rel
    todo = [Path(n) for n in ("model.py", "labeling.py") if Path(n) in sources]
    seen = []
    while todo:
        rel = todo.pop()
        if rel not in seen:
            seen.append(rel)
            todo += [modules[m] for m in sources[rel].imports() if m in modules]
    return [sources[r] for r in seen]


def exists(root, src, value):
    return (root / value).exists() or (root / src.rel.parent / value).exists()


def is_write_open(src, node):
    if dotted(node.func).split(".")[-1] != "open":
        return False
    mode = node.args[1] if len(node.args) > 1 else next((k.value for k in node.keywords if k.arg == "mode"), None)
    mode = src.literal(mode) if mode is not None else ""
    return any(c in (mode or "") for c in "wax+")


def check_artifacts(root, sources):
    for src in sources:
        for node in src.calls():
            if dotted(node.func).split(".")[-1] not in LOADERS or is_write_open(src, node):
                continue
            exprs = node.args[:2] + [k.value for k in node.keywords if k.arg in PATH_KEYWORDS]
            if isinstance(node.func, ast.Attribute):
                exprs.append(node.func.value)
            for value in filter(None, map(src.literal, exprs)):
                p = PurePosixPath(value)
                if ("://" in value or value.startswith(("~", "$")) or re.search(r"[{}*?]", value) or p.is_absolute()
                        or ".." in p.parts or p.suffix.lower() not in ARTIFACT_SUFFIXES
                        or p.name in ("models.txt", "requirements.txt")):
                    continue
                if not exists(root, src, value):
                    fail(f"{src.rel}:{node.lineno} refers to {value!r}, which is not in the zip")


def check_hf_refs(root, sources, models):
    warnings = []
    for src in sources:
        for node in src.calls():
            name = dotted(node.func)
            st = name.split(".")[-1] == "SentenceTransformer"
            if not (st or name.endswith(".from_pretrained")):
                continue
            exprs = node.args[:1] + [k.value for k in node.keywords if k.arg in HF_KEYWORDS]
            ref = next(filter(None, map(src.literal, exprs)), None)
            if ref is None or "://" in ref or exists(root, src, ref):
                continue
            need = f"sentence-transformers/{ref}" if st and "/" not in ref else ref
            if need in models:
                continue
            if ref in models:
                warnings.append(f"{src.rel}:{node.lineno}: {ref!r} — the official validator may require {need!r} in models.txt")
                continue
            fail(f"{src.rel}:{node.lineno} loads hf model {ref!r}, but {need!r} is not in models.txt and is not a local file")
    return warnings


def subject(name, provider="sample-org"):
    return dict(zip(SUBJECT_KEYS, (name, provider, "2025-06-01", "2025-07-01", "sample-harness", "1.0", "high", "")))


def item(text, benchmark=None):
    out = {"item_content": text, "item_features": "kind=sample", "interactors": ""}
    if benchmark is not None:
        out["benchmark_id"] = benchmark
    return out


def smoke_labeled():
    rng = random.Random(0)
    subjects = [subject(f"sample-model-{k}", f"org-{k % 2}") for k in range(4)]
    return [[[subjects[s], item(f"sample item {b}-{i}", b)], rng.randint(0, 1)]
            for s, b in enumerate(["benchmark_101", "benchmark_101", "benchmark_202", "benchmark_202"])
            for i in rng.sample(range(40), 31)], subjects


def child(root):
    import contextlib
    import importlib.util
    import inspect
    import numbers
    import traceback

    def load(name):
        spec = importlib.util.spec_from_file_location(f"submission_{name}", root / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def prob(value, where):
        if isinstance(value, (bool, str, bytes)) or value is None:
            fail(f"{where}: returned {type(value).__name__}, expected a finite number in [0, 1]")
        try:
            x = float(value)
        except (TypeError, ValueError):
            fail(f"{where}: returned {type(value).__name__}, float() does not apply")
        if not math.isfinite(x) or not 0 <= x <= 1:
            fail(f"{where}: returned {x}, expected a finite number in [0, 1]")
        if not isinstance(value, (int, float)):
            report["warnings"].append(f"{where}: returned {type(value).__name__}, prefer float")
        return x

    def timed(where, fn, *args, **kwargs):
        t = time.perf_counter()
        try:
            value = fn(*args, **kwargs)
        except CheckError:
            raise
        except Exception as e:
            sys.stderr.write(traceback.format_exc())
            fail(f"{where}: {type(e).__name__}: {e}")
        report["timings"][where] = round(time.perf_counter() - t, 4)
        return value

    report = {"warnings": [], "timings": {}}
    sys.path.insert(0, str(root))
    try:
        with contextlib.redirect_stdout(sys.stderr):
            model = timed("import model.py", load, "model")
            predict = getattr(model, "predict", None)
            if not callable(predict):
                fail("model.py has no function predict(input, labeled)")
            labeled, subjects = smoke_labeled()
            try:
                inspect.signature(predict).bind([], labeled=[])
            except (TypeError, ValueError):
                fail("predict must accept labeled by keyword: predict(input, labeled)")
            prob(timed("predict, labeled=[], item without benchmark_id", predict, [subject("sample-model"), item("A sample yes/no item.")], labeled=[]), "predict without benchmark_id")
            prob(timed("predict, empty strings", predict, [dict.fromkeys(SUBJECT_KEYS, ""), item("", "")], []), "predict on empty strings")
            target = [copy.deepcopy(subjects[0]), item("sample item benchmark_101-99", "benchmark_101")]
            prob(timed(f"predict, labeled={len(labeled)}", predict, target, copy.deepcopy(labeled)), "predict with non-empty labeled")
            prob(timed("predict, new subject", predict, [subject("unseen-model", "other-org"), item("sample item benchmark_202-3", "benchmark_202")], copy.deepcopy(labeled)), "predict for a new subject")
            if (root / "labeling.py").exists():
                acquisition = getattr(timed("import labeling.py", load, "labeling"), "acquisition_function", None)
                if not callable(acquisition):
                    fail("labeling.py has no function acquisition_function")
                params = inspect.signature(acquisition).parameters
                names = ("prediction", "labeled", "context")
                kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
                accepted = tuple(n for n in names if kwargs or (n in params and params[n].kind != params[n].POSITIONAL_ONLY))
                shape = None
                for positional, keys in [(False, names), (True, ())] + [(False, accepted)] * (accepted not in ((), names)):
                    try:
                        inspect.signature(acquisition).bind(*([None] * (4 if positional else 1)), **dict.fromkeys(keys))
                    except TypeError:
                        continue
                    shape = positional, keys
                    break
                context = {"subject_id": "subject_001", "benchmark_id": "benchmark_101", "labels_acquired": 0,
                           "labels_remaining": 31, "max_labels": 31, "items_remaining": 56}
                if shape is None:
                    try:
                        inspect.signature(acquisition).bind(None)
                    except TypeError:
                        fail("acquisition_function must accept (input, prediction, labeled, context)")
                    report["labeling"] = "legacy_rank"
                    report["warnings"].append("labeling.py uses the one-argument legacy_rank mode (numeric priority), which the competition page does not document")
                    value = timed("acquisition_function", acquisition, copy.deepcopy(target))
                    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
                        fail(f"acquisition_function(input) in legacy_rank mode must return a finite number, returned {value!r}")
                else:
                    positional, keys = shape
                    values = {"prediction": 0.5, "labeled": copy.deepcopy(labeled), "context": context}
                    report["labeling"] = "stream " + ("positional" if positional else "(" + ", ".join(keys) + ")")
                    if not positional and "prediction" not in keys:
                        report["warnings"].append("acquisition_function has no prediction parameter: the platform will not call predict before deciding")
                    args = [values[n] for n in names] if positional else []
                    value = timed("acquisition_function", acquisition, copy.deepcopy(target), *args, **{k: values[k] for k in keys})
                    if type(value) is not bool:
                        fail(f"acquisition_function must return bool, returned {type(value).__name__}")
    except CheckError as e:
        report["error"] = str(e)
    print(MARK + json.dumps(report, ensure_ascii=False), flush=True)


def smoke(root, python, timeout):
    env = dict(os.environ, **{SMOKE_ENV: "1"})
    with tempfile.TemporaryDirectory(prefix="check-zip-cwd-") as cwd:
        try:
            run = subprocess.run([python, str(Path(__file__).resolve()), "--child", str(root)], env=env, cwd=cwd,
                                 capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            fail(f"smoke test exceeded {timeout} s")
    line = next((s for s in reversed(run.stdout.splitlines()) if s.startswith(MARK)), None)
    tail = "\n".join(run.stderr.strip().splitlines()[-15:])
    if line is None:
        fail(f"smoke process exited with code {run.returncode}\n{tail}")
    report = json.loads(line[len(MARK):])
    if "error" in report:
        fail(report["error"] + (f"\n{tail}" if tail else ""))
    return report


def check(zip_path, static_only=False, python=sys.executable, timeout=600):
    zip_path = Path(zip_path)
    if not zip_path.is_file() or not zipfile.is_zipfile(zip_path):
        fail(f"not a zip file: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        names, warnings = check_members(zf)
        models = parse_models(zf, names)
        packages = check_requirements(zf, names)
        with tempfile.TemporaryDirectory(prefix="check-zip-") as tmp:
            root = Path(tmp).resolve()
            zf.extractall(root)
            sources = runtime_sources(root)
            check_artifacts(root, sources)
            warnings += check_hf_refs(root, sources, models)
            report = {} if static_only else smoke(root, python, timeout)
    if packages:
        warnings.append("installation of requirements.txt by the platform is not guaranteed")
    report.update(files=names, models=models, requirements=packages, warnings=warnings + report.get("warnings", []))
    return report


def main():
    parser = argparse.ArgumentParser(description="validate a paiec submission zip")
    parser.add_argument("zip", type=Path)
    parser.add_argument("--static-only", action="store_true", help="static checks only, do not import or run submission code")
    parser.add_argument("--python", default=sys.executable, help="python interpreter for the smoke test")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        return child(args.zip)
    try:
        report = check(args.zip, args.static_only, args.python, args.timeout)
    except CheckError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"ok: {args.zip}" + (" (static checks only, code not run)" if args.static_only else ""))
    print(f"  files: {', '.join(report['files'])}")
    for key in ("models", "requirements"):
        if report[key]:
            print(f"  {key}: {', '.join(report[key])}")
    for where, sec in report.get("timings", {}).items():
        print(f"  {where}: {sec:.4f} s")
    if "labeling" in report:
        print(f"  labeling: {report['labeling']}")
    for w in report["warnings"]:
        print(f"  warning: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
