import hashlib
import tempfile
import textwrap
import unittest
import warnings
import zipfile
from pathlib import Path

from paiec.submission.build_zip import collect, write_zip
from paiec.submission.check_zip import CheckError, check

CONST = "def predict(input, labeled=None):\n    return 0.5\n"


def body(code):
    return f"def predict(input, labeled=None):\n{textwrap.indent(textwrap.dedent(code), '    ')}"


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def zip(self, files):
        path = Path(self.tmp.name) / f"s{len(list(Path(self.tmp.name).iterdir()))}.zip"
        with zipfile.ZipFile(path, "w") as zf:
            for name, data in files.items():
                zf.writestr(name, textwrap.dedent(data) if isinstance(data, str) else data)
        return path

    def ok(self, files, **kw):
        return check(self.zip(files), **kw)

    def bad(self, files, pattern, **kw):
        with self.assertRaisesRegex(CheckError, pattern):
            check(self.zip(files), **kw)


class Valid(Base):
    def test_const(self):
        report = self.ok({"model.py": CONST})
        self.assertEqual(report["files"], ["model.py"])
        self.assertIn("predict, labeled=124", report["timings"])

    def test_full(self):
        report = self.ok({
            "model.py": """
                import json
                from pathlib import Path
                from helpers.prior import prior
                HERE = Path(__file__).resolve().parent
                TABLE = json.loads((HERE / "artifacts" / "prior.json").read_text())

                def load():
                    from transformers import AutoModel
                    return AutoModel.from_pretrained("org/encoder")

                def predict(input, labeled=None):
                    return prior(TABLE, len(labeled or []))
            """,
            "helpers/__init__.py": "",
            "helpers/prior.py": "def prior(table, n):\n    return min(1.0, table['p'] + 0 * n)\n",
            "artifacts/prior.json": '{"p": 0.4}',
            "labeling.py": """
                def acquisition_function(input, *, prediction=None, labeled=None, context=None):
                    return prediction is not None and abs(prediction - 0.5) < 0.2
            """,
            "models.txt": "# encoder\norg/encoder\n",
            "requirements.txt": "numpy>=1.26  # numeric\nscikit-learn==1.5.2\n",
        })
        self.assertEqual(report["labeling"], "stream (prediction, labeled, context)")
        self.assertEqual(report["models"], ["org/encoder"])
        self.assertTrue(any("requirements" in w for w in report["warnings"]))

    def test_positional_labeling(self):
        report = self.ok({"model.py": CONST, "labeling.py": "def acquisition_function(input, prediction, labeled, context, /):\n    return True\n"})
        self.assertEqual(report["labeling"], "stream positional")

    def test_labeling_without_prediction_warns(self):
        report = self.ok({"model.py": CONST, "labeling.py": "def acquisition_function(input, labeled=None, context=None):\n    return False\n"})
        self.assertTrue(any("no prediction parameter" in w for w in report["warnings"]))

    def test_legacy_rank_warns(self):
        report = self.ok({"model.py": CONST, "labeling.py": "def acquisition_function(input):\n    return 0.3\n"})
        self.assertEqual(report["labeling"], "legacy_rank")
        self.assertTrue(any("legacy_rank" in w for w in report["warnings"]))

    def test_smoke_env_set(self):
        self.ok({"model.py": "import os\nassert os.environ['PREDICTIVE_EVAL_LOCAL_SMOKE_TEST'] == '1'\n" + CONST})

    def test_write_open_and_sentence_transformer(self):
        self.ok({"model.py": """
            def dump():
                open("cache.json", "w").write("{}")

            def enc():
                from sentence_transformers import SentenceTransformer
                return SentenceTransformer("all-MiniLM-L6-v2")

            def predict(input, labeled=None):
                return 0.5
        """, "models.txt": "sentence-transformers/all-MiniLM-L6-v2\n"})

    def test_static_only_skips_code(self):
        self.ok({"model.py": "import package_that_is_not_installed\n" + CONST}, static_only=True)

    def test_numpy_like_return_warns(self):
        report = self.ok({"model.py": "from fractions import Fraction\n" + body("return Fraction(1, 3)")})
        self.assertTrue(any("float" in w for w in report["warnings"]))


class Archive(Base):
    def test_not_zip(self):
        path = Path(self.tmp.name) / "x.zip"
        path.write_text("nope")
        with self.assertRaisesRegex(CheckError, "not a zip"):
            check(path)

    def test_no_model(self):
        self.bad({"labeling.py": "x = 1\n"}, "no model.py")

    def test_model_in_folder(self):
        self.bad({"const/model.py": CONST}, "model.py is inside a folder")

    def test_nested_zip(self):
        self.bad({"model.py": CONST, "inner.zip": b"PK"}, "nested zip")

    def test_subjects_txt(self):
        self.bad({"model.py": CONST, "data/subjects.txt": "a\n"}, "subjects.txt")

    def test_parent_path(self):
        self.bad({"model.py": CONST, "../evil.py": "x = 1\n"}, "unsafe path")

    def test_absolute_path(self):
        self.bad({"model.py": CONST, "/etc/evil.txt": "x"}, "unsafe path")

    def test_duplicate_member(self):
        path = Path(self.tmp.name) / "dup.zip"
        with warnings.catch_warnings(), zipfile.ZipFile(path, "w") as zf:
            warnings.simplefilter("ignore")
            zf.writestr("model.py", CONST)
            zf.writestr("model.py", CONST)
        with self.assertRaisesRegex(CheckError, "duplicate members"):
            check(path)


class ModelsTxt(Base):
    def test_too_many(self):
        self.bad({"model.py": CONST, "models.txt": "\n".join(f"org/m{i}" for i in range(6))}, "at most 5")

    def test_duplicate(self):
        self.bad({"model.py": CONST, "models.txt": "org/m\norg/m\n"}, "duplicate in models.txt")

    def test_url(self):
        self.bad({"model.py": CONST, "models.txt": "https://huggingface.co/org/m\n"}, "only hf repo ids")

    def test_local_path(self):
        self.bad({"model.py": CONST, "models.txt": "./weights\n"}, "only hf repo ids")

    def test_token(self):
        self.bad({"model.py": CONST, "models.txt": "hf_abcdef123\n"}, "only hf repo ids")

    def test_nested(self):
        self.bad({"model.py": CONST, "cfg/models.txt": "org/m\n"}, "models.txt must be at the zip root")


class Requirements(Base):
    def test_option(self):
        self.bad({"model.py": CONST, "requirements.txt": "--index-url https://example.org\n"}, "line 1")

    def test_nested_file(self):
        self.bad({"model.py": CONST, "requirements.txt": "numpy\n-r more.txt\n"}, "line 2")

    def test_url(self):
        self.bad({"model.py": CONST, "requirements.txt": "git+https://github.com/x/y\n"}, "line 1")

    def test_local_path(self):
        self.bad({"model.py": CONST, "requirements.txt": "./wheels/pkg.whl\n"}, "line 1")

    def test_garbage(self):
        self.bad({"model.py": CONST, "requirements.txt": "pip install numpy\n"}, "line 1")

    def test_nested(self):
        self.bad({"model.py": CONST, "env/requirements.txt": "numpy\n"}, "requirements.txt must be at the zip root")


class References(Base):
    def test_missing_artifact(self):
        self.bad({"model.py": "import numpy as np\nfrom pathlib import Path\nW = np.load(Path(__file__).parent / 'w.npy')\n" + CONST}, "w.npy")

    def test_missing_artifact_in_helper(self):
        self.bad({"model.py": "from lib import f\n" + CONST, "lib.py": "import pandas as pd\ndef f():\n    return pd.read_parquet('table.parquet')\n"}, "lib.py:3.*table.parquet")

    def test_missing_artifact_via_constant(self):
        self.bad({"model.py": "import os, json\nNAME = 'prior.json'\nP = json.load(open(os.path.join(os.path.dirname(__file__), NAME)))\n" + CONST}, "prior.json")

    def test_undeclared_hf_model(self):
        self.bad({"model.py": "from transformers import AutoTokenizer\nT = AutoTokenizer.from_pretrained('org/lm')\n" + CONST}, "org/lm")

    def test_undeclared_sentence_transformer(self):
        self.bad({"model.py": "from sentence_transformers import SentenceTransformer\nM = SentenceTransformer('all-MiniLM-L6-v2')\n" + CONST}, "sentence-transformers/all-MiniLM-L6-v2")

    def test_syntax_error(self):
        self.bad({"model.py": "def predict(input, labeled=None)\n    return 0.5\n"}, "syntax error", static_only=True)


class Smoke(Base):
    def test_import_error(self):
        self.bad({"model.py": "import package_that_is_not_installed\n" + CONST}, "import model.py: ModuleNotFoundError")

    def test_no_predict(self):
        self.bad({"model.py": "def predicts(input, labeled=None):\n    return 0.5\n"}, "no function predict")

    def test_no_labeled_keyword(self):
        self.bad({"model.py": "def predict(input):\n    return 0.5\n"}, "labeled by keyword")

    def test_needs_benchmark_id(self):
        self.bad({"model.py": body("return 0.5 if input[1]['benchmark_id'] else 0.4")}, "without benchmark_id: KeyError")

    def test_bool(self):
        self.bad({"model.py": body("return True")}, "bool")

    def test_out_of_range(self):
        self.bad({"model.py": body("return 1.5")}, r"1\.5")

    def test_nan(self):
        self.bad({"model.py": body("return float('nan')")}, "nan")

    def test_string(self):
        self.bad({"model.py": body("return '0.5'")}, "str")

    def test_none(self):
        self.bad({"model.py": body("return None")}, "NoneType")

    def test_fails_on_labeled(self):
        self.bad({"model.py": body("return sum(label for _, label in labeled) / len(labeled)")}, "labeled=\\[\\].*ZeroDivisionError")

    def test_breaks_with_labels(self):
        self.bad({"model.py": body("return 0.5 if not labeled else labeled[0][0][1]['missing_key']")}, "labeled=124: KeyError")

    def test_timeout(self):
        self.bad({"model.py": "import time\ntime.sleep(30)\n" + CONST}, "exceeded", timeout=2)

    def test_crash(self):
        self.bad({"model.py": "import os\nos._exit(3)\n"}, "code 3")

    def test_labeling_not_bool(self):
        self.bad({"model.py": CONST, "labeling.py": "def acquisition_function(input, prediction=None, labeled=None, context=None):\n    return 1\n"}, "must return bool")

    def test_labeling_bad_signature(self):
        self.bad({"model.py": CONST, "labeling.py": "def acquisition_function(input, budget, other):\n    return True\n"}, "must accept")

    def test_labeling_missing_function(self):
        self.bad({"model.py": CONST, "labeling.py": "def acquire(input):\n    return True\n"}, "no function acquisition_function")

    def test_legacy_rank_non_number(self):
        self.bad({"model.py": CONST, "labeling.py": "def acquisition_function(input):\n    return 'high'\n"}, "legacy_rank")


class Build(Base):
    def test_deterministic_and_filtered(self):
        src = Path(self.tmp.name) / "s"
        for name, data in {"model.py": CONST, "fit.py": "x", "test_model.py": "x", "README.md": "x", ".DS_Store": "x",
                           "__pycache__/model.cpython-313.pyc": "x", "tests/t.py": "x", "art/w.json": "{}"}.items():
            (src / name).parent.mkdir(parents=True, exist_ok=True)
            (src / name).write_text(data)
        files = collect(src)
        self.assertEqual([p.relative_to(src).as_posix() for p in files], ["art/w.json", "model.py"])
        shas = []
        for k in range(2):
            out = Path(self.tmp.name) / f"b{k}.zip"
            write_zip(src, files, out)
            shas.append(hashlib.sha256(out.read_bytes()).hexdigest())
        self.assertEqual(shas[0], shas[1])
        check(out)


if __name__ == "__main__":
    unittest.main()
