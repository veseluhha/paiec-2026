import unittest
from collections import Counter

import numpy as np

from paiec.harness.data import FOLDS, load
from paiec.harness.sim import BUDGETS, MAX_LABELS, Bench, episode, run

TEST = ["matharena", "real_webagents", "researchcodebench", "appworld", "ikp"]
DATA = load().only(TEST)
BENCH = {f: Bench(DATA, f) for f in TEST}


class Fake:
    def __init__(self, fn, hook=None):
        self.fn, self.hook, self.fresh_calls, self.seen = fn, hook, 0, []

    def fresh(self):
        self.fresh_calls += 1
        k = self.fresh_calls

        def predict(inp, labeled):
            self.seen.append((k, inp, labeled))
            return self.fn(inp, labeled)
        return predict, self.hook


def key(item):
    return tuple(sorted(item.items()))


class SimTest(unittest.TestCase):
    def eps(self, n=6, pairs=2):
        for f in TEST:
            for e in range(n):
                rng = np.random.default_rng([1, FOLDS.index(f), e])
                yield episode(BENCH[f], rng, pairs), rng

    def test_checkpoints_and_leakage(self):
        for ps, rng in self.eps():
            sub = Fake(lambda i, l: 0.5)
            run(sub, ps, rng)
            self.assertEqual(sub.fresh_calls, 1 + len(BUDGETS))
            acq = {p.sid: {key(i) for i, _ in p.acq} for p in ps}
            for k, budget in zip(range(2, 8), BUDGETS):
                labeled = next(l for c, _, l in sub.seen if c == k)
                per = Counter(next(p.sid for p in ps if p.subject == s) for (s, _), _ in labeled)
                for p in ps:
                    self.assertEqual(per[p.sid], min(budget, len(p.acq)))
                for (s, i), y in labeled:
                    self.assertIn(key(i), acq[next(p.sid for p in ps if p.subject == s)])
                    self.assertIn(y, (0, 1))
            for p in ps:
                self.assertGreaterEqual(len(p.acq) + len(p.ev), 80)
                self.assertFalse(set(p.acq_ids) & set(p.ev_ids))
                self.assertEqual(len(set(p.acq_ids + p.ev_ids)), len(p.acq_ids) + len(p.ev_ids))

    def test_nested_and_deterministic(self):
        for (ps, rng), (ps2, rng2) in zip(self.eps(3), self.eps(3)):
            a, b = Fake(lambda i, l: 0.3), Fake(lambda i, l: 0.7)
            ra, rb = run(a, ps, rng), run(b, ps2, rng2)
            la = [l for c, _, l in a.seen if c == 7][0]
            lb = [l for c, _, l in b.seen if c == 7][0]
            self.assertEqual(la, lb)
            prev = []
            for k in range(2, 8):
                cur = next(l for c, _, l in a.seen if c == k)
                self.assertTrue(all(x in cur for x in prev))
                prev = cur

    def test_oracle_brier(self):
        for ps, rng in self.eps(2, 1):
            p = ps[0]
            table = {}
            for i, ys in p.ev:
                table.setdefault(key(i), []).extend(ys)
            table = {k: float(np.mean(v)) for k, v in table.items()}
            out, _, _ = run(Fake(lambda i, l: table.get(key(i[1]), 0.5)), ps, rng)
            want = sum(((table[key(i)] - ys) ** 2).sum() for i, ys in p.ev) / sum(len(ys) for _, ys in p.ev)
            self.assertTrue(np.allclose(out[0], want))

    def test_dedup_inputs(self):
        for ps, rng in self.eps(4, 2):
            sub = Fake(lambda i, l: 0.5)
            _, _, st = run(sub, ps, rng)
            uniq = {(tuple(sorted(p.subject.items())), key(i)) for p in ps for i, _ in p.ev}
            self.assertEqual(st["eval_calls"], len(uniq) * len(BUDGETS))

    def test_hook_context(self):
        ctxs = []

        def hook(inp, prediction=None, labeled=None, context=None):
            ctxs.append(context)
            return context["labels_remaining"] > 0 and prediction == 0.5
        for ps, rng in self.eps(2, 1):
            ctxs.clear()
            _, got, _ = run(Fake(lambda i, l: 0.5, hook), ps, rng)
            self.assertEqual(got[0], min(MAX_LABELS, len(ps[0].acq)))
            self.assertEqual(ctxs[0]["items_remaining"], len(ps[0].acq))
            self.assertEqual(ctxs[-1]["labels_acquired"] + ctxs[-1]["labels_remaining"], MAX_LABELS)

    def test_hook_signatures(self):
        calls = []

        def kwonly(inp, *, labeled=None, context=None):
            calls.append("kwonly")
            return True

        def kwargs(inp, **kw):
            calls.append(set(kw))
            return True

        def nopred(inp, labeled=None, context=None):
            calls.append("nopred")
            return True
        for h in (kwonly, kwargs, nopred):
            ps, rng = next(self.eps(1, 1))
            sub = Fake(lambda i, l: 0.5, h)
            _, got, _ = run(sub, ps, rng)
            self.assertEqual(got[0], MAX_LABELS)
        self.assertIn({"prediction", "labeled", "context"}, calls)


if __name__ == "__main__":
    unittest.main()
