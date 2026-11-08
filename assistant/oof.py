# -*- coding: utf-8 -*-
"""Out-of-fold dev predictions: every dev case is answered by a model that never saw its scenario family.

  python -m assistant.oof --k 5 --seed 7
writes dev_oof_single.jsonl and dev_oof_multi.jsonl (same format as dev_predictions.jsonl / mt_dev.jsonl).
These, not the in-sample dev_predictions.jsonl, are the honest estimate for unseen scenarios.
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
from assistant import train as T
from assistant.engine import Assistant
from assistant.predict import SUBMISSION_KEYS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    cases = T.load_dev()
    sim = T._load_module(os.path.join(ROOT, "sim", "run_conversation.py"), "run_conversation")
    single, multi = {}, {}
    for i, (tr, va) in enumerate(T.cv_folds(cases, a.k, a.seed)):
        pref = os.path.join(T.ART_DIR, "cv_model")
        T.train(tr, verbose=False).save(pref)
        eng = Assistant(pref)
        class Bot:
            def send(self, event, state):
                return eng.step(event, state)
        for c in va:
            ans = eng.predict_single(T.visible(c))
            single[c["case_id"]] = {k: ans.get(k) for k in SUBMISSION_KEYS}
            r = sim.run_one(Bot(), c, 6)
            for k in ("_tier", "_rules"):
                r.pop(k, None)
            multi[c["case_id"]] = r
        print("fold %d done (%d cases)" % (i + 1, len(va)), flush=True)
    for name, d in (("dev_oof_single.jsonl", single), ("dev_oof_multi.jsonl", multi)):
        with open(os.path.join(ROOT, name), "w", encoding="utf-8") as f:
            for c in cases:
                f.write(json.dumps(d[c["case_id"]], ensure_ascii=False) + "\n")
    print("wrote dev_oof_single.jsonl, dev_oof_multi.jsonl")

if __name__ == "__main__":
    main()
