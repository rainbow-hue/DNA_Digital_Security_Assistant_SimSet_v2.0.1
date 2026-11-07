# -*- coding: utf-8 -*-
"""Batch single-turn predictions in the submission format.

  python -m assistant.predict --cases data/test/inputs.jsonl --out test_predictions.jsonl
  python -m assistant.predict --cases data/dev/cases.jsonl  --out dev_predictions.jsonl
  python -m assistant.predict ... --gemini        # consult Gemini when the local model is unsure

Ground truth / simulation fields are stripped before the engine sees a case.
"""
import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from dsakit.simulator import VISIBLE_KEYS  # noqa: E402
from assistant.engine import Assistant, DEFAULT_MODEL  # noqa: E402

SUBMISSION_KEYS = ["case_id", "risk_level", "primary_action", "advice_text", "escalation_required", "escalation_target",
                   "urgency", "steps", "questions_asked", "question_topics", "signals_used", "consent_requested_for",
                   "citations", "confidence", "abstained"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--gemini", action="store_true")
    a = ap.parse_args()
    gem = None
    if a.gemini:
        from assistant.gemini_tier import GeminiTier
        gem = GeminiTier.from_env()
    eng = Assistant(a.model, gemini=gem)
    n, actions = 0, Counter()
    with open(a.cases, encoding="utf-8") as f, open(a.out, "w", encoding="utf-8") as o:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            c = {k: c[k] for k in VISIBLE_KEYS if k in c}       # never peek at labels
            ans = eng.predict_single(c)
            ans = {k: ans.get(k) for k in SUBMISSION_KEYS}
            o.write(json.dumps(ans, ensure_ascii=False) + "\n")
            actions[ans["primary_action"]] += 1
            n += 1
    print("wrote %d predictions to %s" % (n, a.out))
    print("actions:", dict(actions.most_common()))


if __name__ == "__main__":
    main()
