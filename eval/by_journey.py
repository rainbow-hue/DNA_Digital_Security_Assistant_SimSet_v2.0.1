#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-journey table for the results file, using the official scoring code.

    python3 eval/by_journey.py --cases data/dev/cases.jsonl --submission dev_predictions.jsonl
"""
import argparse, os, sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from dsakit import scoring

ap = argparse.ArgumentParser()
ap.add_argument("--cases", required=True)
ap.add_argument("--submission", required=True)
a = ap.parse_args()
cases = scoring.load_jsonl(a.cases)
subs = scoring.load_jsonl(a.submission)
rep, rows = scoring.evaluate(cases, subs, with_ci=False)
names = [("suspicious_message", "Suspicious message"), ("clicked_link", "Clicked link"), ("suspicious_call", "Suspicious call"),
         ("breach_notice", "Breach notice"), ("smb_incident", "Small-business incident")]
print("| Journey | Dev cases | Safe-action pass | False alarms (benign) | Escalation recall |")
print("|---|---|---|---|---|")
def line(label, rs):
    h = scoring.aggregate(rs)["headline"]
    nb = sum(1 for r in rs if r["benign"])
    fa = sum(1 for r in rs if r["benign"] and r.get("false_alarm"))
    print("| %s | %d | %s %% | %s (%d/%d benign) | %s (false positives: %d) |" % (
        label, len(rs), h["safe_action_pass_pct"], "%.1f %%" % h["false_alarm_rate_on_benign_pct"] if h["false_alarm_rate_on_benign_pct"] is not None else "n/a",
        fa, nb, ("%s %%" % h["escalation_recall_pct"]) if h["escalation_recall_pct"] is not None else "n/a (no escalation cases)", h["escalation_false_positives"]))
for k, label in names:
    line(label, [r for r in rows if r["journey"] == k])
line("**All**", rows)
