# -*- coding: utf-8 -*-
"""Train and evaluate the assistant on the dev split.

  # 1) honest estimate: 80/20 split GROUPED BY SCENARIO FAMILY (the 20 % are
  #    families the model never saw, like the 9 test-only families)
  python -m assistant.train --split family --seed 42

  # 2) seen-family estimate: 80/20 split grouped by twin pair (no twin leaks)
  python -m assistant.train --split twin --seed 42

  # 3) 5-fold family cross-validation (more stable numbers, slower)
  python -m assistant.train --cv 5

  # 4) final model on 100 % of dev (after you are happy with 1-3),
  #    saved to assistant/artifacts/dsa_model.{npz,json}
  python -m assistant.train --final

Each evaluation scores the held-out cases with the package's own evaluator
(dsakit.scoring), single-turn AND multi-turn against the simulated user, next
to the rules baseline on exactly the same cases.
"""
import argparse
import importlib.util
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from dsakit import scoring  # noqa: E402
from dsakit.simulator import VISIBLE_KEYS  # noqa: E402

from assistant import knowledge as K  # noqa: E402
from assistant.features import featurize, polarity_features  # noqa: E402
from assistant.model import ModelBundle, MultiLabelHead, SoftmaxHead, Vectorizer  # noqa: E402

ART_DIR = os.path.join(HERE, "artifacts")
FINAL_PREFIX = os.path.join(ART_DIR, "dsa_model")


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------
def load_dev():
    return scoring.load_jsonl(os.path.join(ROOT, "data", "dev", "cases.jsonl"))


def visible(case):
    return {k: json.loads(json.dumps(case[k])) for k in VISIBLE_KEYS if k in case}


def hidden(case):
    c = visible(case)
    pre = set(case["consent"].get("pre_granted", []))
    for path in case["consent"].get("requires_consent", []):
        if path in pre:
            continue
        parts = path.split(".")
        cur = c["signals"]
        for p in parts[:-1]:
            cur = cur.get(p) if isinstance(cur, dict) else None
        if isinstance(cur, dict) and cur.get(parts[-1]) is not None:
            cur[parts[-1]] = K.GATED
    return c


def risk_idx(r):
    return K.RISKS.index(r)


def twin_partners(cases):
    groups = defaultdict(list)
    for c in cases:
        tg = c["ground_truth"].get("twin_group")
        if tg:
            groups[tg].append(c)
    partner = {}
    for tg, l in groups.items():
        for c in l:
            others = [o for o in l if o is not c and o["ground_truth"]["branch"] != c["ground_truth"]["branch"]]
            if others:
                partner[c["case_id"]] = others[0]
    return partner


def severity(c):
    gt = c["ground_truth"]
    return (K.INTERVENTION[gt["primary_action"]], risk_idx(gt["risk_level"]))


def target_of(c, partner):
    gt = c["ground_truth"]
    t = gt["escalation"].get("target") or (gt["escalation"].get("acceptable_targets") or [None])[0]
    if not t and c["case_id"] in partner:
        pg = partner[c["case_id"]]["ground_truth"]
        t = pg["escalation"].get("target") or (pg["escalation"].get("acceptable_targets") or [None])[0]
    return t


def labels(c, partner):
    """Single-turn ('st') and after-the-answer ('mt') labels."""
    gt = c["ground_truth"]
    twin = bool(gt.get("twin_group")) and c["case_id"] in partner
    mt = {
        "action": gt["primary_action"], "risk": gt["risk_level"],
        "escalate": "yes" if gt["escalation"]["required"] else "no",
        "target": target_of(c, partner) if gt["escalation"]["required"] else None,
        "steps": list(gt["required_steps"]), "cond": [],
    }
    st = dict(mt)
    if twin:
        p = partner[c["case_id"]]
        risky, benign = (c, p) if severity(c) >= severity(p) else (p, c)
        r_req, b_req = set(risky["ground_truth"]["required_steps"]), set(benign["ground_truth"]["required_steps"])
        stt = gt["single_turn"]
        st.update(action=stt["acceptable_actions"][0], risk=stt["risk_range"][1],
                  escalate="yes" if stt.get("escalation_required") else "no",
                  target=target_of(c, partner) if stt.get("escalation_required") else None,
                  steps=sorted((r_req & b_req) | (b_req - r_req - {"explain_benign"})),
                  cond=sorted(r_req - b_req) + (["SAFE:explain_benign"] if "explain_benign" in b_req else []))
        is_risky = c is risky
    else:
        is_risky = None
    ask = (gt["must_ask_topics"] or gt["should_ask_topics"] or ["none"])[0]
    nec = [s for s in gt["necessary_signals"] if s in c["consent"]["requires_consent"]]
    return {"st": st, "mt": mt, "ask": ask, "nec": nec, "cites": list(gt["guidance_refs"]),
            "journey": c["journey"], "twin": twin, "is_risky": is_risky}


def must_facts(c):
    gt = c["ground_truth"]
    facts = c["simulation"]["facts"]
    return {t: facts[t] for t in gt["must_ask_topics"] if t in facts}


# ---------------------------------------------------------------------------
# polarity (reading yes / no answers) training data
# ---------------------------------------------------------------------------
LEX_AUG = {
    "risky": {"en": ["yes", "yes I did", "yeah, I did that", "I already did", "unfortunately yes"],
              "fi": ["kyllä", "joo tein", "juu, valitettavasti", "tein sen jo", "kyllä tein"],
              "sv": ["ja", "ja, det gjorde jag", "tyvärr ja", "jo, jag gjorde det"]},
    "safe": {"en": ["no", "no I didn't", "nope, nothing", "I did not", "no, I hung up"],
             "fi": ["ei", "en", "en tehnyt mitään", "en, lopetin puhelun", "en antanut mitään"],
             "sv": ["nej", "nej, det gjorde jag inte", "nej, ingenting", "jag gjorde inte det"]},
}


def polarity_rows(cases, partner):
    rows, ys = [], []
    for c in cases:
        lab = labels(c, partner)
        if not lab["twin"]:
            continue
        for t, txt in must_facts(c).items():
            rows.append(polarity_features(t, txt))
            ys.append("risky" if lab["is_risky"] else "safe")
    for t in K.TOPIC_IDS:
        flip = t in K.YES_IS_SAFE
        for pol, by_lang in LEX_AUG.items():
            lab = pol if not flip else ("safe" if pol == "risky" else "risky")
            for l, xs in by_lang.items():
                for x in xs:
                    rows.append(polarity_features(t, x))
                    ys.append(lab)
    return rows, ys


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def build_rows(cases, partner, pol_head=None, pol_vec=None):
    R = defaultdict(list)   # head -> list of (feats, label)

    def add(heads, feats, lab):
        for h in heads:
            R[h].append((feats, lab))

    for c in cases:
        L = labels(c, partner)
        st, mt = L["st"], L["mt"]
        full, hid, = visible(c), hidden(c)
        views_st = [featurize(full), featurize(hid), featurize(full, view="chat", include_journey=False)]
        for i, f in enumerate(views_st):
            add(["action"], f, st["action"]); add(["risk"], f, st["risk"]); add(["escalate"], f, st["escalate"])
            if st["target"]:
                add(["target"], f, st["target"])
            add(["steps"], f, st["steps"]); add(["cond"], f, st["cond"])
            add(["ask"], f, L["ask"]); add(["cites"], f, L["cites"])
            if i < 2:
                add(["necessary"], f, L["nec"])
        add(["journey"], featurize(full, include_journey=False), L["journey"])
        add(["journey"], featurize(full, view="chat", include_journey=False), L["journey"])
        facts = must_facts(c)
        if facts:
            if L["twin"]:
                pol = {t: ("risky" if L["is_risky"] else "safe") for t in facts}
            else:
                pol = {}
                for t, txt in facts.items():
                    if pol_head is not None:
                        p = pol_head.dist(pol_vec.transform([polarity_features(t, txt)])[0])["risky"]
                        pol[t] = "risky" if p >= 0.5 else "safe"
            for f in (featurize(full, facts=facts, polarity=pol), featurize(hid, facts=facts, polarity=pol),
                      featurize(full, facts=facts, polarity=pol, view="chat", include_journey=False)):
                add(["action"], f, mt["action"]); add(["risk"], f, mt["risk"]); add(["escalate"], f, mt["escalate"])
                if mt["target"]:
                    add(["target"], f, mt["target"])
                add(["steps"], f, mt["steps"]); add(["cond"], f, [])
    return R


SPEC = {
    "action": ("softmax", K.ACTIONS), "risk": ("softmax", K.RISKS), "escalate": ("softmax", ["no", "yes"]),
    "target": ("softmax", K.ESC_TARGETS), "journey": ("softmax", list(K.ACTION_VOCAB["journeys"])),
    "ask": ("softmax", ["none"] + K.TOPIC_IDS), "steps": ("multi", K.STEP_IDS),
    "cond": ("multi", K.STEP_IDS + ["SAFE:explain_benign"]), "necessary": ("multi", K.SENSITIVE_PATHS),
    "cites": ("multi", [g["id"] for g in K.GUIDANCE]),
}


def train(cases, l2=3e-3, iters=250, verbose=True):
    t0 = time.time()
    partner = twin_partners(cases)
    bundle = ModelBundle()
    # polarity head first (used to label non-twin facts)
    prow, py = polarity_rows(cases, partner)
    pvec = Vectorizer(min_df=1).fit(prow)
    phead = SoftmaxHead(["safe", "risky"], l2=1e-3, iters=300).fit(pvec.transform(prow), py)
    bundle.vecs["pol"] = pvec
    bundle.add("polarity", "pol", phead)
    R = build_rows(cases, partner, phead, pvec)
    allrows = [f for h in R for f, _ in R[h]]
    vec = Vectorizer(min_df=2).fit(allrows)
    bundle.vecs["main"] = vec
    for name, (kind, classes) in SPEC.items():
        data = R[name]
        X = vec.transform([f for f, _ in data])
        y = [l for _, l in data]
        if kind == "softmax":
            present = [c for c in classes if c in set(y)] or classes
            head = SoftmaxHead(present, l2=l2, iters=iters).fit(X, y)
        else:
            head = MultiLabelHead(classes, l2=l2, iters=iters).fit(X, y)
        bundle.add(name, "main", head)
    # learned priors per action (used when the expert layer overrides the action)
    act_steps, act_risk, act_esc = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    n_act = Counter()
    for c in cases:
        st = labels(c, partner)["st"]
        key = st["action"] + "|" + c["audience"]
        n_act[key] += 1
        act_steps[key].update(st["steps"])
        act_risk[key][st["risk"]] += 1
        act_esc[key][st["escalate"]] += 1
    priors = {}
    for key, n in n_act.items():
        priors[key] = {"steps": [s for s, k in act_steps[key].most_common() if k / n >= 0.3][:4],
                       "risk": act_risk[key].most_common(1)[0][0],
                       "escalate": act_esc[key]["yes"] / n >= 0.5}
    # which consent-gated signals each generic expert reason needed in training
    from assistant.expert import expert as _expert
    rn, rc = Counter(), defaultdict(Counter)
    for c in cases:
        _, _, reason = _expert(visible(c))
        rn[reason] += 1
        rc[reason].update(labels(c, partner)["nec"])
    nec_priors = {r: [s for s, k in rc[r].items() if k / n >= 0.5] for r, n in rn.items()}
    bundle.extra = {"priors": priors, "nec_priors": nec_priors, "trained_on": len(cases), "features": len(vec.index), "families": sorted({c["scenario_family"] for c in cases})}
    if verbose:
        print("  trained %d heads on %d cases (%d features, %d polarity rows) in %.1fs"
              % (len(bundle.heads), len(cases), len(vec.index), len(prow), time.time() - t0))
    return bundle


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------
def split(cases, mode="family", seed=42, frac=0.2):
    rnd = random.Random(seed)
    if mode == "family":
        key = lambda c: c["scenario_family"]
    else:  # keep each twin pair together, otherwise random by case
        key = lambda c: c["ground_truth"].get("twin_group") or c["case_id"]
    groups = sorted({key(c) for c in cases})
    rnd.shuffle(groups)
    n_val = max(1, round(len(groups) * frac))
    val_g = set(groups[:n_val])
    return [c for c in cases if key(c) not in val_g], [c for c in cases if key(c) in val_g]


def cv_folds(cases, k=5, seed=42):
    rnd = random.Random(seed)
    fams = sorted({c["scenario_family"] for c in cases})
    rnd.shuffle(fams)
    for i in range(k):
        val_f = set(fams[i::k])
        yield [c for c in cases if c["scenario_family"] not in val_f], [c for c in cases if c["scenario_family"] in val_f]


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def evaluate(engine, val, multi=True, label="model"):
    rules = _load_module(os.path.join(ROOT, "baselines", "rules_baseline.py"), "rules_baseline")
    sim = _load_module(os.path.join(ROOT, "sim", "run_conversation.py"), "run_conversation")
    out = {}
    preds = [engine.predict_single(visible(c)) for c in val]
    out[label + " single"] = scoring.evaluate(val, preds, with_ci=False)[0]
    base = [rules.strategy_answer(visible(c), "rules") for c in val]
    out["rules single"] = scoring.evaluate(val, base, with_ci=False)[0]
    if multi:
        class Bot:
            def send(self, event, state):
                return engine.step(event, state)
        mt = []
        for c in val:
            r = sim.run_one(Bot(), c, 6)
            for k in ("_tier", "_rules"):
                r.pop(k, None)
            mt.append(r)
        out[label + " multi"] = scoring.evaluate(val, mt, with_ci=False)[0]

        class RBot:
            def send(self, event, state):
                return rules.step(event, state)
        out["rules multi"] = scoring.evaluate(val, [sim.run_one(RBot(), c, 6) for c in val], with_ci=False)[0]
    return out, preds


COLS = [("pass", lambda r: r["headline"]["safe_action_pass_pct"]),
        ("falseAlarm", lambda r: r["headline"]["false_alarm_rate_on_benign_pct"]),
        ("escRecall", lambda r: r["headline"]["escalation_recall_pct"]),
        ("escFP", lambda r: r["headline"]["escalation_false_positives"]),
        ("decisiveQ", lambda r: r["headline"]["asked_decisive_question_pct"]),
        ("stepCov", lambda r: r["details"]["required_step_coverage_pct"]),
        ("correct", lambda r: r["brief_scorecard_proxy"]["correctness"]),
        ("clarity", lambda r: r["brief_scorecard_proxy"]["clarity"]),
        ("uncert", lambda r: r["brief_scorecard_proxy"]["uncertainty_escalation"]),
        ("privacy", lambda r: r["brief_scorecard_proxy"]["privacy_consent"]),
        ("PROXY", lambda r: r["brief_scorecard_proxy"]["weighted_proxy_over_assessed_dimensions"])]


def table(reports, title):
    print("\n" + title)
    print("  %-16s" % "" + "".join("%11s" % c for c, _ in COLS))
    for name, r in reports.items():
        print("  %-16s" % name + "".join("%11s" % ("-" if f(r) is None else f(r)) for _, f in COLS))


def show_failures(rep, n=15):
    print("  failure types:", rep["failure_counts"])
    for f in rep["failure_cases"][:n]:
        extra = {k: v for k, v in f.items() if k not in ("case_id", "type", "journey")}
        print("    %-11s %-22s %s" % (f["case_id"], f["type"], json.dumps(extra, ensure_ascii=False)[:120]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["family", "twin"], default="family")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv", type=int, default=0)
    ap.add_argument("--final", action="store_true", help="train on 100 %% of dev and save the production model")
    ap.add_argument("--no-multi", action="store_true")
    ap.add_argument("--failures", type=int, default=15)
    a = ap.parse_args()
    from assistant.engine import Assistant
    os.makedirs(ART_DIR, exist_ok=True)
    cases = load_dev()

    if a.final:
        b = train(cases)
        b.save(FINAL_PREFIX)
        print("saved final model to %s.{npz,json}" % FINAL_PREFIX)
        return

    if a.cv:
        agg = defaultdict(list)
        for i, (tr, va) in enumerate(cv_folds(cases, a.cv, a.seed)):
            print("fold %d: train %d / val %d (%d unseen families)" % (i + 1, len(tr), len(va), len({c['scenario_family'] for c in va})))
            pref = os.path.join(ART_DIR, "cv_model")
            train(tr).save(pref)
            reps, _ = evaluate(Assistant(pref), va, multi=not a.no_multi)
            table(reps, "  fold %d" % (i + 1))
            for k, r in reps.items():
                agg[k].append(r)
        print("\nMEAN OVER FOLDS (unseen families)")
        print("  %-16s" % "" + "".join("%11s" % c for c, _ in COLS))
        for k, rs in agg.items():
            vals = []
            for _, f in COLS:
                xs = [f(r) for r in rs if f(r) is not None]
                vals.append(round(sum(xs) / len(xs), 1) if xs else "-")
            print("  %-16s" % k + "".join("%11s" % v for v in vals))
        return

    tr, va = split(cases, a.split, a.seed)
    print("80/20 split by %s (seed %d): train %d cases / validation %d cases" % (a.split, a.seed, len(tr), len(va)))
    print("  validation families:", sorted({c["scenario_family"] for c in va}))
    pref = os.path.join(ART_DIR, "split_model")
    train(tr).save(pref)
    reps, preds = evaluate(Assistant(pref), va, multi=not a.no_multi)
    table(reps, "VALIDATION (%s split) - dsakit.scoring on held-out cases" % a.split)
    print("\nmodel single-turn failures:")
    show_failures(reps["model single"], a.failures)
    if "model multi" in reps:
        print("\nmodel multi-turn failures:")
        show_failures(reps["model multi"], a.failures)


if __name__ == "__main__":
    main()
