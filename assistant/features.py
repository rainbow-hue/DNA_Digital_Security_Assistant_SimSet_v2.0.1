# -*- coding: utf-8 -*-
"""Turns a case (exactly the fields of a test input) plus what was learned in
the conversation into a sparse feature dict.

Feature families (prefix):
  w:   word uni/bigrams of the user's own words (opening message + chat notes)
  a:   word unigrams of the shared artifact (message / call notes)
  C:   concept hits in the user's words          (knowledge.CONCEPTS)
  AC:  concept hits in the artifact
  REQ: what the artifact asks the user to do     (unsafe_claims.json patterns)
  AST: advice-like phrases inside the artifact   (step_vocabulary.json patterns)
  T:   question topics the user already talks about (question_topics.json)
  S:   structured signals (link check, call warning, subscription events, device ...)
  G:   guidance passages retrieved for the text   (guidance_corpus.jsonl)
  J: / AUD:  journey and audience
  POL: polarity of the user's answers to our questions (+ conjunctions)
"""
import math
import re
from collections import Counter

from . import knowledge as K
from .expert import rule_features

_URL = re.compile(r"https?://\S+|\b[\w.-]+\.(?:example(?:\.\w+)?|invalid|test|net|com|fi)\b\S*", re.I)
_TOK = re.compile(r"[a-zåäöéü0-9€]+", re.I)
_NUM = re.compile(r"\d+([.,]\d+)?")


def _norm(text):
    t = (text or "").lower().replace("’", "'")
    t = _URL.sub(" urltoken ", t)
    t = _NUM.sub("0", t)
    return t


def tokens(text):
    return _TOK.findall(_norm(text))


def _ngrams(toks, prefix, bigrams=True):
    out = Counter(prefix + t for t in toks if len(t) > 1)
    if bigrams:
        out.update(prefix + a + "_" + b for a, b in zip(toks, toks[1:]))
    return out


def _vocab_patterns():
    """Compiled regexes from the package's reference vocabularies."""
    from dsakit import textcheck as tc
    return tc._get("unsafe"), tc._get("steps"), tc._get("topics")


_VOCAB = None


def _vocab():
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = _vocab_patterns()
    return _VOCAB


# ---------------------------------------------------------------------------
# Guidance retrieval (TF-IDF over the 55 guidance passages, fi + en)
# ---------------------------------------------------------------------------
class GuidanceIndex:
    def __init__(self):
        self.ids = [g["id"] for g in K.GUIDANCE]
        docs = []
        for g in K.GUIDANCE:
            txt = " ".join([g.get("title_en", ""), g.get("text_en", ""), g.get("title_fi", ""), g.get("text_fi", "")])
            docs.append(Counter(tokens(txt)))
        df = Counter(t for d in docs for t in d)
        n = len(docs)
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}
        self.vecs = [self._vec(d) for d in docs]

    def _vec(self, counts):
        v = {t: (1 + math.log(c)) * self.idf.get(t, 0.0) for t, c in counts.items() if t in self.idf}
        nrm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / nrm for t, x in v.items()}

    def search(self, text, k=3):
        q = self._vec(Counter(tokens(text)))
        sims = []
        for gid, v in zip(self.ids, self.vecs):
            s = sum(w * v.get(t, 0.0) for t, w in q.items())
            sims.append((s, gid))
        sims.sort(reverse=True)
        return [(gid, round(s, 4)) for s, gid in sims[:k]]


_GINDEX = None


def guidance_index():
    global _GINDEX
    if _GINDEX is None:
        _GINDEX = GuidanceIndex()
    return _GINDEX


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
def _get(d, *path):
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def signal_features(case):
    f = {}
    sig = case.get("signals") or {}
    avail = case.get("signal_availability") or {}

    def add(k, v=1.0):
        f["S:" + k] = v

    art = sig.get("artifact")
    if isinstance(art, dict):
        add("art_type=" + str(art.get("type")))
        if art.get("links"):
            add("art_has_links")
        sender = str(art.get("sender") or "")
        if sender.startswith("+") or sender[:1].isdigit():
            add("sender_number")
        elif "@" in sender:
            add("sender_email")
        elif sender:
            add("sender_name")
        cl = art.get("channel_language")
        if cl and cl != case.get("language"):
            add("art_foreign_language")
    lc = sig.get("link_check")
    if isinstance(lc, dict):
        add("link_verdict=" + str(lc.get("verdict")))
        v = lc.get("domain_age_days")
        if isinstance(v, (int, float)):
            add("link_young" if v < 30 else "link_old")
        for cat in lc.get("categories") or []:
            add("link_cat=" + str(cat))
    for k, v in avail.items():
        add("avail_%s=%s" % (k, v))
    cw = sig.get("call_warning")
    if isinstance(cw, dict):
        add("call_warning=" + str(cw.get("warning")))
        if cw.get("caller_id_possibly_spoofed"):
            add("call_spoofed")
        if cw.get("answered") is False:
            add("call_unanswered")
        d = cw.get("duration_s")
        if isinstance(d, (int, float)) and d < 10:
            add("call_very_short")
    sub = sig.get("subscription") or {}
    ev = sub.get("recent_events") if isinstance(sub, dict) else None
    if ev == K.GATED:
        add("hidden:subscription.recent_events")
    elif isinstance(ev, list):
        for e in ev:
            if not isinstance(e, dict):
                continue
            t = str(e.get("type"))
            add("ev=" + t)
            for key in ("initiated_by_verified_user", "authentication", "login_device", "new_device", "requested_via", "ordered_by", "status"):
                if key in e:
                    add("ev=%s&%s=%s" % (t, key, e.get(key)))
            if e.get("delivery"):
                add("ev=%s&delivery=%s" % (t, e.get("delivery")))
            if e.get("channel"):
                add("ev=%s&channel=%s" % (t, e.get("channel")))
            m = e.get("minutes_ago")
            h = e.get("hours_ago")
            if (isinstance(m, (int, float)) and m < 180) or (isinstance(h, (int, float)) and h <= 24) or e.get("days_ago") == 0:
                add("ev_recent=" + t)
    ns = sub.get("network_status") if isinstance(sub, dict) else None
    if ns == K.GATED:
        add("hidden:subscription.network_status")
    elif isinstance(ns, dict):
        add("net=" + str(ns.get("service")))
    dev = sig.get("device") or {}
    if isinstance(dev, dict):
        summ = dev.get("summary") or {}
        if isinstance(summ, dict):
            add("platform=" + str(summ.get("platform")))
            if summ.get("patch_status") and summ.get("patch_status") != "current":
                add("unpatched")
        apps = dev.get("recent_app_installs")
        if apps == K.GATED:
            add("hidden:device.recent_app_installs")
        elif isinstance(apps, list):
            for a in apps:
                if isinstance(a, dict):
                    add("app_source=" + str(a.get("source")))
                    for p in a.get("permissions") or []:
                        add("app_perm=" + str(p))
            if apps:
                add("apps_installed")
        sf = dev.get("security_findings")
        if sf == K.GATED:
            add("hidden:device.security_findings")
        elif isinstance(sf, dict):
            for k, v in sf.items():
                if v:
                    add("finding=" + k)
    oa = sig.get("operator_account") or {}
    lg = oa.get("recent_logins") if isinstance(oa, dict) else None
    if lg == K.GATED:
        add("hidden:operator_account.recent_logins")
    elif isinstance(lg, list):
        for l in lg:
            if isinstance(l, dict) and l.get("country") not in (None, "FI") and not l.get("device_known"):
                add("login_foreign_unknown_device")
                if (l.get("days_ago") or 0) <= 1:
                    add("login_foreign_recent")
    ce = sig.get("credential_exposure")
    if ce == K.GATED:
        add("hidden:credential_exposure")
    elif isinstance(ce, dict):
        add("exposure_found=%s" % ce.get("found"))
        for src in ce.get("sources") or []:
            if isinstance(src, dict):
                for x in src.get("data_classes") or src.get("data") or src.get("exposed") or []:
                    add("exposed=" + str(x))
                d = src.get("days_ago")
                if isinstance(d, (int, float)):
                    add("exposure_old" if d > 180 else "exposure_recent")
    biz = sig.get("business") or {}
    if isinstance(biz, dict) and biz:
        prof = biz.get("profile") or {}
        if isinstance(prof, dict):
            add("it_support=" + str(prof.get("it_support")))
            add("approval=" + str(prof.get("payment_approval")))
        sa = biz.get("staff_accounts")
        if sa == K.GATED:
            add("hidden:business.staff_accounts")
        elif isinstance(sa, list):
            for s in sa:
                if isinstance(s, dict) and s.get("role") == "former_employee":
                    add("former_staff_status=" + str(s.get("status")))
                    if (s.get("last_active_days_ago") or 99) < 14:
                        add("former_staff_recently_active")
    return f


# ---------------------------------------------------------------------------
# Main featurizer
# ---------------------------------------------------------------------------
def user_text(case, notes=()):
    return " ".join([case.get("opening_message") or ""] + [n for n in (notes or []) if n])


def artifact_text(case):
    art = (case.get("signals") or {}).get("artifact")
    if not isinstance(art, dict):
        return ""
    return " ".join(str(art.get(k) or "") for k in ("sender", "body"))


def featurize(case, facts=None, polarity=None, notes=None, view="full", include_journey=True):
    """view: 'full' uses artifact + signals; 'chat' uses only the user's words
    (the mobile chat case where the app has no signals)."""
    facts = facts or {}
    polarity = polarity or {}
    f = Counter()
    utext = user_text(case, notes)
    atext = artifact_text(case) if view == "full" else ""
    f.update(_ngrams(tokens(utext), "w:"))
    if atext:
        f.update(_ngrams(tokens(atext), "a:", bigrams=False))
    for k in f:
        f[k] = min(f[k], 3) / 3.0
    ucon = K.concepts_in(utext)
    for c in ucon:
        f["C:" + c] = 1.0
    for c in K.concepts_in(atext):
        f["AC:" + c] = 1.0
    unsafe, steps, topics = _vocab()
    an = atext.lower()
    if an:
        for cid, pats in unsafe.items():
            if any(p.search(an) for p in pats):
                f["REQ:" + cid] = 1.0
        for sid, pats in steps.items():
            if any(p.search(an) for p in pats):
                f["AST:" + sid] = 1.0
    un = utext.lower()
    for tid, pats in topics.items():
        if any(p.search(un) for p in pats):
            f["T:" + tid] = 1.0
    if view == "full":
        f.update(signal_features(case))
        f.update(rule_features(case, notes))
    else:
        f.update(rule_features({k: v for k, v in case.items() if k != "signals"}, notes))
    for gid, s in guidance_index().search(utext + " " + atext, k=3):
        if s > 0.05:
            f["G:" + gid] = min(1.0, s * 3)
    if include_journey and case.get("journey"):
        f["J:" + case["journey"]] = 1.0
    if case.get("audience"):
        f["AUD:" + case["audience"]] = 1.0
    # what the user told us in answers to our questions
    if facts:
        f["HAS_FACTS"] = 1.0
        for t, txt in facts.items():
            for tok in set(tokens(txt)):
                f["F:" + tok] = 1.0
    for t, pol in polarity.items():
        key = "POL:%s:%s" % (t, pol)
        f[key] = 1.0
        f["POL:any:" + pol] = 1.0
        if case.get("journey"):
            f[key + "&J:" + case["journey"]] = 1.0
        for c in ucon:
            f["POL:%s&C:%s" % (pol, c)] = 1.0
    f["BIAS"] = 1.0
    return dict(f)


# ---------------------------------------------------------------------------
# Answer-polarity features (for reading the user's yes / no replies)
# ---------------------------------------------------------------------------
def polarity_features(topic, reply):
    t = (reply or "").lower()
    f = Counter()
    toks = tokens(t)
    for tok in toks:
        f["w:" + tok] = 1.0
    for a, b in zip(toks, toks[1:]):
        f["b:%s_%s" % (a, b)] = 1.0
    padded = " %s " % " ".join(toks)
    for n in (3, 4):
        for i in range(len(padded) - n + 1):
            f["c:" + padded[i:i + n]] = 0.5
    yes = bool(K.YES_RX.search(t))
    no = bool(K.NO_RX.search(t))
    uns = bool(K.UNSURE_RX.search(t))
    flip = topic in K.YES_IS_SAFE
    for name, val in (("yes", yes), ("no", no), ("unsure", uns)):
        if val:
            f["LEX:" + name] = 1.0
            f["LEX:%s&flip=%s" % (name, flip)] = 1.0
            f["LEX:%s&T:%s" % (name, topic)] = 1.0
    f["T:" + topic] = 1.0
    f["BIAS"] = 1.0
    return dict(f)
