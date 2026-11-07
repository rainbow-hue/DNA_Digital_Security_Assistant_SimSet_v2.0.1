# -*- coding: utf-8 -*-
"""Digital Security Assistant: inference engine.

One engine serves three callers:
  * batch single-turn predictions   (predict.py -> test_predictions.jsonl)
  * the multi-turn simulator          (sim_bot.py -> sim/run_conversation.py)
  * the mobile chat API               (api_server.py)

Pipeline per turn
  1. featurize the visible case + what the user told us   (features.py)
  2. learned heads give distributions                      (model.py, trained by train.py)
  3. guidance playbooks for families the model never saw   (knowledge.MANUAL_PLAYBOOKS)
  4. deterministic safety rules (never overridden)
  5. optional Gemini tier when the local model is unsure   (gemini_tier.py)
  6. compose the answer from validated fi/en/sv templates, then re-check it
     with the package's own unsafe-statement detector before returning it.
"""
import copy
import os
import re

from . import knowledge as K
from .features import featurize, polarity_features, guidance_index
from .model import ModelBundle
from .expert import expert, playbook

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "dsa_model")

ASK_THRESHOLD = 0.40          # ask the decisive question when p(topic) >= this
STEP_THRESHOLD = 0.35
EXPERT_OVERRIDE = 0.55       # expert rules replace the model's action below this confidence
COND_THRESHOLD = 0.45
NEC_THRESHOLD = 0.50
LOW_CONFIDENCE = 0.40         # below this the Gemini tier may be consulted
MAX_STEPS = 6


def _lang(case):
    l = (case.get("language") or "").lower()
    if l in K.LANGS:
        return l
    from dsakit import textcheck as tc
    d = tc.detect_language(case.get("opening_message") or "")
    return d if d in K.LANGS else "en"


def _lower_first(s):
    return s[:1].lower() + s[1:] if s else s


def _strip_end(s):
    return s.rstrip(" .!")


class Assistant:
    def __init__(self, model_prefix=DEFAULT_MODEL, gemini=None):
        self.m = ModelBundle.load(model_prefix)
        self.gemini = gemini

    # ------------------------------------------------------------------
    # model access
    # ------------------------------------------------------------------
    def _dist(self, head, feats):
        vn, h = self.m.heads[head]
        return h.dist(self.m.vector(vn, feats))

    @staticmethod
    def _argmax(d):
        return max(d.items(), key=lambda kv: kv[1])

    # ------------------------------------------------------------------
    # reading a reply to one of our questions
    # ------------------------------------------------------------------
    def reply_polarity(self, topic, text, lang="en"):
        """-> ('risky' | 'safe' | 'unsure', p_risky, tier)"""
        t = (text or "").strip()
        if not t or K.UNSURE_RX.search(t):
            return "unsure", 0.5, "lexicon"
        p = self._dist("polarity", polarity_features(topic, t)).get("risky", 0.5)
        tier = "local"
        if 0.35 < p < 0.65 and self.gemini is not None:
            g = self.gemini.interpret_reply(topic, K.QUESTION_TEXT.get(topic, {}).get(lang, ""), t, lang)
            if g in ("risky", "safe", "unsure"):
                return g, p, "gemini"
        if p >= 0.6:
            return "risky", p, tier
        if p <= 0.4:
            return "safe", p, tier
        return "unsure", p, tier

    # ------------------------------------------------------------------
    # analysis
    # ------------------------------------------------------------------
    def _view(self, case):
        sig = case.get("signals") or {}
        return "full" if isinstance(sig.get("artifact"), dict) or len(sig) > 1 else "chat"

    def hidden_case(self, case, granted=()):
        """Case with consent-gated signals hidden unless pre-granted / granted."""
        c = copy.deepcopy(case)
        allowed = set(granted) | set((case.get("consent") or {}).get("pre_granted", []))
        for path in (case.get("consent") or {}).get("requires_consent", []):
            if path in allowed:
                continue
            parts = path.split(".")
            cur = c.get("signals") or {}
            for p in parts[:-1]:
                cur = cur.get(p) if isinstance(cur, dict) else None
            if isinstance(cur, dict) and parts[-1] in cur and cur[parts[-1]] is not None:
                cur[parts[-1]] = K.GATED
        return c

    def predict_journey(self, case):
        feats = featurize(case, view=self._view(case), include_journey=False)
        j, p = self._argmax(self._dist("journey", feats))
        return j, p

    def needed_consents(self, case):
        """Consent-gated signals this case needs (predicted), excluding pre-granted."""
        view = self._view(case)
        feats = featurize(self.hidden_case(case), view=view, include_journey=view == "full")
        d = self._dist("necessary", feats)
        req = set((case.get("consent") or {}).get("requires_consent", []))
        pre = set((case.get("consent") or {}).get("pre_granted", []))
        out = [p for p, v in sorted(d.items(), key=lambda kv: -kv[1]) if v >= NEC_THRESHOLD]
        _, _, reason = expert(case if view == "full" else {k: v for k, v in case.items() if k != "signals"})
        out += self.m.extra.get("nec_priors", {}).get(reason, []) + playbook(reason, None).get("nec", [])
        out = [p for p in dict.fromkeys(out) if p in req and p not in pre]
        return out[:2]

    def decisive_topic(self, case):
        view = self._view(case)
        feats = featurize(self.hidden_case(case), view=view, include_journey=view == "full")
        d = self._dist("ask", feats)
        t, p = self._argmax({k: v for k, v in d.items() if k != "none"})
        rule = self._rule_topic(case)
        if rule:
            return rule, 0.99
        if p >= ASK_THRESHOLD and d.get("none", 0) < p + 0.15:
            return t, p
        _, et, _ = expert(case if view == "full" else {k: v for k, v in case.items() if k != "signals"})
        if et:
            return et, 0.5
        return None, d.get("none", 0)

    def _rule_topic(self, case):
        ucon = set(K.concepts_in(case.get("opening_message") or ""))
        if "ex_partner" in ucon and ("monitoring" in ucon or "app_profile" in ucon):
            return "user_safety"
        return None

    def analyse(self, case, facts=None, polarity=None, notes=None, granted=(), refused=(), allow_gemini=True):
        """Return a decision dict (no text yet)."""
        view = self._view(case)
        visible = self.hidden_case(case, granted=granted)
        # single-turn: we request consent for what we need in the same answer,
        # so those signals may be used; everything else stays hidden.
        feats = featurize(visible, facts=facts, polarity=polarity, notes=notes, view=view, include_journey=view == "full")
        dist = {h: self._dist(h, feats) for h in ("action", "risk", "escalate", "target", "steps", "cond", "cites")}
        action, p_action = self._argmax(dist["action"])
        dec = {
            "action": action, "p_action": p_action, "action_dist": dist["action"],
            "risk": self._argmax(dist["risk"])[0],
            "escalate": dist["escalate"].get("yes", 0) >= 0.5,
            "target": self._argmax(dist["target"])[0],
            "steps": [s for s, p in dist["steps"].items() if p >= STEP_THRESHOLD],
            "cond_steps": [s for s, p in dist["cond"].items() if p >= COND_THRESHOLD and not s.startswith("SAFE:")],
            "safe_cond": dist["cond"].get("SAFE:explain_benign", 0) >= COND_THRESHOLD,
            "cites": [g for g, p in sorted(dist["cites"].items(), key=lambda kv: -kv[1]) if p >= 0.3][:2] or [self._argmax(dist["cites"])[0]],
            "tier": "local_model", "rules": [], "view": view,
        }
        self._apply_expert(case, dec, notes, facts, polarity, view)
        self._apply_playbook(case, feats, dec, notes)
        if allow_gemini and self.gemini is not None and dec["p_action"] < LOW_CONFIDENCE and dec["tier"] == "local_model":
            g = self.gemini.decide(case, dec, notes=notes, facts=facts)
            if g:
                dec.update(g)
                dec["tier"] = "gemini_fallback"
        self._safety_rules(case, dec, notes, facts, polarity)
        return dec

    def _prior(self, action, case):
        pri = self.m.extra.get("priors", {})
        return pri.get(action + "|" + (case.get("audience") or "consumer")) or pri.get(action + "|consumer") or pri.get(action + "|small_business") or {}

    def _apply_expert(self, case, dec, notes, facts, polarity, view):
        c = case if view == "full" else {k: v for k, v in case.items() if k != "signals"}
        answered = bool(facts or polarity)
        act, et, reason = expert(c, list(notes or []) + list((facts or {}).values()))
        dec["expert"] = (act, reason)
        limit = 0.9 if answered else (0.8 if act == "no_action_needed" else EXPERT_OVERRIDE)
        if act and act != dec["action"] and dec["p_action"] < limit:
            pr = self._prior(act, case)
            dec.update(action=act, tier="expert_rules", p_action=max(dec["action_dist"].get(act, 0.0), 0.45),
                       risk=pr.get("risk", dec["risk"]), escalate=pr.get("escalate", dec["escalate"]),
                       steps=list(dict.fromkeys(pr.get("steps", []))), cond_steps=[])
        if et and not answered and act and act != dec["action"] and dec["tier"] == "local_model":
            # single-turn with an open decisive question: give the action that is
            # safe if the riskier answer is true (expert rule for that ambiguity)
            pr = self._prior(act, case)
            dec.update(action=act, tier="expert_rules", risk=pr.get("risk", dec["risk"]),
                       escalate=pr.get("escalate", dec["escalate"]) or dec["escalate"])
        pr = self._prior(dec["action"], case)
        pb = playbook(reason, dec["action"]) if act == dec["action"] else {}
        if pb:
            if pb.get("risk") and (dec["tier"] == "expert_rules" or not answered and abs(K.RISKS.index(pb["risk"]) - K.RISKS.index(dec["risk"])) >= 2):
                dec["risk"] = pb["risk"]
            dec["steps"] = list(dict.fromkeys(pb.get("steps", []) + dec["steps"]))
            if pb.get("cond"):
                dec["cond_steps"] = list(dict.fromkeys(pb["cond"] + dec["cond_steps"]))
            dec["safe_cond"] = dec.get("safe_cond") or bool(pb.get("safe_cond"))
            if pb.get("esc") and not (answered and dec["action"] in ("no_action_needed", "do_not_interact_and_delete", "verify_via_official_channel")):
                dec["escalate"] = True
                smb = case.get("audience") == "small_business"
                dec["target"] = ("business_security_support" if smb else "security_specialist") if pb["esc"] == "auto" else pb["esc"]
        else:
            dec["steps"] = list(dict.fromkeys(dec["steps"] + pr.get("steps", [])[:3]))
            apb = playbook(reason, act) if act else {}
            if apb.get("esc") and K.INTERVENTION.get(dec["action"], 0) >= 3 and not dec["escalate"]:
                smb = case.get("audience") == "small_business"
                dec["escalate"] = True
                dec["target"] = ("business_security_support" if smb else "security_specialist") if apb["esc"] == "auto" else apb["esc"]

    def _apply_playbook(self, case, feats, dec, notes):
        ucon = set(K.concepts_in(" ".join([case.get("opening_message") or ""] + list(notes or []))))
        acon = {k[3:] for k in feats if k.startswith("AC:")}
        con = ucon | acon
        hits = [(g, 1.0) for g in K.MANUAL_PLAYBOOKS] or guidance_index().search(" ".join([case.get("opening_message") or "", ((case.get("signals") or {}).get("artifact") or {}).get("body", "") if isinstance((case.get("signals") or {}).get("artifact"), dict) else ""] + list(notes or [])), k=3)
        for gid, sim in hits:
            trig = K.PLAYBOOK_TRIGGERS.get(gid)
            if gid in K.MANUAL_PLAYBOOKS and trig and all(t in con for t in trig):
                pb = K.MANUAL_PLAYBOOKS[gid]
                dec.update(action=pb["action"], risk=pb["risk"], escalate=bool(pb["esc"]), target=pb["esc"] or dec["target"],
                           steps=list(pb["steps"]), cond_steps=[], safe_cond=False, cites=[gid], tier="guidance_playbook",
                           p_action=max(dec["p_action"], 0.6))
                return

    def _safety_rules(self, case, dec, notes, facts, polarity):
        text = " ".join([case.get("opening_message") or ""] + list(notes or []))
        ucon = set(K.concepts_in(text))
        art = (case.get("signals") or {}).get("artifact")
        acon = set(K.concepts_in(art.get("body", "") if isinstance(art, dict) else ""))
        smb = case.get("audience") == "small_business"
        if "trace_request" in ucon:
            dec.update(action="decline_out_of_scope", risk="medium", escalate=False,
                       steps=["decline_request", "report_police"], cond_steps=[], safe_cond=False)
            dec["rules"].append("out_of_scope_trace")
        if "ex_partner" in ucon and ("monitoring" in ucon or "app_profile" in ucon):
            dec.update(action="escalate_to_human", risk="high", escalate=True, target="human_advisor",
                       steps=["plan_before_removing", "seek_support_org"], cond_steps=[], safe_cond=False)
            dec["rules"].append("safety_sensitive_monitoring")
        if "ransom" in (ucon | acon) and "breach" not in (ucon | acon):
            dec.update(action="isolate_device_and_scan", risk="critical", escalate=True,
                       target="business_security_support" if smb else "security_specialist")
            dec["steps"] = list(dict.fromkeys(["disconnect_device", "dont_pay_or_transfer"] + dec["steps"]))
            dec["rules"].append("ransomware")
        if "injection" in acon and dec["action"] == "no_action_needed":
            dec.update(action="verify_via_official_channel", risk="high", escalate=False)
            dec["rules"].append("prompt_injection_never_benign")
        # consistency
        if dec["action"] == "no_action_needed":
            dec["escalate"] = False
            if dec["risk"] not in ("none", "low"):
                dec["risk"] = "low"
            keep = {"explain_benign", "verify_official_channel", "review_sessions", "check_statements"}
            dec["steps"] = [s for s in dec["steps"] if s in keep]
            dec["cond_steps"] = []
        if dec["action"] == "escalate_to_human":
            dec["escalate"] = True
        if dec["escalate"] and not dec.get("target"):
            dec["target"] = "business_security_support" if smb else "security_specialist"
        if dec["escalate"] and dec["target"] == "business_security_support" and not smb:
            dec["target"] = "operator_support_agent" if dec["action"] == "contact_operator_support" else "security_specialist"
        if "block_card" in dec["steps"] and dec["action"] not in ("contact_bank_and_block_cards",):
            dec["steps"].remove("block_card")
        if dec["action"] == "escalate_to_human" and "remove_app_or_profile" in dec["steps"]:
            dec["steps"].remove("remove_app_or_profile")
        for s in K.ACTION_CORE_STEPS.get(dec["action"], []):
            if s not in dec["steps"] and s not in dec["cond_steps"]:
                dec["steps"].append(s)
        if dec["action"] == "decline_out_of_scope":
            dec["steps"] = [s for s in dec["steps"] if s in ("decline_request", "report_police", "keep_evidence", "seek_support_org")]
        if not dec["steps"] and not dec["cond_steps"]:
            dec["steps"] = ["verify_official_channel"]

    # ------------------------------------------------------------------
    # writing the answer
    # ------------------------------------------------------------------
    def compose(self, case, dec, topic=None, question_text=None, mode="single_turn", asked_topics=(),
                consent_requested=(), granted=(), refused=(), twin_conditional=False):
        lang = _lang(case)
        order = {s: i for i, s in enumerate(K.STEP_PRIORITY)}
        steps = sorted(dict.fromkeys(dec["steps"]), key=lambda s: order.get(s, 99))[:MAX_STEPS]
        cond = [s for s in sorted(dict.fromkeys(dec["cond_steps"]), key=lambda s: order.get(s, 99)) if s not in steps][:3]
        sentences = [K.STEP_TEXT[s][lang] for s in steps]
        cond_sentences = []
        if twin_conditional and topic in K.COND_PREFIX:
            if cond:
                body = " and also ".join(_strip_end(_lower_first(K.STEP_TEXT[s][lang])).replace(",", "") for s in cond)
                if lang == "fi":
                    body = " ja myös ".join(_strip_end(_lower_first(K.STEP_TEXT[s][lang])).replace(",", "") for s in cond)
                elif lang == "sv":
                    body = " och även ".join(_strip_end(_lower_first(K.STEP_TEXT[s][lang])).replace(",", "") for s in cond)
                cond_sentences.append("%s %s." % (K.COND_PREFIX[topic][lang], body))
            if dec.get("safe_cond"):
                cond_sentences.append(K.SAFE_COND.get(topic, K.SAFE_COND["_default"])[lang])
        elif cond:  # facts known and risky: say the risky-branch steps plainly
            sentences += [K.STEP_TEXT[s][lang] for s in cond]
        why = K.WHY[dec["action"]][lang]
        if dec["action"] == "do_not_interact_and_delete" and case.get("journey") == "suspicious_call":
            why = K.WHY_CALL[lang]
        if sentences:
            body = sentences[:1] + cond_sentences[:1] + [why] + sentences[1:] + cond_sentences[1:]
        else:
            body = cond_sentences[:1] + [why] + cond_sentences[1:]
        if dec["escalate"]:
            body.append(K.ESCALATION_TEXT[dec["target"]][lang])
        advice = " ".join(body)
        advice, dropped = self._scrub(advice, lang)
        all_steps = [K.STEP_TEXT[s][lang] for s in steps] + cond_sentences
        all_steps = [s for s in all_steps if self._scrub(s, lang)[0]]
        return advice, all_steps, dropped

    @staticmethod
    def _scrub(text, lang):
        """Drop any sentence the package's unsafe-statement detector flags."""
        from dsakit import textcheck as tc
        keep, dropped = [], []
        for s in tc.sentences(text):
            (dropped if tc.find_unsafe(s, lang=lang, claim_ids=[c for c in _ALWAYS_UNSAFE]) else keep).append(s)
        return " ".join(keep), dropped

    def build_answer(self, case, dec, mode="single_turn", topic=None, asked_topics=(), consent_requested=(),
                     granted=(), refused=(), twin_conditional=False, question=True):
        lang = _lang(case)
        advice, steps, dropped = self.compose(case, dec, topic=topic, mode=mode, twin_conditional=twin_conditional)
        sig = case.get("signals") or {}
        used = []
        if isinstance(sig.get("artifact"), dict):
            used.append("artifact")
        for k in ("link_check", "call_warning"):
            if isinstance(sig.get(k), dict):
                used.append(k)
        if isinstance((sig.get("subscription") or {}).get("products"), list):
            pass  # visible, but only list it when it shaped the advice
        pre = set((case.get("consent") or {}).get("pre_granted", []))
        allowed = (pre | set(granted) | (set(consent_requested) if mode == "single_turn" else set())) - set(refused)
        needed = set(dec.get("needed_signals", []))
        used += sorted(p for p in needed if p in allowed)
        qs, qt = [], []
        if question and topic and mode == "single_turn":
            qs = [self.question_text(case, topic)]
            qt = [topic]
        risk = dec["risk"]
        ans = {
            "case_id": case.get("case_id"),
            "risk_level": risk,
            "primary_action": dec["action"],
            "advice_text": advice,
            "escalation_required": bool(dec["escalate"]),
            "escalation_target": dec["target"] if dec["escalate"] else None,
            "urgency": "immediate" if risk == "critical" else K.URGENCY_FOR_RISK.get(risk, "routine"),
            "steps": steps,
            "questions_asked": qs,
            "question_topics": qt if mode == "single_turn" else list(asked_topics),
            "signals_used": used,
            "consent_requested_for": sorted(set(consent_requested) - pre),
            "citations": dec["cites"],
            "confidence": round(min(0.97, max(0.2, dec["p_action"])), 2),
            "abstained": False,
        }
        return ans

    def question_text(self, case, topic):
        lang = _lang(case)
        con = set(K.concepts_in(" ".join([case.get("opening_message") or "", ((case.get("signals") or {}).get("artifact") or {}).get("body", "") if isinstance((case.get("signals") or {}).get("artifact"), dict) else ""])))
        for (t, c), d in K.QUESTION_VARIANTS.items():
            if t == topic and c in con:
                return d[lang]
        return K.QUESTION_TEXT[topic][lang]

    def consent_text(self, case, fields):
        lang = _lang(case)
        what = K.AND_WORD[lang].join(K.CONSENT_WHAT[f][lang] for f in fields if f in K.CONSENT_WHAT)
        return K.CONSENT_FRAME[lang].format(what=what)

    # ------------------------------------------------------------------
    # single-turn
    # ------------------------------------------------------------------
    def predict_single(self, case):
        needed = self.needed_consents(case)
        topic, p_topic = self.decisive_topic(case)
        dec = self.analyse(case, granted=needed)
        dec["needed_signals"] = needed
        return self.build_answer(case, dec, mode="single_turn", topic=topic, consent_requested=needed,
                                 twin_conditional=bool(topic))

    # ------------------------------------------------------------------
    # multi-turn protocol (sim/PROTOCOL.md); also used by the API
    # ------------------------------------------------------------------
    def step(self, event, state):
        et = event.get("type")
        if et == "start":
            state.clear()
            case = copy.deepcopy(event["case"])
            state.update(case=case, facts={}, polarity={}, notes=[], asked=[], pending=None,
                         consent_requested=[], granted=list((case.get("consent") or {}).get("pre_granted", [])),
                         refused=[], consent_done=False, final=None, turns=0,
                         max_turns=event.get("max_assistant_turns", 6))
            topic, _ = self.decisive_topic(case)
            state["topic"] = topic
            state["needed"] = [p for p in self.needed_consents(case) if self._is_hidden(case, p)]
        elif et == "user_reply":
            topic = state.get("pending")
            if topic:
                state["facts"][topic] = event.get("text", "")
                pol, _, _ = self.reply_polarity(topic, event.get("text", ""), _lang(state["case"]))
                state["polarity"][topic] = pol
            else:
                state["notes"].append(event.get("text", ""))
            state["pending"] = None
        elif et == "consent_reply":
            for path, val in (event.get("signals") or {}).items():
                self._set_path(state["case"], path, val)
            state["granted"] += list(event.get("granted") or [])
            state["refused"] += list(event.get("refused") or [])
            state["pending"] = None
        elif et == "user_message":     # free text from the app outside a question (API only)
            state["notes"].append(event.get("text", ""))
        case = state["case"]
        lang = _lang(case)
        if et != "turn_limit" and state["turns"] < state.get("max_turns", 6) and state.get("final") is None:
            t = state.get("topic")
            if t and t not in state["asked"]:
                state["asked"].append(t)
                state["pending"] = t
                state["turns"] += 1
                return {"type": "question", "text": self.question_text(case, t), "topics": [t]}
            if state["needed"] and not state["consent_done"]:
                state["consent_done"] = True
                state["consent_requested"] = list(state["needed"])
                state["turns"] += 1
                return {"type": "consent_request", "fields": list(state["needed"]), "text": self.consent_text(case, state["needed"])}
        ans = self.final_answer(state)
        state["final"] = ans
        return {"type": "final", "answer": ans}

    def final_answer(self, state):
        case = state["case"]
        topic = state.get("topic")
        pol = dict(state["polarity"])
        unsure = topic and (topic not in pol or pol.get(topic) == "unsure")
        facts = {k: v for k, v in state["facts"].items() if pol.get(k) != "unsure"}
        pol = {k: v for k, v in pol.items() if v != "unsure"}
        granted = [g for g in state["granted"] if g not in state["refused"]]
        dec = self.analyse(case, facts=facts or None, polarity=pol or None, notes=state["notes"] or None, granted=granted, refused=state["refused"])
        dec["needed_signals"] = [p for p in state.get("needed", []) + self.needed_consents(case) if p in granted]
        ans = self.build_answer(case, dec, mode="multi_turn", topic=topic, asked_topics=state["asked"],
                                consent_requested=state["consent_requested"], granted=granted, refused=state["refused"],
                                twin_conditional=bool(unsure), question=False)
        ans["mode"] = "multi_turn"
        ans["questions_asked"] = [self.question_text(case, t) for t in state["asked"]]
        ans["_tier"] = dec["tier"]
        ans["_rules"] = dec["rules"]
        return ans

    @staticmethod
    def _is_hidden(case, path):
        cur = case.get("signals") or {}
        for p in path.split("."):
            if not isinstance(cur, dict):
                return False
            cur = cur.get(p)
        return cur == K.GATED

    @staticmethod
    def _set_path(case, path, val):
        cur = case.setdefault("signals", {})
        parts = path.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = val


# Claims that are never acceptable in our own advice text, whatever the case.
_ALWAYS_UNSAFE = ["pay_the_fee", "open_link_to_check", "log_in_via_link", "call_number_in_message", "share_code",
                  "approve_prompt", "install_from_link", "give_remote_access", "keep_using_device", "thread_proves_genuine",
                  "false_reassurance", "transfer_money", "safe_account", "update_bank_details", "wait_and_see",
                  "network_glitch", "hack_back", "remove_monitoring_immediately", "disable_security", "open_attachment",
                  "reply_to_sender", "block_card_unneeded", "claim_compromised"]

_ENGINE = None


def get_engine(model_prefix=DEFAULT_MODEL, gemini=None):
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = Assistant(model_prefix, gemini=gemini)
    return _ENGINE
