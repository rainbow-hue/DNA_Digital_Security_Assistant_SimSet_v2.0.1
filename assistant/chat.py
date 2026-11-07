# -*- coding: utf-8 -*-
"""Chat sessions for the mobile app (framework-free; api_server.py wraps it).

Turns free chat messages into the simulator protocol events the engine
understands, so the app and the evaluated multi-turn bot share one brain:

  first message            -> "start"  (case built from message + optional app signals)
  answer to our question   -> "user_reply"
  answer to consent ask    -> "consent_reply"  (yes/no read from the text, or explicit flags)
  anything after the final -> "user_message"   (new info: re-assess and answer again)

Each session is keyed by session_id (returned to the app on every reply).
There is NO global "latest session": two users can never see each other's case.
"""
import copy
import re
import threading
import time
import uuid

from . import knowledge as K
from .engine import get_engine, _lang

SESSION_TTL_S = 60 * 60
_YES = re.compile(r"^\s*(yes|yeah|ok|okay|sure|go ahead|fine|kyllä|joo|juu|ok|selvä|saa|saat|katso|ja\b|japp|okej|visst|går bra)", re.I)
_NO = re.compile(r"^\s*(no|nope|don't|do not|rather not|ei|en|älä|mieluummin ei|nej|helst inte|inte)\b", re.I)
_SMB = re.compile(r"\b(our company|on behalf of our|our business|webshop|employees?|staff|yritys|yrityksen|yrityksemme|firma|företag|vårt företag)\b", re.I)


def detect_language(text):
    from dsakit import textcheck as tc
    t = (text or "").lower()
    d = tc.detect_language(t)
    if d in K.LANGS:
        return d
    if re.search(r"\b(jag|inte|och|är|vad|mitt|min)\b|å", t):
        return "sv"
    if re.search(r"[äö]|\b(sain|minä|mitä|onko|tämä|en|ei|kyllä|pitäisi|viesti|puhelu)\b", t):
        return "fi"
    return "en"


class ChatSessions:
    def __init__(self, engine=None):
        self.engine = engine or get_engine()
        self.sessions = {}
        self.lock = threading.Lock()

    # -- session store ------------------------------------------------------
    def _gc(self):
        now = time.time()
        for sid in [s for s, v in self.sessions.items() if now - v["touched"] > SESSION_TTL_S]:
            del self.sessions[sid]

    def reset(self, session_id):
        with self.lock:
            self.sessions.pop(session_id, None)

    # -- main entry ---------------------------------------------------------
    def handle(self, message, session_id=None, language=None, journey=None, audience=None,
               signals=None, pre_granted=None, consent=None, reset=False):
        """consent: optional explicit {"granted": [...], "refused": [...]} from app buttons."""
        message = (message or "").strip()
        with self.lock:
            self._gc()
            if reset or not session_id or session_id not in self.sessions:
                session_id = session_id if (session_id and not reset) else "S-" + uuid.uuid4().hex[:10].upper()
                sess = self._new_session(session_id, message, language, journey, audience, signals, pre_granted)
                self.sessions[session_id] = sess
                if sess["collecting"]:
                    return self._reply(sess, {"type": "clarify"})
                reply = self.engine.step({"type": "start", "case": sess["visible_case"], "max_assistant_turns": 4}, sess["state"])
                return self._reply(sess, reply)
            sess = self.sessions[session_id]
            sess["touched"] = time.time()
            st = sess["state"]
            if sess["collecting"]:                     # still gathering the story
                sess["visible_case"]["opening_message"] += " " + message
                sess["full_case"]["opening_message"] = sess["visible_case"]["opening_message"]
                sess["collecting"] = False
                self._fill_case(sess, journey)
                reply = self.engine.step({"type": "start", "case": sess["visible_case"], "max_assistant_turns": 4}, st)
                return self._reply(sess, reply)
            if sess.get("last_type") == "consent_request":
                fields = sess["last_fields"]
                if consent:
                    granted = [f for f in consent.get("granted", []) if f in fields]
                else:
                    granted = fields if (_YES.search(message) and not _NO.search(message)) else []
                refused = [f for f in fields if f not in granted]
                sigs = {f: self._get_path(sess["full_case"], f) for f in granted}
                ev = {"type": "consent_reply", "granted": granted, "refused": refused, "text": message, "signals": sigs}
                sess["ack"] = "consent_yes" if granted else "consent_no"
            elif sess.get("last_type") == "question":
                ev = {"type": "user_reply", "text": message}
                sess["ack"] = "thanks"
            else:                                       # after a final answer: new information
                st["final"] = None
                ev = {"type": "user_message", "text": message}
                sess["ack"] = "thanks"
            return self._reply(sess, self.engine.step(ev, st))

    # -- helpers --------------------------------------------------------------
    def _new_session(self, sid, message, language, journey, audience, signals, pre_granted):
        lang = language if language in K.LANGS else detect_language(message)
        signals = copy.deepcopy(signals or {})
        req = [p for p in K.SENSITIVE_PATHS if self._get_path({"signals": signals}, p) is not None]
        case = {"case_id": sid, "language": lang, "journey": journey, "opening_message": message,
                "audience": audience or ("small_business" if _SMB.search(message) else "consumer"),
                "entry_point": "operator_app_chat", "user_identified": True, "signals": signals,
                "signal_availability": {}, "consent": {"requires_consent": req, "pre_granted": list(pre_granted or [])}}
        sess = {"id": sid, "full_case": case, "visible_case": None, "state": {}, "touched": time.time(),
                "collecting": False, "ack": None, "last_type": None}
        has_artifact = isinstance(signals.get("artifact"), dict) and signals["artifact"].get("body")
        if len(message.split()) < 4 and not K.concepts_in(message) and not has_artifact:
            sess["collecting"] = True               # "help!" -> ask what happened first
            sess["visible_case"] = copy.deepcopy(case)
            return sess
        self._fill_case(sess, journey)
        return sess

    def _fill_case(self, sess, journey):
        case = sess["full_case"]
        if not journey:
            case["journey"], _ = self.engine.predict_journey(case)
        visible = copy.deepcopy(case)
        allowed = set(case["consent"]["pre_granted"])
        for p in case["consent"]["requires_consent"]:
            if p not in allowed:
                self._set_path(visible, p, K.GATED)
        sess["visible_case"] = visible

    @staticmethod
    def _get_path(case, path):
        cur = case.get("signals") or {}
        for p in path.split("."):
            if not isinstance(cur, dict):
                return None
            cur = cur.get(p)
        return cur

    @staticmethod
    def _set_path(case, path, val):
        cur = case.setdefault("signals", {})
        parts = path.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = val

    def _reply(self, sess, reply):
        case = sess["visible_case"]
        lang = _lang(case)
        rtype = reply["type"]
        out = {"session_id": sess["id"], "reply_type": rtype, "language": lang, "journey": case.get("journey"),
               "question": None, "consent_request": None, "answer": None}
        ack = K.ACK[sess["ack"]][lang] + " " if sess.get("ack") else ""
        sess["ack"] = None
        if rtype == "clarify":
            out["reply_type"] = "question"
            out["display_text"] = K.ACK["ask_more"][lang]
            out["question"] = {"text": out["display_text"], "topic": "message_content"}
        elif rtype == "question":
            out["display_text"] = ack + reply["text"]
            out["question"] = {"text": reply["text"], "topic": (reply.get("topics") or [None])[0]}
        elif rtype == "consent_request":
            out["display_text"] = ack + reply["text"]
            out["consent_request"] = {"fields": reply["fields"], "text": reply["text"]}
            sess["last_fields"] = reply["fields"]
        else:
            ans = dict(reply["answer"])
            out["tier_used"] = ans.pop("_tier", "local_model")
            out["safety_rules"] = ans.pop("_rules", [])
            ans.pop("mode", None)
            out["answer"] = ans
            out["display_text"] = ack + ans["advice_text"]
        sess["last_type"] = rtype if rtype != "clarify" else "clarify"
        return out
