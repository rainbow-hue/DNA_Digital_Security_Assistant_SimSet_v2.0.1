# -*- coding: utf-8 -*-
"""Generic expert rules: proportionate-response logic that does not depend on
any one scenario family. Used twice:
  * as features (RULE:action=..., RULE_ASK:...) so the learned model can learn
    how far to trust them, and
  * as the fallback decision when the learned model is unsure (unseen families).

Returns (action or None, decisive_topic or None, reason).
"""
from . import knowledge as K


def _sig(case, *path):
    cur = case.get("signals") or {}
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def expert(case, notes=()):
    text = " ".join([case.get("opening_message") or ""] + list(notes or []))
    art = _sig(case, "artifact")
    body = art.get("body", "") if isinstance(art, dict) else ""
    u = set(K.concepts_in(text))
    a = set(K.concepts_in(body))
    described = not body.strip()        # chat: the user describes the message in their own words
    if described:
        a = set(u) - {"did_enter", "did_pay", "did_click", "negated", "own_action", "expected", "asks_if_safe"}
    ua = u | a
    smb = case.get("audience") == "small_business"
    lc = _sig(case, "link_check")
    verdict = lc.get("verdict") if isinstance(lc, dict) else None
    cw = _sig(case, "call_warning")
    warning = cw.get("warning") if isinstance(cw, dict) else None
    events = _sig(case, "subscription", "recent_events")
    ev_types = {e.get("type") for e in events if isinstance(e, dict)} if isinstance(events, list) else set()
    acted = bool(u & {"did_enter", "did_pay"}) and "negated" not in u
    has_links = bool(isinstance(art, dict) and art.get("links")) or "http" in body.lower() or verdict is not None
    own = "own_action" in u or "expected" in u

    if "trace_request" in u:
        return "decline_out_of_scope", None, "out_of_scope"
    if "ex_partner" in ua and ("monitoring" in ua or "app_profile" in ua):
        return "escalate_to_human", "user_safety", "monitoring"
    if "ransom" in ua and "breach" not in ua:
        return "isolate_device_and_scan", None, "ransomware"
    if "safe_account" in ua or ("police" in ua and "did_pay" in u):
        return "contact_bank_and_block_cards", None, "safe_account"
    if "approval_prompt" in ua:
        if own and "repeated" not in ua:
            return "no_action_needed", None, "own_login_prompt"
        return "change_password_and_enable_mfa", None, "prompt_storm"
    if "porting" in ua or "no_service" in ua or ev_types & {"esim_activation", "esim_transfer_code_sent", "port_out_request"}:
        if own and "negated" not in u:
            return "no_action_needed", None, "own_sim_change"
        return "contact_operator_support", ("self_initiated" if "negated" not in u else None), "sim_takeover"
    if "order" in ua and ("sim" in ua or "new_subscription_order" in ev_types):
        bl = body.lower()
        remote_delivery = any(w in bl for w in ("pickup point", "another city", "noutopiste", "toisessa kaupungissa", "toiseen kaupunkiin", "utlämningsställe", "annan stad"))             or any(isinstance(e, dict) and str(e.get("delivery", "")).startswith("pickup") for e in (events if isinstance(events, list) else []))
        if not remote_delivery and ("admin" in bl or "pääkäyttäjä" in bl or "administratör" in bl or own):
            return "no_action_needed", None, "own_order"
        return "contact_operator_support", "self_initiated", "unexpected_order"
    if "login_alert" in ua:
        return "change_password_and_enable_mfa", (None if own else "self_initiated"), "login_alert"
    if "breach" in ua:
        if "only_email" in ua and "hashed_pw" not in ua:
            return "no_action_needed", None, "breach_email_only"
        if "link" in a and "urgency" in a:
            return "verify_via_official_channel", None, "fake_breach_notice"
        return "change_password_and_enable_mfa", None, "breach_passwords"
    if "former_employee" in ua:
        return "change_password_and_enable_mfa", None, "offboarding"
    if "forwarding" in ua:
        return "change_password_and_enable_mfa", None, "mailbox_takeover"
    if "remote" in ua:
        if own:
            return "no_action_needed", None, "booked_it_support"
        if "remote" in u and acted and ("bank" in u or "bank_login" in u):
            return "contact_bank_and_block_cards", None, "remote_banking"
        if "installed" in u or "did_enter" in u or "remote" in u and "negated" not in u:
            return "isolate_device_and_scan", "credentials_entered", "remote_access"
        return "verify_via_official_channel", None, "remote_request"
    if "terminal" in ua:
        return "verify_via_official_channel", None, "terminal_support_call"
    if "installed" in u or ("certificate_profile" in ua and "did_enter" not in u):
        return "isolate_device_and_scan", None, "installed_app"
    if "attachment" in ua and "did_click" in u and "negated" in u and "enabled_content" not in u:
        return "do_not_interact_and_delete", None, "attachment_not_enabled"
    if "attachment" in ua and ("did_click" in u or "enabled_content" in u):
        return "isolate_device_and_scan", (None if "enabled_content" in u else "app_installed"), "attachment_opened"
    if "code" in ua and "voice_call" in ua:
        if acted:
            return ("contact_operator_support" if "sim" in ua else "contact_bank_and_block_cards"), None, "code_given"
        if "negated" in u:
            return "verify_via_official_channel", None, "code_not_given"
        return ("contact_operator_support" if "sim" in ua else "contact_bank_and_block_cards"), "code_shared", "code_call"
    if acted and ("bank_login" in ua or "code" in ua or "card" in ua):
        return "contact_bank_and_block_cards", None, "bank_details_entered"
    if acted and "password" in ua:
        return "change_password_and_enable_mfa", None, "password_entered"
    if "did_click" in u and "link" in ua and not own and "negated" in u:
        return "do_not_interact_and_delete", None, "link_opened"
    if "did_click" in u and "link" in ua and not own:
        if "card" in ua or "pay_request" in ua:
            return "contact_bank_and_block_cards", "card_details_entered", "link_opened_payment"
        return "do_not_interact_and_delete", None, "link_opened"
    if "ceo" in ua and "pay_request" in ua and "did_pay" in u and "negated" not in u:
        return "contact_bank_and_block_cards", None, "ceo_paid"
    if "ceo" in ua and "pay_request" in ua and "negated" in u:
        return "verify_via_official_channel", None, "ceo_not_paid"
    if "ceo" in ua and "pay_request" in ua:
        return "contact_bank_and_block_cards", "payment_status", "ceo_fraud"
    if "bank_change" in ua:
        if "verified_by_phone" in ua or "known_number" in ua:
            return "no_action_needed", None, "verified_bank_change"
        return "verify_via_official_channel", None, "bank_change"
    if "voice_call" in ua and "no_request" in ua and "bank" in ua:
        return "verify_via_official_channel", None, "bank_call_asked_nothing"
    if "voice_call" in ua and not ("code" in ua or "pay_request" in ua or "bank_login" in ua):
        if "negated" in u or "remote" in ua:
            return "do_not_interact_and_delete", None, "hung_up"
        if warning in ("possible_spam", "none", None):
            return "no_action_needed", None, "genuine_call"
    if "voice_call" in ua and "no_request" in ua:
        return "verify_via_official_channel", None, "bank_call_asked_nothing"
    if "family" in ua and "pay_request" in ua:
        return "verify_via_official_channel", None, "family_money_request"
    if "injection" in a:
        return "verify_via_official_channel", None, "prompt_injection"
    if a & {"prize", "investment"} or ("parcel" in a and "pay_request" in a):
        return "do_not_interact_and_delete", None, "lure"
    if "password" in a and has_links and "did_enter" not in u:
        return "verify_via_official_channel", None, "unrequested_reset"
    if "bank_login" in a and "link" in a:
        return "verify_via_official_channel", None, "credential_lure"
    if verdict in ("malicious", "suspicious"):
        return "verify_via_official_channel", None, "bad_link"
    if described:
        if a & {"link", "pay_request", "card", "code", "password", "bank_login", "install_req"}:
            return "verify_via_official_channel", None, "described_lure"
        return None, None, "no_rule"
    if not has_links and "bank_login" not in a and not ("pay_request" in a and "urgency" in a):
        return "no_action_needed", None, "no_link_no_request"
    if verdict == "clean" and own:
        return "no_action_needed", None, "own_clean_link"
    if "pay_request" in a or "card" in a:
        return "verify_via_official_channel", None, "payment_lure"
    return None, None, "no_rule"


# Generic playbook per rule reason: unconditional steps, steps that apply only if
# the riskier answer to the decisive question is true, and the hand-off.
# "auto" = security_specialist for consumers, business_security_support for SMBs.
REASON_PLAYBOOK = {
    "out_of_scope": {"risk": "medium", "steps": ["decline_request", "report_police"]},
    "monitoring": {"risk": "high", "nec": ['device.recent_app_installs'], "steps": ["plan_before_removing", "seek_support_org"], "esc": "human_advisor"},
    "ransomware": {"risk": "critical", "steps": ["disconnect_device", "dont_pay_or_transfer", "get_device_checked"], "esc": "auto"},
    "safe_account": {"risk": "critical", "steps": ["contact_bank_now", "report_police"], "esc": "human_advisor"},
    "own_login_prompt": {"risk": "none", "steps": ["explain_benign"]},
    "prompt_storm": {"risk": "critical", "steps": ["dont_approve_prompts", "change_password", "enable_mfa"]},
    "own_sim_change": {"risk": "none", "steps": ["explain_benign"]},
    "sim_takeover": {"risk": "critical", "nec": ['subscription.recent_events'], "steps": [], "cond": ["contact_operator", "lock_subscription"], "safe_cond": True, "esc": "operator_support_agent"},
    "unexpected_order": {"risk": "critical", "nec": ['subscription.recent_events', 'operator_account.recent_logins'], "steps": ["contact_operator"], "cond": ["change_password"], "esc": "business_security_support"},
    "own_order": {"risk": "none", "nec": ['subscription.recent_events'], "steps": ["explain_benign"]},
    "login_alert": {"risk": "high", "nec": ['operator_account.recent_logins'], "steps": ["review_sessions"], "cond": ["change_password", "enable_mfa"]},
    "breach_email_only": {"risk": "low", "steps": ["explain_benign"]},
    "fake_breach_notice": {"risk": "medium", "steps": ["dont_open_link", "verify_official_channel"]},
    "breach_passwords": {"risk": "high", "steps": ["change_password", "check_password_reuse", "enable_mfa"]},
    "offboarding": {"risk": "high", "nec": ['business.staff_accounts'], "steps": ["revoke_access", "change_password", "review_sessions"]},
    "mailbox_takeover": {"risk": "critical", "steps": ["change_password", "check_forwarding_rules", "inform_staff"], "esc": "business_security_support"},
    "booked_it_support": {"risk": "none", "steps": ["explain_benign"]},
    "remote_access": {"risk": "high", "nec": ['device.security_findings'], "steps": ["disconnect_device", "remove_app_or_profile", "get_device_checked"], "cond": ["contact_bank_now"], "esc": "auto"},
    "remote_request": {"risk": "high", "steps": ["verify_official_channel", "dont_share_codes"]},
    "terminal_support_call": {"risk": "high", "steps": ["verify_official_channel", "dont_share_codes"]},
    "installed_app": {"risk": "critical", "nec": ['device.recent_app_installs'], "steps": ["disconnect_device", "remove_app_or_profile", "avoid_banking_on_device", "contact_bank_now"], "esc": "auto"},
    "attachment_opened": {"risk": "high", "steps": ["dont_open_attachment"], "cond": ["disconnect_device", "get_device_checked"], "esc": "auto"},
    "code_given": {"risk": "critical", "steps": ["contact_bank_now", "dont_share_codes"]},
    "code_not_given": {"risk": "high", "steps": ["dont_share_codes", "verify_official_channel"]},
    "code_call": {"risk": "critical", "steps": ["dont_share_codes", "verify_official_channel"], "cond": ["contact_bank_now"]},
    "bank_details_entered": {"risk": "critical", "steps": ["contact_bank_now", "block_card", "check_statements"]},
    "password_entered": {"risk": "high", "steps": ["change_password", "check_password_reuse", "enable_mfa"]},
    "link_opened_payment": {"risk": "critical", "steps": ["delete_or_report_message"], "cond": ["contact_bank_now", "block_card"]},
    "link_opened": {"risk": "low", "steps": ["delete_or_report_message", "dont_reply"]},
    "ceo_fraud": {"risk": "critical", "steps": ["dont_pay_or_transfer", "verify_out_of_band"], "cond": ["contact_bank_now"], "esc": "business_security_support"},
    "verified_bank_change": {"risk": "low", "steps": ["explain_benign"]},
    "remote_banking": {"risk": "critical", "nec": ['device.security_findings'], "steps": ["contact_bank_now", "disconnect_device", "remove_app_or_profile"], "esc": "auto"},
    "attachment_not_enabled": {"risk": "medium", "steps": ["dont_open_attachment", "delete_or_report_message"]},
    "ceo_paid": {"risk": "critical", "steps": ["contact_bank_now", "verify_out_of_band", "report_police"], "esc": "business_security_support"},
    "ceo_not_paid": {"risk": "high", "steps": ["dont_pay_or_transfer", "verify_out_of_band"]},
    "bank_change": {"risk": "high", "steps": ["dont_pay_or_transfer", "verify_out_of_band"]},
    "bank_call_asked_nothing": {"risk": "low", "steps": ["verify_official_channel"]},
    "hung_up": {"risk": "low", "steps": ["hang_up_dont_call_back"]},
    "genuine_call": {"risk": "none", "steps": ["explain_benign"]},
    "family_money_request": {"risk": "high", "steps": ["dont_pay_or_transfer", "verify_out_of_band"]},
    "prompt_injection": {"risk": "high", "steps": ["dont_open_link", "verify_official_channel"]},
    "lure": {"risk": "medium", "steps": ["dont_open_link", "dont_pay_or_transfer", "delete_or_report_message"]},
    "unrequested_reset": {"risk": "low", "steps": ["dont_open_link", "verify_official_channel"]},
    "credential_lure": {"risk": "high", "steps": ["dont_open_link", "verify_official_channel"]},
    "bad_link": {"risk": "high", "steps": ["dont_open_link", "verify_official_channel"]},
    "no_link_no_request": {"risk": "none", "steps": ["explain_benign"]},
    "own_clean_link": {"risk": "none", "steps": ["explain_benign"]},
    "described_lure": {"risk": "high", "steps": ["dont_open_link", "dont_pay_or_transfer", "verify_official_channel"]},
    "payment_lure": {"risk": "medium", "steps": ["dont_open_link", "dont_pay_or_transfer", "verify_official_channel"]},
}
CODE_CALL_OPERATOR = {"risk": "critical", "nec": ["subscription.recent_events"], "steps": ["dont_share_codes"], "cond": ["contact_operator", "lock_subscription"], "esc": "operator_support_agent"}


def playbook(reason, action):
    if reason == "code_call" and action == "contact_operator_support":
        return CODE_CALL_OPERATOR
    if reason == "code_given" and action == "contact_operator_support":
        return {"risk": "critical", "steps": ["contact_operator", "lock_subscription"], "esc": "operator_support_agent"}
    return REASON_PLAYBOOK.get(reason, {})


def rule_features(case, notes=()):
    act, topic, reason = expert(case, notes)
    f = {"RULE:reason=" + reason: 1.0}
    if act:
        f["RULE:action=" + act] = 1.0
    if topic:
        f["RULE_ASK:" + topic] = 1.0
    return f
