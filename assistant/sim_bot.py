# -*- coding: utf-8 -*-
"""Plug the assistant into the package's multi-turn simulator:

  python sim/run_conversation.py --cases data/dev/cases.jsonl \
      --bot-py assistant/sim_bot.py:step --out mt.jsonl --show --limit 10
  python eval/evaluate.py --cases data/dev/cases.jsonl --submission mt.jsonl
"""
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from assistant.engine import get_engine  # noqa: E402


def step(event, state):
    reply = get_engine().step(event, state)
    if reply.get("type") == "final":
        ans = dict(reply["answer"])
        ans.pop("_tier", None)
        ans.pop("_rules", None)
        reply = {"type": "final", "answer": ans}
    return reply
