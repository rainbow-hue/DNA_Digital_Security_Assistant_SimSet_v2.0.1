# Digital Security Assistant v3: context-aware, multi-turn

Everything runs from the package root (`DNA_Digital_Security_Assistant_SimSet_v2.0.1/`).
Only the package's own data is used: dev cases, vocabularies, guidance corpus, and the evaluator and simulator.

## Files

| File | What it does |
|---|---|
| `assistant/knowledge.py` | User-facing fi/en/sv sentences for every step, question, consent request and hand-off. Each one is checked against the package's own step, topic and unsafe-claim regexes. Also holds the multilingual concept lexicon and the playbooks for guidance topics with no dev cases. |
| `assistant/features.py` | Turns a case plus the conversation into features: words, concepts, the vocabulary regexes applied to the artifact, signals, guidance retrieval, expert-rule votes and answer polarity. |
| `assistant/expert.py` | Generic proportionate-response rules (about 45 "reasons"). Each reason has a playbook: steps, steps that only apply in the riskier case, risk level, needed consents and hand-off. |
| `assistant/model.py` | numpy-only softmax and one-vs-rest heads. About 1 MB; small enough to ship. |
| `assistant/train.py` | 80/20 grouped split, 5-fold family CV, final training, and evaluation with `dsakit.scoring` plus the simulator, next to the rules baseline. |
| `assistant/engine.py` | One brain for single-turn, multi-turn and the chat app: model → expert → playbooks → safety rules → optional Gemini → validated text. |
| `assistant/gemini_tier.py` | Optional Tier 3 with a system prompt. Only used when the local model is unsure; its output is validated against the vocabularies. |
| `assistant/chat.py` | Chat sessions for the mobile app. Turns free messages into simulator protocol events. |
| `assistant/predict.py` | Batch single-turn predictions → `test_predictions.jsonl`. |
| `assistant/sim_bot.py` | `step()` for `sim/run_conversation.py`. |
| `api_server.py` | FastAPI wrapper for the app. |

## Steps

```bash
# 0. install (numpy is enough for steps 1-4)
pip install -r requirements-assistant.txt

# 1. honest check: 80/20 split grouped by SCENARIO FAMILY (held-out families never seen in training)
python -m assistant.train --split family --seed 42

# 2. more stable: 5-fold family cross-validation (single-turn + multi-turn vs rules baseline)
python -m assistant.train --cv 5 --seed 7

# 3. seen-family check: 80/20 split that keeps each twin pair together
python -m assistant.train --split twin --seed 42

# 4. final model on 100 % of dev -> assistant/artifacts/dsa_model.{npz,json}
python -m assistant.train --final

# 5. submission file + package validator
python -m assistant.predict --cases data/test/inputs.jsonl --out test_predictions.jsonl
python tools/validate.py --submission test_predictions.jsonl

# 6. multi-turn against the simulated user, scored by the official evaluator
python sim/run_conversation.py --cases data/dev/cases.jsonl --bot-py assistant/sim_bot.py:step --out mt_dev.jsonl --show --limit 10
python eval/evaluate.py --cases data/dev/cases.jsonl --submission mt_dev.jsonl

# 7. the app backend (Gemini is optional)
set GEMINI_API_KEY=...            # PowerShell: $env:GEMINI_API_KEY="..."
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

Why the final model is trained on 100%: the 80/20 split (and the CV) is how you *measure* overfitting and choose settings. Once you've chosen, retraining on all the labelled data is standard practice. Report the split/CV numbers as your estimate, not the score on data the model was trained on.

## Mobile app contract (`POST /api/chat`)

Request: `{"message": "...", "session_id": "<from last reply or omit>", "language": "fi|en|sv (optional)", "signals": {...optional...}, "pre_granted": [...], "consent": {"granted": [...]}}`

Response: `session_id`, `reply_type` (`question` | `consent_request` | `final`), `display_text` (show this), `question{text,topic}`, `consent_request{fields,text}`, and `answer`. On `final`, `answer` is exactly the assignment's submission object: `risk_level, primary_action, advice_text, escalation_required, escalation_target, urgency, steps, questions_asked, question_topics, signals_used, consent_requested_for, citations, confidence, abstained`.

UI suggestions: render `consent_request` with Allow / Don't allow buttons. On `final`, show `steps` as a checklist and a "Talk to a person" button when `escalation_required`. Always send back `session_id`. A new problem means `reset: true`.

## Safety design

- Consent-gated signals stay hidden until the user agrees. A refusal still gets useful advice.
- Every answer is re-checked with the package's own unsafe-statement detector before it is returned; flagged sentences are dropped.
- When the decisive fact is unknown, the answer covers the riskier case. "Only if X" steps are written as one conditional clause so they never read as a general instruction.
- Text inside a forwarded message is treated as untrusted data. Prompt injection can never make a message "benign".
- Gemini never writes user-facing text. It only proposes vocabulary IDs, which are validated before use.
