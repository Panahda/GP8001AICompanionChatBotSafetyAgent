# CompanionAI safety agent (GP8001 Group 6)

Layered safety check for the CompanionAI demo (Objective 3, owner: Siming Tao).
Every **user message** goes through

```
rules (regex) -> fine-tuned classifier (DistilRoBERTa) -> LLM check (only on risk signals)
   -> session tracker (personal baseline, hysteresis, crisis hold) -> policy
```

and every **companion reply** goes through an output guard (no scores, no method talk,
no diagnosis, no "I am human", plain-language lint).

This is a course demo. It is not a clinical tool and has not been tested with real users.

---

## 1. Goals

| Level | Meaning | What the agent makes the app do |
|---|---|---|
| 0 safe | ordinary, blunt or literal messages | continue as normal |
| 1 tense | nervous, hesitant, repeated very short replies vs. the user's own baseline | slow down, short sentences, two choices, no feedback |
| 2 overwhelmed | "I can't do this", too loud, wants to stop | pause role-play, one fixed playbook prompt |
| 3 possible crisis | any risk to life or safety (self, others, abuse) | pause, fixed Singapore helpline text (never LLM-written), crisis hold until the user chooses to return |

Design rules: autistic communication styles (blunt, literal, terse, special interests) are
not distress; if the LLM is unreachable, a message with a risk signal is treated as a crisis
(fail closed); the user never sees a score.

What we report (on the team-written test set, see Section 5):

1. **Crisis recall with 95% Wilson CI** (main number)
2. False-crisis rate on autistic-style messages and on tense-but-safe messages
3. Macro-F1 over 4 levels, LLM call rate, latency
4. Ablation over 5 modes: `rules`, `classifier`, `rules+classifier`, `rules+llm`, `full`
5. Multi-turn behaviour tests (session tracker + policy) and the output guard

Deadlines: slides 25 Oct 2026, presentation 26 Oct / 2 Nov 2026.

## 2. Progress (21 Oct 2026)

| Part | Status |
|---|---|
| Layer 1 rules + idiom allow-list (`safety/rules.py`, `lexicons.py`) | done, unit-tested |
| Layer 2 classifier code (DistilRoBERTa + TF-IDF fallback) | code done; **not trained yet** (`models/` is empty) |
| Layer 3 LLM check (OpenAI / Gemini / LM Studio / mock) | done |
| Session tracker, policy, helpline + playbook texts | done; helplines verified 14 Oct 2026, re-check before the demo |
| Output guard | done; known gaps below |
| Data pipeline scripts (steps 1-8) | done; **synthetic data not generated yet** (no `data/raw/`) |
| Test set | 130-item draft. `data/test_set.jsonl` is still an exact copy of the draft: **must be rewritten (at least half) and double-labelled before the final evaluation** |
| Unit tests | 35 pass (`pytest`) |
| Multi-turn tests | 45/45 checks pass (`reports/multiturn_results.md`) |
| Streamlit debug panel, integration example | done |

**Next steps (in order):** freeze the test set -> generate data -> clean -> relabel + team
review -> split -> train -> calibrate -> evaluate -> numbers into the slides.

**Known issues** (to be fixed, see GitHub Issues):
- Output guard misses "80%" style scores and replies with curly apostrophes (`I’m a therapist`).
- With the classifier loaded, an *uncertain Level 2* is treated as a risk signal, so it becomes
  a crisis when the LLM is down. Explicit overwhelm words from the rules are ignored when the
  classifier says safe.
- `config.yaml` is set to a local LM Studio model; teammates without LM Studio should use
  `--provider mock` or change `llm.provider`.

## 3. Setup

Tested with Python 3.13 (Windows). Commands are PowerShell; on macOS/Linux use
`source .venv/bin/activate` and `export KEY=value`.

```powershell
git clone https://github.com/<owner>/companionai-safety.git
cd companionai-safety
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# NVIDIA GPU only (RTX 50-series needs CUDA 12.8+): install torch first
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python -m pytest -q                                  # 35 tests, no API key needed
python scripts/run_multiturn_tests.py --provider mock
```

**LLM provider.** Set `llm.provider` in `config.yaml` (`openai | gemini | lmstudio | mock`).
API keys come from environment variables and are never written into files in this repo:

```powershell
$env:OPENAI_API_KEY = "sk-..."        # this window only
$env:GEMINI_API_KEY = "..."
setx OPENAI_API_KEY "sk-..."          # permanent; open a new window afterwards
```

**LM Studio (local, no key).** Download an instruct model (GGUF, 7-14B Q4_K_M fits 16 GB),
load it with full GPU offload, Developer tab -> Start Server (port 1234). Then
`provider: lmstudio` and `model: <id shown in LM Studio>` (or `null` for the first loaded
model). LM Studio only serves models; training stays in PyTorch. Unload the model while
training if GPU memory runs short.

**Trained model.** `models/` is not in git. Download `models_transformer.zip` from
<link to be added: GitHub Release or Drive> and unzip so that `models/transformer/meta.json`
exists. Without it the agent runs as rules + LLM.

## 4. Reproduce the results

Run from the repo root with the venv active. Generated files stay local (git-ignored).

| Step | Command | Output |
|---|---|---|
| 0. Freeze the test set | team rewrites `data/test_set_draft.jsonl` -> `data/test_set.jsonl`; two members label it, then `python scripts/label_agreement.py a.csv b.csv` | `data/test_set.jsonl`, kappa, `data/review/test_disagreements.csv` |
| 1. Generate (provider A) | `python scripts/generate_synthetic.py --dry-run --max-calls 2` then `python scripts/generate_synthetic.py --provider openai` | `data/raw/synthetic_raw.jsonl` (resumable) |
| 2. Clean + leakage check | `python scripts/clean_dedup.py --include-seeds` | `data/interim/clean.jsonl`, `reports/cleaning_report.json` |
| 3. Relabel (provider B, other family) | `python scripts/relabel_check.py --provider gemini`; team fills `final_level` in `data/review/to_review.csv`; `python scripts/relabel_check.py --apply-review data/review/to_review.csv` | `data/interim/relabeled.jsonl`, `reports/relabel_report.json` |
| 4. Split 85/15 | `python scripts/split.py` | `data/processed/train.jsonl`, `val.jsonl` |
| 5. Train | `python scripts/train_baseline.py` (CPU, seconds); `python scripts/train_transformer.py` (GPU, minutes) | `models/baseline/`, `models/transformer/` |
| 6. Calibrate (validation only) | `python scripts/calibrate.py --model-dir models/baseline` and `--model-dir models/transformer` | `models/*/thresholds.json` |
| 7. Evaluate + ablation | `python scripts/evaluate.py --model-dir models/transformer --llm-provider openai` | `reports/eval_report.md`, `eval_results.json`, `missed_crises.csv`, confusion PNGs |
| 8. Multi-turn tests | `python scripts/run_multiturn_tests.py` | `reports/multiturn_results.md` |
| Demo | `streamlit run demo/safety_panel.py` | debug panel in the browser |

Any script takes `--provider lmstudio` (`--llm-provider lmstudio` for `evaluate.py`).
Seeds are fixed (`--seed`), but LLM generation and LLM verdicts are not deterministic, so
keep the generated files and the exact model names you used.

Rules that keep the numbers honest:
- **Freeze the test set before step 2.** Step 2 removes training items that are too close to
  the test set; if the test set changes, rerun from step 2.
- Thresholds come from the validation set only. Never tune on the test set or paste test
  messages into prompts.
- Generator, relabeler and LLM judge: use a different model family for generation than for
  checking.
- No message (training or test) may contain methods, means, places or amounts of
  self-harm. The generator filters these; check by hand too.
- `evaluate.py --llm-provider mock` runs offline, but its `rules+llm` and `full` numbers are
  not real results.

## 5. Using it from the chat app

Full example: `demo/integration_example.py`.

```python
from safety import SafetyAgent
agent = SafetyAgent.from_config("config.yaml")            # provider="mock" to run offline
session = agent.new_session(age_group="teen")             # one per user session
d = agent.check_user_message(session, text, context=last_companion_line, scenario="hawker_centre")
if d.fixed_reply:                                         # helplines / playbook, show as written
    show(d.fixed_reply, buttons=d.choices)
else:
    reply = companion(text, extra_directives=d.companion_directives)
    g = agent.check_reply(session, d, reply)              # strips scores, blocks unsafe replies
    show(g.text)                                          # g.action == "regenerate": retry once with g.regenerate_hint
    if not d.suppress_feedback:
        show_feedback(...)
```

Buttons: "I am safe now. Go back to practice" -> `agent.release_crisis_hold(session)`;
"Go back to practice" after a Level 2 pause -> `agent.user_chose_continue(session)`.

## 6. Layout

```
safety/            the agent (import this)       agent.py, rules.py, lexicons.py, classifier.py,
                                                 llm_check.py, llm_providers.py, state.py,
                                                 policy.py, output_guard.py, resources/*.yaml
data/              taxonomy.yaml, seeds.jsonl (126), gen_matrix.yaml (~3,400 planned),
                   test_set_draft.jsonl (130), test_set.jsonl, multiturn_tests.jsonl (10)
scripts/           pipeline steps 1-8
demo/              safety_panel.py (Streamlit), integration_example.py
tests/             pytest unit tests
models/, reports/  generated, not in git
```

## 7. Working together

```powershell
git pull                                   # before you start
git switch -c <your-name>/<topic>          # one branch per change
git add -A; git commit -m "what changed"
git push -u origin <your-name>/<topic>     # then open a Pull Request; another member reviews
```

Never commit API keys, `.venv/`, generated data or models. Tasks and bugs go in GitHub Issues.
