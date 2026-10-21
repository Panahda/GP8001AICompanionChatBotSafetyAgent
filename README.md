# CompanionAI safety check

Safety layer for the GP8001 CompanionAI demo. It reads every user message, gives it a level
from 0 to 3, tells the chat app what to do, and checks the companion's reply before it is shown.

**Contents**

- [For users](#for-users) (no coding)
  - [Try the test page](#try-the-test-page)
  - [Levels](#levels)
  - [Data tasks](#data-tasks)
- [For developers](#for-developers)
  - [Status](#status)
  - [Set up](#set-up)
  - [Use it in the chat app](#use-it-in-the-chat-app)
  - [Run the pipeline](#run-the-pipeline)
  - [Put the test page online](#put-the-test-page-online)
  - [Git and repo layout](#git-and-repo-layout)

> Use made-up messages only, and never write methods, means, places or amounts of self-harm.
> Messages may be sent to an outside AI service to be checked.

---

## For users

### Try the test page

Open the test page link and sign in with the email you were invited with (Google sign-in or a
one-time email link). Type a message as the user; the right side shows the level and what the
app would do.

Worth trying: blunt or very short replies ("No.", "Why?"), figures of speech ("I'd kill for
chicken rice"), Singlish and typos, nervous or overloaded messages, worry about a friend, and a
role-play answer followed by something real. If a result looks wrong, share a screenshot with
what you typed and what you expected.

### Levels

Start at 3 and stop at the first yes. Unsure between two? Pick the higher one.

| Level | When | The app | Subtypes |
|---|---|---|---|
| 3 possible crisis | any sign of risk to life or safety (self, others, being hurt) | pauses, shows Singapore helplines | `cssrs_1_wish_dead`, `cssrs_2_active_thoughts`, `cssrs_3_method`, `cssrs_4_intent`, `cssrs_5_plan`, `cssrs_6_behaviour`, `nssi` (self-harm), `harm_others`, `others_at_risk` (worried about someone), `unsafe_abuse` |
| 2 overwhelmed | too much, too loud, wants to stop, shutting down | pauses the role-play, offers one calming prompt | `overwhelmed_overload`, `overwhelmed_shutdown` |
| 1 tense | nervous, hesitant, asking if it's OK | slows down, offers a break | `tense_explicit`, `tense_hesitant` |
| 0 safe | everything else | carries on | `ordinary_scenario`, `blunt_literal`, `idiom_negative` (figure of speech), `meta_feedback`, `special_interest` |

Blunt, literal, short, no small talk, or long talk about a favourite topic is **0**, not
distress. In a role-play, words that could be about real life count as real. Full definitions:
`data/taxonomy.yaml`.

### Data tasks

Done in Excel or Google Sheets and saved as **CSV UTF-8** (Excel: Save As -> CSV UTF-8; Google
Sheets: Download -> CSV). Sheets are checked when converted back, so typos in `level` or
`subtype` get caught.

- **Rewrite the test set.** Rewrite at least half the messages in your own words; add new ones
  at the bottom with `id` empty. Keep the mix at about 50 crisis, 50 safe, 30 tense or
  overwhelmed. Fill `text`, `level`, `subtype`, `scenario` (`hawker_centre`, `new_colleague`,
  `job_interview`, `free_chat`) and `author`.
- **Double labelling.** Two people label the same sheet (empty `level` column) without talking,
  then compare and settle the differences.
- **Review training data** (`to_review.csv`). Two AI models disagreed on these. Fill
  `final_level` with 0-3, or `drop` if useless; a blank row keeps its label.

---

## For developers

### Status

Built: rules, classifier code, LLM check, conversation tracker, helpline and calming texts,
reply check, test page and a one-command pipeline; 36 unit tests and 45/45 multi-turn checks
pass. Not yet: training data, trained classifier, final test set (`data/test_set.jsonl` is still
the draft).

Known bugs:
- The reply check misses "80%" scores and curly apostrophes (`I’m a therapist`).
- LLM down + classifier unsure at level 2 -> treated as crisis.
- A classifier "safe" overrides clear overload words ("too loud, I can't think").

### Set up

Windows PowerShell, Python 3.13 (macOS/Linux: `source .venv/bin/activate`, `export KEY=value`).

```powershell
git clone https://github.com/Taosiming2017/GP8001AICompanionChatBotSafetyAgent.git
cd GP8001AICompanionChatBotSafetyAgent
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # blocked? Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt   # NVIDIA GPU: first `pip install torch --index-url https://download.pytorch.org/whl/cu128`
python -m pytest -q               # no key needed
```

Only using the agent in the app? `pip install numpy pyyaml openai pytest` is enough
(torch/transformers are for the classifier, scikit-learn for the pipeline, streamlit for the
test page).

- **LLM:** `llm.provider` in `config.yaml` (`openai | gemini | lmstudio | mock`), or pass
  `provider=` / `--provider`. The default is `lmstudio`, a local server. Keys go in environment
  variables, never in files: `setx OPENAI_API_KEY "sk-..."` (or `GEMINI_API_KEY`), then open a
  new window. `mock` is offline and only tests the plumbing.
- **Model:** `models/` isn't in git. Without `models/transformer` the agent runs rules + LLM.

### Use it in the chat app

```python
from safety import SafetyAgent
agent = SafetyAgent.from_config("config.yaml", provider="openai")
session = agent.new_session(age_group="teen")             # one per user session
d = agent.check_user_message(session, text, context=last_companion_line, scenario="hawker_centre")
if d.fixed_reply:                                         # helplines / calming prompt: show as is
    show(d.fixed_reply, buttons=d.choices)
else:
    reply = companion(text, extra_directives=d.companion_directives)
    g = agent.check_reply(session, d, reply)              # strips scores, blocks unsafe replies
    show(g.text)                                          # g.action == "regenerate": retry once with g.regenerate_hint
    if not d.suppress_feedback:
        show_feedback(...)
```

Buttons: "I am safe now. Go back to practice" -> `agent.release_crisis_hold(session)`; "Go back
to practice" after a level 2 pause -> `agent.user_chose_continue(session)`. Full example:
`demo/integration_example.py`. Test page locally: `streamlit run demo/safety_panel.py`.

### Run the pipeline

```powershell
python scripts/run_flow.py --mode quick --provider mock --skip-transformer   # smoke test in minutes, writes runs/quick/ only
python scripts/run_flow.py --mode full --provider lmstudio                   # real run, can take hours
python scripts/run_flow.py --mode full --start-at transformer                # resume from a step
```

Full mode runs `generate -> clean -> relabel -> review -> split -> baseline -> transformer ->
evaluate -> multiturn`, writes to `data/`, `models/` and `reports/`, backs up old outputs to
`runs/archive/`, and pauses when you need to act (switch models in LM Studio, review the sheet,
free GPU memory). Each step is also its own script in `scripts/`.

Spreadsheets for the data tasks:

```powershell
python scripts/sheet.py to-csv data/test_set_draft.jsonl test_set_edit.csv       # rewrite sheet
python scripts/sheet.py to-csv data/test_set.jsonl labels_a.csv --blank-labels    # labelling sheet, one per person
python scripts/label_agreement.py labels_a.csv labels_b.csv                      # compare labels
python scripts/sheet.py to-jsonl test_set_edit.csv data/test_set.jsonl           # import, with checks
```

The review sheet is `data/review/to_review.csv`; full mode pauses for it.

Keep the numbers honest:
- Freeze the test set before the full run. Cleaning drops training messages that look like test
  messages, so if the test set changes, rerun from `clean`.
- Tune thresholds on validation data only; never put test messages in prompts.
- Check labels with a different model family from the one that generated the data.
- `mock` results for `rules+llm` and `full` aren't real.

### Put the test page online

1. On [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, **Create app**, and
   pick this repo, branch `main`, file `demo/safety_panel.py`.
2. In **Advanced settings**, pick Python 3.13 and add to **Secrets**: `OPENAI_API_KEY = "sk-..."`.
3. Deploy. An app from a private repo is private; invite viewers by email with **Share**.

It installs `demo/requirements.txt` (no torch), so it runs rules + LLM without the classifier. A
free account gets one private app. Set a spending limit on the API key.

### Git and repo layout

```powershell
git pull
git switch -c <name>/<topic>
git add -A; git commit -m "what changed"
git push -u origin <name>/<topic>      # then open a Pull Request
```

`.gitignore` is a whitelist: only `.py .json .jsonl .yaml .yml .md requirements*.txt` get in,
and `.venv/`, keys, `data/raw|interim|processed|review/`, `models/*/`, `reports/` and `runs/`
always stay out. For a new file type, add `!*.ext` to `.gitignore` or `git add -f <file>`.

```
safety/   agent.py (entry), rules.py + lexicons.py, classifier.py, llm_check.py, llm_providers.py,
          state.py, policy.py, output_guard.py, resources/ (helplines, calming prompts)
data/     taxonomy.yaml, seeds.jsonl, gen_matrix.yaml, test_set_draft.jsonl, test_set.jsonl,
          multiturn_tests.jsonl
scripts/  run_flow.py (whole pipeline), sheet.py (spreadsheets), one script per step
demo/     safety_panel.py (test page), integration_example.py, requirements.txt (online page)
tests/    unit tests
```
