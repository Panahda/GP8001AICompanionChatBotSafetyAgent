"""Older Streamlit version of the safety test page (optional; the main page is demo/web_demo.py).

    pip install -r demo/streamlit/requirements.txt
    streamlit run demo/streamlit/safety_panel.py

Type user messages and see what the safety check decided. This page has no companion
LLM: where the companion would reply, it shows the instructions it would get.
The real chat app calls the same two functions, see demo/integration_example.py.

Online (Streamlit Community Cloud): put the API key in the app's Secrets, e.g.
    OPENAI_API_KEY = "sk-..."
Root-level secrets become environment variables, which is where the providers look.
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from safety.agent import SafetyAgent  # noqa: E402

st.set_page_config(page_title="CompanionAI safety check", layout="wide")

LEVELS = {0: "0 · safe", 1: "1 · tense", 2: "2 · overwhelmed", 3: "3 · possible crisis"}
ACTIONS = {
    "continue": "Carry on as normal.",
    "slow_down": "Slow down: short sentences, one question, offer a break.",
    "pause_regulate": "Pause the role-play and offer one calming prompt.",
    "crisis_pause": "Pause and show the Singapore helplines.",
    "crisis_support": "Stay in support mode until the user says they are safe.",
}
KEYS = {"openai": ("OPENAI_API_KEY",), "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY")}


def provider_options() -> list[str]:
    """Default order: a provider with an API key (the online page), else the one in config.yaml
    (e.g. LM Studio on your own PC), else mock."""
    with_key = [p for p, names in KEYS.items() if any(os.environ.get(n) for n in names)]
    try:
        in_file = (yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")).get("llm") or {}).get("provider")
    except Exception:
        in_file = None
    order = with_key + ([in_file] if in_file else []) + ["mock", "lmstudio", *KEYS]
    return list(dict.fromkeys(p for p in order if p))


@st.cache_data(ttl=30, show_spinner=False)
def lmstudio_problem(base_url: str) -> str | None:
    """None if the LM Studio server answers and has a model loaded, else what is wrong."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=3) as r:
            models = [m["id"] for m in json.load(r).get("data", [])]
    except Exception:
        return f"LM Studio is not reachable at {base_url}. Open LM Studio, load a model, then Developer -> Start Server."
    return None if models else "LM Studio is running but no model is loaded."


@st.cache_resource
def load_agent(provider: str, mode: str) -> SafetyAgent:
    return SafetyAgent.from_config(ROOT / "config.yaml", provider=provider, mode=mode)


with st.sidebar:
    st.header("Settings")
    scenario = st.selectbox("Scenario", ["hawker_centre", "new_colleague", "job_interview", "free_chat"])
    age = st.radio("User", ["teen", "adult"], horizontal=True)
    consent = st.checkbox("Adult agreed to carer alerts", value=False, disabled=age == "teen")
    if st.button("Start a new conversation"):
        st.session_state.clear()
    with st.expander("Developer settings"):
        provider = st.selectbox("LLM check", provider_options(), index=0,
                                help="mock = offline stand-in, lmstudio = local server on this computer")
        mode = st.selectbox("Mode", ["full", "rules+classifier", "rules+llm", "classifier", "rules"], index=0)

agent = load_agent(provider, mode)

st.title("CompanionAI safety check")
st.info("Type a message as if you were the user. On the right you see the safety level "
        "(0 safe to 3 possible crisis) and what the app would do. "
        "Please type made-up messages only, never real personal stories.")
if agent.llm is None or agent.llm.provider.name == "mock":
    st.warning("The LLM check is not connected (offline test mode), so unclear messages are judged roughly. "
               "Tell the person who runs this page.")
elif agent.llm.provider.name == "lmstudio":
    problem = lmstudio_problem(str(agent.llm.provider.client.base_url))
    if problem:
        st.error(problem + " Until then, unclear messages are treated as a crisis (fail-safe).")
st.caption(f"classifier: {'loaded (' + agent.clf.clf.name + ')' if agent.clf else 'not loaded'} · "
           f"LLM: {agent.llm.provider.name + '/' + agent.llm.provider.model if agent.llm else 'none'}")

if "session" not in st.session_state:
    st.session_state.session = agent.new_session(age_group=age, carer_alert_consent=consent)
    st.session_state.log = []
    st.session_state.companion_line = ""
s = st.session_state.session
s.age_group, s.carer_alert_consent = age, consent

left, right = st.columns([3, 2])
with left:
    st.subheader("Conversation")
    st.session_state.companion_line = st.text_input(
        "What the companion said last (optional)", st.session_state.companion_line,
        placeholder="e.g. Hello! What would you like to order?")
    for role, text in st.session_state.log:
        with st.chat_message(role):
            st.markdown(text)
    msg = st.chat_input("Type as the user")

if msg:
    d = agent.check_user_message(s, msg, context=st.session_state.companion_line, scenario=scenario)
    st.session_state.last = d
    st.session_state.log.append(("user", msg))
    if d.fixed_reply:
        st.session_state.log.append(("assistant", d.fixed_reply.replace("\n", "  \n")))
    else:
        extra = "  \n".join(f"- {x}" for x in d.companion_directives) or "- (normal reply)"
        st.session_state.log.append(("assistant", f"*[the companion would reply with these instructions]*  \n{extra}"))
    st.rerun()

with right:
    d = st.session_state.get("last")
    st.subheader("What the safety check decided")
    st.metric("Conversation level", LEVELS[int(s.level)], help="Stays up for a few calm messages before it goes down")
    if d:
        st.write(f"**This message:** {LEVELS[int(d.message_level)]}")
        st.write(f"**The app would:** {ACTIONS.get(d.action, d.action)}")
        st.caption(f"action `{d.action}` · category {d.category or '-'} · LLM asked: {d.llm_called} · "
                   f"carer alert: {d.carer_alert}")
        if d.choices:
            st.write("Buttons shown to the user:")
            cols = st.columns(len(d.choices))
            for c, label in zip(cols, d.choices):
                if c.button(label):
                    if label.startswith("I am safe now"):
                        agent.release_crisis_hold(s)
                    elif label in ("Go back to practice", "Keep going", "I did it", "I am back"):
                        agent.user_chose_continue(s)
                    st.session_state.log.append(("user", f"*[button] {label}*"))
                    st.rerun()
        st.write("Details (for developers):")
        for layer in d.layers:
            with st.expander(f"{layer.layer}: level {None if layer.level is None else int(layer.level)}"
                             f"{'  (decisive)' if layer.decisive else ''}{'  → asks LLM' if layer.wants_llm else ''}"):
                if layer.probs:
                    st.bar_chart(pd.Series(layer.probs, name="probability"))
                if layer.reasons:
                    st.write(layer.reasons)
                if layer.error:
                    st.error(layer.error)
        if d.notes:
            with st.expander("Notes"):
                st.write(d.notes)

    st.subheader("Check a companion reply")
    reply = st.text_area("Paste a reply to see what the output guard does",
                         "Good job! That was 8/10. Can you say it again slower?")
    if st.button("Check reply"):
        # A fixed reply (helplines / playbook) is never checked, so test against a normal turn in that case.
        use = d if d is not None and d.fixed_reply is None else None
        g = agent.guard.check(reply, use)
        st.write(f"action: `{g.action}` · ok: {g.ok}")
        st.code(g.text)
        if g.violations or g.warnings:
            st.write({"violations": g.violations, "warnings": g.warnings, "hint": g.regenerate_hint})
