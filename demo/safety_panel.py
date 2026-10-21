"""Safety agent debug panel (for testing and for the live demo).

    streamlit run demo/safety_panel.py

Type user messages and see what every layer decided. This page has no companion
LLM: where the companion would reply, it shows the directives it would receive.
The real chat app (built by C) calls the same two functions, see integration_example.py.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safety.agent import SafetyAgent  # noqa: E402
from safety.schema import Action  # noqa: E402

st.set_page_config(page_title="CompanionAI safety panel", layout="wide")


@st.cache_resource
def load_agent(provider: str, mode: str) -> SafetyAgent:
    return SafetyAgent.from_config(ROOT / "config.yaml", provider=provider, mode=mode)


with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("LLM check", ["mock", "lmstudio", "openai", "gemini"], index=0)
    mode = st.selectbox("Mode", ["full", "rules+classifier", "rules+llm", "classifier", "rules"], index=0)
    scenario = st.selectbox("Scenario", ["hawker_centre", "new_colleague", "job_interview", "free_chat"])
    age = st.radio("User", ["teen", "adult"], horizontal=True)
    consent = st.checkbox("Adult agreed to carer alerts", value=False, disabled=age == "teen")
    if st.button("New session"):
        st.session_state.clear()

agent = load_agent(provider, mode)
st.caption(f"classifier: {'loaded (' + agent.clf.clf.name + ')' if agent.clf else 'not trained yet'} · "
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
    st.session_state.companion_line = st.text_input("Companion's previous line (context)", st.session_state.companion_line)
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
        st.session_state.log.append(("assistant", f"*[companion reply generated with directives]*  \n{extra}"))
    st.rerun()

with right:
    d = st.session_state.get("last")
    st.subheader("Safety decision")
    st.metric("Session level", int(s.level), help="0 safe · 1 tense · 2 overwhelmed · 3 crisis")
    if d:
        st.write(f"**Message level:** {int(d.message_level)} · **Action:** `{d.action}` · **Category:** {d.category or '-'}"
                 f" · **LLM called:** {d.llm_called} · **Carer alert:** {d.carer_alert}")
        for layer in d.layers:
            with st.expander(f"{layer.layer}: level {None if layer.level is None else int(layer.level)}"
                             f"{'  (decisive)' if layer.decisive else ''}{'  → asks LLM' if layer.wants_llm else ''}", expanded=True):
                if layer.probs:
                    st.bar_chart(pd.Series(layer.probs, name="probability"))
                if layer.reasons:
                    st.write(layer.reasons)
                if layer.error:
                    st.error(layer.error)
        if d.notes:
            st.info("\n".join(d.notes))
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

    st.subheader("Check a companion reply")
    reply = st.text_area("Paste a reply to run the output guard", "Good job! That was 8/10. Can you say it again slower?")
    if st.button("Check reply"):
        # A fixed reply (helplines / playbook) is never checked, so test against a normal turn in that case.
        use = d if d is not None and d.fixed_reply is None else None
        g = agent.guard.check(reply, use)
        st.write(f"action: `{g.action}` · ok: {g.ok}")
        st.code(g.text)
        if g.violations or g.warnings:
            st.write({"violations": g.violations, "warnings": g.warnings, "hint": g.regenerate_hint})
