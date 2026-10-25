"""Streamlit interactive safety panel and conversation practice demo for CompanionAI.

    pip install -r demo/streamlit/requirements.txt
    streamlit run demo/streamlit/safety_panel.py
    # or: streamlit run demo/streamlit/app.py

Ethical Disclaimer:
    CompanionAI is an educational social practice tool, non-clinical and not a human or therapist.
    Please use fictional, made-up messages only.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import streamlit as st
except ImportError:
    st = None

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safety.agent import SafetyAgent  # noqa: E402
from safety.companion import CompanionAgent  # noqa: E402
from safety.env import load_env  # noqa: E402
from safety.feedback import FeedbackAgent  # noqa: E402
from safety.orchestrator import TurnResult, handle_turn  # noqa: E402
from safety.scenarios import Scenario, get_scenario, list_scenarios  # noqa: E402

load_env()

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
    """Default order: a provider with an API key, else from config.yaml, else mock."""
    with_key = [p for p, names in KEYS.items() if any(os.environ.get(n) for n in names)]
    try:
        in_file = (yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")).get("llm") or {}).get("provider")
    except Exception:
        in_file = None
    order = with_key + ([in_file] if in_file else []) + ["mock", "lmstudio", *KEYS]
    return list(dict.fromkeys(p for p in order if p))


def lmstudio_problem(base_url: str) -> str | None:
    """None if the LM Studio server answers and has a model loaded, else what is wrong."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=3) as r:
            models = [m["id"] for m in json.load(r).get("data", [])]
    except Exception:
        return f"LM Studio is not reachable at {base_url}. Open LM Studio, load a model, then Developer -> Start Server."
    return None if models else "LM Studio is running but no model is loaded."


def load_agent(provider: str, mode: str) -> SafetyAgent:
    if st is not None and hasattr(st, "cache_resource"):
        @st.cache_resource
        def _get_agent(p: str, m: str) -> SafetyAgent:
            return SafetyAgent.from_config(ROOT / "config.yaml", provider=p, mode=m)
        return _get_agent(provider, mode)
    return SafetyAgent.from_config(ROOT / "config.yaml", provider=provider, mode=mode)


def render_safety_panel(
    agent: Optional[SafetyAgent] = None,
    session: Optional[Any] = None,
    last: Optional[Any] = None,
) -> None:
    """Renders the interactive safety inspector and scenario practice panel in Streamlit."""
    if st is None:
        return

    st.set_page_config(page_title="CompanionAI conversation practice & safety check", layout="wide")

    # 1. Ethical Disclaimer Banner (Non-clinical AI practice tool notice)
    st.info(
        "⚠️ **Notice & Disclaimer:** CompanionAI is an educational social practice tool. "
        "It is an AI, not a human, and not a therapist or clinical counselor. "
        "This is a non-clinical practice demo — please use fictional or made-up messages only."
    )

    with st.sidebar:
        st.header("Practice Settings")
        available_scenarios = [s.id for s in list_scenarios()]
        if not available_scenarios:
            available_scenarios = ["hawker_centre", "new_colleague", "job_interview", "free_chat"]
        scenario_id = st.selectbox("Scenario", available_scenarios, index=0)

        age = st.radio("User Age Group", ["teen", "adult"], horizontal=True)
        consent = st.checkbox("Adult agreed to carer alerts", value=False, disabled=age == "teen")

        if st.button("Start a new conversation"):
            st.session_state.clear()
            st.rerun()

        with st.expander("Developer settings"):
            provider = st.selectbox(
                "LLM check",
                provider_options(),
                index=0,
                help="mock = offline stand-in, lmstudio = local server on this computer",
            )
            mode = st.selectbox(
                "Mode",
                ["full", "rules+classifier", "rules+llm", "classifier", "rules"],
                index=0,
            )

        st.markdown("---")
        end_practice = st.button("End Practice & View Transfer Prompt")

    if agent is None:
        agent = load_agent(provider, mode)

    sc_obj = get_scenario(scenario_id)

    # Scenario details card
    with st.expander(f"📖 Scenario Info: {sc_obj.title}", expanded=True):
        st.write(f"**Setting:** {sc_obj.setting}")
        st.write(f"**Role Played by AI:** {sc_obj.companion_role}")
        st.write(f"**Your Communication Goal:** {sc_obj.user_goal}")
        st.markdown("**Suggested Example Replies (Ideas to try):**")
        for ex in sc_obj.example_replies:
            st.markdown(f"- *\"{ex.text}\"* — {ex.explanation}")

    if end_practice:
        st.success(f"🌟 **Real-Life Practice Recommendation:** {sc_obj.real_life_prompt}")

    # Provider status banner
    if agent.llm is None or (hasattr(agent.llm, "provider") and agent.llm.provider.name == "mock"):
        st.warning(
            "The LLM check is in offline mock mode (stand-in). Results are deterministic for testing."
        )
    elif hasattr(agent.llm, "provider") and agent.llm.provider.name == "lmstudio":
        problem = lmstudio_problem(str(agent.llm.provider.client.base_url))
        if problem:
            st.error(problem + " Until then, unclear messages are treated as a crisis (fail-safe).")

    st.caption(
        f"Classifier: {'loaded (' + agent.clf.clf.name + ')' if agent.clf else 'not loaded'} · "
        f"LLM: {agent.llm.provider.name + '/' + agent.llm.provider.model if agent.llm else 'none'}"
    )

    # Session State management
    if "session" not in st.session_state or st.session_state.get("current_scenario") != scenario_id:
        st.session_state.session = agent.new_session(age_group=age, carer_alert_consent=consent)
        st.session_state.current_scenario = scenario_id
        st.session_state.log = [("assistant", sc_obj.opening_line)]
        st.session_state.companion_line = sc_obj.opening_line
        st.session_state.last = None

    s = st.session_state.session
    s.age_group, s.carer_alert_consent = age, consent

    # Orchestrator agents
    llm_prov = agent.llm.provider if agent.llm else None
    companion_agent = CompanionAgent(provider=llm_prov, safety_agent=agent)
    feedback_agent = FeedbackAgent(provider=llm_prov)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Conversation")
        st.session_state.companion_line = st.text_input(
            "What the companion said last (context)",
            st.session_state.companion_line,
            placeholder="e.g. Hello! Eat here or take away?",
        )

        for role, text in st.session_state.log:
            with st.chat_message(role):
                st.markdown(text)

        with st.form("chat_form", clear_on_submit=True):
            f_col1, f_col2 = st.columns([5, 1])
            user_input_box = f_col1.text_input("Type as the user", placeholder="Type your message and press Enter or Send...", label_visibility="collapsed")
            send_clicked = f_col2.form_submit_button("Send", use_container_width=True)

    chat_bar_msg = st.chat_input("Or type as the user here...")
    msg = (user_input_box.strip() if send_clicked and user_input_box else None) or chat_bar_msg

    if msg:
        history = [
            {"role": "user" if r == "user" else "companion", "text": t}
            for r, t in st.session_state.log
            if not t.startswith("💡") and not t.startswith("*[button]")
        ]
        turn = handle_turn(
            agent,
            session=s,
            user_text=msg,
            last_companion_line=st.session_state.companion_line,
            scenario=sc_obj,
            companion_agent=companion_agent,
            feedback_agent=feedback_agent,
            history=history,
        )
        st.session_state.last = turn.decision
        st.session_state.companion_line = turn.reply
        st.session_state.log.append(("user", msg))

        # Companion response bubble
        st.session_state.log.append(("assistant", turn.reply.replace("\n", "  \n")))

        # Feedback suggestion bubble on calm turns
        if turn.feedback:
            st.session_state.log.append(
                ("assistant", f"💡 **Coach suggestion (optional):** {turn.feedback}")
            )

        st.rerun()

    with right:
        d = st.session_state.get("last")
        st.subheader("What the safety check decided")
        st.metric(
            "Conversation level",
            LEVELS[int(s.level)],
            help="Stays up for a few calm messages before it goes down",
        )
        if d:
            st.write(f"**This message:** {LEVELS[int(d.message_level)]}")
            st.write(f"**The app would:** {ACTIONS.get(d.action, d.action)}")
            st.caption(
                f"Action: `{d.action}` · Category: {d.category or '-'} · "
                f"LLM asked: {d.llm_called} · Carer alert: {d.carer_alert}"
            )

            # Interactive action choices (e.g. "I am safe now", "Keep going", "Stop for today")
            buttons = list(d.choices)
            if buttons:
                st.write("Buttons shown to the user:")
                cols = st.columns(len(buttons))
                for c, label in zip(cols, buttons):
                    if c.button(label):
                        if label.startswith("I am safe now"):
                            agent.release_crisis_hold(s)
                        elif label in ("Go back to practice", "Keep going", "I did it", "I am back", "Keep talking"):
                            agent.user_chose_continue(s)
                        st.session_state.log.append(("user", f"*[button] {label}*"))
                        st.rerun()

            st.write("Details (for developers):")
            for layer in d.layers:
                layer_lvl = "–" if layer.level is None else int(layer.level)
                with st.expander(
                    f"{layer.layer}: level {layer_lvl}"
                    f"{'  (decisive)' if layer.decisive else ''}"
                    f"{'  → asks LLM' if layer.wants_llm else ''}"
                ):
                    if layer.probs and pd is not None:
                        st.bar_chart(pd.Series(layer.probs, name="probability"))
                    elif layer.probs:
                        st.write(layer.probs)
                    if layer.reasons:
                        st.write(layer.reasons)
                    if layer.error:
                        st.error(layer.error)
            if d.notes:
                with st.expander("Notes"):
                    st.write(d.notes)

        st.subheader("Check a companion reply")
        reply = st.text_area(
            "Paste a candidate reply to test OutputGuard score stripping",
            "Good job! That was 8/10. Can you say it again slower?",
        )
        if st.button("Check reply"):
            use = d if d is not None and d.fixed_reply is None else None
            g = agent.guard.check(reply, use)
            st.write(f"Action: `{g.action}` · OK: {g.ok}")
            st.code(g.text)
            if g.violations or g.warnings:
                st.write({"violations": g.violations, "warnings": g.warnings, "hint": g.regenerate_hint})


render_panel = render_safety_panel


def main() -> None:
    if st is not None:
        render_safety_panel()
    else:
        print("Streamlit is not installed. Install requirements with: pip install -r demo/streamlit/requirements.txt")


if __name__ == "__main__":
    main()
