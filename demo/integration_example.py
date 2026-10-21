"""How the orchestrator (Streamlit chat app) should call the safety agent.

    python demo/integration_example.py

`companion_generate` and `feedback_generate` stand in for C's companion and
feedback prompts. Only the safety calls are real.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safety import SafetyAgent  # noqa: E402
from safety.output_guard import SAFE_FALLBACK  # noqa: E402


def companion_generate(user_text, scenario, extra_directives):
    """Placeholder for the companion LLM call. Directives go at the end of its system prompt."""
    return "Okay. One chicken rice. Do you want chilli? You did great, 9/10!"


def feedback_generate(user_text, scenario):
    return "(one optional suggestion from the feedback agent)"


def handle_turn(agent, session, user_text, last_companion_line, scenario):
    # 1. check the user's message before anything else
    d = agent.check_user_message(session, user_text, context=last_companion_line, scenario=scenario)

    # 2. fixed replies (helplines, playbook) are shown as written, with buttons
    if d.fixed_reply:
        return {"reply": d.fixed_reply, "buttons": d.choices, "feedback": None, "decision": d}

    # 3. otherwise the companion writes a reply, steered by the safety directives
    reply = companion_generate(user_text, scenario, d.companion_directives)
    g = agent.check_reply(session, d, reply)
    if g.action == "regenerate":  # one retry with a hint, then fall back
        reply = companion_generate(user_text, scenario, d.companion_directives + [g.regenerate_hint])
        g = agent.check_reply(session, d, reply)
        if g.action == "regenerate" and not g.ok:
            g.text = SAFE_FALLBACK

    # 4. feedback only when the session is calm
    fb = None if d.suppress_feedback else feedback_generate(user_text, scenario)
    return {"reply": g.text, "buttons": d.choices, "feedback": fb, "decision": d}


if __name__ == "__main__":
    agent = SafetyAgent.from_config(ROOT / "config.yaml", provider="mock")
    s = agent.new_session(age_group="teen")
    for text in ["One chicken rice please.", "i want to die"]:
        out = handle_turn(agent, s, text, "Hello! What would you like to order?", "hawker_centre")
        print(f"\nUSER: {text}\nACTION: {out['decision'].action}\nREPLY:\n{out['reply']}\nBUTTONS: {out['buttons']}")
