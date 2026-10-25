"""How the orchestrator (web demo & Streamlit chat app) calls the safety agent and LLMs.

    python demo/integration_example.py

Integrates the multi-tiered SafetyAgent, plain-language CompanionAgent, and
rubric-constrained FeedbackAgent into a unified practice turn coordinator.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safety import SafetyAgent  # noqa: E402
from safety.companion import CompanionAgent  # noqa: E402
from safety.feedback import FeedbackAgent  # noqa: E402
from safety.llm_providers import MockProvider  # noqa: E402
from safety.orchestrator import TurnResult, handle_turn  # noqa: E402
from safety.output_guard import SAFE_FALLBACK  # noqa: E402
from safety.scenarios import get_scenario  # noqa: E402


def companion_generate(user_text, scenario, extra_directives=None, safety_agent=None):
    """Generates a plain-language in-character companion reply vetted by OutputGuard."""
    agent = CompanionAgent(safety_agent=safety_agent)
    return agent.generate(
        scenario=scenario,
        user_text=user_text,
        directives=extra_directives or [],
    )


def feedback_generate(user_text, scenario, companion_reply=""):
    """Generates at most one rubric-grounded coaching suggestion."""
    agent = FeedbackAgent()
    return agent.generate(
        scenario=scenario,
        user_text=user_text,
        companion_reply=companion_reply,
    )


__all__ = [
    "handle_turn",
    "CompanionAgent",
    "FeedbackAgent",
    "TurnResult",
    "companion_generate",
    "feedback_generate",
]


if __name__ == "__main__":
    agent = SafetyAgent.from_config(ROOT / "config.yaml", provider="mock")
    comp = CompanionAgent(provider=MockProvider(), safety_agent=agent)
    fb = FeedbackAgent(provider=MockProvider())
    s = agent.new_session(age_group="teen")
    for text in ["One chicken rice please.", "i want to die"]:
        out = handle_turn(agent, s, text, "Hello! What would you like to order?", "hawker_centre", companion_agent=comp, feedback_agent=fb)
        print(f"\nUSER: {text}\nACTION: {out['decision'].action}\nREPLY:\n{out['reply']}\nBUTTONS: {out['buttons']}")
