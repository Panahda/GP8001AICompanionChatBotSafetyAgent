"""Turn orchestrator coordinating SafetyAgent, CompanionAgent, and FeedbackAgent.

Coordinates pre-turn safety classification across Levels 0-3, conditional companion
dialogue generation with OutputGuard verification, and rubric-constrained feedback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from .agent import SafetyAgent
from .companion import CompanionAgent
from .feedback import FeedbackAgent
from .scenarios import Scenario, get_scenario
from .schema import Action, Level, SafetyDecision
from .state import SessionState


@dataclass
class TurnResult:
    """Result of a single conversational practice turn."""

    reply: str
    buttons: List[str]
    feedback: Optional[str]
    decision: SafetyDecision
    scenario_id: str

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reply": self.reply,
            "buttons": list(self.buttons),
            "feedback": self.feedback,
            "decision": self.decision,
            "scenario_id": self.scenario_id,
        }


def handle_turn(
    agent: Any,
    *args,
    companion_agent: Optional[Any] = None,
    feedback_agent: Optional[Any] = None,
    session: Optional[Any] = None,
    user_text: Optional[str] = None,
    last_companion_line: Optional[str] = None,
    scenario: Optional[Union[str, Scenario]] = None,
    scenario_id: Optional[str] = None,
    history: Optional[List[dict]] = None,
    **kwargs,
) -> TurnResult:
    """Coordinates a practice turn across safety triage, companion dialogue, and coaching."""
    # Handle flexible parameter ordering
    if len(args) > 0:
        first = args[0]
        # Check if first positional arg is session
        if isinstance(first, SessionState) or (hasattr(first, "crisis_hold") and not hasattr(first, "generate")):
            session = first
            if len(args) > 1:
                user_text = args[1]
            if len(args) > 2:
                last_companion_line = args[2]
            if len(args) > 3:
                scenario = args[3]
            if len(args) > 4:
                companion_agent = args[4]
            if len(args) > 5:
                feedback_agent = args[5]
            if len(args) > 6:
                history = args[6]
        else:
            # First arg is companion_agent
            companion_agent = first
            if len(args) > 1:
                feedback_agent = args[1]
            if len(args) > 2:
                session = args[2]
            if len(args) > 3:
                user_text = args[3]
            if len(args) > 4:
                last_companion_line = args[4]
            if len(args) > 5:
                scenario = args[5]
            if len(args) > 6:
                history = args[6]

    if user_text is None:
        user_text = kwargs.get("text", "")
    if scenario is None and scenario_id is not None:
        scenario = scenario_id
    if scenario is None:
        scenario = "hawker_centre"

    sc_id = scenario.id if isinstance(scenario, Scenario) else str(scenario)
    try:
        sc_obj = get_scenario(sc_id)
    except KeyError:
        sc_obj = None

    if session is None:
        session = agent.new_session() if hasattr(agent, "new_session") else SessionState()
    if companion_agent is None:
        companion_agent = CompanionAgent(safety_agent=agent)
    if feedback_agent is None:
        feedback_agent = FeedbackAgent()

    # Check for crisis release or continue button actions passed as text
    text_clean = (user_text or "").strip().lower()
    if text_clean.startswith("i am safe now") and getattr(session, "crisis_hold", False):
        if hasattr(agent, "release_crisis_hold"):
            agent.release_crisis_hold(session)
    elif text_clean in ("i did it", "go back to practice", "keep going", "i am back") and getattr(session, "level", 0) == 2:
        if hasattr(agent, "user_chose_continue"):
            agent.user_chose_continue(session)

    # 1. Pre-turn safety check
    decision = agent.check_user_message(
        session, user_text, context=last_companion_line, scenario=sc_id
    )
    # Ensure decision has .level attribute pointing to session_level
    setattr(decision, "level", decision.session_level)

    # 2. Level 2 (Overwhelmed) / Level 3 (Crisis): fixed reply returned directly, companion & feedback suppressed
    if decision.fixed_reply:
        return TurnResult(
            reply=decision.fixed_reply,
            buttons=list(decision.choices),
            feedback=None,
            decision=decision,
            scenario_id=sc_id,
        )

    # 3. Level 1 (Tense): slow down directives, feedback suppressed, choices presented
    if decision.action == Action.SLOW_DOWN or decision.session_level == Level.TENSE:
        reply = companion_agent.generate(
            session=session,
            scenario=sc_obj or sc_id,
            user_text=user_text,
            directives=decision.companion_directives,
            history=history,
            decision=decision,
        )
        return TurnResult(
            reply=reply,
            buttons=list(decision.choices),
            feedback=None,
            decision=decision,
            scenario_id=sc_id,
        )

    # 4. Level 0 (Safe / Calm): normal role-play with companion and optional feedback
    reply = companion_agent.generate(
        session=session,
        scenario=sc_obj or sc_id,
        user_text=user_text,
        directives=decision.companion_directives,
        history=history,
        decision=decision,
    )

    fb: Optional[str] = None
    if not decision.suppress_feedback:
        fb = feedback_agent.generate(
            scenario=sc_obj or sc_id,
            user_text=user_text,
            companion_reply=reply,
            decision=decision,
        )

    return TurnResult(
        reply=reply,
        buttons=list(decision.choices),
        feedback=fb,
        decision=decision,
        scenario_id=sc_id,
    )


__all__ = ["TurnResult", "handle_turn"]
