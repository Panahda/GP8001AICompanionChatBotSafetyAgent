"""Unit tests for safety.orchestrator, safety.companion, and safety.feedback."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safety.agent import SafetyAgent
from safety.companion import CompanionAgent
from safety.feedback import FeedbackAgent
from safety.llm_providers import MockProvider
from safety.orchestrator import TurnResult, handle_turn
from safety.output_guard import SAFE_FALLBACK, OutputGuard
from safety.scenarios import get_scenario
from safety.schema import Action, Level, SafetyDecision
from safety.state import SessionState

# Zero score leak assertion helper
SCORE_LEAK_REGEX = re.compile(
    r"\b\d+(\.\d+)?\s*/\s*(10|5|100)\b|\b\d{1,3}\s*(%|percent)(?!\w)|\b(score|scored|scoring|grade|graded|rating)\b|"
    r"\b\d+ (marks|points)\b|\bincorrect\b|\b(wrong|right|correct) (answer|reply|response)\b|"
    r"\b(that is|that's|thats|that was|this is) (wrong|incorrect)\b|"
    r"\b(your|the|that) (answer|reply|response) (is|was) (wrong|right|correct|good|bad|perfect)\b|"
    r"\byou (passed|failed|got it right|got it wrong)\b|[\u2605\u2606]",
    re.IGNORECASE,
)


def assert_no_score_leak(text: str | None) -> None:
    if not text:
        return
    match = SCORE_LEAK_REGEX.search(text)
    assert not match, f"Score leak detected: {match.group(0)!r} in {text!r}"


@pytest.fixture
def agent():
    return SafetyAgent.from_config(ROOT / "config.yaml", provider="mock", mode="rules+llm")


@pytest.fixture
def companion(agent):
    return CompanionAgent(provider=MockProvider(), safety_agent=agent)


@pytest.fixture
def feedback():
    return FeedbackAgent(provider=MockProvider())


# -----------------------------------------------------------------------------
# 1. Calm Turn (Level 0 Safe)
# -----------------------------------------------------------------------------
def test_level0_calm_turn(agent, companion, feedback):
    """Calm turn returns in-character reply, OutputGuard ok, at most one feedback suggestion."""
    session = agent.new_session(user_id="test_u1", age_group="teen")
    sc = get_scenario("hawker_centre")

    res = handle_turn(
        agent,
        companion,
        feedback,
        session,
        "One chicken rice please.",
        sc.opening_line,
        sc.id,
    )

    assert isinstance(res, TurnResult)
    assert res.decision.session_level == Level.SAFE
    assert res.decision.level == 0
    assert res.decision.action == Action.CONTINUE
    assert len(res.reply) > 0
    assert agent.guard.check(res.reply).ok is True

    # Feedback generated on calm turn
    assert res.feedback is not None
    assert isinstance(res.feedback, str)
    assert len(res.feedback.split()) <= 50
    assert "Goal:" in res.feedback or "chicken rice" in res.feedback.lower()

    # Score-free invariant
    assert_no_score_leak(res.reply)
    assert_no_score_leak(res.feedback)


# -----------------------------------------------------------------------------
# 2. Tense Turn (Level 1 Tense)
# -----------------------------------------------------------------------------
def test_level1_tense_turn(agent, companion, feedback):
    """Tense turn passes slow down directives, suppresses feedback, surfaces choices."""
    session = agent.new_session(user_id="test_u2", age_group="teen")
    sc = get_scenario("new_colleague")

    res = handle_turn(
        agent,
        companion,
        feedback,
        session,
        "I'm feeling very anxious and nervous about talking to people...",
        sc.opening_line,
        sc.id,
    )

    assert res.decision.session_level == Level.TENSE
    assert res.decision.level == 1
    assert res.decision.action == Action.SLOW_DOWN
    assert res.decision.suppress_feedback is True

    # Feedback strictly suppressed
    assert res.feedback is None

    # Choices offered
    assert len(res.buttons) >= 2
    assert "Keep going" in res.buttons or "Take a short break" in res.buttons

    # Output guard and score invariant
    assert agent.guard.check(res.reply, res.decision).ok is True
    assert_no_score_leak(res.reply)


# -----------------------------------------------------------------------------
# 3. Overwhelmed Turn (Level 2 Overwhelmed)
# -----------------------------------------------------------------------------
def test_level2_overwhelmed_turn(agent, companion, feedback):
    """Overwhelmed turn pauses roleplay, suppresses LLMs, returns playbook regulation prompt."""
    session = agent.new_session(user_id="test_u3", age_group="teen")
    sc = get_scenario("job_interview")

    res = handle_turn(
        agent,
        companion,
        feedback,
        session,
        "Too loud, can't think, stop!",
        sc.opening_line,
        sc.id,
    )

    assert res.decision.session_level == Level.OVERWHELMED
    assert res.decision.level == 2
    assert res.decision.action == Action.PAUSE_REGULATE
    assert res.decision.pause_roleplay is True
    assert res.decision.suppress_feedback is True
    assert res.feedback is None

    # Returns playbook prompt with self-regulation choices
    assert "breath" in res.reply.lower() or "feeling" in res.reply.lower() or "break" in res.reply.lower()
    assert len(res.buttons) > 0
    assert_no_score_leak(res.reply)


# -----------------------------------------------------------------------------
# 4. Crisis Turn (Level 3 Crisis) & Crisis Hold
# -----------------------------------------------------------------------------
def test_level3_crisis_turn_and_hold(agent, companion, feedback):
    """Crisis turn pauses roleplay, returns Singapore helplines, engages crisis hold."""
    session = agent.new_session(user_id="test_u4", age_group="teen")
    sc = get_scenario("hawker_centre")

    res = handle_turn(
        agent,
        companion,
        feedback,
        session,
        "i want to die",
        sc.opening_line,
        sc.id,
    )

    assert res.decision.session_level == Level.CRISIS
    assert res.decision.level == 3
    assert res.decision.pause_roleplay is True
    assert res.decision.suppress_feedback is True
    assert res.feedback is None

    # Singapore helplines
    assert "1767" in res.reply or "1771" in res.reply
    assert "SOS" in res.reply or "mindline" in res.reply
    assert_no_score_leak(res.reply)

    # Crisis hold is engaged
    assert session.crisis_hold is True

    # Subsequent turn with calm text must remain locked in crisis support
    res2 = handle_turn(
        agent,
        companion,
        feedback,
        session,
        "I am fine now",
        sc.opening_line,
        sc.id,
    )
    assert res2.decision.action == Action.CRISIS_SUPPORT
    assert session.crisis_hold is True
    assert res2.feedback is None
    assert "1767" in res2.reply or "1771" in res2.reply

    # Explicit release resets crisis hold
    res3 = handle_turn(
        agent,
        companion,
        feedback,
        session,
        "I am safe now. Go back to practice",
        sc.opening_line,
        sc.id,
    )
    assert session.crisis_hold is False
    assert res3.decision.session_level == Level.TENSE


# -----------------------------------------------------------------------------
# 5. Output Invariant: Zero Scores Across Turns
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("scenario_id", ["hawker_centre", "new_colleague", "job_interview", "free_chat"])
def test_output_invariant_zero_scores(agent, companion, feedback, scenario_id):
    """Zero scores, percentages, grades, or right/wrong marks in replies or feedback."""
    session = agent.new_session()
    sc = get_scenario(scenario_id)

    turns = [
        "Hello!",
        "Can I order something please?",
        "I would like to have chicken rice.",
        "Take away please.",
    ]
    for text in turns:
        res = handle_turn(agent, companion, feedback, session, text, sc.opening_line, sc.id)
        assert_no_score_leak(res.reply)
        assert_no_score_leak(res.feedback)


# -----------------------------------------------------------------------------
# 6. Companion Guard Retry & Safe Fallback
# -----------------------------------------------------------------------------
def test_companion_guard_retry_and_safe_fallback(agent):
    """Verifies that soft warnings trigger 1 retry, and persistent violations return SAFE_FALLBACK."""
    session = agent.new_session()
    sc = get_scenario("hawker_centre")

    # A mock provider that leaks a score on first attempt, then fixes it on retry with hint
    class LeakingThenFixingProvider:
        name = "mock_leak_fix"
        model = "mock"
        calls = 0

        def complete_json(self, system, user, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"reply": "Great job! That was 8/10. What would you like next?"}
            return {"reply": "Okay. One chicken rice. Steamed or roasted?"}

    prov = LeakingThenFixingProvider()
    comp = CompanionAgent(provider=prov, safety_agent=agent)
    reply = comp.generate(session, sc, "One chicken rice please.", directives=[], history=[])
    assert "8/10" not in reply
    assert_no_score_leak(reply)

    # An always-violating provider returning medical diagnosis
    class PersistentViolationProvider:
        name = "mock_violating"
        model = "mock"
        calls = 0

        def complete_json(self, system, user, *args, **kwargs):
            self.calls += 1
            return {"reply": "You seem to have clinical depression and should take antidepressants."}

    bad_prov = PersistentViolationProvider()
    bad_comp = CompanionAgent(provider=bad_prov, safety_agent=agent)
    fallback_reply = bad_comp.generate(session, sc, "I feel sad.", directives=[], history=[])
    assert fallback_reply == SAFE_FALLBACK
    assert bad_prov.calls == 2  # 1 initial + 1 retry


# -----------------------------------------------------------------------------
# 7. TurnResult Dataclass and Subscript Compatibility
# -----------------------------------------------------------------------------
def test_turn_result_compatibility(agent, companion, feedback):
    """TurnResult supports both attribute access and dictionary subscripting."""
    session = agent.new_session()
    res = handle_turn(
        agent,
        companion,
        feedback,
        session,
        "Hello!",
        "Hi!",
        "hawker_centre",
    )
    # Attribute access
    assert hasattr(res, "reply")
    assert hasattr(res, "buttons")
    assert hasattr(res, "feedback")
    assert hasattr(res, "decision")
    assert hasattr(res, "scenario_id")

    # Subscript access (as used in demo)
    assert res["reply"] == res.reply
    assert res["buttons"] == res.buttons
    assert res["feedback"] == res.feedback
    assert res["decision"] == res.decision
    assert res["scenario_id"] == res.scenario_id

    # to_dict
    d = res.to_dict()
    assert isinstance(d, dict)
    assert d["reply"] == res.reply


# -----------------------------------------------------------------------------
# 8. Dual Calling Convention Compatibility
# -----------------------------------------------------------------------------
def test_handle_turn_dual_calling_conventions(agent, companion, feedback):
    """Verifies that handle_turn supports both parameter ordering styles."""
    sc = get_scenario("hawker_centre")

    # Style A: (agent, companion, feedback, session, text, last_line, scenario)
    s1 = agent.new_session()
    res_a = handle_turn(agent, companion, feedback, s1, "One chicken rice.", sc.opening_line, sc.id)

    # Style B: (agent, session, text, last_line, scenario, companion, feedback)
    s2 = agent.new_session()
    res_b = handle_turn(agent, s2, "One chicken rice.", sc.opening_line, sc.id, companion_agent=companion, feedback_agent=feedback)

    assert res_a.decision.level == res_b.decision.level
    assert res_a.reply == res_b.reply
    assert res_a.buttons == res_b.buttons
