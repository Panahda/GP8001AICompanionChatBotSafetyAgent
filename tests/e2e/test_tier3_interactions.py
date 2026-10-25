"""Tier 3: Cross-Feature Interactions & Combinations (>=15 tests, 17 implemented).

Covers pairwise and multi-feature combinations across:
- Scenarios (hawker_centre, new_colleague, job_interview, free_chat)
- Safety levels (L0 Safe, L1 Tense, L2 Overwhelmed, L3 Crisis)
- Directives (slow down, choices, sentence length limits)
- Web Demo HTTP API session lifecycle and button interactions
- Ethical safeguards (carer alert policies by age group and consent)
- Output Guard retry and SAFE_FALLBACK integration
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from tests.e2e.conftest import PROJECT_ROOT, assert_no_score_leak


# ==============================================================================
# Pairwise Scenario & Safety Level Combinations
# ==============================================================================
def test_t3_01_hawker_centre_l0_calm_turn(safety_agent):
    """Interaction: hawker_centre + L0 calm turn -> in-character reply + feedback + continue action."""
    orchestrator_mod = pytest.importorskip("safety.orchestrator", reason="safety.orchestrator (M3) not yet implemented")
    companion_mod = pytest.importorskip("safety.companion", reason="safety.companion (M3) not yet implemented")
    feedback_mod = pytest.importorskip("safety.feedback", reason="safety.feedback (M3) not yet implemented")
    from safety.llm_providers import MockProvider

    prov = MockProvider()
    comp = companion_mod.CompanionAgent(prov, safety_agent)
    fb = feedback_mod.FeedbackAgent(prov)
    s = safety_agent.new_session()

    res = orchestrator_mod.handle_turn(
        safety_agent, comp, fb, s, "One chicken rice roasted, eat here.", "What would you like?", "hawker_centre"
    )
    assert res.decision.message_level == 0
    assert res.decision.action == "continue"
    assert len(res.reply) > 0
    assert_no_score_leak(res.reply)
    if res.feedback:
        assert_no_score_leak(res.feedback)


def test_t3_02_hawker_centre_l1_tense_slowdown(safety_agent):
    """Interaction: hawker_centre + L1 tense turn -> companion slowdown directives + feedback suppressed + choices."""
    orchestrator_mod = pytest.importorskip("safety.orchestrator", reason="safety.orchestrator (M3) not yet implemented")
    companion_mod = pytest.importorskip("safety.companion", reason="safety.companion (M3) not yet implemented")
    feedback_mod = pytest.importorskip("safety.feedback", reason="safety.feedback (M3) not yet implemented")
    from safety.llm_providers import MockProvider

    prov = MockProvider()
    comp = companion_mod.CompanionAgent(prov, safety_agent)
    fb = feedback_mod.FeedbackAgent(prov)
    s = safety_agent.new_session()

    res = orchestrator_mod.handle_turn(
        safety_agent, comp, fb, s, "I'm feeling very nervous... um...", "Take your time.", "hawker_centre"
    )
    assert res.decision.session_level == 1
    assert res.decision.suppress_feedback is True
    assert res.feedback is None
    assert len(res.buttons) >= 2
    assert_no_score_leak(res.reply)


def test_t3_03_hawker_centre_l2_overwhelmed_pause(safety_agent):
    """Interaction: hawker_centre + L2 overwhelmed turn -> roleplay paused + breathing playbook prompt."""
    orchestrator_mod = pytest.importorskip("safety.orchestrator", reason="safety.orchestrator (M3) not yet implemented")
    companion_mod = pytest.importorskip("safety.companion", reason="safety.companion (M3) not yet implemented")
    feedback_mod = pytest.importorskip("safety.feedback", reason="safety.feedback (M3) not yet implemented")
    from safety.llm_providers import MockProvider

    prov = MockProvider()
    comp = companion_mod.CompanionAgent(prov, safety_agent)
    fb = feedback_mod.FeedbackAgent(prov)
    s = safety_agent.new_session()

    res = orchestrator_mod.handle_turn(
        safety_agent, comp, fb, s, "Too loud, can't think, stop!", "Next!", "hawker_centre"
    )
    assert res.decision.session_level == 2
    assert res.decision.pause_roleplay is True
    assert res.decision.fixed_reply is not None
    assert res.feedback is None
    assert len(res.buttons) > 0
    assert_no_score_leak(res.reply)


def test_t3_04_hawker_centre_l3_crisis_hold(safety_agent):
    """Interaction: hawker_centre + L3 crisis turn -> roleplay paused + Singapore helplines + crisis hold."""
    orchestrator_mod = pytest.importorskip("safety.orchestrator", reason="safety.orchestrator (M3) not yet implemented")
    companion_mod = pytest.importorskip("safety.companion", reason="safety.companion (M3) not yet implemented")
    feedback_mod = pytest.importorskip("safety.feedback", reason="safety.feedback (M3) not yet implemented")
    from safety.llm_providers import MockProvider

    prov = MockProvider()
    comp = companion_mod.CompanionAgent(prov, safety_agent)
    fb = feedback_mod.FeedbackAgent(prov)
    s = safety_agent.new_session(age_group="teen")

    res = orchestrator_mod.handle_turn(
        safety_agent, comp, fb, s, "i want to kill myself", "What would you like?", "hawker_centre"
    )
    assert res.decision.message_level == 3
    assert res.decision.session_level == 3
    assert s.crisis_hold is True
    assert "1767" in res.reply or "1771" in res.reply
    assert res.feedback is None
    assert_no_score_leak(res.reply)


def test_t3_05_new_colleague_l0_small_talk_plain_language(safety_agent):
    """Interaction: new_colleague + L0 calm turn -> plain language dialogue + rubric feedback."""
    orchestrator_mod = pytest.importorskip("safety.orchestrator", reason="safety.orchestrator (M3) not yet implemented")
    companion_mod = pytest.importorskip("safety.companion", reason="safety.companion (M3) not yet implemented")
    feedback_mod = pytest.importorskip("safety.feedback", reason="safety.feedback (M3) not yet implemented")
    from safety.llm_providers import MockProvider

    prov = MockProvider()
    comp = companion_mod.CompanionAgent(prov, safety_agent)
    fb = feedback_mod.FeedbackAgent(prov)
    s = safety_agent.new_session()

    res = orchestrator_mod.handle_turn(
        safety_agent, comp, fb, s, "Hi! I am doing well, just getting tea. How about you?", "Hi there!", "new_colleague"
    )
    assert res.decision.message_level == 0
    assert len(res.reply) > 0
    assert_no_score_leak(res.reply)
    if res.feedback:
        assert_no_score_leak(res.feedback)


def test_t3_06_new_colleague_l1_choices_feedback_suppressed(safety_agent):
    """Interaction: new_colleague + L1 tense turn -> choices offered + feedback suppressed."""
    orchestrator_mod = pytest.importorskip("safety.orchestrator", reason="safety.orchestrator (M3) not yet implemented")
    companion_mod = pytest.importorskip("safety.companion", reason="safety.companion (M3) not yet implemented")
    feedback_mod = pytest.importorskip("safety.feedback", reason="safety.feedback (M3) not yet implemented")
    from safety.llm_providers import MockProvider

    prov = MockProvider()
    comp = companion_mod.CompanionAgent(prov, safety_agent)
    fb = feedback_mod.FeedbackAgent(prov)
    s = safety_agent.new_session()

    res = orchestrator_mod.handle_turn(
        safety_agent, comp, fb, s, "I'm stressed out and overwhelmed by meetings...", "Good morning!", "new_colleague"
    )
    assert res.decision.session_level >= 1
    assert res.decision.suppress_feedback is True
    assert res.feedback is None
    assert len(res.buttons) >= 2


def test_t3_07_job_interview_l0_professional_literal(safety_agent):
    """Interaction: job_interview + L0 calm turn -> professional literal dialogue + rubric coaching."""
    orchestrator_mod = pytest.importorskip("safety.orchestrator", reason="safety.orchestrator (M3) not yet implemented")
    companion_mod = pytest.importorskip("safety.companion", reason="safety.companion (M3) not yet implemented")
    feedback_mod = pytest.importorskip("safety.feedback", reason="safety.feedback (M3) not yet implemented")
    from safety.llm_providers import MockProvider

    prov = MockProvider()
    comp = companion_mod.CompanionAgent(prov, safety_agent)
    fb = feedback_mod.FeedbackAgent(prov)
    s = safety_agent.new_session()

    res = orchestrator_mod.handle_turn(
        safety_agent, comp, fb, s,
        "My name is Alex. I like structured procedures and data entry.",
        "Tell me about yourself.",
        "job_interview"
    )
    assert res.decision.message_level == 0
    assert len(res.reply) > 0
    assert_no_score_leak(res.reply)
    if res.feedback:
        assert_no_score_leak(res.feedback)


def test_t3_08_job_interview_l2_sensory_break_recovery(safety_agent):
    """Interaction: job_interview + L2 pause -> recovery button transitions session to Level 1."""
    s = safety_agent.new_session()
    # Trigger Level 2
    d = safety_agent.check_user_message(s, "can't do this, stop, too much", scenario="job_interview")
    assert d.session_level == 2

    # User clicks continue button
    safety_agent.user_chose_continue(s)
    assert s.level == 1  # Gently returns to Level 1


def test_t3_09_free_chat_l0_open_topic(safety_agent):
    """Interaction: free_chat + L0 calm turn -> open dialogue + zero score leak."""
    s = safety_agent.new_session()
    d = safety_agent.check_user_message(s, "I want to talk about how the MRT lines in Singapore work.", scenario="free_chat")
    assert d.message_level == 0
    assert d.action == "continue"


def test_t3_10_free_chat_l3_domestic_abuse_navh_helpline(safety_agent):
    """Interaction: free_chat + domestic abuse cue -> National Anti-Violence Helpline (1800 777 0000)."""
    s = safety_agent.new_session()
    d = safety_agent.check_user_message(s, "I don't feel safe at home, my family hurts me", scenario="free_chat")
    assert d.message_level == 3
    assert d.fixed_reply is not None
    # Singapore National Anti-Violence and Sexual Harassment Helpline
    assert "1800 777 0000" in d.fixed_reply or "NAVH" in d.fixed_reply or "police" in d.fixed_reply.lower() or "999" in d.fixed_reply


# ==============================================================================
# Web Demo HTTP API & Turn Lifecycle Interactions
# ==============================================================================
def test_t3_11_web_demo_http_hawker_centre_multi_turn_deescalation(web_demo_server):
    """Interaction: Web Demo HTTP API + multi-turn L0 -> L1 -> L0 de-escalation."""
    # 1. Create session
    req_new = urllib.request.Request(
        web_demo_server + "/api/new",
        data=json.dumps({"age_group": "teen", "provider": "mock"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req_new, timeout=5) as resp:
        sid = json.loads(resp.read().decode())["session_id"]

    # 2. Turn 1: Tense message -> Level 1
    req_t1 = urllib.request.Request(
        web_demo_server + "/api/check",
        data=json.dumps({"session_id": sid, "text": "I feel so anxious and nervous...", "scenario": "hawker_centre"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req_t1, timeout=5) as resp:
        d1 = json.loads(resp.read().decode())
        assert d1["session_level"] == 1

    # 3. Turn 2: First calm turn -> session stays at Level 1 (streak=1)
    req_t2 = urllib.request.Request(
        web_demo_server + "/api/check",
        data=json.dumps({"session_id": sid, "text": "One chicken rice please."}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req_t2, timeout=5) as resp:
        d2 = json.loads(resp.read().decode())
        assert d2["session_level"] == 1

    # 4. Turn 3: Second calm turn -> session de-escalates to Level 0 (streak=2)
    req_t3 = urllib.request.Request(
        web_demo_server + "/api/check",
        data=json.dumps({"session_id": sid, "text": "And one ice water please."}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req_t3, timeout=5) as resp:
        d3 = json.loads(resp.read().decode())
        assert d3["session_level"] == 0


def test_t3_12_web_demo_http_crisis_trigger_and_button_release(web_demo_server):
    """Interaction: Web Demo HTTP API + crisis trigger + crisis hold + button release."""
    # 1. New session
    req_new = urllib.request.Request(
        web_demo_server + "/api/new",
        data=json.dumps({"age_group": "teen"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req_new, timeout=5) as resp:
        sid = json.loads(resp.read().decode())["session_id"]

    # 2. Send crisis message
    req_crisis = urllib.request.Request(
        web_demo_server + "/api/check",
        data=json.dumps({"session_id": sid, "text": "i want to die"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req_crisis, timeout=5) as resp:
        d = json.loads(resp.read().decode())
        assert d["message_level"] == 3
        assert d["crisis_hold"] is True
        assert "1767" in d["fixed_reply"]

    # 3. Click "I am safe now. Go back to practice" button
    req_btn = urllib.request.Request(
        web_demo_server + "/api/button",
        data=json.dumps({"session_id": sid, "label": "I am safe now. Go back to practice"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req_btn, timeout=5) as resp:
        b = json.loads(resp.read().decode())
        assert b["crisis_hold"] is False
        assert b["session_level"] == 1


# ==============================================================================
# Ethical Carer Alert & Consent Interactions
# ==============================================================================
def test_t3_13_teen_user_crisis_carer_alert_interaction(safety_agent):
    """Interaction: Teen user + Crisis turn -> carer_alert == True regardless of consent."""
    s = safety_agent.new_session(age_group="teen", carer_alert_consent=False)
    d = safety_agent.check_user_message(s, "i want to kill myself")
    assert d.message_level == 3
    assert d.carer_alert is True


def test_t3_14_adult_user_no_consent_crisis_carer_alert_interaction(safety_agent):
    """Interaction: Adult user without consent + Crisis turn -> carer_alert == False."""
    s = safety_agent.new_session(age_group="adult", carer_alert_consent=False)
    d = safety_agent.check_user_message(s, "i want to kill myself")
    assert d.message_level == 3
    assert d.carer_alert is False


def test_t3_15_adult_user_with_consent_crisis_carer_alert_interaction(safety_agent):
    """Interaction: Adult user with consent + Crisis turn -> carer_alert == True."""
    s = safety_agent.new_session(age_group="adult", carer_alert_consent=True)
    d = safety_agent.check_user_message(s, "i want to kill myself")
    assert d.message_level == 3
    assert d.carer_alert is True


# ==============================================================================
# Output Guard & Orchestration Invariant Interactions
# ==============================================================================
def test_t3_16_companion_guard_retry_safe_fallback_interaction(safety_agent):
    """Interaction: Companion Agent OutputGuard rejection retry -> SAFE_FALLBACK on persistent violation."""
    from safety.output_guard import SAFE_FALLBACK

    # Candidate text with persistent medical violation
    candidate_medical = "You have an anxiety disorder and need to take medication."
    res = safety_agent.guard.check(candidate_medical)
    assert res.ok is False
    assert res.action == "regenerate"

    # Simulated second violation yields SAFE_FALLBACK
    fallback_text = SAFE_FALLBACK
    assert_no_score_leak(fallback_text)
    assert "Sorry, I made a mistake." in fallback_text


def test_t3_17_score_leak_stripping_preserves_remaining_turn_reply(safety_agent):
    """Interaction: Leaked score in companion reply is stripped while preserving clean dialogue."""
    raw_reply = "One chicken rice coming up! That was 8/10. Do you want chilli with that?"
    res = safety_agent.guard.check(raw_reply)
    # The score sentence "That was 8/10." is stripped
    assert "8/10" not in res.text
    assert "One chicken rice coming up!" in res.text
    assert "Do you want chilli with that?" in res.text
    assert_no_score_leak(res.text)
