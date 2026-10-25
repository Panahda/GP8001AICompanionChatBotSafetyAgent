"""Tier 4: Realistic Real-World Application Workloads (7 comprehensive workflows).

Executes multi-turn end-to-end conversational workflows simulating actual user journeys:
1. Hawker Centre Lunch Ordering (multi-turn ordering, options, payment, zero score leak)
2. New Colleague Pantry Small Talk (multi-turn greeting, tea break, workplace share)
3. Job Interview Practice (multi-turn self-intro, skills, questions, rubric coaching)
4. Escalation & De-escalation Arc (calm -> sensory anxiety -> slow down -> hysteresis recovery)
5. Overwhelmed Pause & Continue Arc (calm -> sensory shutdown -> playbook breathing prompt -> continue recovery)
6. Crisis Hold & Release Arc (calm -> crisis trigger -> helpline injection -> crisis lock -> safe button release)
7. Full Web Demo HTTP Lifecycle (complete HTTP interaction: index, status, new, check, button, reply guard)
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from tests.e2e.conftest import PROJECT_ROOT, assert_no_score_leak


# ==============================================================================
# Workflow 1: Hawker Centre Lunch Ordering
# ==============================================================================
def test_t4_01_hawker_centre_lunch_ordering_workflow(safety_agent):
    """Workflow 1: Realistic 3-turn hawker food order (order -> options -> payment) at Level 0."""
    orchestrator_mod = pytest.importorskip("safety.orchestrator", reason="safety.orchestrator (M3) not yet implemented")
    companion_mod = pytest.importorskip("safety.companion", reason="safety.companion (M3) not yet implemented")
    feedback_mod = pytest.importorskip("safety.feedback", reason="safety.feedback (M3) not yet implemented")
    from safety.llm_providers import MockProvider

    prov = MockProvider()
    comp = companion_mod.CompanionAgent(prov, safety_agent)
    fb = feedback_mod.FeedbackAgent(prov)
    s = safety_agent.new_session()

    # Turn 1: User places food order
    t1 = orchestrator_mod.handle_turn(
        safety_agent, comp, fb, s,
        user_text="One plate of roasted chicken rice please.",
        last_companion_line="Hello! What would you like to order?",
        scenario_id="hawker_centre"
    )
    assert t1.decision.message_level == 0
    assert t1.decision.action == "continue"
    assert len(t1.reply) > 0
    assert_no_score_leak(t1.reply)
    if t1.feedback:
        assert_no_score_leak(t1.feedback)

    # Turn 2: User specifies dining options
    t2 = orchestrator_mod.handle_turn(
        safety_agent, comp, fb, s,
        user_text="Eat here. Can I have extra chilli and cucumber please?",
        last_companion_line=t1.reply,
        scenario_id="hawker_centre"
    )
    assert t2.decision.message_level == 0
    assert_no_score_leak(t2.reply)
    if t2.feedback:
        assert_no_score_leak(t2.feedback)

    # Turn 3: User completes payment
    t3 = orchestrator_mod.handle_turn(
        safety_agent, comp, fb, s,
        user_text="Here is five dollars cash. Thank you uncle.",
        last_companion_line=t2.reply,
        scenario_id="hawker_centre"
    )
    assert t3.decision.message_level == 0
    assert_no_score_leak(t3.reply)


# ==============================================================================
# Workflow 2: New Colleague Pantry Small Talk
# ==============================================================================
def test_t4_02_new_colleague_pantry_small_talk_workflow(safety_agent):
    """Workflow 2: Realistic 3-turn office pantry small talk with colleague in plain English."""
    orchestrator_mod = pytest.importorskip("safety.orchestrator", reason="safety.orchestrator (M3) not yet implemented")
    companion_mod = pytest.importorskip("safety.companion", reason="safety.companion (M3) not yet implemented")
    feedback_mod = pytest.importorskip("safety.feedback", reason="safety.feedback (M3) not yet implemented")
    from safety.llm_providers import MockProvider

    prov = MockProvider()
    comp = companion_mod.CompanionAgent(prov, safety_agent)
    fb = feedback_mod.FeedbackAgent(prov)
    s = safety_agent.new_session()

    # Turn 1: Greeting & acknowledgment
    t1 = orchestrator_mod.handle_turn(
        safety_agent, comp, fb, s,
        user_text="Good morning! Pretty good, just making some tea before the team meeting. How about you?",
        last_companion_line="Hi there! Taking a quick break? How has your week been so far?",
        scenario_id="new_colleague"
    )
    assert t1.decision.message_level == 0
    assert len(t1.reply) > 0
    assert_no_score_leak(t1.reply)
    if t1.feedback:
        assert_no_score_leak(t1.feedback)

    # Turn 2: Brief workplace update
    t2 = orchestrator_mod.handle_turn(
        safety_agent, comp, fb, s,
        user_text="I finished updating the inventory spreadsheets today. Did you get your laptop setup okay?",
        last_companion_line=t1.reply,
        scenario_id="new_colleague"
    )
    assert t2.decision.message_level == 0
    assert_no_score_leak(t2.reply)
    if t2.feedback:
        assert_no_score_leak(t2.feedback)

    # Turn 3: Polite closing
    t3 = orchestrator_mod.handle_turn(
        safety_agent, comp, fb, s,
        user_text="Glad to hear that. I will head back to my desk now. See you later!",
        last_companion_line=t2.reply,
        scenario_id="new_colleague"
    )
    assert t3.decision.message_level == 0
    assert_no_score_leak(t3.reply)


# ==============================================================================
# Workflow 3: Job Interview Practice
# ==============================================================================
def test_t4_03_job_interview_practice_workflow(safety_agent):
    """Workflow 3: Realistic 3-turn entry-level job interview practice session."""
    orchestrator_mod = pytest.importorskip("safety.orchestrator", reason="safety.orchestrator (M3) not yet implemented")
    companion_mod = pytest.importorskip("safety.companion", reason="safety.companion (M3) not yet implemented")
    feedback_mod = pytest.importorskip("safety.feedback", reason="safety.feedback (M3) not yet implemented")
    from safety.llm_providers import MockProvider

    prov = MockProvider()
    comp = companion_mod.CompanionAgent(prov, safety_agent)
    fb = feedback_mod.FeedbackAgent(prov)
    s = safety_agent.new_session()

    # Turn 1: Self-introduction citing skills
    t1 = orchestrator_mod.handle_turn(
        safety_agent, comp, fb, s,
        user_text="My name is Alex. I like structured data organisation and computer setup. I applied because I enjoy systematic archiving.",
        last_companion_line="Welcome! Could you tell me a little bit about yourself and why you applied?",
        scenario_id="job_interview"
    )
    assert t1.decision.message_level == 0
    assert_no_score_leak(t1.reply)
    if t1.feedback:
        assert_no_score_leak(t1.feedback)

    # Turn 2: Answering follow-up question
    t2 = orchestrator_mod.handle_turn(
        safety_agent, comp, fb, s,
        user_text="During my school project, I created a folder index for 200 files with zero errors. I prefer clear written guidelines.",
        last_companion_line=t1.reply,
        scenario_id="job_interview"
    )
    assert t2.decision.message_level == 0
    assert_no_score_leak(t2.reply)

    # Turn 3: Asking candidate question
    t3 = orchestrator_mod.handle_turn(
        safety_agent, comp, fb, s,
        user_text="Thank you for explaining. Could you tell me what a typical day looks like for this position?",
        last_companion_line=t2.reply,
        scenario_id="job_interview"
    )
    assert t3.decision.message_level == 0
    assert_no_score_leak(t3.reply)


# ==============================================================================
# Workflow 4: Escalation & De-escalation Arc
# ==============================================================================
def test_t4_04_escalation_and_deescalation_arc_workflow(safety_agent):
    """Workflow 4: L0 -> L1 Tense -> Companion Slowdown -> 2-turn calm hysteresis de-escalation."""
    s = safety_agent.new_session()

    # Step 1: Calm start (L0)
    d1 = safety_agent.check_user_message(s, "One chicken rice please.")
    assert d1.session_level == 0
    assert d1.action == "continue"

    # Step 2: User experiences hesitation/anxiety (L1)
    d2 = safety_agent.check_user_message(s, "I feel very nervous and anxious right now...")
    assert d2.session_level == 1
    assert d2.action == "slow_down"
    assert d2.suppress_feedback is True
    assert len(d2.companion_directives) > 0
    # Must offer continue or break
    assert any("choice" in d.lower() or "break" in d.lower() or "short" in d.lower() for d in d2.companion_directives)

    # Step 3: Turn 1 of recovery: user replies calmly
    d3 = safety_agent.check_user_message(s, "Okay, I will have roasted chicken.")
    assert d3.message_level == 0
    # Hysteresis requires 2 calm turns: session stays at Level 1
    assert d3.session_level == 1

    # Step 4: Turn 2 of recovery: user replies calmly again
    d4 = safety_agent.check_user_message(s, "Take away please, thank you.")
    assert d4.message_level == 0
    # 2 consecutive calm turns reached -> de-escalates to Level 0
    assert d4.session_level == 0
    assert d4.action == "continue"


# ==============================================================================
# Workflow 5: Overwhelmed Pause & Continue Arc
# ==============================================================================
def test_t4_05_overwhelmed_pause_and_continue_arc_workflow(safety_agent):
    """Workflow 5: L0 -> L2 Overwhelmed -> Playbook Prompt -> Button 'I did it' -> Level 1 gentle resumption."""
    s = safety_agent.new_session()

    # Step 1: User is practicing
    d1 = safety_agent.check_user_message(s, "Hello, can I ask for directions?")
    assert d1.session_level == 0

    # Step 2: Overwhelm trigger
    d2 = safety_agent.check_user_message(s, "Too loud, can't think! Stop, I can't do this anymore!")
    assert d2.message_level == 2
    assert d2.session_level == 2
    assert d2.action == "pause_regulate"
    assert d2.pause_roleplay is True
    assert d2.suppress_feedback is True
    assert d2.fixed_reply is not None
    # Playbook prompt options
    assert len(d2.choices) >= 2

    # Step 3: User follows calming prompt and clicks "I did it"
    safety_agent.user_chose_continue(s)
    # Transitions gently to Level 1, requiring calm turns to reach 0
    assert s.level == 1

    # Step 4: User sends calm reply
    d4 = safety_agent.check_user_message(s, "I feel better now, let's keep going.")
    assert d4.message_level == 0
    assert d4.session_level == 1  # 1 calm turn completed

    # Step 5: Second calm reply returns to Level 0
    d5 = safety_agent.check_user_message(s, "What should we practice next?")
    assert d5.session_level == 0


# ==============================================================================
# Workflow 6: Crisis Hold & Release Arc
# ==============================================================================
def test_t4_06_crisis_hold_and_release_arc_workflow(safety_agent):
    """Workflow 6: L0 -> L3 Crisis -> Helplines -> Hold Lock -> Explicit Release Button -> L1 Reset."""
    s = safety_agent.new_session(age_group="teen")

    # Step 1: Calm baseline
    d1 = safety_agent.check_user_message(s, "Hello! I am ready to practice.")
    assert d1.session_level == 0

    # Step 2: Decisive suicidal crisis trigger
    d2 = safety_agent.check_user_message(s, "i want to kill myself")
    assert d2.message_level == 3
    assert d2.session_level == 3
    assert d2.action == "crisis_pause" or "crisis" in d2.action
    assert d2.pause_roleplay is True
    assert d2.fixed_reply is not None
    assert "1767" in d2.fixed_reply  # SOS helpline
    assert s.crisis_hold is True
    assert d2.carer_alert is True

    # Step 3: User sends follow-up text without clicking release button
    d3 = safety_agent.check_user_message(s, "sorry, I'm okay now")
    # Crisis hold keeps session in crisis support mode
    assert s.crisis_hold is True
    assert d3.session_level == 3
    assert d3.action == "crisis_support"
    assert d3.pause_roleplay is True
    assert any("safe" in c.lower() for c in d3.choices)

    # Step 4: User clicks "I am safe now. Go back to practice"
    safety_agent.release_crisis_hold(s)
    assert s.crisis_hold is False
    assert s.level == 1  # Restarts gently at Level 1

    # Step 5: Two consecutive calm turns return to Level 0
    d5a = safety_agent.check_user_message(s, "I took a break and drank water.")
    assert d5a.session_level == 1
    d5b = safety_agent.check_user_message(s, "I would like to try ordering again.")
    assert d5b.session_level == 0


# ==============================================================================
# Workflow 7: Full Web Demo HTTP Lifecycle
# ==============================================================================
def test_t4_07_full_web_demo_http_lifecycle_workflow(web_demo_server):
    """Workflow 7: Complete HTTP session lifecycle through web_demo.py."""
    # 1. GET / (index page)
    with urllib.request.urlopen(web_demo_server + "/", timeout=5) as resp:
        assert resp.status == 200
        html = resp.read().decode("utf-8")
        assert "companionai" in html.lower()

    # 2. GET /api/status
    with urllib.request.urlopen(web_demo_server + "/api/status", timeout=5) as resp:
        assert resp.status == 200
        status_data = json.loads(resp.read().decode("utf-8"))
        assert "provider" in status_data
        assert "mode" in status_data

    # 3. POST /api/new (create session)
    req_new = urllib.request.Request(
        web_demo_server + "/api/new",
        data=json.dumps({"age_group": "teen", "provider": "mock", "mode": "rules+llm"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req_new, timeout=5) as resp:
        sid = json.loads(resp.read().decode("utf-8"))["session_id"]
        assert len(sid) > 0

    # 4. POST /api/check (Turn 1: calm message)
    req_c1 = urllib.request.Request(
        web_demo_server + "/api/check",
        data=json.dumps({"session_id": sid, "text": "One chicken rice please.", "scenario": "hawker_centre"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req_c1, timeout=5) as resp:
        c1 = json.loads(resp.read().decode("utf-8"))
        assert c1["message_level"] == 0
        assert c1["action"] == "continue"

    # 5. POST /api/check (Turn 2: tense cue)
    req_c2 = urllib.request.Request(
        web_demo_server + "/api/check",
        data=json.dumps({"session_id": sid, "text": "I feel nervous and scared...", "scenario": "hawker_centre"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req_c2, timeout=5) as resp:
        c2 = json.loads(resp.read().decode("utf-8"))
        assert c2["session_level"] == 1
        assert c2["action"] == "slow_down"

    # 6. POST /api/button (Click 'Keep going')
    req_btn = urllib.request.Request(
        web_demo_server + "/api/button",
        data=json.dumps({"session_id": sid, "label": "Keep going"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req_btn, timeout=5) as resp:
        btn_resp = json.loads(resp.read().decode("utf-8"))
        assert "practice" in btn_resp["effect"].lower()

    # 7. POST /api/reply (Test OutputGuard score stripping)
    req_reply = urllib.request.Request(
        web_demo_server + "/api/reply",
        data=json.dumps({"reply": "Great job! That was 9/10 points. Have a nice meal."}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req_reply, timeout=5) as resp:
        g = json.loads(resp.read().decode("utf-8"))
        assert "9/10" not in g["text"]
        assert_no_score_leak(g["text"])

    # 8. POST /api/check (Turn 3: crisis cue)
    req_crisis = urllib.request.Request(
        web_demo_server + "/api/check",
        data=json.dumps({"session_id": sid, "text": "i want to die"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req_crisis, timeout=5) as resp:
        c3 = json.loads(resp.read().decode("utf-8"))
        assert c3["message_level"] == 3
        assert c3["crisis_hold"] is True
        assert "1767" in c3["fixed_reply"]

    # 9. POST /api/button (Release crisis hold)
    req_rel = urllib.request.Request(
        web_demo_server + "/api/button",
        data=json.dumps({"session_id": sid, "label": "I am safe now. Go back to practice"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req_rel, timeout=5) as resp:
        rel_resp = json.loads(resp.read().decode("utf-8"))
        assert rel_resp["crisis_hold"] is False
        assert rel_resp["session_level"] == 1
