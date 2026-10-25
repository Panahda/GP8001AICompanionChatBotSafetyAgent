"""The local test page's API (demo/web_demo.py), run with the offline mock LLM."""
from __future__ import annotations

import json
import re
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import web_demo  # noqa: E402

OPEN = urllib.request.build_opener(urllib.request.ProxyHandler({})).open  # never send localhost through a proxy

SCORE_REGEX = re.compile(
    r"\b(\d+(\.\d+)?\s*/\s*\d+|\d{1,3}\s*%|\d+\s*(points?|marks?)|grade\s+[A-F][+-]?)\b",
    re.IGNORECASE,
)


def assert_no_scores(text: str | None) -> None:
    if not text:
        return
    match = SCORE_REGEX.search(text)
    assert match is None, f"Detected forbidden score leak in user-facing text: {match.group(0)!r} in '{text}'"


@pytest.fixture(scope="module")
def base():
    server = web_demo.make_server(web_demo.Demo("mock", "rules+llm"), 0)  # port 0 = any free port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://localhost:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def get(base, path):
    req = urllib.request.Request(base + path)
    with OPEN(req, timeout=10) as r:
        return json.load(r)


def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with OPEN(req, timeout=10) as r:
        return json.load(r)


def test_page_and_status(base):
    with OPEN(base + "/", timeout=10) as r:
        assert b"CompanionAI safety check" in r.read()
    with OPEN(base + "/api/status", timeout=10) as r:
        assert json.load(r)["provider"] == "mock"


def test_crisis_then_release(base):
    sid = post(base, "/api/new", {"age_group": "teen"})["session_id"]
    d = post(base, "/api/check", {"session_id": sid, "text": "One chicken rice please.", "scenario": "hawker_centre"})
    assert d["message_level"] == 0 and d["action"] == "continue"
    d = post(base, "/api/check", {"session_id": sid, "text": "i want to die"})
    assert d["message_level"] == 3 and "1767" in d["fixed_reply"] and d["crisis_hold"] and d["carer_alert"]
    j = post(base, "/api/button", {"session_id": sid, "label": "I am safe now. Go back to practice"})
    assert j["crisis_hold"] is False


def test_reply_check_strips_scores(base):
    g = post(base, "/api/reply", {"session_id": None, "reply": "Good job! That was 8/10."})
    assert "8/10" not in g["text"]


def test_refuses_other_hosts_and_unknown_sessions(base):
    req = urllib.request.Request(base + "/api/status", headers={"Host": "evil.example"})
    with pytest.raises(urllib.error.HTTPError) as e:
        OPEN(req, timeout=10)
    assert e.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as e:
        post(base, "/api/check", {"session_id": "nope", "text": "hi"})
    assert e.value.code == 400


def test_api_scenarios_listing(base):
    """Verifies /api/scenarios returns all registered scenarios with complete metadata."""
    data_get = get(base, "/api/scenarios")
    assert "scenarios" in data_get
    sc_list = data_get["scenarios"]
    sc_ids = [s["id"] for s in sc_list]
    for required in ["hawker_centre", "new_colleague", "job_interview", "free_chat"]:
        assert required in sc_ids

    # Also verify POST /api/scenarios works identically
    data_post = post(base, "/api/scenarios", {})
    assert "scenarios" in data_post

    # Verify metadata fields
    hawker = next(s for s in sc_list if s["id"] == "hawker_centre")
    assert "setting" in hawker and len(hawker["setting"]) > 10
    assert "opening_line" in hawker and len(hawker["opening_line"]) > 5
    assert "user_goal" in hawker and len(hawker["user_goal"]) > 5
    assert "rubric" in hawker and len(hawker["rubric"]) >= 3
    assert "example_replies" in hawker and len(hawker["example_replies"]) >= 3
    assert "real_life_prompt" in hawker and len(hawker["real_life_prompt"]) > 10


def test_api_new_scenario_initialization(base):
    """Verifies POST /api/new initializes session with scenario opening line and metadata."""
    res = post(base, "/api/new", {"scenario": "hawker_centre", "age_group": "teen"})
    assert "session_id" in res
    assert "status" in res
    assert "opening_line" in res
    assert "user_goal" in res
    assert "setting" in res
    assert "example_replies" in res
    assert "real_life_prompt" in res
    assert "chicken" in res["opening_line"].lower() or "eat here" in res["opening_line"].lower()


def test_multi_turn_scenario_chat_calm_and_feedback(base):
    """Verifies multi-turn scenario chat returns plain companion dialogue, feedback, and prompt."""
    sid = post(base, "/api/new", {"scenario": "hawker_centre"})["session_id"]

    # Turn 1: user orders food
    t1 = post(base, "/api/check", {
        "session_id": sid,
        "text": "One plate of roasted chicken rice, please. Eat here.",
        "scenario": "hawker_centre",
    })
    assert t1["message_level"] == 0
    assert t1["action"] == "continue"
    assert "reply" in t1 and len(t1["reply"]) > 0
    assert "feedback" in t1 and t1["feedback"] is not None
    assert "end_of_session_prompt" in t1 and len(t1["end_of_session_prompt"]) > 10
    assert "buttons" in t1
    assert "decision" in t1
    assert_no_scores(t1["reply"])
    assert_no_scores(t1["feedback"])

    # Turn 2: user responds to payment
    t2 = post(base, "/api/check", {
        "session_id": sid,
        "text": "I will pay with cash.",
        "scenario": "hawker_centre",
    })
    assert t2["message_level"] == 0
    assert "reply" in t2 and len(t2["reply"]) > 0
    assert_no_scores(t2["reply"])
    if t2["feedback"]:
        assert_no_scores(t2["feedback"])


def test_tense_turn_slow_down_and_feedback_suppression(base):
    """Verifies Level 1 tense turns return slow-down directives, suppress feedback, and offer choices."""
    sid = post(base, "/api/new", {"scenario": "job_interview"})["session_id"]
    d = post(base, "/api/check", {
        "session_id": sid,
        "text": "I feel really stressed and nervous about answering this.",
        "scenario": "job_interview",
    })
    assert d["message_level"] == 1
    assert d["action"] == "slow_down"
    # Feedback MUST be suppressed at Level 1
    assert d["feedback"] is None
    # Reply must be generated adhering to slow down / choices
    assert "reply" in d and len(d["reply"]) > 0
    assert_no_scores(d["reply"])
    # Buttons must contain choices
    assert len(d["buttons"]) > 0 or len(d["choices"]) > 0


def test_overwhelmed_turn_level_2_playbook_pause(base):
    """Verifies Level 2 overwhelmed turns pause roleplay, suppress feedback, and return playbook."""
    sid = post(base, "/api/new", {"scenario": "new_colleague"})["session_id"]
    d = post(base, "/api/check", {
        "session_id": sid,
        "text": "I cannot take this anymore, everything is too loud and sensory overload!",
        "scenario": "new_colleague",
    })
    assert d["message_level"] == 2
    assert d["action"] == "pause_regulate"
    assert d["feedback"] is None
    assert d["fixed_reply"] is not None
    assert "reply" in d and d["reply"] == d["fixed_reply"]
    assert any("it" in b.lower() or "stop" in b.lower() or "something" in b.lower() for b in d["buttons"])


def test_api_action_endpoints(base):
    """Verifies /api/action handles user action selections (continue, release, stop)."""
    sid = post(base, "/api/new", {"scenario": "hawker_centre"})["session_id"]

    # 1. Trigger crisis
    post(base, "/api/check", {"session_id": sid, "text": "i want to kill myself"})

    # 2. Call /api/action to release crisis hold
    act1 = post(base, "/api/action", {"session_id": sid, "label": "I am safe now. Go back to practice"})
    assert act1["crisis_hold"] is False
    assert "released" in act1["effect"].lower()

    # 3. Call /api/action with continue
    act2 = post(base, "/api/action", {"session_id": sid, "action": "Keep going"})
    assert "effect" in act2

    # 4. Call /api/action with Stop for today
    act3 = post(base, "/api/action", {"session_id": sid, "label": "Stop for today"})
    assert "end" in act3["effect"].lower()


def test_api_guard_test_endpoint(base):
    """Verifies /api/guard_test strips scores from candidate text."""
    res1 = post(base, "/api/guard_test", {"reply": "Good attempt! That was 9/10 points."})
    assert "9/10" not in res1["text"]
    assert_no_scores(res1["text"])

    res2 = post(base, "/api/guard_test", {"text": "Great! 100% accurate ordering."})
    assert "100%" not in res2["text"]
    assert_no_scores(res2["text"])
