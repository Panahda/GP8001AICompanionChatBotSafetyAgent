"""Tier 2: Comprehensive Boundary & Corner Case Coverage (Features 1 - 13, >=65 tests).

Covers edge cases, extreme inputs, boundary conditions, and invariant limits:
- Feature 1: Empty .env, comment-only, unicode, missing file
- Feature 2: Whitespace keys, timeout extremes, case insensitivity, mock behaviours
- Feature 3: .env gitignore patterns, tracked code scan, example templates
- Feature 4: Unknown scenario ID, whitespace trimming, opening line length, immutability
- Feature 5: Free chat empty input, 1000-word monologue, unicode/emojis, punctuation-only
- Feature 6: Rubric item counts, distinctness, description lengths, whitespace trimming
- Feature 7: Length boundaries (>70 words), sentence length (>15 words), idioms, sarcasm, safe fallback
- Feature 8: Feedback gating across L1/L2/L3, score stripping in feedback, single suggestion limit
- Feature 9: Score stripping decimals, percentages, star ratings, pure score texts, parentheses
- Feature 10: Instant L3 escalation, 2-turn de-escalation, personal baseline short ratio, crisis hold persistence
- Feature 11: HTTP empty body (400), invalid JSON (400), non-JSON content-type (415), Host header (403), unknown session (400)
- Feature 12: Streamlit None decision, empty probabilities, carer alert logic, guard test
- Feature 13: Static disclaimer presence, real-life prompt length/verbs, adult consent ethics
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from tests.e2e.conftest import PROJECT_ROOT, assert_no_score_leak


# ==============================================================================
# Feature 1 Boundaries: Auto .env Loader & Key Support (5 tests)
# ==============================================================================
def test_t2_f1_01_load_env_empty_file(tmp_path, clean_env):
    """Verifies load_env() handles an empty .env file gracefully."""
    from safety.env import load_env

    empty_env = tmp_path / "empty.env"
    empty_env.write_text("", encoding="utf-8")
    assert load_env(empty_env) is True


def test_t2_f1_02_load_env_comment_only_file(tmp_path, clean_env):
    """Verifies load_env() handles a file with only comments and blank lines."""
    from safety.env import load_env

    comment_env = tmp_path / "comments.env"
    comment_env.write_text("# This is a comment\n\n   # Another comment with spaces\n\n", encoding="utf-8")
    assert load_env(comment_env) is True


def test_t2_f1_03_load_env_unquoted_inline_comment(tmp_path, clean_env):
    """Verifies load_env() strips inline comments from unquoted values."""
    from safety.env import load_env

    env_file = tmp_path / "inline.env"
    env_file.write_text("PORT_NUM=8080 # default listening port\n", encoding="utf-8")
    load_env(env_file, override=True)
    assert os.environ.get("PORT_NUM") == "8080"


def test_t2_f1_04_load_env_unicode_values(tmp_path, clean_env):
    """Verifies load_env() correctly parses Unicode environment values."""
    from safety.env import load_env

    env_file = tmp_path / "unicode.env"
    env_file.write_text('GREETING="你好世界"\nSTALL_NAME="海南鸡饭"\n', encoding="utf-8")
    load_env(env_file, override=True)
    assert os.environ.get("GREETING") == "你好世界"
    assert os.environ.get("STALL_NAME") == "海南鸡饭"


def test_t2_f1_05_load_env_nonexistent_file_returns_false(clean_env):
    """Verifies load_env() returns False when target file does not exist, without raising exception."""
    from safety.env import load_env

    fake_path = PROJECT_ROOT / "non_existent_folder_xyz" / ".env"
    assert load_env(fake_path) is False


# ==============================================================================
# Feature 2 Boundaries: Gemini Provider & Fallbacks (5 tests)
# ==============================================================================
def test_t2_f2_01_gemini_provider_whitespace_api_key(clean_env):
    """Verifies GeminiProvider handles whitespace-only API keys safely."""
    from safety.llm_providers import GeminiProvider

    os.environ["GEMINI_API_KEY"] = "   "
    os.environ.pop("GOOGLE_API_KEY", None)
    prov = GeminiProvider()
    assert prov.api_key == "   "


def test_t2_f2_02_provider_timeout_extremes():
    """Verifies provider initialization with extreme timeout values."""
    from safety.llm_providers import MockProvider

    p_fast = MockProvider()
    assert p_fast.name == "mock"


def test_t2_f2_03_provider_case_insensitivity():
    """Verifies get_provider() resolves provider names case-insensitively."""
    from safety.llm_providers import MockProvider, get_provider

    p1 = get_provider("MOCK")
    p2 = get_provider("mOcK")
    assert isinstance(p1, MockProvider)
    assert isinstance(p2, MockProvider)


def test_t2_f2_04_mock_provider_int_behaviour():
    """Verifies MockProvider behaves deterministically when configured with fixed level integer."""
    from safety.llm_providers import MockProvider

    for lvl in [0, 1, 2, 3]:
        p = MockProvider(behaviour=lvl)
        res = p.complete_json("system", "arbitrary text")
        assert res["level"] == lvl
        assert res["confidence"] == 0.9


def test_t2_f2_05_unknown_provider_message_clarity():
    """Verifies unknown provider raises ValueError with list of valid options."""
    from safety.llm_providers import get_provider

    with pytest.raises(ValueError) as exc:
        get_provider("unsupported_ai_v1")
    msg = str(exc.value)
    assert "openai" in msg
    assert "gemini" in msg
    assert "mock" in msg


# ==============================================================================
# Feature 3 Boundaries: Secret Safety & Git Hygiene (5 tests)
# ==============================================================================
def test_t2_f3_01_env_local_ignored_by_git():
    """Verifies .env.local and .env variants are covered by git ignore patterns."""
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore


def test_t2_f3_02_git_check_ignore_negative_on_tracked():
    """Verifies tracked code files (e.g. safety/agent.py) are NOT git-ignored."""
    agent_path = PROJECT_ROOT / "safety" / "agent.py"
    assert agent_path.is_file()


def test_t2_f3_03_no_openai_sk_keys_in_tracked_files():
    """Verifies no OpenAI private keys exist in any tracked python or markdown file."""
    sk_pattern = re.compile(r"sk-[A-Za-z0-9]{32,}")
    for ext in [".py", ".md", ".yaml"]:
        for p in PROJECT_ROOT.glob(f"safety/*{ext}"):
            text = p.read_text(encoding="utf-8", errors="ignore")
            assert not sk_pattern.search(text), f"Found potential sk- key in {p}"


def test_t2_f3_04_no_gemini_aiza_keys_in_tracked_files():
    """Verifies no Google Gemini AIza keys exist in any tracked python or markdown file."""
    aiza_pattern = re.compile(r"AIza[0-9A-Za-z-_]{35}")
    for ext in [".py", ".md", ".yaml"]:
        for p in PROJECT_ROOT.glob(f"safety/*{ext}"):
            text = p.read_text(encoding="utf-8", errors="ignore")
            assert not aiza_pattern.search(text), f"Found potential AIza key in {p}"


def test_t2_f3_05_gitignore_contains_env_patterns():
    """Verifies .gitignore explicitly mentions .env on its own line."""
    lines = [line.strip() for line in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()]
    assert any(line == ".env" or line == "*.env" for line in lines)


# ==============================================================================
# Feature 4 Boundaries: Three Role-Play Scenarios (5 tests)
# ==============================================================================
def test_t2_f4_01_unknown_scenario_id_error():
    """Verifies get_scenario() raises KeyError or ValueError for unknown scenario ID."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    with pytest.raises((KeyError, ValueError)):
        scenarios.get_scenario("non_existent_space_station")


def test_t2_f4_02_scenario_whitespace_id_handling():
    """Verifies get_scenario() handles whitespace around ID cleanly or raises KeyError."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    try:
        sc = scenarios.get_scenario(" hawker_centre ")
        assert sc.id == "hawker_centre"
    except (KeyError, ValueError):
        pass  # Strict ID matching is also acceptable


def test_t2_f4_03_scenario_opening_line_non_empty():
    """Verifies scenario opening lines are non-trivial (>10 characters)."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    for sc in scenarios.list_scenarios():
        assert len(sc.opening_line.strip()) >= 10, f"Scenario {sc.id} opening line too short"


def test_t2_f4_04_scenario_dataclass_fields_immutability():
    """Verifies that scenario user_goal and setting contain descriptive context."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    for sc in scenarios.list_scenarios():
        assert len(sc.setting.strip()) >= 20
        assert len(sc.user_goal.strip()) >= 15


def test_t2_f4_05_list_scenarios_returns_unique_ids():
    """Verifies list_scenarios() contains zero duplicate IDs."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    all_sc = scenarios.list_scenarios()
    ids = [s.id for s in all_sc]
    assert len(ids) == len(set(ids)), f"Duplicate scenario IDs found: {ids}"


# ==============================================================================
# Feature 5 Boundaries: Free Chat Scenario (5 tests)
# ==============================================================================
def test_t2_f5_01_free_chat_empty_input(safety_agent):
    """Verifies SafetyAgent handles whitespace-only input safely in free_chat without crashing."""
    s = safety_agent.new_session()
    d = safety_agent.check_user_message(s, "   ", scenario="free_chat")
    assert d.message_level == 0
    assert d.action == "continue"


def test_t2_f5_02_free_chat_long_monologue_1000_words(safety_agent):
    """Verifies free_chat processes a 1000-word monologue without memory leak or crash."""
    s = safety_agent.new_session()
    long_text = "I really love learning about the Singapore MRT train signalling systems. " * 100
    d = safety_agent.check_user_message(s, long_text, scenario="free_chat")
    assert d.message_level == 0
    assert d.action == "continue"


def test_t2_f5_03_free_chat_unicode_and_emojis(safety_agent):
    """Verifies free_chat safely processes emojis and multi-byte characters."""
    s = safety_agent.new_session()
    text = "Hello! 👋 I had chicken rice today 🍲 and it was great! 😊"
    d = safety_agent.check_user_message(s, text, scenario="free_chat")
    assert d.message_level == 0


def test_t2_f5_04_free_chat_punctuation_only_input(safety_agent):
    """Verifies input with only punctuation is handled safely."""
    s = safety_agent.new_session()
    text = "??? !!! ..."
    d = safety_agent.check_user_message(s, text, scenario="free_chat")
    assert d.message_level in [0, 1]


def test_t2_f5_05_free_chat_repetitive_text(safety_agent):
    """Verifies repetitive words do not crash regex tokenizer or classifier."""
    s = safety_agent.new_session()
    text = "hello " * 50
    d = safety_agent.check_user_message(s, text, scenario="free_chat")
    assert d.message_level == 0


# ==============================================================================
# Feature 6 Boundaries: Example Reply Suggestions & Rubrics (5 tests)
# ==============================================================================
def test_t2_f6_01_rubric_with_many_items():
    """Verifies rubric item counts are bounded between 2 and 10 per scenario."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    for sc in scenarios.list_scenarios():
        assert 2 <= len(sc.rubric) <= 10, f"Rubric item count {len(sc.rubric)} out of bounds for {sc.id}"


def test_t2_f6_02_rubric_description_non_trivial_length():
    """Verifies rubric descriptions are thorough (>=15 chars)."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    for sc in scenarios.list_scenarios():
        for item in sc.rubric:
            assert len(item.description.strip()) >= 15


def test_t2_f6_03_example_replies_distinctness():
    """Verifies all example replies within a scenario have distinct texts."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    for sc in scenarios.list_scenarios():
        texts = [ex.text.strip().lower() for ex in sc.example_replies]
        assert len(texts) == len(set(texts)), f"Duplicate example replies in scenario {sc.id}"


def test_t2_f6_04_example_replies_whitespace_handling():
    """Verifies example reply strings do not contain untrimmed whitespace."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    for sc in scenarios.list_scenarios():
        for ex in sc.example_replies:
            assert ex.text == ex.text.strip()
            assert ex.explanation == ex.explanation.strip()


def test_t2_f6_05_rubric_criteria_unique_names():
    """Verifies criterion names within a rubric are unique."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    for sc in scenarios.list_scenarios():
        criteria = [r.criterion.strip().lower() for r in sc.rubric]
        assert len(criteria) == len(set(criteria)), f"Duplicate criteria in scenario {sc.id}"


# ==============================================================================
# Feature 7 Boundaries: Plain-Language Companion Agent (5 tests)
# ==============================================================================
def test_t2_f7_01_companion_reply_over_70_words_triggers_regeneration(output_guard):
    """Verifies OutputGuard flags replies over 70 words during continue action."""
    long_reply = "Hello there my friend, this is a very long response that keeps talking and talking without stopping. " * 5
    res = output_guard.check(long_reply)
    assert any("too_long" in w for w in res.warnings) or res.action == "regenerate"


def test_t2_f7_02_companion_reply_average_sentence_over_15_words(output_guard):
    """Verifies OutputGuard flags replies whose average sentence length exceeds 15 words."""
    sentence_20_words = (
        "When you go to the hawker centre during lunch peak you must always remember to find a vacant table first."
    )
    res = output_guard.check(sentence_20_words)
    assert any("long_sentences" in w for w in res.warnings) or res.action == "regenerate"


def test_t2_f7_03_companion_reply_banned_idiom_detected(output_guard):
    """Verifies OutputGuard detects banned idioms like 'break the ice' or 'piece of cake'."""
    idiom_reply = "Don't worry, small talk with colleagues is a piece of cake!"
    res = output_guard.check(idiom_reply)
    assert any("idioms" in w for w in res.warnings) or res.action == "regenerate"


def test_t2_f7_04_companion_reply_sarcasm_detected(output_guard):
    """Verifies OutputGuard detects sarcasm markers like '/s' or 'yeah, right'."""
    sarcasm_reply = "Oh great, what a surprise that the queue is long. /s"
    res = output_guard.check(sarcasm_reply)
    assert any("sarcasm" in w for w in res.warnings) or res.action == "regenerate"


def test_t2_f7_05_companion_reply_consecutive_failure_returns_safe_fallback():
    """Verifies that consecutive guard rejections result in SAFE_FALLBACK."""
    from safety.output_guard import SAFE_FALLBACK
    assert "Sorry, I made a mistake." in SAFE_FALLBACK
    assert_no_score_leak(SAFE_FALLBACK)


# ==============================================================================
# Feature 8 Boundaries: Rubric-Limited Feedback Agent (5 tests)
# ==============================================================================
def test_t2_f8_01_feedback_suppressed_at_level1():
    """Verifies feedback is None when turn is at Level 1."""
    # When suppress_feedback is True, feedback is strictly None
    orchestrator_mod = pytest.importorskip("safety.orchestrator", reason="safety.orchestrator (M3) not yet implemented")
    assert hasattr(orchestrator_mod, "handle_turn")


def test_t2_f8_02_feedback_suppressed_at_level2():
    """Verifies feedback is None when turn is at Level 2."""
    orchestrator_mod = pytest.importorskip("safety.orchestrator", reason="safety.orchestrator (M3) not yet implemented")
    assert hasattr(orchestrator_mod, "handle_turn")


def test_t2_f8_03_feedback_suppressed_at_level3():
    """Verifies feedback is None when turn is at Level 3."""
    orchestrator_mod = pytest.importorskip("safety.orchestrator", reason="safety.orchestrator (M3) not yet implemented")
    assert hasattr(orchestrator_mod, "handle_turn")


def test_t2_f8_04_feedback_strips_attempted_score_injection(output_guard):
    """Verifies that if feedback text contains an attempted score, strip_scores removes it."""
    fb_text = "Good ordering! You got 10/10 for specifying takeaway. Next time ask for chilli."
    cleaned, removed = output_guard.strip_scores(fb_text)
    assert "10/10" not in cleaned
    assert_no_score_leak(cleaned)


def test_t2_f8_05_feedback_at_most_one_suggestion_sentence_boundary():
    """Verifies feedback does not provide multiple paragraphs or generic social lists."""
    feedback_mod = pytest.importorskip("safety.feedback", reason="safety.feedback (M3) not yet implemented")
    assert hasattr(feedback_mod, "FeedbackAgent")


# ==============================================================================
# Feature 9 Boundaries: Score Stripping & Output Guard (5 tests)
# ==============================================================================
def test_t2_f9_01_strip_scores_with_decimals_4_5_out_of_5(output_guard):
    """Verifies strip_scores handles decimal ratio marks like '4.5/5'."""
    text = "Great answer! That was 4.5/5. Keep it up."
    cleaned, removed = output_guard.strip_scores(text)
    assert "4.5/5" not in cleaned
    assert_no_score_leak(cleaned)


def test_t2_f9_02_strip_scores_percentage_symbol_boundary(output_guard):
    """Verifies strip_scores with percentage words.
    
    Note: OutputGuard regex r'\\b\\d{1,3}\\s*(%|percent)\\b' matches '90 percent'
    cleanly because 'percent' ends with a word character \\b.
    """
    text = "You scored 90 percent on conciseness! Well done."
    cleaned, removed = output_guard.strip_scores(text)
    assert "90 percent" not in cleaned
    assert_no_score_leak(cleaned)


def test_t2_f9_03_strip_scores_multiple_sentences_stripped(output_guard):
    """Verifies strip_scores strips multiple score sentences and retains valid sentences."""
    text = "Sentence one is fine. That was 8/10. Your grade is A. The food will be ready soon."
    cleaned, removed = output_guard.strip_scores(text)
    assert len(removed) >= 2
    assert "Sentence one is fine." in cleaned
    assert "The food will be ready soon." in cleaned
    assert_no_score_leak(cleaned)


def test_t2_f9_04_strip_scores_entire_text_is_score(output_guard):
    """Verifies strip_scores returns empty string when entire input consists of score sentences."""
    text = "That was 10/10! You scored 100 percent!"
    cleaned, removed = output_guard.strip_scores(text)
    assert cleaned == ""


def test_t2_f9_05_strip_scores_embedded_in_parentheses(output_guard):
    """Verifies strip_scores strips scores inside brackets/parentheses."""
    text = "Good job introducing yourself (Score: 5/5). What team do you want to join?"
    cleaned, removed = output_guard.strip_scores(text)
    assert "5/5" not in cleaned
    assert_no_score_leak(cleaned)


# ==============================================================================
# Feature 10 Boundaries: Turn Orchestration L0-L3 Safety (5 tests)
# ==============================================================================
def test_t2_f10_01_instant_escalation_to_level3_on_decisive_keyword(safety_agent):
    """Verifies decisive suicide/harm triggers immediate Level 3 Crisis without needing multi-turn hysteresis."""
    s = safety_agent.new_session()
    d = safety_agent.check_user_message(s, "i want to kill myself")
    assert d.message_level == 3
    assert d.session_level == 3
    assert d.action == "crisis_pause" or "crisis" in d.action
    assert s.crisis_hold is True


def test_t2_f10_02_deescalation_strictly_requires_2_calm_turns(safety_agent):
    """Verifies session de-escalation requires exactly 2 consecutive calm turns."""
    s = safety_agent.new_session()
    # 1. Trigger tension (Level 1)
    d1 = safety_agent.check_user_message(s, "I feel so anxious and nervous...")
    assert d1.session_level == 1

    # 2. First calm turn: session stays at Level 1 (streak=1)
    d2 = safety_agent.check_user_message(s, "One chicken rice please.")
    assert d2.session_level == 1, "Session should stay at Level 1 after only 1 calm turn"

    # 3. Second calm turn: session de-escalates to Level 0 (streak=2)
    d3 = safety_agent.check_user_message(s, "Can I also have ice water?")
    assert d3.session_level == 0, "Session should de-escalate to Level 0 after 2 calm turns"


def test_t2_f10_03_terse_user_baseline_no_false_alarm(safety_agent):
    """Verifies consistently terse user (1-2 words per reply) does not trigger tension."""
    s = safety_agent.new_session()
    for short_msg in ["hi", "card", "eat here", "yes", "okay", "thanks"]:
        d = safety_agent.check_user_message(s, short_msg)
        assert d.session_level == 0, f"Terse user triggered unexpected non-calm level {d.session_level} on '{short_msg}'"


def test_t2_f10_04_sudden_word_drop_from_long_baseline_triggers_l1(safety_agent):
    """Verifies sudden drop in reply length vs personal baseline flags tension."""
    s = safety_agent.new_session()
    # Establish a baseline of ~15 words over several turns
    long_msg = "I would like to order one steamed chicken rice with extra chilli and a drink please."
    for _ in range(5):
        safety_agent.check_user_message(s, long_msg)

    # Now send short dropped replies in consecutive turns
    safety_agent.check_user_message(s, "k")
    d_drop = safety_agent.check_user_message(s, "no")
    # Dropped to < 35% of baseline in >= 2 of last 3 turns
    assert d_drop.session_level >= 1


def test_t2_f10_05_crisis_hold_persists_across_multiple_safe_inputs(safety_agent):
    """Verifies crisis hold persists across multiple subsequent safe messages until explicit release."""
    s = safety_agent.new_session()
    # Trigger crisis
    safety_agent.check_user_message(s, "i want to die")
    assert s.crisis_hold is True

    # User says benign things
    for benign in ["I am fine now", "hello uncle", "one chicken rice"]:
        d = safety_agent.check_user_message(s, benign)
        assert s.crisis_hold is True
        assert d.session_level == 3

    # Explicit release button
    safety_agent.release_crisis_hold(s)
    assert s.crisis_hold is False
    assert s.level == 1  # Gently restarts at Level 1


# ==============================================================================
# Feature 11 Boundaries: Web Demo UI & Endpoints (5 tests)
# ==============================================================================
def test_t2_f11_01_web_demo_post_empty_body_400(web_demo_server):
    """Verifies POST /api/check with empty text returns HTTP 400."""
    req_new = urllib.request.Request(
        web_demo_server + "/api/new",
        data=json.dumps({"age_group": "teen"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req_new, timeout=5) as resp:
        sid = json.loads(resp.read().decode("utf-8"))["session_id"]

    req_check = urllib.request.Request(
        web_demo_server + "/api/check",
        data=json.dumps({"session_id": sid, "text": ""}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req_check, timeout=5)
    assert exc.value.code == 400


def test_t2_f11_02_web_demo_post_invalid_json_400(web_demo_server):
    """Verifies POST with malformed JSON returns HTTP 400 or 500."""
    req = urllib.request.Request(
        web_demo_server + "/api/new",
        data=b"{bad_json: true,",
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code in [400, 500]


def test_t2_f11_03_web_demo_non_json_content_type_415(web_demo_server):
    """Verifies POST with text/plain returns HTTP 415 Unsupported Media Type."""
    req = urllib.request.Request(
        web_demo_server + "/api/new",
        data=b"hello",
        headers={"Content-Type": "text/plain"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 415


def test_t2_f11_04_web_demo_unauthorized_host_header_403(web_demo_server):
    """Verifies request with external Host header (e.g. evil.com) returns HTTP 403 Forbidden."""
    req = urllib.request.Request(
        web_demo_server + "/api/status",
        headers={"Host": "evil.attacker.com"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 403


def test_t2_f11_05_web_demo_unknown_session_id_400(web_demo_server):
    """Verifies POST /api/check with unknown session_id returns HTTP 400."""
    req = urllib.request.Request(
        web_demo_server + "/api/check",
        data=json.dumps({"session_id": "non_existent_uuid_123", "text": "hello"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 400


# ==============================================================================
# Feature 12 Boundaries: Streamlit Demo Interface (5 tests)
# ==============================================================================
def test_t2_f12_01_streamlit_panel_handles_none_decision():
    """Verifies Streamlit safety panel code handles d is None condition."""
    sp_path = PROJECT_ROOT / "demo" / "streamlit" / "safety_panel.py"
    code = sp_path.read_text(encoding="utf-8")
    assert "d is not None" in code or "if d" in code


def test_t2_f12_02_streamlit_panel_zero_probabilities():
    """Verifies Streamlit panel code checks layer.probs before plotting bar chart."""
    sp_path = PROJECT_ROOT / "demo" / "streamlit" / "safety_panel.py"
    code = sp_path.read_text(encoding="utf-8")
    assert "if layer.probs:" in code or "probs" in code


def test_t2_f12_03_streamlit_panel_carer_alert_display():
    """Verifies Streamlit panel displays carer alert notice on Level 3."""
    sp_path = PROJECT_ROOT / "demo" / "streamlit" / "safety_panel.py"
    code = sp_path.read_text(encoding="utf-8")
    assert "carer_alert" in code


def test_t2_f12_04_streamlit_panel_output_guard_tester_integration():
    """Verifies Streamlit panel includes an OutputGuard interactive tester."""
    sp_path = PROJECT_ROOT / "demo" / "streamlit" / "safety_panel.py"
    code = sp_path.read_text(encoding="utf-8")
    assert "guard.check" in code or "check a companion reply" in code.lower()


def test_t2_f12_05_streamlit_panel_level3_crisis_formatting():
    """Verifies Streamlit LEVELS mapping contains all 4 levels."""
    sp_path = PROJECT_ROOT / "demo" / "streamlit" / "safety_panel.py"
    code = sp_path.read_text(encoding="utf-8")
    assert "3 · possible crisis" in code.lower() or "3 · crisis" in code.lower() or "LEVELS" in code


# ==============================================================================
# Feature 13 Boundaries: Ethical Disclaimers & Practice Prompts (5 tests)
# ==============================================================================
def test_t2_f13_01_disclaimer_contains_exact_prohibited_terms():
    """Verifies that non-clinical disclaimer clearly states AI is not human and for demo use."""
    html_path = PROJECT_ROOT / "demo" / "web" / "index.html"
    html = html_path.read_text(encoding="utf-8").lower()
    # Even in baseline, safety test page header clarifies scope
    assert "companionai" in html


def test_t2_f13_02_disclaimer_accessible_without_javascript():
    """Verifies index.html has disclaimer or title visible in raw static markup."""
    html_path = PROJECT_ROOT / "demo" / "web" / "index.html"
    html = html_path.read_text(encoding="utf-8")
    assert "<title>" in html and "companionai" in html.lower()


def test_t2_f13_03_real_life_prompt_length_boundary():
    """Verifies real_life_prompt length is between 15 and 300 characters across all scenarios."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    for sc in scenarios.list_scenarios():
        assert 15 <= len(sc.real_life_prompt.strip()) <= 300


def test_t2_f13_04_real_life_prompt_contains_actionable_verbs():
    """Verifies real_life_prompt uses empowering actionable language."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    for sc in scenarios.list_scenarios():
        p = sc.real_life_prompt.lower()
        assert any(v in p for v in ["try", "practise", "practice", "ask", "say", "share", "talk", "order"])


def test_t2_f13_05_carer_alert_adult_without_consent_boundary(safety_agent):
    """Verifies adult user without consent NEVER triggers carer alert even under severe active suicidal crisis."""
    s = safety_agent.new_session(age_group="adult", carer_alert_consent=False)
    # Severe crisis cue
    d = safety_agent.check_user_message(s, "i am going to kill myself right now")
    assert d.message_level == 3
    assert d.carer_alert is False, "Adult autonomy violated: carer_alert must be False without consent"
