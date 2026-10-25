"""Unit tests for safety.scenarios module."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety.scenarios import (
    FREE_CHAT,
    HAWKER_CENTRE,
    JOB_INTERVIEW,
    NEW_COLLEAGUE,
    SCENARIOS,
    ExampleReply,
    RubricItem,
    Scenario,
    get_scenario,
    list_scenario_ids,
    list_scenarios,
)

SCORE_LEAK_PATTERN = re.compile(
    r"\b\d+(\.\d+)?\s*/\s*(10|5|100)\b|\b\d{1,3}\s*(%|percent)\b|\b(score|grade|rating|marks|points)\b|"
    r"\b(wrong|right|correct) (answer|reply)\b|[★☆]",
    re.IGNORECASE,
)


def assert_no_score_leak(text: str) -> None:
    """Helper asserting that text contains zero evaluation score or grade markers."""
    match = SCORE_LEAK_PATTERN.search(text)
    assert not match, f"Score or rating leak detected: {match.group(0)!r} in {text!r}"


# -----------------------------------------------------------------------------
# Scenario Registry & Existence Tests
# -----------------------------------------------------------------------------

def test_all_four_scenarios_exist_in_registry():
    """Verify all 4 core scenarios are present in SCENARIOS registry."""
    expected_ids = {"hawker_centre", "new_colleague", "job_interview", "free_chat"}
    assert set(SCENARIOS.keys()) == expected_ids
    assert len(SCENARIOS) == 4


def test_list_scenarios_returns_all_four():
    """Verify list_scenarios() returns all 4 Scenario instances in order."""
    scenarios = list_scenarios()
    assert len(scenarios) == 4
    ids = [sc.id for sc in scenarios]
    assert ids == ["hawker_centre", "new_colleague", "job_interview", "free_chat"]
    for sc in scenarios:
        assert isinstance(sc, Scenario)


def test_list_scenario_ids_returns_correct_ids():
    """Verify list_scenario_ids() returns matching list of IDs."""
    ids = list_scenario_ids()
    assert ids == ["hawker_centre", "new_colleague", "job_interview", "free_chat"]


@pytest.mark.parametrize("scenario_id", ["hawker_centre", "new_colleague", "job_interview", "free_chat"])
def test_get_scenario_retrieval(scenario_id: str):
    """Verify each scenario can be retrieved by its ID."""
    sc = get_scenario(scenario_id)
    assert isinstance(sc, Scenario)
    assert sc.id == scenario_id


def test_get_scenario_whitespace_tolerance():
    """Verify get_scenario handles leading/trailing whitespace cleanly."""
    sc1 = get_scenario("  hawker_centre  ")
    assert sc1.id == "hawker_centre"

    sc2 = get_scenario("new_colleague\n")
    assert sc2.id == "new_colleague"

    sc3 = get_scenario("\tjob_interview ")
    assert sc3.id == "job_interview"


@pytest.mark.parametrize("bad_id", ["", "   ", "non_existent", "space_station", "hawker", "interview"])
def test_get_scenario_unknown_id_raises_keyerror(bad_id: str):
    """Verify unknown scenario ID raises KeyError with descriptive error message."""
    with pytest.raises(KeyError) as exc_info:
        get_scenario(bad_id)
    assert "Unknown scenario ID" in str(exc_info.value)
    assert "hawker_centre" in str(exc_info.value)


@pytest.mark.parametrize("bad_type", [None, 123, ["hawker_centre"], {"id": "hawker_centre"}])
def test_get_scenario_invalid_type_raises_keyerror(bad_type):
    """Verify non-string scenario ID raises KeyError."""
    with pytest.raises(KeyError) as exc_info:
        get_scenario(bad_type)  # type: ignore
    assert "Invalid scenario ID type" in str(exc_info.value)


# -----------------------------------------------------------------------------
# Scenario Schema & Content Validation Tests
# -----------------------------------------------------------------------------

def test_scenario_field_types_and_immutability():
    """Verify Scenario dataclass fields have expected types and frozen immutability."""
    sc = get_scenario("hawker_centre")
    assert isinstance(sc.id, str)
    assert isinstance(sc.title, str)
    assert isinstance(sc.setting, str)
    assert isinstance(sc.companion_role, str)
    assert isinstance(sc.user_role, str)
    assert isinstance(sc.opening_line, str)
    assert isinstance(sc.user_goal, str)
    assert isinstance(sc.rubric, list)
    assert isinstance(sc.example_replies, list)
    assert isinstance(sc.real_life_prompt, str)

    # Immutability check
    with pytest.raises((AttributeError, TypeError)):
        sc.id = "changed"  # type: ignore


def test_all_scenarios_meet_structural_bounds():
    """Verify all scenarios satisfy length, content, and safety invariants."""
    for sc in list_scenarios():
        # Title
        assert len(sc.title.strip()) >= 5, f"{sc.id}: title too short"
        assert_no_score_leak(sc.title)

        # Setting
        assert len(sc.setting.strip()) >= 20, f"{sc.id}: setting too short"
        assert_no_score_leak(sc.setting)

        # Roles
        assert len(sc.companion_role.strip()) > 0, f"{sc.id}: companion_role empty"
        assert len(sc.user_role.strip()) > 0, f"{sc.id}: user_role empty"
        assert_no_score_leak(sc.companion_role)
        assert_no_score_leak(sc.user_role)

        # Opening line
        assert len(sc.opening_line.strip()) >= 10, f"{sc.id}: opening_line too short"
        assert_no_score_leak(sc.opening_line)

        # User goal
        assert len(sc.user_goal.strip()) >= 15, f"{sc.id}: user_goal too short"
        assert_no_score_leak(sc.user_goal)

        # Rubric
        min_rubric = 2 if sc.id == "free_chat" else 3
        assert min_rubric <= len(sc.rubric) <= 10, f"{sc.id}: rubric count {len(sc.rubric)} out of bounds"
        criteria = [r.criterion.strip().lower() for r in sc.rubric]
        assert len(criteria) == len(set(criteria)), f"{sc.id}: duplicate rubric criterion names"

        for item in sc.rubric:
            assert isinstance(item, RubricItem)
            assert len(item.criterion.strip()) > 0
            assert len(item.description.strip()) >= 15, f"{sc.id}: description too short for {item.criterion}"
            assert_no_score_leak(item.criterion)
            assert_no_score_leak(item.description)

        # Example replies
        assert len(sc.example_replies) >= 2, f"{sc.id}: fewer than 2 example replies"
        texts = [ex.text.strip().lower() for ex in sc.example_replies]
        assert len(texts) == len(set(texts)), f"{sc.id}: duplicate example reply texts"

        for ex in sc.example_replies:
            assert isinstance(ex, ExampleReply)
            assert ex.text == ex.text.strip(), f"{sc.id}: untrimmed example reply text"
            assert ex.explanation == ex.explanation.strip(), f"{sc.id}: untrimmed example explanation"
            assert len(ex.text) > 0
            assert len(ex.explanation) > 0
            assert "only correct" not in ex.explanation.lower()
            assert_no_score_leak(ex.text)
            assert_no_score_leak(ex.explanation)

        # Real-life prompt
        assert 15 <= len(sc.real_life_prompt.strip()) <= 300, f"{sc.id}: real_life_prompt length out of bounds"
        assert any(
            v in sc.real_life_prompt.lower()
            for v in ["try", "practise", "practice", "ask", "say", "share", "talk", "order", "visit"]
        ), f"{sc.id}: real_life_prompt lacks actionable verb"
        assert_no_score_leak(sc.real_life_prompt)


# -----------------------------------------------------------------------------
# Specific Scenario Content Assertions
# -----------------------------------------------------------------------------

def test_hawker_centre_specifics():
    """Verify hawker_centre specific details."""
    sc = get_scenario("hawker_centre")
    assert "hawker" in sc.title.lower() or "food" in sc.title.lower()
    assert "chicken rice" in sc.setting.lower()
    assert "eat here or take away" in sc.opening_line.lower()

    # Rubric checks
    criteria = {r.criterion.lower() for r in sc.rubric}
    rubric_text = " ".join([r.criterion.lower() + " " + r.description.lower() for r in sc.rubric])
    assert any(w in rubric_text for w in ["food", "order", "takeaway", "clarity"])

    # Example replies
    assert any("chicken rice" in ex.text.lower() for ex in sc.example_replies)
    assert any("take away" in ex.text.lower() for ex in sc.example_replies)


def test_new_colleague_specifics():
    """Verify new_colleague specific details."""
    sc = get_scenario("new_colleague")
    assert "colleague" in sc.title.lower() or "small talk" in sc.title.lower()
    assert "pantry" in sc.setting.lower()
    assert "alex" in sc.opening_line.lower()

    # Rubric checks
    criteria = {r.criterion.lower() for r in sc.rubric}
    assert any("polite" in c or "acknowledgment" in c for c in criteria)
    assert any("relevance" in c for c in criteria)


def test_job_interview_specifics():
    """Verify job_interview specific details."""
    sc = get_scenario("job_interview")
    assert "interview" in sc.title.lower()
    assert "ms. tan" in sc.companion_role.lower()
    assert "tell me a little about yourself" in sc.opening_line.lower()

    # Rubric checks
    rubric_text = " ".join([r.criterion.lower() + " " + r.description.lower() for r in sc.rubric])
    assert any(w in rubric_text for w in ["skill", "experience", "role", "clarity"])


def test_free_chat_specifics():
    """Verify free_chat specific details."""
    sc = get_scenario("free_chat")
    assert "free" in sc.title.lower() or "chat" in sc.title.lower()
    assert "?" in sc.opening_line

    # Rubric criteria
    criteria_names = [item.criterion.lower() for item in sc.rubric]
    assert any("clarity" in c or "topic" in c or "expression" in c for c in criteria_names)

    # Example replies
    assert any("mrt" in ex.text.lower() for ex in sc.example_replies)


def test_to_dict_method():
    """Verify to_dict returns complete dictionary representation."""
    for sc in list_scenarios():
        d = sc.to_dict()
        assert isinstance(d, dict)
        assert d["id"] == sc.id
        assert d["title"] == sc.title
        assert d["setting"] == sc.setting
        assert d["companion_role"] == sc.companion_role
        assert d["user_role"] == sc.user_role
        assert d["opening_line"] == sc.opening_line
        assert d["user_goal"] == sc.user_goal
        assert isinstance(d["rubric"], list)
        assert len(d["rubric"]) == len(sc.rubric)
        assert isinstance(d["example_replies"], list)
        assert len(d["example_replies"]) == len(sc.example_replies)
        assert d["real_life_prompt"] == sc.real_life_prompt
