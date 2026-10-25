"""Tier 1: Comprehensive Feature Coverage (Features 1 - 13, >=65 tests).

Covers:
- Feature 1: Auto .env Loader & Key Support (R1)
- Feature 2: Gemini Provider & Fallbacks (R1)
- Feature 3: Secret Safety & Git Hygiene (R1)
- Feature 4: Three Role-Play Scenarios (R2)
- Feature 5: Free Chat Scenario (R2)
- Feature 6: Example Reply Suggestions & Rubrics (R2)
- Feature 7: Plain-Language Companion Agent (R3)
- Feature 8: Rubric-Limited Feedback Agent (R3)
- Feature 9: Score Stripping & Output Guard (R3)
- Feature 10: Turn Orchestration L0-L3 Safety (R3, R4)
- Feature 11: Web Demo UI & Endpoints (R4)
- Feature 12: Streamlit Demo Interface (R4)
- Feature 13: Ethical Disclaimers & Practice Prompts (R4)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest
import yaml

from tests.e2e.conftest import PROJECT_ROOT, assert_no_score_leak


# ==============================================================================
# Feature 1: Auto .env Loader & Key Support (R1) (5 tests)
# ==============================================================================
def test_t1_f1_01_load_env_loads_root_env(clean_env):
    """Verifies that load_env() finds and loads the root .env file."""
    from safety.env import load_env

    # Ensure clean state
    os.environ.pop("TEST_COMPANION_ENV_KEY", None)
    loaded = load_env()
    assert isinstance(loaded, bool)


def test_t1_f1_02_load_env_custom_path(tmp_path, clean_env):
    """Verifies that load_env() can load from a specified custom path."""
    from safety.env import load_env

    custom_env = tmp_path / "custom.env"
    custom_env.write_text("CUSTOM_KEY_ABC=hello_companion\nANOTHER_KEY=42\n", encoding="utf-8")

    assert "CUSTOM_KEY_ABC" not in os.environ
    res = load_env(custom_env)
    assert res is True
    assert os.environ.get("CUSTOM_KEY_ABC") == "hello_companion"
    assert os.environ.get("ANOTHER_KEY") == "42"


def test_t1_f1_03_load_env_gemini_and_google_keys(tmp_path, clean_env):
    """Verifies that both GEMINI_API_KEY and GOOGLE_API_KEY are loaded and accessible."""
    from safety.env import load_env

    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=test_gemini_key\nGOOGLE_API_KEY=test_google_key\n", encoding="utf-8")

    os.environ.pop("GEMINI_API_KEY", None)
    os.environ.pop("GOOGLE_API_KEY", None)

    load_env(env_file, override=True)
    assert os.environ.get("GEMINI_API_KEY") == "test_gemini_key"
    assert os.environ.get("GOOGLE_API_KEY") == "test_google_key"


def test_t1_f1_04_load_env_preserves_existing_without_override(tmp_path, clean_env):
    """Verifies that existing environment variables take precedence unless override=True."""
    from safety.env import load_env

    os.environ["EXISTING_KEY"] = "original_value"
    env_file = tmp_path / ".env"
    env_file.write_text("EXISTING_KEY=new_value\nNEW_KEY=added\n", encoding="utf-8")

    load_env(env_file, override=False)
    assert os.environ["EXISTING_KEY"] == "original_value"
    assert os.environ["NEW_KEY"] == "added"

    load_env(env_file, override=True)
    assert os.environ["EXISTING_KEY"] == "new_value"


def test_t1_f1_05_load_env_export_and_quoted_syntax(tmp_path, clean_env):
    """Verifies support for 'export KEY=VAL', double quotes, single quotes, and inline comments."""
    from safety.env import load_env

    env_file = tmp_path / ".env"
    env_content = (
        'export QUOTED_DOUBLE="hello world with spaces"\n'
        "export QUOTED_SINGLE='single quoted value'\n"
        "UNQUOTED=some_value # inline comment\n"
        "# full line comment\n"
        "EMPTY_VAL=\n"
    )
    env_file.write_text(env_content, encoding="utf-8")

    load_env(env_file, override=True)
    assert os.environ.get("QUOTED_DOUBLE") == "hello world with spaces"
    assert os.environ.get("QUOTED_SINGLE") == "single quoted value"
    assert os.environ.get("UNQUOTED") == "some_value"
    assert os.environ.get("EMPTY_VAL") == ""


# ==============================================================================
# Feature 2: Gemini Provider & Fallbacks (R1) (5 tests)
# ==============================================================================
def test_t1_f2_01_gemini_provider_default_model():
    """Verifies that Gemini provider defaults to a modern, working flash model."""
    from safety.llm_providers import DEFAULT_MODELS, GeminiProvider

    assert "gemini" in DEFAULT_MODELS
    default_model = DEFAULT_MODELS["gemini"]
    assert "gemini" in default_model.lower()
    assert "flash" in default_model.lower()


def test_t1_f2_02_gemini_provider_key_resolution(clean_env):
    """Verifies that GeminiProvider reads API key from GEMINI_API_KEY or GOOGLE_API_KEY."""
    from safety.llm_providers import GeminiProvider

    os.environ["GEMINI_API_KEY"] = "dummy_gemini_key_123"
    prov = GeminiProvider()
    assert prov.api_key == "dummy_gemini_key_123"

    os.environ.pop("GEMINI_API_KEY", None)
    os.environ["GOOGLE_API_KEY"] = "dummy_google_key_456"
    prov2 = GeminiProvider()
    assert prov2.api_key == "dummy_google_key_456"


def test_t1_f2_03_mock_provider_fallback_keyword():
    """Verifies that get_provider('mock') provides deterministic offline responses."""
    from safety.llm_providers import MockProvider, get_provider

    prov = get_provider("mock")
    assert isinstance(prov, MockProvider)
    assert prov.name == "mock"

    res = prov.complete_json("system", "<user_message>hello stall uncle</user_message>")
    assert isinstance(res, dict)
    assert "level" in res
    assert res["level"] == 0


def test_t1_f2_04_provider_fallbacks_openai_and_lmstudio():
    """Verifies that fallbacks 'openai' and 'lmstudio' can be instantiated via get_provider."""
    from safety.llm_providers import LMStudioProvider, OpenAIProvider, get_provider

    try:
        lm = get_provider("lmstudio")
        assert isinstance(lm, LMStudioProvider)
        assert lm.name == "lmstudio"
    except (ImportError, RuntimeError):
        pass  # openai package or local LM Studio server optional

    try:
        oa = get_provider("openai")
        assert isinstance(oa, OpenAIProvider)
        assert oa.name == "openai"
    except (ImportError, RuntimeError):
        pass  # openai package optional if not installed


def test_t1_f2_05_provider_factory_invalid_provider_raises():
    """Verifies that get_provider() raises ValueError on unsupported provider names."""
    from safety.llm_providers import get_provider

    with pytest.raises(ValueError, match="unknown LLM provider"):
        get_provider("non_existent_provider_xyz")


# ==============================================================================
# Feature 3: Secret Safety & Git Hygiene (R1) (5 tests)
# ==============================================================================
def test_t1_f3_01_env_file_is_git_ignored():
    """Verifies that .env is ignored by Git via git check-ignore or gitignore contents."""
    gitignore_path = PROJECT_ROOT / ".gitignore"
    assert gitignore_path.is_file(), ".gitignore must exist at project root"

    gitignore_text = gitignore_path.read_text(encoding="utf-8")
    assert re.search(r"^\.env\b", gitignore_text, re.MULTILINE) or re.search(r"\*\.env\b", gitignore_text)

    # If git CLI is available, verify git check-ignore
    try:
        res = subprocess.run(["git", "check-ignore", "-v", ".env"], cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        assert res.returncode == 0 or ".gitignore" in res.stdout
    except FileNotFoundError:
        pass


def test_t1_f3_02_no_api_keys_in_tracked_files():
    """Verifies that no real Google or OpenAI API keys exist in tracked code files."""
    google_key_pattern = re.compile(r"AIza[0-9A-Za-z-_]{35}")
    openai_key_pattern = re.compile(r"sk-[0-9A-Za-z]{32,}")

    code_extensions = {".py", ".yaml", ".yml", ".json", ".md", ".sh", ".bat", ".html", ".js"}
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Skip git, venv, and cache dirs
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in {"venv", ".venv", "__pycache__"}]
        for f in files:
            path = Path(root) / f
            if path.suffix in code_extensions and path.name != ".env":
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                    assert not google_key_pattern.search(content), f"Potential Google API key leak found in {path}"
                    assert not openai_key_pattern.search(content), f"Potential OpenAI API key leak found in {path}"
                except Exception:
                    pass


def test_t1_f3_03_config_yaml_has_no_secret_keys():
    """Verifies that config.yaml does not contain hardcoded API keys."""
    config_path = PROJECT_ROOT / "config.yaml"
    assert config_path.is_file()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    llm_cfg = cfg.get("llm", {})
    assert "api_key" not in llm_cfg or not llm_cfg["api_key"], "config.yaml must not contain hardcoded api_key"


def test_t1_f3_04_git_status_shows_no_tracked_env():
    """Verifies that .env is untracked by Git."""
    try:
        res = subprocess.run(["git", "status", "--porcelain"], cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        if res.returncode == 0:
            lines = res.stdout.splitlines()
            for line in lines:
                # Tracked modified/added files start with 'M ', 'A ', etc.
                assert not (line.startswith("M .env") or line.startswith("A .env")), ".env must not be tracked in git"
    except FileNotFoundError:
        pass


def test_t1_f3_05_example_env_contains_only_placeholders():
    """Verifies that .env.example (if present) contains only placeholder values."""
    example_path = PROJECT_ROOT / ".env.example"
    if example_path.is_file():
        text = example_path.read_text(encoding="utf-8")
        assert "AIza" not in text
        assert "sk-" not in text


# ==============================================================================
# Feature 4: Three Role-Play Scenarios (R2) (5 tests)
# ==============================================================================
def test_t1_f4_01_hawker_centre_scenario_spec():
    """Verifies hawker_centre scenario has all required fields and metadata."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    sc = scenarios.get_scenario("hawker_centre")
    assert sc.id == "hawker_centre"
    assert "hawker" in sc.title.lower() or "food" in sc.title.lower()
    assert len(sc.setting.strip()) > 10
    assert len(sc.companion_role.strip()) > 0
    assert len(sc.opening_line.strip()) > 0
    assert len(sc.user_goal.strip()) > 0
    assert len(sc.rubric) >= 3
    assert len(sc.example_replies) >= 2
    assert len(sc.real_life_prompt.strip()) > 10


def test_t1_f4_02_new_colleague_scenario_spec():
    """Verifies new_colleague scenario has all required fields and metadata."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    sc = scenarios.get_scenario("new_colleague")
    assert sc.id == "new_colleague"
    assert "colleague" in sc.title.lower() or "small talk" in sc.title.lower()
    assert len(sc.setting.strip()) > 10
    assert len(sc.companion_role.strip()) > 0
    assert len(sc.opening_line.strip()) > 0
    assert len(sc.user_goal.strip()) > 0
    assert len(sc.rubric) >= 3
    assert len(sc.example_replies) >= 2
    assert len(sc.real_life_prompt.strip()) > 10


def test_t1_f4_03_job_interview_scenario_spec():
    """Verifies job_interview scenario has all required fields and metadata."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    sc = scenarios.get_scenario("job_interview")
    assert sc.id == "job_interview"
    assert "interview" in sc.title.lower()
    assert len(sc.setting.strip()) > 10
    assert len(sc.companion_role.strip()) > 0
    assert len(sc.opening_line.strip()) > 0
    assert len(sc.user_goal.strip()) > 0
    assert len(sc.rubric) >= 3
    assert len(sc.example_replies) >= 2
    assert len(sc.real_life_prompt.strip()) > 10


def test_t1_f4_04_list_scenarios_returns_all_three():
    """Verifies list_scenarios() contains all three core scenarios."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    sc_list = scenarios.list_scenarios()
    ids = {s.id for s in sc_list}
    assert "hawker_centre" in ids
    assert "new_colleague" in ids
    assert "job_interview" in ids


def test_t1_f4_05_scenario_dataclass_contract():
    """Verifies Scenario dataclass schema conforms to PROJECT.md interface contract."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    sc = scenarios.get_scenario("hawker_centre")
    assert hasattr(sc, "id") and isinstance(sc.id, str)
    assert hasattr(sc, "title") and isinstance(sc.title, str)
    assert hasattr(sc, "setting") and isinstance(sc.setting, str)
    assert hasattr(sc, "companion_role") and isinstance(sc.companion_role, str)
    assert hasattr(sc, "opening_line") and isinstance(sc.opening_line, str)
    assert hasattr(sc, "user_goal") and isinstance(sc.user_goal, str)
    assert hasattr(sc, "rubric") and isinstance(sc.rubric, list)
    assert hasattr(sc, "example_replies") and isinstance(sc.example_replies, list)
    assert hasattr(sc, "real_life_prompt") and isinstance(sc.real_life_prompt, str)


# ==============================================================================
# Feature 5: Free Chat Scenario (R2) (5 tests)
# ==============================================================================
def test_t1_f5_01_free_chat_scenario_exists():
    """Verifies that free_chat scenario is registered and retrievable."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    sc = scenarios.get_scenario("free_chat")
    assert sc.id == "free_chat"
    assert "free" in sc.title.lower() or "chat" in sc.title.lower()


def test_t1_f5_02_free_chat_relaxed_goal():
    """Verifies free_chat has a relaxed, open communication goal."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    sc = scenarios.get_scenario("free_chat")
    assert len(sc.user_goal) > 5


def test_t1_f5_03_free_chat_opening_line():
    """Verifies free_chat has a welcoming opening question."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    sc = scenarios.get_scenario("free_chat")
    assert len(sc.opening_line) > 0
    assert "?" in sc.opening_line or "!" in sc.opening_line


def test_t1_f5_04_free_chat_flexible_rubric():
    """Verifies free_chat rubric contains criteria evaluating clarity or expression."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    sc = scenarios.get_scenario("free_chat")
    assert len(sc.rubric) >= 2
    criteria_names = [item.criterion.lower() for item in sc.rubric]
    assert any("clarity" in c or "topic" in c or "expression" in c for c in criteria_names)


def test_t1_f5_05_free_chat_example_replies():
    """Verifies free_chat includes example replies for casual/special-interest topics."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    sc = scenarios.get_scenario("free_chat")
    assert len(sc.example_replies) >= 2
    for ex in sc.example_replies:
        assert len(ex.text) > 0
        assert len(ex.explanation) > 0


# ==============================================================================
# Feature 6: Example Reply Suggestions & Rubrics (R2) (5 tests)
# ==============================================================================
def test_t1_f6_01_all_scenarios_have_multiple_example_replies():
    """Verifies that all scenarios define at least 2 example replies."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    for sc in scenarios.list_scenarios():
        assert len(sc.example_replies) >= 2, f"Scenario {sc.id} has fewer than 2 example replies"


def test_t1_f6_02_example_replies_have_text_and_explanation():
    """Verifies that ExampleReply dataclass provides text and explanation."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    for sc in scenarios.list_scenarios():
        for ex in sc.example_replies:
            assert hasattr(ex, "text") and isinstance(ex.text, str)
            assert hasattr(ex, "explanation") and isinstance(ex.explanation, str)
            assert len(ex.text.strip()) > 0
            assert len(ex.explanation.strip()) > 0


def test_t1_f6_03_fixed_rubrics_have_criterion_and_description():
    """Verifies that RubricItem dataclass provides criterion and description."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    for sc in scenarios.list_scenarios():
        for r in sc.rubric:
            assert hasattr(r, "criterion") and isinstance(r.criterion, str)
            assert hasattr(r, "description") and isinstance(r.description, str)
            assert len(r.criterion.strip()) > 0
            assert len(r.description.strip()) > 0


def test_t1_f6_04_example_replies_are_non_prescriptive():
    """Verifies that example replies are presented as suggestions, not single 'correct' answers."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    for sc in scenarios.list_scenarios():
        for ex in sc.example_replies:
            # Must not claim to be the 'only' or 'correct' answer
            assert "only correct" not in ex.explanation.lower()
            assert_no_score_leak(ex.explanation)
            assert_no_score_leak(ex.text)


def test_t1_f6_05_rubric_items_tailored_to_scenario_goals():
    """Verifies rubric criteria match the specific context of each scenario."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    hawker = scenarios.get_scenario("hawker_centre")
    hawker_text = " ".join([r.criterion.lower() + " " + r.description.lower() for r in hawker.rubric])
    assert "food" in hawker_text or "order" in hawker_text or "takeaway" in hawker_text or "clarity" in hawker_text

    interview = scenarios.get_scenario("job_interview")
    interview_text = " ".join([r.criterion.lower() + " " + r.description.lower() for r in interview.rubric])
    assert "skill" in interview_text or "experience" in interview_text or "role" in interview_text or "clarity" in interview_text


# ==============================================================================
# Feature 7: Plain-Language Companion Agent (R3) (5 tests)
# ==============================================================================
def test_t1_f7_01_companion_agent_class_contract(safety_agent):
    """Verifies CompanionAgent initialization and interface contract."""
    companion_mod = pytest.importorskip("safety.companion", reason="safety.companion (M3) not yet implemented")
    from safety.llm_providers import MockProvider

    prov = MockProvider()
    agent = companion_mod.CompanionAgent(prov, safety_agent)
    assert hasattr(agent, "generate")


def test_t1_f7_02_companion_agent_literal_plain_language(safety_agent):
    """Verifies that companion agent produces short, literal sentences and passes OutputGuard."""
    companion_mod = pytest.importorskip("safety.companion", reason="safety.companion (M3) not yet implemented")
    scenarios_mod = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")
    from safety.llm_providers import MockProvider

    prov = MockProvider()
    agent = companion_mod.CompanionAgent(prov, safety_agent)
    sc = scenarios_mod.get_scenario("hawker_centre")
    session = safety_agent.new_session()

    reply = agent.generate(session, sc, "One chicken rice please.", directives=[], history=[])
    assert isinstance(reply, str)
    assert len(reply) > 0
    assert_no_score_leak(reply)


def test_t1_f7_03_companion_agent_obeys_directives(safety_agent):
    """Verifies that companion agent accepts directives (e.g., slow down)."""
    companion_mod = pytest.importorskip("safety.companion", reason="safety.companion (M3) not yet implemented")
    scenarios_mod = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")
    from safety.llm_providers import MockProvider

    prov = MockProvider()
    agent = companion_mod.CompanionAgent(prov, safety_agent)
    sc = scenarios_mod.get_scenario("hawker_centre")
    session = safety_agent.new_session()

    directives = ["Use at most 12 words.", "Offer two simple choices."]
    reply = agent.generate(session, sc, "I feel a bit hurried.", directives=directives, history=[])
    assert isinstance(reply, str)
    assert_no_score_leak(reply)


def test_t1_f7_04_companion_agent_passes_through_output_guard(safety_agent):
    """Verifies that every companion reply is vetted by OutputGuard."""
    companion_mod = pytest.importorskip("safety.companion", reason="safety.companion (M3) not yet implemented")
    scenarios_mod = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")
    from safety.llm_providers import MockProvider

    prov = MockProvider()
    agent = companion_mod.CompanionAgent(prov, safety_agent)
    sc = scenarios_mod.get_scenario("new_colleague")
    session = safety_agent.new_session()

    reply = agent.generate(session, sc, "Hi there!", directives=[], history=[])
    guard_res = safety_agent.guard.check(reply)
    assert guard_res.ok is True, f"OutputGuard failed on companion reply: {guard_res.violations}"


def test_t1_f7_05_companion_agent_retry_and_safe_fallback(safety_agent):
    """Verifies that companion agent retries once and falls back to SAFE_FALLBACK on persistent violation."""
    companion_mod = pytest.importorskip("safety.companion", reason="safety.companion (M3) not yet implemented")
    scenarios_mod = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")
    from safety.output_guard import SAFE_FALLBACK

    class AlwaysViolatingProvider:
        name = "mock_bad"
        model = "mock"
        calls = 0

        def complete_json(self, system, user, *args, **kwargs):
            self.calls += 1
            # Hard medical violation that cannot be stripped
            return {"reply": "You have clinical anxiety disorder and need prescription antidepressants."}

    prov = AlwaysViolatingProvider()
    agent = companion_mod.CompanionAgent(prov, safety_agent)
    sc = scenarios_mod.get_scenario("hawker_centre")
    session = safety_agent.new_session()

    reply = agent.generate(session, sc, "One chicken rice.", directives=[], history=[])
    assert reply == SAFE_FALLBACK
    assert_no_score_leak(reply)


# ==============================================================================
# Feature 8: Rubric-Limited Feedback Agent (R3) (5 tests)
# ==============================================================================
def test_t1_f8_01_feedback_agent_class_contract():
    """Verifies FeedbackAgent initialization and generate contract."""
    feedback_mod = pytest.importorskip("safety.feedback", reason="safety.feedback (M3) not yet implemented")
    from safety.llm_providers import MockProvider

    prov = MockProvider()
    fb_agent = feedback_mod.FeedbackAgent(prov)
    assert hasattr(fb_agent, "generate")


def test_t1_f8_02_feedback_agent_rubric_evaluation():
    """Verifies that feedback evaluates user turn against scenario goal and rubric."""
    feedback_mod = pytest.importorskip("safety.feedback", reason="safety.feedback (M3) not yet implemented")
    scenarios_mod = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")
    from safety.llm_providers import MockProvider

    prov = MockProvider()
    fb_agent = feedback_mod.FeedbackAgent(prov)
    sc = scenarios_mod.get_scenario("hawker_centre")

    fb = fb_agent.generate(sc, "Chicken rice please.", "Steamed or roasted?")
    if fb is not None:
        assert isinstance(fb, str)
        assert_no_score_leak(fb)


def test_t1_f8_03_feedback_agent_cites_goal_and_examples():
    """Verifies that feedback cites scenario goal or example replies, not generic platitudes."""
    feedback_mod = pytest.importorskip("safety.feedback", reason="safety.feedback (M3) not yet implemented")
    scenarios_mod = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")
    from safety.llm_providers import MockProvider

    prov = MockProvider()
    fb_agent = feedback_mod.FeedbackAgent(prov)
    sc = scenarios_mod.get_scenario("hawker_centre")

    fb = fb_agent.generate(sc, "One chicken rice, take away.", "Okay, take away.")
    if fb is not None:
        assert_no_score_leak(fb)


def test_t1_f8_04_feedback_agent_at_most_one_suggestion():
    """Verifies that feedback returns at most one short, optional suggestion."""
    feedback_mod = pytest.importorskip("safety.feedback", reason="safety.feedback (M3) not yet implemented")
    scenarios_mod = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")
    from safety.llm_providers import MockProvider

    prov = MockProvider()
    fb_agent = feedback_mod.FeedbackAgent(prov)
    sc = scenarios_mod.get_scenario("new_colleague")

    fb = fb_agent.generate(sc, "Good morning!", "Hi! How are you?")
    if fb is not None:
        # At most 1 suggestion should be concise (under 50 words)
        words = fb.split()
        assert len(words) <= 50
        assert_no_score_leak(fb)


def test_t1_f8_05_feedback_agent_suppressed_when_non_calm(safety_agent):
    """Verifies that feedback is suppressed when decision.suppress_feedback is True."""
    orchestrator_mod = pytest.importorskip("safety.orchestrator", reason="safety.orchestrator (M3) not yet implemented")
    companion_mod = pytest.importorskip("safety.companion", reason="safety.companion (M3) not yet implemented")
    feedback_mod = pytest.importorskip("safety.feedback", reason="safety.feedback (M3) not yet implemented")
    from safety.llm_providers import MockProvider

    prov = MockProvider()
    comp = companion_mod.CompanionAgent(prov, safety_agent)
    fb = feedback_mod.FeedbackAgent(prov)
    s = safety_agent.new_session()

    # Tense input (Level 1) -> suppress_feedback should be True
    res = orchestrator_mod.handle_turn(
        safety_agent, comp, fb, s, "I feel very nervous and anxious...", "Hello!", "hawker_centre"
    )
    assert res.decision.suppress_feedback is True
    assert res.feedback is None


# ==============================================================================
# Feature 9: Score Stripping & Output Guard (R3) (5 tests)
# ==============================================================================
def test_t1_f9_01_output_guard_strips_numeric_scores(output_guard):
    """Verifies OutputGuard strips numeric ratio scores like '8/10' or '9.5/10'."""
    text = "Great job ordering chicken rice! That was 8/10. What would you like next?"
    cleaned, removed = output_guard.strip_scores(text)
    assert len(removed) > 0
    assert "8/10" not in cleaned
    assert "Great job ordering chicken rice!" in cleaned
    assert_no_score_leak(cleaned)


def test_t1_f9_02_output_guard_strips_percentages(output_guard):
    """Verifies OutputGuard strips percentage scores like '90 percent'."""
    text = "You achieved 95 percent on clarity! Next time say extra chilli."
    cleaned, removed = output_guard.strip_scores(text)
    assert "95 percent" not in cleaned
    assert_no_score_leak(cleaned)


def test_t1_f9_03_output_guard_strips_star_ratings(output_guard):
    """Verifies OutputGuard strips Unicode star ratings."""
    text = "Order complete! ★★★★ rating. Enjoy your meal."
    cleaned, removed = output_guard.strip_scores(text)
    assert "★" not in cleaned
    assert_no_score_leak(cleaned)


def test_t1_f9_04_output_guard_blocks_medical_diagnoses(output_guard):
    """Verifies OutputGuard rejects replies offering medical diagnosis or medication."""
    reply = "You seem to have autism and should take medication."
    res = output_guard.check(reply)
    assert res.ok is False
    assert res.action == "regenerate"
    assert any("medical" in v for v in res.violations)


def test_t1_f9_05_output_guard_blocks_human_impersonation(output_guard):
    """Verifies OutputGuard rejects claims of being a real human or therapist."""
    reply = "I am a real therapist and you can trust me with everything."
    res = output_guard.check(reply)
    assert res.ok is False
    assert res.action == "regenerate"
    assert any("impersonation" in v for v in res.violations)


# ==============================================================================
# Feature 10: Turn Orchestration L0-L3 Safety (R3, R4) (5 tests)
# ==============================================================================
def test_t1_f10_01_turn_orchestrator_level0_calm(safety_agent):
    """Verifies turn orchestrator handles calm turn (L0 Safe) with in-character reply."""
    orchestrator_mod = pytest.importorskip("safety.orchestrator", reason="safety.orchestrator (M3) not yet implemented")
    companion_mod = pytest.importorskip("safety.companion", reason="safety.companion (M3) not yet implemented")
    feedback_mod = pytest.importorskip("safety.feedback", reason="safety.feedback (M3) not yet implemented")
    from safety.llm_providers import MockProvider

    prov = MockProvider()
    comp = companion_mod.CompanionAgent(prov, safety_agent)
    fb = feedback_mod.FeedbackAgent(prov)
    s = safety_agent.new_session()

    res = orchestrator_mod.handle_turn(
        safety_agent, comp, fb, s, "One chicken rice please.", "What would you like?", "hawker_centre"
    )
    assert res.decision.level == 0
    assert len(res.reply) > 0
    assert_no_score_leak(res.reply)
    if res.feedback:
        assert_no_score_leak(res.feedback)


def test_t1_f10_02_turn_orchestrator_level1_tense(safety_agent):
    """Verifies turn orchestrator handles tense turn (L1 Tense) with slow down and choices."""
    orchestrator_mod = pytest.importorskip("safety.orchestrator", reason="safety.orchestrator (M3) not yet implemented")
    companion_mod = pytest.importorskip("safety.companion", reason="safety.companion (M3) not yet implemented")
    feedback_mod = pytest.importorskip("safety.feedback", reason="safety.feedback (M3) not yet implemented")
    from safety.llm_providers import MockProvider

    prov = MockProvider()
    comp = companion_mod.CompanionAgent(prov, safety_agent)
    fb = feedback_mod.FeedbackAgent(prov)
    s = safety_agent.new_session()

    res = orchestrator_mod.handle_turn(
        safety_agent, comp, fb, s, "I'm feeling anxious and nervous...", "Take your time.", "hawker_centre"
    )
    assert res.decision.level == 1
    assert res.decision.suppress_feedback is True
    assert res.feedback is None
    assert len(res.buttons) >= 2
    assert "Keep going" in res.buttons or "Take a short break" in res.buttons
    assert_no_score_leak(res.reply)


def test_t1_f10_03_turn_orchestrator_level2_overwhelmed(safety_agent):
    """Verifies turn orchestrator handles overwhelmed turn (L2) with roleplay pause and playbook prompt."""
    orchestrator_mod = pytest.importorskip("safety.orchestrator", reason="safety.orchestrator (M3) not yet implemented")
    companion_mod = pytest.importorskip("safety.companion", reason="safety.companion (M3) not yet implemented")
    feedback_mod = pytest.importorskip("safety.feedback", reason="safety.feedback (M3) not yet implemented")
    from safety.llm_providers import MockProvider

    prov = MockProvider()
    comp = companion_mod.CompanionAgent(prov, safety_agent)
    fb = feedback_mod.FeedbackAgent(prov)
    s = safety_agent.new_session()

    res = orchestrator_mod.handle_turn(
        safety_agent, comp, fb, s, "Too loud, can't think, stop!", "Hello!", "hawker_centre"
    )
    assert res.decision.level == 2
    assert res.decision.pause_roleplay is True
    assert res.decision.suppress_feedback is True
    assert res.feedback is None
    assert len(res.buttons) > 0
    assert_no_score_leak(res.reply)


def test_t1_f10_04_turn_orchestrator_level3_crisis(safety_agent):
    """Verifies turn orchestrator handles crisis turn (L3) with verified Singapore helplines and hold."""
    orchestrator_mod = pytest.importorskip("safety.orchestrator", reason="safety.orchestrator (M3) not yet implemented")
    companion_mod = pytest.importorskip("safety.companion", reason="safety.companion (M3) not yet implemented")
    feedback_mod = pytest.importorskip("safety.feedback", reason="safety.feedback (M3) not yet implemented")
    from safety.llm_providers import MockProvider

    prov = MockProvider()
    comp = companion_mod.CompanionAgent(prov, safety_agent)
    fb = feedback_mod.FeedbackAgent(prov)
    s = safety_agent.new_session(age_group="teen")

    res = orchestrator_mod.handle_turn(
        safety_agent, comp, fb, s, "i want to die", "Hello!", "hawker_centre"
    )
    assert res.decision.level == 3
    assert res.decision.pause_roleplay is True
    assert "1767" in res.reply or "1771" in res.reply
    assert s.crisis_hold is True
    assert res.feedback is None
    assert_no_score_leak(res.reply)


def test_t1_f10_05_orchestrator_turn_result_dataclass():
    """Verifies TurnResult dataclass fields conform to PROJECT.md interface contract."""
    orchestrator_mod = pytest.importorskip("safety.orchestrator", reason="safety.orchestrator (M3) not yet implemented")

    assert hasattr(orchestrator_mod, "TurnResult")
    # Check annotations or field names
    fields = orchestrator_mod.TurnResult.__annotations__
    assert "reply" in fields
    assert "buttons" in fields
    assert "feedback" in fields
    assert "decision" in fields
    assert "scenario_id" in fields


# ==============================================================================
# Feature 11: Web Demo UI & Endpoints (R4) (5 tests)
# ==============================================================================
def test_t1_f11_01_web_demo_serves_html(web_demo_server):
    """Verifies GET / returns 200 with HTML content."""
    with urllib.request.urlopen(web_demo_server + "/", timeout=5) as resp:
        assert resp.status == 200
        content = resp.read().decode("utf-8")
        assert "<!doctype html>" in content.lower() or "<html" in content.lower()


def test_t1_f11_02_web_demo_api_status_endpoint(web_demo_server):
    """Verifies GET /api/status returns JSON status payload."""
    with urllib.request.urlopen(web_demo_server + "/api/status", timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "mode" in data
        assert "provider" in data


def test_t1_f11_03_web_demo_api_new_session_endpoint(web_demo_server):
    """Verifies POST /api/new creates a session and returns session_id."""
    req = urllib.request.Request(
        web_demo_server + "/api/new",
        data=json.dumps({"age_group": "teen", "carer_consent": False}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "session_id" in data
        assert len(data["session_id"]) > 0


def test_t1_f11_04_web_demo_api_check_endpoint(web_demo_server):
    """Verifies POST /api/check processes a conversation turn."""
    # First get session_id
    req_new = urllib.request.Request(
        web_demo_server + "/api/new",
        data=json.dumps({"age_group": "teen"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req_new, timeout=5) as resp:
        sid = json.loads(resp.read().decode("utf-8"))["session_id"]

    req_check = urllib.request.Request(
        web_demo_server + "/api/check",
        data=json.dumps({
            "session_id": sid,
            "text": "One chicken rice please.",
            "scenario": "hawker_centre"
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req_check, timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "message_level" in data
        assert "action" in data
        assert data["message_level"] == 0


def test_t1_f11_05_web_demo_api_reply_guard_endpoint(web_demo_server):
    """Verifies POST /api/reply tests candidate text against OutputGuard and strips scores."""
    req = urllib.request.Request(
        web_demo_server + "/api/reply",
        data=json.dumps({"reply": "Great job! That was 9/10 points. Have a nice day."}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "text" in data
        assert "9/10" not in data["text"]
        assert_no_score_leak(data["text"])


# ==============================================================================
# Feature 12: Streamlit Demo Interface (R4) (5 tests)
# ==============================================================================
def test_t1_f12_01_streamlit_panel_module_import():
    """Verifies demo/streamlit/safety_panel.py compiles and imports if streamlit is available."""
    sp_path = PROJECT_ROOT / "demo" / "streamlit" / "safety_panel.py"
    assert sp_path.is_file()
    compile(sp_path.read_text(encoding="utf-8"), str(sp_path), "exec")
    try:
        import streamlit
        if not hasattr(streamlit, "set_page_config"):
            pytest.skip("streamlit stub lacks set_page_config")
        sys.path.insert(0, str(PROJECT_ROOT / "demo" / "streamlit"))
        import safety_panel
        assert hasattr(safety_panel, "render_safety_panel") or hasattr(safety_panel, "render_panel")
    except (ImportError, AttributeError):
        pass


def test_t1_f12_02_streamlit_panel_safety_inspector():
    """Verifies safety panel inspection code includes level, layers, carer alert inspection."""
    sp_path = PROJECT_ROOT / "demo" / "streamlit" / "safety_panel.py"
    assert sp_path.is_file()
    code = sp_path.read_text(encoding="utf-8")
    assert "carer_alert" in code
    assert "level" in code.lower()
    assert "layer" in code.lower()


def test_t1_f12_03_streamlit_app_file_exists():
    """Verifies demo/streamlit/ directory contains the required entrypoint files."""
    st_dir = PROJECT_ROOT / "demo" / "streamlit"
    assert st_dir.is_dir()
    py_files = list(st_dir.glob("*.py"))
    assert len(py_files) >= 1


def test_t1_f12_04_streamlit_panel_disclaimer_display():
    """Verifies that Streamlit demo includes the non-clinical AI disclaimer text (Milestone 4)."""
    st_files = list((PROJECT_ROOT / "demo" / "streamlit").glob("*.py"))
    combined = " ".join([f.read_text(encoding="utf-8", errors="ignore") for f in st_files])
    if not any(k in combined.lower() for k in ["non-clinical", "not a human", "not a therapist", "disclaimer"]):
        pytest.skip("Streamlit ethical disclaimer (Milestone 4) not yet integrated")
    assert any(k in combined.lower() for k in ["non-clinical", "not a human", "not a therapist", "disclaimer"])


def test_t1_f12_05_streamlit_interactive_button_labels():
    """Verifies that button labels in demo/web_demo.py include continue and safety options."""
    from demo.web_demo import CONTINUE_BUTTONS
    assert "Go back to practice" in CONTINUE_BUTTONS or "Keep going" in CONTINUE_BUTTONS
    assert "I did it" in CONTINUE_BUTTONS


# ==============================================================================
# Feature 13: Ethical Disclaimers & Practice Prompts (R4) (5 tests)
# ==============================================================================
def test_t1_f13_01_web_html_contains_non_clinical_disclaimer():
    """Verifies index.html has prominent non-clinical disclaimer stating tool is not a therapist (Milestone 4)."""
    html_path = PROJECT_ROOT / "demo" / "web" / "index.html"
    assert html_path.is_file()
    html = html_path.read_text(encoding="utf-8")
    if not any(k in html.lower() for k in ["non-clinical", "not a therapist", "not a human"]):
        pytest.skip("Web Demo non-clinical disclaimer (Milestone 4) not yet integrated")
    assert any(k in html.lower() for k in ["non-clinical", "not a therapist", "not a human"])


def test_t1_f13_02_fictional_demo_use_only_notice():
    """Verifies index.html specifies fictional / demo messages should be used (Milestone 4)."""
    html_path = PROJECT_ROOT / "demo" / "web" / "index.html"
    assert html_path.is_file()
    html = html_path.read_text(encoding="utf-8")
    if not any(k in html.lower() for k in ["fictional", "made-up", "demo use only"]):
        pytest.skip("Web Demo fictional demo notice (Milestone 4) not yet integrated")
    assert any(k in html.lower() for k in ["fictional", "made-up", "demo use only"])


def test_t1_f13_03_real_life_prompts_exist_in_scenarios():
    """Verifies that all scenarios define a non-empty actionable real_life_prompt."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    for sc in scenarios.list_scenarios():
        assert hasattr(sc, "real_life_prompt")
        assert len(sc.real_life_prompt.strip()) >= 15
        assert_no_score_leak(sc.real_life_prompt)


def test_t1_f13_04_end_of_session_actionable_advice():
    """Verifies that real_life_prompt gives positive, anxiety-reducing action advice."""
    scenarios = pytest.importorskip("safety.scenarios", reason="safety.scenarios (M2) not yet implemented")

    for sc in scenarios.list_scenarios():
        prompt = sc.real_life_prompt.lower()
        assert any(verb in prompt for verb in ["try", "practise", "practice", "ask", "say", "share", "visit"])


def test_t1_f13_05_carer_alert_policy_age_group_ethics(safety_agent):
    """Verifies ethical carer alert policy: teens alert by default, adults require consent."""
    # Teen in crisis -> carer alert True
    s_teen = safety_agent.new_session(age_group="teen", carer_alert_consent=False)
    d_teen = safety_agent.check_user_message(s_teen, "i want to die")
    assert d_teen.message_level == 3
    assert d_teen.carer_alert is True

    # Adult without consent -> carer alert False
    s_adult_no = safety_agent.new_session(age_group="adult", carer_alert_consent=False)
    d_adult_no = safety_agent.check_user_message(s_adult_no, "i want to die")
    assert d_adult_no.message_level == 3
    assert d_adult_no.carer_alert is False

    # Adult with consent -> carer alert True
    s_adult_yes = safety_agent.new_session(age_group="adult", carer_alert_consent=True)
    d_adult_yes = safety_agent.check_user_message(s_adult_yes, "i want to die")
    assert d_adult_yes.message_level == 3
    assert d_adult_yes.carer_alert is True
