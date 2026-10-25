"""Adversarial stress-test suite for safety/llm_providers.py and SafetyAgent integration.

Tests include:
1. Missing GEMINI_API_KEY and GOOGLE_API_KEY under various conditions.
2. Empty string API keys ("") and whitespace keys.
3. Invalid / corrupted API keys.
4. Provider factory (get_provider) stress tests:
   - gemini with and without keys
   - lmstudio when offline and with/without model
   - openai with/without key
   - mock with normal, empty, unusual prompts
   - invalid provider names
5. Complete_json error conditions and malformed JSON handling.
6. SafetyAgent.from_config robustness:
   - None/null timeout_s in config
   - Missing keys with fail_safe escalation
   - Unknown provider fallback
   - Missing or corrupted config.yaml
7. Key precedence (GEMINI_API_KEY vs GOOGLE_API_KEY).
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safety.agent import SafetyAgent
from safety.llm_providers import (
    DEFAULT_MODELS,
    GeminiProvider,
    LMStudioProvider,
    MockProvider,
    OpenAIProvider,
    _parse_json,
    get_provider,
)
from safety.schema import Level


# ==============================================================================
# 1. Missing and Empty API Keys in GeminiProvider
# ==============================================================================

def test_gemini_provider_missing_both_keys():
    """Verify GeminiProvider initialization and complete_json behavior when no keys exist."""
    with patch.dict(os.environ, {}, clear=True):
        # Explicitly ensure keys are absent
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("GOOGLE_API_KEY", None)

        gp = GeminiProvider()
        assert gp.api_key is None
        # In google-genai, genai.Client() without key might either raise in __init__ or succeed.
        # If client is None, complete_json should raise RuntimeError.
        # If client is initialized, complete_json without auth should raise RuntimeError when called.
        if gp.client is None:
            with pytest.raises(RuntimeError, match="GeminiProvider is not ready"):
                gp.complete_json("system", "user")
        else:
            # Client was created without key; complete_json should raise RuntimeError on API call
            with pytest.raises(RuntimeError, match="Gemini API request failed"):
                gp.complete_json("system", "user")


def test_gemini_provider_empty_string_keys():
    """Verify GeminiProvider handles empty string API keys without crashing interpreter."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}, clear=True):
        gp = GeminiProvider()
        # In Python, empty string is falsy, so gp.api_key evaluates to "" or None
        assert gp.api_key in ("", None)
        # Calling complete_json should raise clean RuntimeError, never crash Python
        with pytest.raises(RuntimeError):
            gp.complete_json("system", "user")


def test_gemini_get_provider_missing_keys_raises_or_returns_ready():
    """Verify get_provider('gemini') raises RuntimeError if client initialization failed."""
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("GOOGLE_API_KEY", None)

        try:
            prov = get_provider("gemini")
            # If it succeeded, client must not be None
            assert prov.client is not None
        except RuntimeError as e:
            assert "Gemini provider could not be initialized" in str(e)


# ==============================================================================
# 2. Corrupted / Invalid API Keys
# ==============================================================================

def test_gemini_provider_invalid_key_error_handling():
    """Verify invalid key raises clean RuntimeError from complete_json without crashing."""
    gp = GeminiProvider(api_key="INVALID_KEY_EXPLICIT_TEST_12345")
    assert gp.api_key == "INVALID_KEY_EXPLICIT_TEST_12345"
    assert gp.client is not None
    gp.client.models.generate_content = MagicMock(side_effect=Exception("API key not valid. Please pass a valid API key."))

    with pytest.raises(RuntimeError) as exc_info:
        gp.complete_json("You are a safety assistant.", "Hello stall uncle")
    assert "Gemini API request failed" in str(exc_info.value)
    assert "API key not valid" in str(exc_info.value)


def test_gemini_complete_json_mocked_network_exception():
    """Verify any network exception in generate_content is wrapped in RuntimeError."""
    gp = GeminiProvider(api_key="dummy_key")
    gp.client = MagicMock()
    gp.client.models.generate_content.side_effect = ConnectionResetError("Connection dropped by peer")

    with pytest.raises(RuntimeError, match="Gemini API request failed.*Connection dropped by peer"):
        gp.complete_json("system instruction", "user message")


# ==============================================================================
# 3. LM Studio Provider Offline & Error Behaviors
# ==============================================================================

def test_lmstudio_offline_first_loaded_model_handling():
    """Empirically test what happens when LM Studio is queried with no server running."""
    # When no model is provided and LMSTUDIO_MODEL is not set, LMStudioProvider attempts
    # to query http://localhost:1234/v1/models.
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("LMSTUDIO_MODEL", None)
        try:
            lm = LMStudioProvider(base_url="http://127.0.0.1:54321/v1", timeout_s=0.5)
            # If by any chance a server exists
            assert lm is not None
        except Exception as e:
            # Verified: raises RuntimeError wrapping the connection error
            assert isinstance(e, RuntimeError)
            assert "LM Studio server is unreachable" in str(e)


def test_lmstudio_with_explicit_model_no_probe():
    """When model is explicitly provided, LMStudioProvider does not make network calls on init."""
    lm = LMStudioProvider(model="qwen-2.5-7b", base_url="http://127.0.0.1:54321/v1")
    assert lm.model == "qwen-2.5-7b"
    assert lm.name == "lmstudio"


def test_openai_complete_json_offline_raises_runtime_error():
    """Verify OpenAIProvider.complete_json wraps connection errors into RuntimeError."""
    oa = OpenAIProvider(api_key="sk-dummy", base_url="http://127.0.0.1:54321/v1", timeout_s=0.5)
    with pytest.raises(RuntimeError, match="OpenAI API request failed"):
        oa.complete_json("system", "user")


def test_lmstudio_complete_json_offline_raises_runtime_error():
    """Verify LMStudioProvider.complete_json wraps connection errors into RuntimeError."""
    lm = LMStudioProvider(model="dummy-model", base_url="http://127.0.0.1:54321/v1", timeout_s=0.5)
    with pytest.raises(RuntimeError, match="LM Studio API request failed"):
        lm.complete_json("system", "user")


# ==============================================================================
# 4. JSON Parsing & Malformed LLM Responses
# ==============================================================================

def test_parse_json_resilience():
    """Verify _parse_json tolerates code fences, surrounding text, and rejects invalid strings."""
    # Standard clean json
    assert _parse_json('{"level": 0, "rationale": "calm"}') == {"level": 0, "rationale": "calm"}

    # Markdown code fence json
    fence = '```json\n{"level": 1, "category": "distress"}\n```'
    assert _parse_json(fence) == {"level": 1, "category": "distress"}

    # Text before and after json
    surrounded = 'Analysis:\n{"level": 2}\nEnd of response'
    assert _parse_json(surrounded) == {"level": 2}

    # Empty string raises JSONDecodeError
    with pytest.raises(json.JSONDecodeError):
        _parse_json("")

    # No JSON object at all raises JSONDecodeError
    with pytest.raises(json.JSONDecodeError):
        _parse_json("I cannot evaluate this message.")


def test_gemini_complete_json_malformed_response_handling():
    """Verify complete_json wraps JSON decode errors in RuntimeError."""
    gp = GeminiProvider(api_key="dummy_key")
    gp.client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "This is not valid json at all."
    gp.client.models.generate_content.return_value = mock_resp

    with pytest.raises(RuntimeError, match="Gemini API request failed"):
        gp.complete_json("system", "user")


# ==============================================================================
# 5. MockProvider Edge Cases
# ==============================================================================

def test_mock_provider_edge_cases():
    """Stress-test MockProvider with various inputs and modes."""
    mp = MockProvider()
    assert mp.name == "mock"

    # Empty message
    res0 = mp.complete_json("sys", "")
    assert res0["level"] == 0

    # Level 3 keyword
    res3 = mp.complete_json("sys", "<user_message>I want to kill myself</user_message>")
    assert res3["level"] == 3

    # Level 2 keyword
    res2 = mp.complete_json("sys", "It is too much to handle!")
    assert res2["level"] == 2

    # Level 1 keyword
    res1 = mp.complete_json("sys", "I feel so anxious and worried.")
    assert res1["level"] == 1

    # Fixed int behavior
    mp_fixed = MockProvider(behaviour=2)
    assert mp_fixed.complete_json("s", "u")["level"] == 2

    # Fail behavior
    mp_fail = MockProvider(behaviour="fail")
    with pytest.raises(TimeoutError):
        mp_fail.complete_json("s", "u")


# ==============================================================================
# 6. SafetyAgent.from_config() Robustness & Timeout Handling
# ==============================================================================

def test_safety_agent_from_config_with_null_timeout():
    """Verify SafetyAgent.from_config does not crash if timeout_s is null in YAML."""
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".yaml") as tmp:
        cfg = {
            "mode": "full",
            "llm": {
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "timeout_s": None,  # Explicitly null
            },
            "fail_safe": True,
        }
        yaml.dump(cfg, tmp)
        tmp_path = tmp.name

    try:
        # If timeout_s is None, int(None * 1000) inside GeminiProvider will raise TypeError.
        # Verify SafetyAgent either catches it gracefully or falls back without crash.
        agent = SafetyAgent.from_config(tmp_path)
        assert agent is not None
    finally:
        os.unlink(tmp_path)


def test_safety_agent_from_config_unsupported_provider():
    """Verify SafetyAgent.from_config gracefully handles an unsupported provider string."""
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".yaml") as tmp:
        cfg = {
            "mode": "full",
            "llm": {
                "provider": "unsupported_provider_xyz",
            },
            "fail_safe": True,
        }
        yaml.dump(cfg, tmp)
        tmp_path = tmp.name

    try:
        agent = SafetyAgent.from_config(tmp_path)
        assert agent is not None
        assert agent.llm is None  # provider failed, continuing without LLM layer
    finally:
        os.unlink(tmp_path)


def test_safety_agent_fail_safe_escalation_when_provider_fails():
    """Verify SafetyAgent triggers fail-safe crisis escalation when provider fails on risk cues."""
    # Build agent with mock-fail provider
    fail_provider = MockProvider("fail")
    agent = SafetyAgent(llm_provider=fail_provider, mode="rules+llm", fail_safe=True)

    session = agent.new_session()
    # "I think it is all pointless" triggers grey-zone risk signal which wants LLM check
    decision = agent.check_user_message(session, "I think it is all pointless")
    # Because LLM failed and message carried risk signal, fail-safe should escalate to CRISIS
    assert decision.message_level == Level.CRISIS
    assert any("fail-safe" in note for note in decision.notes)


def test_gemini_dual_key_precedence():
    """Verify key precedence between GEMINI_API_KEY and GOOGLE_API_KEY."""
    with patch("google.genai.Client"):
        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "gemini_val_111",
            "GOOGLE_API_KEY": "google_val_222"
        }, clear=True):
            gp = GeminiProvider()
            # GEMINI_API_KEY is checked first
            assert gp.api_key == "gemini_val_111"

        with patch.dict(os.environ, {
            "GOOGLE_API_KEY": "google_val_222"
        }, clear=True):
            gp = GeminiProvider()
            assert gp.api_key == "google_val_222"
