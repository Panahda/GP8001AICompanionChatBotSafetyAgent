"""Unit tests for safety.env and safety.llm_providers."""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety.agent import SafetyAgent
from safety.env import load_env, parse_env_line
from safety.llm_providers import (
    DEFAULT_MODELS,
    GeminiProvider,
    MockProvider,
    OpenAIProvider,
    LMStudioProvider,
    get_provider,
)


def test_parse_env_line():
    assert parse_env_line("") is None
    assert parse_env_line("# comment") is None
    assert parse_env_line("INVALID") is None
    assert parse_env_line("KEY=val") == ("KEY", "val")
    assert parse_env_line("export FOO=bar") == ("FOO", "bar")
    assert parse_env_line('QUOTED="hello world"') == ("QUOTED", "hello world")
    assert parse_env_line("SINGLE='single quote'") == ("SINGLE", "single quote")
    assert parse_env_line("WITH_COMMENT=value # inline comment") == ("WITH_COMMENT", "value")


def test_load_env_temporary_file():
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".env") as tmp:
        tmp.write("TEST_LOAD_ENV_KEY_1=loaded_val\n")
        tmp.write("TEST_LOAD_ENV_KEY_EXISTING=new_val\n")
        tmp_path = tmp.name

    try:
        # Pre-set existing key to test non-destructive loading
        os.environ["TEST_LOAD_ENV_KEY_EXISTING"] = "original_val"
        if "TEST_LOAD_ENV_KEY_1" in os.environ:
            del os.environ["TEST_LOAD_ENV_KEY_1"]

        assert load_env(tmp_path) is True
        assert os.environ["TEST_LOAD_ENV_KEY_1"] == "loaded_val"
        assert os.environ["TEST_LOAD_ENV_KEY_EXISTING"] == "original_val"

        # Test override=True
        load_env(tmp_path, override=True)
        assert os.environ["TEST_LOAD_ENV_KEY_EXISTING"] == "new_val"
    finally:
        os.unlink(tmp_path)
        os.environ.pop("TEST_LOAD_ENV_KEY_1", None)
        os.environ.pop("TEST_LOAD_ENV_KEY_EXISTING", None)


def test_load_env_nonexistent_file():
    assert load_env("/nonexistent/path/to/.env") is False


def test_default_models_gemini():
    assert DEFAULT_MODELS["gemini"] == "gemini-2.5-flash"


def test_gemini_provider_dual_key_support():
    with patch("google.genai.Client") as mock_client:
        # Test GEMINI_API_KEY
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-gemini"}, clear=True):
            gp = GeminiProvider()
            assert gp.api_key == "test-key-gemini"
            assert gp.model == "gemini-2.5-flash"
            assert mock_client.call_args[1].get("api_key") == "test-key-gemini"

        # Test GOOGLE_API_KEY fallback
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key-google"}, clear=True):
            gp = GeminiProvider()
            assert gp.api_key == "test-key-google"
            assert mock_client.call_args[1].get("api_key") == "test-key-google"

        # Test explicit api_key param
        with patch.dict(os.environ, {}, clear=True):
            gp = GeminiProvider(api_key="direct-param-key")
            assert gp.api_key == "direct-param-key"
            assert mock_client.call_args[1].get("api_key") == "direct-param-key"


def test_gemini_provider_graceful_missing_key():
    with patch("google.genai.Client", side_effect=ValueError("No API key provided")):
        with patch.dict(os.environ, {}, clear=True):
            gp = GeminiProvider()
            assert gp.api_key is None
            assert gp.client is None
            with pytest.raises(RuntimeError) as exc_info:
                gp.complete_json("system", "user")
            assert "GeminiProvider is not ready" in str(exc_info.value)


def test_fallbacks_intact():
    mock_p = get_provider("mock")
    assert isinstance(mock_p, MockProvider)
    assert mock_p.name == "mock"

    mock_fail = get_provider("mock-fail")
    assert isinstance(mock_fail, MockProvider)
    with pytest.raises(TimeoutError):
        mock_fail.complete_json("s", "u")

    with patch("openai.OpenAI"):
        openai_p = OpenAIProvider(api_key="sk-dummy")
        assert openai_p.name == "openai"

        lmstudio_p = LMStudioProvider(model="local-model")
        assert lmstudio_p.name == "lmstudio"
        assert lmstudio_p.model == "local-model"


def test_safety_agent_from_config_defaults():
    # Verify SafetyAgent loads config.yaml with gemini default without crashing
    agent = SafetyAgent.from_config("config.yaml", provider="mock")
    assert agent is not None
    assert agent.mode == "full"
