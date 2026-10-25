"""E2E Test Configuration & Fixtures for CompanionAI.

Provides:
- assert_no_score_leak: Zero-Score-Leak invariant checker across any string.
- Mock providers: deterministic offline LLM providers for plain language and safety checks.
- Test server fixture: ephemeral HTTP server running web_demo for HTTP lifecycle tests.
- Session and scenario helper fixtures.
"""
from __future__ import annotations

import os
import re
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Generator, Optional

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEMO_DIR = PROJECT_ROOT / "demo"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from safety.agent import SafetyAgent
from safety.llm_providers import LLMProvider, MockProvider
from safety.output_guard import SCORE, OutputGuard
from safety.schema import SafetyDecision
from safety.state import SessionState, SessionTracker


# ==============================================================================
# Invariant Verifier: Zero Score Leak
# ==============================================================================
# Comprehensive patterns matching scores, grades, percentages, marks, and right/wrong judgments
_SCORE_LEAK_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in SCORE
] + [
    re.compile(r"\b\d+(\.\d+)?\s*/\s*\d+\b", re.IGNORECASE),          # e.g. 8/10, 4.5/5, 95/100
    re.compile(r"\b\d{1,3}\s*(%|percent)\b", re.IGNORECASE),         # e.g. 85%, 100 percent
    re.compile(r"\b(score|scored|scoring|grade|graded|rating|ratings)\b", re.IGNORECASE),
    re.compile(r"\b\d+\s*(marks?|points?)\b", re.IGNORECASE),
    re.compile(r"\b(wrong|right|correct|incorrect)\s+(answer|reply|response|order)\b", re.IGNORECASE),
    re.compile(r"\b(that is|that's|thats|that was|this is)\s+(wrong|incorrect)\b", re.IGNORECASE),
    re.compile(r"\byou\s+(passed|failed|got it right|got it wrong)\b", re.IGNORECASE),
    re.compile(r"[\u2605\u2606]", re.IGNORECASE),                    # star symbols
    re.compile(r"\bgrade\s+[A-F][+-]?\b", re.IGNORECASE),            # letter grades
]


def assert_no_score_leak(text: Optional[str]) -> None:
    """Verifies that text contains zero scores, grades, percentages, or right/wrong judgments.
    
    Raises AssertionError if any forbidden pattern is detected.
    """
    if not text:
        return
    text_str = str(text)
    for pattern in _SCORE_LEAK_PATTERNS:
        match = pattern.search(text_str)
        if match:
            raise AssertionError(
                f"ZERO-SCORE-LEAK INVARIANT VIOLATION: detected leaked score/judgment "
                f"'{match.group(0)}' in user-facing text:\n{text_str}"
            )


# ==============================================================================
# Mock LLM Providers
# ==============================================================================
class ScriptedMockProvider(MockProvider):
    """Offline mock LLM provider returning predetermined scripted responses."""

    def __init__(self, responses: Optional[list[dict]] = None, default_level: int = 0):
        super().__init__(behaviour=default_level)
        self.responses = list(responses or [])
        self.call_history: list[dict] = []

    def complete_json(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 400,
                      schema: Optional[dict] = None) -> dict:
        self.calls += 1
        self.call_history.append({"system": system, "user": user})
        if self.responses:
            return self.responses.pop(0)
        return super().complete_json(system, user, temperature, max_tokens, schema)


class ScoreLeakingMockProvider(MockProvider):
    """Mock provider that deliberately returns a score leak, for testing guard stripping."""

    def complete_json(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 400,
                      schema: Optional[dict] = None) -> dict:
        self.calls += 1
        return {
            "level": 0,
            "category": None,
            "confidence": 0.95,
            "rationale": "mock",
            "reply": "Good job ordering! That was 9/10. What would you like next?",
        }


# ==============================================================================
# Fixtures
# ==============================================================================
@pytest.fixture
def mock_llm() -> MockProvider:
    return MockProvider(behaviour="keyword")


@pytest.fixture
def safety_agent(mock_llm) -> SafetyAgent:
    config_path = PROJECT_ROOT / "config.yaml"
    return SafetyAgent.from_config(config_path, provider="mock", mode="rules+llm")


@pytest.fixture
def session_teen(safety_agent) -> SessionState:
    return safety_agent.new_session(age_group="teen", carer_alert_consent=False)


@pytest.fixture
def session_adult_no_consent(safety_agent) -> SessionState:
    return safety_agent.new_session(age_group="adult", carer_alert_consent=False)


@pytest.fixture
def session_adult_with_consent(safety_agent) -> SessionState:
    return safety_agent.new_session(age_group="adult", carer_alert_consent=True)


@pytest.fixture
def output_guard() -> OutputGuard:
    return OutputGuard()


@pytest.fixture
def clean_env():
    """Context manager to preserve and restore os.environ."""
    old_env = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(old_env)


def find_free_port() -> int:
    """Finds an unused TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def web_demo_server() -> Generator[str, None, None]:
    """Runs a local Demo HTTPServer on a free port for HTTP API testing."""
    import web_demo

    port = find_free_port()
    demo = web_demo.Demo("mock", "rules+llm")
    server = web_demo.make_server(demo, port)
    actual_port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)  # allow server socket to accept

    base_url = f"http://127.0.0.1:{actual_port}"
    yield base_url

    server.shutdown()
    server.server_close()
