"""Pluggable LLM back-ends. Every provider exposes one method:

    complete_json(system, user, temperature=0.0, max_tokens=400, schema=None) -> dict

`schema` is an optional JSON schema for the answer. LM Studio uses it to force valid
JSON (grammar-constrained decoding); the cloud providers ignore it.

Keys come from the environment: OPENAI_API_KEY or GEMINI_API_KEY (or GOOGLE_API_KEY).
LM Studio needs no key: start its local server (default http://localhost:1234/v1).
Model names change often; override them in config.yaml (llm.model) or with --model.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional, Protocol

from .env import load_env

# Automatically load environment variables from .env on import
load_env()

DEFAULT_MODELS = {
    "openai": "gpt-5-mini",        # check https://platform.openai.com/docs/models
    "gemini": "gemini-3.5-flash",  # check https://ai.google.dev/gemini-api/docs/models
}


class LLMProvider(Protocol):
    name: str
    model: str

    def complete_json(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 400,
                      schema: Optional[dict] = None) -> dict: ...


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)  # tolerate ```json fences
        if not m:
            raise
        return json.loads(m.group(0))


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: Optional[str] = None, timeout_s: float = 20.0, api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        from openai import OpenAI

        self.model = model or DEFAULT_MODELS["openai"]
        kwargs = dict(timeout=timeout_s, max_retries=1)
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if key:
            kwargs["api_key"] = key
        if base_url:
            kwargs["base_url"] = base_url
        try:
            self.client = OpenAI(**kwargs)
        except Exception as e:
            raise RuntimeError(f"OpenAI client initialization failed: {e}") from e

    def complete_json(self, system, user, temperature=0.0, max_tokens=400, schema=None):
        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            max_completion_tokens=max_tokens,
        )
        try:
            try:
                resp = self.client.chat.completions.create(temperature=temperature, **kwargs)
            except Exception as e:  # some reasoning models only accept the default temperature
                if "temperature" not in str(e):
                    raise
                resp = self.client.chat.completions.create(**kwargs)
            return _parse_json(resp.choices[0].message.content)
        except Exception as e:
            raise RuntimeError(f"OpenAI API request failed: {e}") from e


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: Optional[str] = None, timeout_s: float = 20.0, api_key: Optional[str] = None):
        self.model = model or DEFAULT_MODELS["gemini"]
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.client = None
        self.types = None
        self._init_error = None

        try:
            from google import genai
            from google.genai import types

            self.types = types
            client_kwargs = {
                "http_options": types.HttpOptions(timeout=int(timeout_s * 1000))
            }
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
                if self.api_key.startswith("AQ.") or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes"):
                    client_kwargs["vertexai"] = True
            self.client = genai.Client(**client_kwargs)
        except Exception as e:
            self._init_error = str(e)

    def complete_json(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 400,
                      schema: Optional[dict] = None) -> dict:
        if self.client is None or self.types is None:
            err_msg = self._init_error or "Gemini client not initialized (missing API key or google-genai library)"
            raise RuntimeError(f"GeminiProvider is not ready: {err_msg}")

        try:
            cfg = self.types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            resp = self.client.models.generate_content(model=self.model, contents=user, config=cfg)
            return _parse_json(resp.text)
        except Exception as e:
            raise RuntimeError(f"Gemini API request failed: {e}") from e


class LMStudioProvider:
    """A model served locally by LM Studio through its OpenAI-compatible server.

    In LM Studio: load a model, then Developer tab -> Start Server. The model id is the
    one shown there (or set LMSTUDIO_MODEL); if none is given, the first loaded model is used.
    """

    name = "lmstudio"

    def __init__(self, model: Optional[str] = None, timeout_s: float = 60.0, base_url: Optional[str] = None):
        from openai import OpenAI

        url = base_url or os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        self.client = OpenAI(base_url=url, api_key="lm-studio", timeout=timeout_s, max_retries=0)
        self.model = model or os.environ.get("LMSTUDIO_MODEL") or self._first_loaded_model()

    def _first_loaded_model(self) -> str:
        try:
            models = [m.id for m in self.client.models.list().data]
            if not models:
                raise RuntimeError("LM Studio server is running but no model is loaded")
            return models[0]
        except Exception as e:
            raise RuntimeError(f"LM Studio server is unreachable: {e}") from e

    def complete_json(self, system, user, temperature=0.0, max_tokens=400, schema=None):
        fmt = {"type": "json_schema",
               "json_schema": {"name": "answer", "strict": True, "schema": schema or {"type": "object"}}}
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format=fmt, temperature=temperature, max_tokens=max_tokens,
            )
            return _parse_json(resp.choices[0].message.content)
        except Exception as e:
            raise RuntimeError(f"LM Studio API request failed: {e}") from e


class MockProvider:
    """Offline stand-in for tests and for running the pipeline without an API key.

    behaviour:
      "keyword" - crude keyword guess (NOT a real safety check)
      "fail"    - raises, to test the fail-safe path
      int 0..3  - always returns that level
    """

    name = "mock"

    def __init__(self, behaviour="keyword", model: str = "mock"):
        self.behaviour = behaviour
        self.model = model
        self.calls = 0

    def complete_json(self, system, user, temperature=0.0, max_tokens=400, schema=None):
        self.calls += 1
        if self.behaviour == "fail":
            raise TimeoutError("mock provider failure")
        if isinstance(self.behaviour, int):
            return {"level": self.behaviour, "category": None, "confidence": 0.9, "rationale": "mock"}
        m = re.search(r"<user_message>(.*?)</user_message>", user, re.DOTALL)
        msg = (m.group(1) if m else user).lower()
        if re.search(r"\b(die|dead|suicid\w*|kill myself|end it|hurt myself|burden|disappear\w*|without me|goodbye)\b", msg):
            return {"level": 3, "category": None, "confidence": 0.6, "rationale": "mock keyword"}
        if re.search(r"\b(can'?t do this|too much|too loud|stop|overwhelm\w*|leave me alone)\b", msg):
            return {"level": 2, "category": None, "confidence": 0.6, "rationale": "mock keyword"}
        if re.search(r"\b(nervous|anxious|scared|worried|stressed|um+|uh+|is this ok)\b", msg):
            return {"level": 1, "category": None, "confidence": 0.6, "rationale": "mock keyword"}
        return {"level": 0, "category": None, "confidence": 0.6, "rationale": "mock keyword"}


def get_provider(name: str, model: Optional[str] = None, timeout_s: float = 20.0,
                 base_url: Optional[str] = None, api_key: Optional[str] = None) -> LLMProvider:
    name = (name or "mock").lower()
    if name == "openai":
        return OpenAIProvider(model, timeout_s, api_key=api_key, base_url=base_url)
    if name == "gemini":
        prov = GeminiProvider(model, timeout_s, api_key=api_key)
        if prov.client is None:
            raise RuntimeError(f"Gemini provider could not be initialized: {prov._init_error}")
        return prov
    if name == "lmstudio":
        return LMStudioProvider(model, max(timeout_s, 60.0), base_url)
    if name == "mock":
        return MockProvider()
    if name == "mock-fail":
        return MockProvider("fail")
    raise ValueError(f"unknown LLM provider {name!r} (use openai | gemini | lmstudio | mock)")
