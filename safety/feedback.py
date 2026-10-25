"""Rubric-constrained Feedback Agent for CompanionAI.

Evaluates user conversational turns strictly against the scenario goal and fixed rubric,
citing only the scenario goal and suggested example replies on calm turns, with complete
suppression on non-calm turns and mandatory score stripping.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from .llm_providers import LLMProvider, MockProvider, get_provider
from .output_guard import OutputGuard
from .scenarios import Scenario, get_scenario
from .schema import SafetyDecision


class FeedbackAgent:
    """Rubric-constrained coach providing at most one optional suggestion on calm turns."""

    def __init__(self, provider: Optional[LLMProvider] = None):
        if provider is None:
            try:
                self.provider = get_provider()
            except Exception:
                self.provider = MockProvider()
        else:
            self.provider = provider

        self.guard = OutputGuard()

    def _build_system_prompt(self, scenario: Scenario) -> str:
        rubric_lines = "\n".join(f"- {r.criterion}: {r.description}" for r in scenario.rubric)
        examples_lines = "\n".join(f"- '{ex.text}': {ex.explanation}" for ex in scenario.example_replies)
        return (
            "You are a supportive, rubric-constrained communication coach for verbal autistic individuals.\n"
            f"Scenario Goal: {scenario.user_goal}\n\n"
            f"Scenario Rubric:\n{rubric_lines}\n\n"
            f"Example Replies (Suggestions):\n{examples_lines}\n\n"
            "Rules for your feedback:\n"
            "1. Return at most ONE short, optional suggestion (under 40 words total).\n"
            "2. Ground your suggestion ONLY in the scenario goal and the example replies above.\n"
            "3. NEVER invent generic social platitudes or unsolicited advice.\n"
            "4. NEVER assign grades, scores, percentages, points, or say 'right/wrong' / 'correct/incorrect'.\n"
            "5. Frame the suggestion gently as an optional phrasing to try.\n\n"
            'Output format: Return ONLY a JSON object: {"suggestion": "<one optional suggestion>"}'
        )

    def _offline_feedback(self, scenario: Scenario, user_text: str) -> str:
        """Deterministic rubric-grounded suggestion citing scenario goal and example replies."""
        user_lower = user_text.lower().strip()

        if scenario.id == "hawker_centre":
            if any(w in user_lower for w in ["take", "dabao", "away"]):
                return "Goal: complete food and drink details. For example, you can try: 'Chicken rice eat here, and one iced barley water please.'"
            return "Goal: state your food choice and takeaway option. For example, you can try: 'One plate of chicken rice, roasted, no cucumber please.'"

        elif scenario.id == "new_colleague":
            if any(w in user_lower for w in ["tea", "water", "break", "busy"]):
                return "Goal: ask a friendly return question. For example, you can try: 'Hi Alex, nice to meet you too! Welcome to the team. How are you settling in?'"
            return "Goal: share a brief routine detail. For example, you can try: 'Hello Alex! I am doing well, just getting some tea before my afternoon meeting.'"

        elif scenario.id == "job_interview":
            if any(w in user_lower for w in ["sam", "alex", "name", "diploma"]):
                return "Goal: connect your abilities to this role. For example, you can try: 'Hello Ms. Tan. I am very organised and detail-oriented. I applied because I want to support your team.'"
            return "Goal: introduce yourself and mention one strength. For example, you can try: 'Good morning Ms. Tan. My name is Sam. I have experience with data entry.'"

        else:  # free_chat or default
            return "Goal: share a topic or interest you enjoy. For example, you can try: 'I would like to talk about how the MRT train signalling system works in Singapore.'"

    def generate(
        self,
        scenario: Union[Scenario, str, None],
        user_text: str,
        companion_reply: str = "",
        decision: Optional[SafetyDecision] = None,
        **kwargs,
    ) -> Optional[str]:
        """Generates at most one short optional suggestion when calm, else returns None."""
        # 1. Feedback gating: strictly suppressed on non-calm turns
        if decision is not None and getattr(decision, "suppress_feedback", False):
            return None

        # Resolve scenario object
        if isinstance(scenario, str):
            try:
                scenario_obj = get_scenario(scenario)
            except KeyError:
                scenario_obj = get_scenario("hawker_centre")
        elif isinstance(scenario, Scenario):
            scenario_obj = scenario
        else:
            scenario_obj = get_scenario("hawker_centre")

        raw_suggestion: Optional[str] = None

        # 2. Invoke provider if not plain mock
        if not (isinstance(self.provider, MockProvider) and self.provider.name == "mock"):
            try:
                sys_prompt = self._build_system_prompt(scenario_obj)
                user_content = f"User said: {user_text}\nCompanion replied: {companion_reply}"
                res = self.provider.complete_json(
                    system=sys_prompt,
                    user=user_content,
                    temperature=0.0,
                    max_tokens=150,
                    schema={"type": "object", "properties": {"suggestion": {"type": "string"}}, "required": ["suggestion"]},
                )
                if isinstance(res, dict) and "suggestion" in res and res["suggestion"]:
                    raw_suggestion = str(res["suggestion"]).strip()
            except Exception:
                raw_suggestion = None

        if not raw_suggestion:
            raw_suggestion = self._offline_feedback(scenario_obj, user_text)

        # 3. Always strip any accidental scores, grades, percentages, or marks
        cleaned, _ = self.guard.strip_scores(raw_suggestion)
        cleaned = cleaned.strip()

        if not cleaned:
            return None

        # 4. Limit length: at most 50 words
        words = cleaned.split()
        if len(words) > 50:
            cleaned = " ".join(words[:45]) + "..."

        return cleaned


__all__ = ["FeedbackAgent"]
