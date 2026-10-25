"""Plain-language Companion Agent for CompanionAI.

Generates in-character conversational practice replies tailored for verbal autistic
individuals, constrained by autistic communication principles and safety directives,
and verified through OutputGuard with retry and fallback mechanisms.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from .llm_providers import LLMProvider, MockProvider, get_provider
from .output_guard import SAFE_FALLBACK, OutputGuard
from .scenarios import Scenario, get_scenario
from .schema import Action, GuardResult, Level, SafetyDecision
from .state import SessionState


class CompanionAgent:
    """Conversational partner adhering to plain-language principles and safety directives."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        safety_agent: Optional[Any] = None,
    ):
        if provider is None:
            try:
                self.provider = get_provider()
            except Exception:
                self.provider = MockProvider()
        else:
            self.provider = provider

        self.safety_agent = safety_agent
        self.guard = getattr(safety_agent, "guard", None) or OutputGuard()

    def _check_reply(
        self,
        session: Optional[SessionState],
        decision: Optional[SafetyDecision],
        reply: str,
    ) -> GuardResult:
        if self.safety_agent and hasattr(self.safety_agent, "check_reply"):
            return self.safety_agent.check_reply(session, decision, reply)
        return self.guard.check(reply, decision)

    def _build_system_prompt(self, scenario: Scenario, directives: List[str]) -> str:
        directives_text = "\n".join(f"- {d}" for d in directives) if directives else "- None"
        return (
            "You are an AI companion role-play partner for verbal autistic teenagers and adults in Singapore.\n"
            f"Role: {scenario.companion_role}\n"
            f"Scenario Setting: {scenario.setting}\n"
            f"User Goal: {scenario.user_goal}\n\n"
            "Communication Principles:\n"
            "1. Stay strictly in character.\n"
            "2. Use plain, literal English only.\n"
            "3. Keep each sentence short (under 12 to 15 words).\n"
            "4. NEVER use idioms, figures of speech, or sarcasm.\n"
            "5. NEVER make medical or clinical diagnoses or suggest medication.\n"
            "6. NEVER give scores, grades, points, percentages, or say 'right/wrong' / 'correct/incorrect'.\n"
            "7. Ask at most 1 question per turn.\n"
            "8. You MUST strictly obey all safety directives below.\n\n"
            f"Safety Directives:\n{directives_text}\n\n"
            'Output format: Return ONLY a JSON object: {"reply": "<in-character reply>"}'
        )

    def _offline_reply(
        self,
        scenario: Scenario,
        user_text: str,
        directives: List[str],
        hint: Optional[str] = None,
    ) -> str:
        """Deterministic in-character fallback dialogue following autistic communication principles."""
        is_slow_down = any(
            "slow down" in d.lower() or "12 words" in d.lower() or "two choices" in d.lower() or "break" in d.lower()
            for d in directives
        )
        user_lower = user_text.lower().strip()

        if is_slow_down:
            # Short plain sentences (< 12 words), literal, offering choices / break
            if scenario.id == "hawker_centre":
                return "Take your time. Would you like to keep going or take a short break?"
            elif scenario.id == "new_colleague":
                return "No rush. Take your time. Would you like to keep chatting or take a break?"
            elif scenario.id == "job_interview":
                return "Take your time. Would you like to continue or take a short break?"
            else:
                return "No problem. Take your time. Would you like to keep going or take a short break?"

        # Level 0 (Calm / Normal turn): in-character literal dialogue
        if scenario.id == "hawker_centre":
            if any(w in user_lower for w in ["chick", "rice"]):
                if any(w in user_lower for w in ["roast", "steam"]):
                    return "Okay, got it. Eat here or take away?"
                return "Okay, one chicken rice. Steamed or roasted?"
            elif any(w in user_lower for w in ["take", "dabao", "away"]):
                return "Take away, okay. Any drink for you today?"
            elif any(w in user_lower for w in ["drink", "water", "tea", "barley", "kopi"]):
                return "Got it. That is four dollars. Cash or card?"
            elif any(w in user_lower for w in ["cash", "card", "paynow", "qr"]):
                return "Payment received. Here is your food. Enjoy your meal!"
            else:
                return "Hello. What food would you like to order today?"

        elif scenario.id == "new_colleague":
            if any(w in user_lower for w in ["hi", "hello", "morning", "alex"]):
                return "Hi! Nice to meet you too. How is your week going?"
            elif any(w in user_lower for w in ["project", "work", "busy"]):
                return "I see. Our team is working on reports today. It is going well."
            elif any(w in user_lower for w in ["tea", "coffee", "pantry", "break"]):
                return "The pantry is nice and quiet. What kind of tea do you like?"
            else:
                return "That is nice. How is your afternoon going so far?"

        elif scenario.id == "job_interview":
            if any(w in user_lower for w in ["name", "sam", "alex", "apply", "applied"]):
                return "Thank you. What experience do you have with computer systems?"
            elif any(w in user_lower for w in ["comput", "data", "admin", "organis", "organiz"]):
                return "That is very helpful. How do you check your work for accuracy?"
            elif any(w in user_lower for w in ["accura", "detail", "careful"]):
                return "Thank you. Do you have any questions for me about this role?"
            else:
                return "Thank you for sharing that. What position are you applying for?"

        else:  # free_chat or default
            if any(w in user_lower for w in ["mrt", "train", "signal"]):
                return "MRT train systems are fascinating. Which MRT line do you like best?"
            elif any(w in user_lower for w in ["book", "read", "home"]):
                return "Reading at home sounds relaxing. What book are you reading?"
            elif any(w in user_lower for w in ["direct", "bus", "practise", "practice"]):
                return "Sure, we can practise asking for bus directions. What would you like to ask?"
            else:
                return "I understand. What topic would you like to talk about next?"

    def _call_provider(
        self,
        scenario: Scenario,
        user_text: str,
        directives: List[str],
        history: Optional[List[dict]] = None,
        hint: Optional[str] = None,
    ) -> str:
        """Invokes provider or falls back to deterministic in-character dialogue."""
        # Check if provider is a basic MockProvider without scripted reply
        if isinstance(self.provider, MockProvider) and self.provider.name == "mock":
            return self._offline_reply(scenario, user_text, directives, hint)

        all_directives = list(directives)
        if hint:
            all_directives.append(hint)

        sys_prompt = self._build_system_prompt(scenario, all_directives)
        user_content = f"User said: {user_text}"
        if history:
            hist_str = "\n".join(
                f"{h.get('role', 'user')}: {h.get('content', '') or h.get('text', '')}"
                for h in history[-4:]
            )
            user_content = f"Recent dialogue:\n{hist_str}\n\nUser said: {user_text}"

        try:
            res = self.provider.complete_json(
                system=sys_prompt,
                user=user_content,
                temperature=0.0,
                max_tokens=200,
                schema={"type": "object", "properties": {"reply": {"type": "string"}}, "required": ["reply"]},
            )
            if isinstance(res, dict) and "reply" in res and res["reply"]:
                return str(res["reply"]).strip()
            if isinstance(res, dict) and "text" in res and res["text"]:
                return str(res["text"]).strip()
        except Exception:
            # If API call fails or times out, fall back to deterministic offline dialogue
            pass

        return self._offline_reply(scenario, user_text, all_directives, hint)

    def generate(
        self,
        session: Optional[SessionState] = None,
        scenario: Union[Scenario, str, None] = None,
        user_text: str = "",
        directives: Optional[List[str]] = None,
        history: Optional[List[dict]] = None,
        decision: Optional[SafetyDecision] = None,
    ) -> str:
        """Generates an in-character reply, verified by OutputGuard with retry/fallback."""
        if isinstance(scenario, str):
            try:
                scenario_obj = get_scenario(scenario)
            except KeyError:
                scenario_obj = get_scenario("hawker_centre")
        elif isinstance(scenario, Scenario):
            scenario_obj = scenario
        else:
            scenario_obj = get_scenario("hawker_centre")

        active_directives = list(directives or [])
        if decision and decision.companion_directives:
            for d in decision.companion_directives:
                if d not in active_directives:
                    active_directives.append(d)

        # First generation attempt
        candidate = self._call_provider(scenario_obj, user_text, active_directives, history)
        guard_res = self._check_reply(session, decision, candidate)

        if guard_res.ok and guard_res.action != "regenerate":
            return guard_res.text

        # If guard demands regeneration, retry ONCE with hint
        hint = guard_res.regenerate_hint or "Use short literal sentences. Do not mention medical or harm topics."
        retry_candidate = self._call_provider(scenario_obj, user_text, active_directives, history, hint=hint)
        retry_guard = self._check_reply(session, decision, retry_candidate)

        if retry_guard.ok and retry_guard.action != "regenerate":
            return retry_guard.text

        # If retry still fails or is a hard violation, return safe fallback
        return SAFE_FALLBACK


__all__ = ["CompanionAgent"]
