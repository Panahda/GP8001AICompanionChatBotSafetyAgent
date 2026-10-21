"""SafetyAgent: the one object the orchestrator talks to.

    agent = SafetyAgent.from_config("config.yaml")
    session = agent.new_session(age_group="teen")
    decision = agent.check_user_message(session, text, context=last_companion_line, scenario="hawker_centre")
    ... if decision.fixed_reply: show it; else generate a reply with decision.companion_directives
    guard = agent.check_reply(session, decision, reply)

Layer order for each user message (proposal Section 4.1, plus the trained classifier):
    rules  ->  classifier  ->  LLM (only when needed)  ->  fusion  ->  session tracker  ->  policy
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import yaml

from .classifier import ClassifierLayer, load_classifier
from .llm_check import LLMCheckLayer
from .llm_providers import get_provider
from .output_guard import OutputGuard
from .policy import Policy
from .rules import RuleEngine
from .schema import GuardResult, LayerResult, Level, SafetyDecision, Turn
from .state import SessionState, SessionTracker, StateConfig
from .text import word_count

MODES = ("rules", "classifier", "rules+classifier", "rules+llm", "full")


class SafetyAgent:
    def __init__(self, classifier=None, llm_provider=None, mode: str = "full",
                 state_cfg: Optional[StateConfig] = None, fail_safe: bool = True):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.rules = RuleEngine()
        self.clf = ClassifierLayer(classifier) if classifier is not None else None
        self.llm = LLMCheckLayer(llm_provider) if llm_provider is not None else None
        self.tracker = SessionTracker(state_cfg)
        self.policy = Policy()
        self.guard = OutputGuard()
        self.fail_safe = fail_safe

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, path: Union[str, Path] = "config.yaml", provider: Optional[str] = None,
                    mode: Optional[str] = None) -> "SafetyAgent":
        """Build from config.yaml. `provider` / `mode` override the file (e.g. provider="mock")."""
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if provider and provider != cfg.get("llm", {}).get("provider"):
            cfg.setdefault("llm", {})["provider"] = provider
            cfg["llm"]["model"] = None  # the model in the file belongs to the other provider; use the default
        if mode:
            cfg["mode"] = mode
        root = Path(path).resolve().parent
        clf = None
        for key in ("model_dir", "fallback_dir"):
            d = cfg.get("classifier", {}).get(key)
            if d and (root / d / "meta.json").exists():
                clf = load_classifier(root / d)
                break
        llm_cfg = cfg.get("llm", {})
        provider = None
        if llm_cfg.get("provider"):
            try:
                provider = get_provider(llm_cfg["provider"], llm_cfg.get("model"), llm_cfg.get("timeout_s", 20),
                                        llm_cfg.get("base_url"))
            except Exception as e:  # missing package or key: run without the LLM layer (fail-safe applies)
                print(f"[safety] LLM provider unavailable ({e}); continuing without it")
        return cls(classifier=clf, llm_provider=provider, mode=cfg.get("mode", "full"),
                   state_cfg=StateConfig(**cfg.get("state", {})), fail_safe=cfg.get("fail_safe", True))

    def new_session(self, user_id: str = "demo", age_group: str = "adult",
                    carer_alert_consent: bool = False) -> SessionState:
        return SessionState(user_id=user_id, age_group=age_group, carer_alert_consent=carer_alert_consent)

    # ------------------------------------------------------------------ stateless message assessment
    def assess_message(self, text: str, context: Optional[str] = None, scenario: Optional[str] = None
                       ) -> tuple[Level, Optional[str], list[LayerResult], bool, list[str]]:
        """Judge one message without touching any session. Used directly by evaluate.py."""
        use_rules = self.mode in ("rules", "rules+classifier", "rules+llm", "full")
        use_clf = self.mode in ("classifier", "rules+classifier", "full") and self.clf is not None
        use_llm = self.mode in ("rules+llm", "full") and self.llm is not None
        layers: list[LayerResult] = []
        notes: list[str] = []

        r = self.rules.assess(text)
        if use_rules:
            layers.append(r.to_layer())
            if r.is_decisive:  # high-precision rule: act now, skip the slower layers
                return Level.CRISIS, r.category, layers, False, ["decided by rules"]

        c = self.clf.assess(text, context, scenario) if use_clf else None
        if c:
            layers.append(c)

        if self.mode == "rules":
            return r.rules_only_level(), None, layers, False, notes
        if self.mode == "classifier":
            return (c.level if c else Level.SAFE), ("unspecified" if c and c.level == Level.CRISIS else None), layers, False, notes

        risk_signal = (use_rules and r.wants_llm) or bool(c and c.wants_llm)
        need_llm = risk_signal or (c is None and use_rules and r.cue_level is not None)

        if need_llm and use_llm:
            v = self.llm.assess(text, context, scenario)
            layers.append(v)
            if v.error is None:
                return v.level, v.category, layers, True, ["decided by LLM"]
            notes.append(f"LLM failed: {v.error}")
            if self.fail_safe and risk_signal:
                notes.append("fail-safe: risk signal without a working LLM -> crisis")
                return Level.CRISIS, "unspecified", layers, True, notes
        elif need_llm and risk_signal and self.fail_safe and self.mode in ("rules+llm", "full"):
            notes.append("fail-safe: risk signal but no LLM configured -> crisis")
            return Level.CRISIS, "unspecified", layers, False, notes

        if c is not None:
            level = c.level
        elif use_rules:
            level = r.rules_only_level()
        else:
            level = Level.SAFE
        return level, ("unspecified" if level == Level.CRISIS else None), layers, need_llm and use_llm, notes

    # ------------------------------------------------------------------ session-level API
    def check_user_message(self, s: SessionState, text: str,
                           context: Union[None, str, list[Turn]] = None,
                           scenario: Optional[str] = None) -> SafetyDecision:
        if isinstance(context, list):  # take the last companion line from the history
            context = next((t.text for t in reversed(context) if t.role == "companion"), None)
        level, cat, layers, llm_called, notes = self.assess_message(text, context, scenario)
        prev = s.level
        _, state_notes = self.tracker.update(s, level, word_count(text))
        d = self.policy.decide(s, level, prev, cat, layers, llm_called, notes + state_notes)
        s.history.append((int(level), int(s.level), d.action))
        return d

    def check_reply(self, s: SessionState, decision: SafetyDecision, reply: str) -> GuardResult:
        return self.guard.check(reply, decision)

    def release_crisis_hold(self, s: SessionState) -> None:
        self.tracker.release_crisis_hold(s)

    def user_chose_continue(self, s: SessionState) -> None:
        self.tracker.user_chose_continue(s)
