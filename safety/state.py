"""Session state tracker.

The detection layers judge one message at a time. This module turns that stream
into a session level that does not flicker, using three ideas:

1. Personal baseline. A reply is "short" only compared with this user's own
   recent replies, so a person who always writes tersely is never flagged for it.
   A short-reply drop counts as tension only if it happens in 2 of the last 3
   turns (proposal Level 1: "tension, e.g. very short replies").
2. Fast up, slow down (hysteresis). Escalation is immediate; de-escalation needs
   N calm messages in a row and goes down one level at a time.
3. Crisis hold. After a crisis the session stays in support mode until the user
   explicitly chooses to go back to practice (release_crisis_hold).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import median
from typing import Optional

from .schema import Level


@dataclass
class StateConfig:
    baseline_window: int = 10        # replies used for the personal baseline
    min_baseline_turns: int = 3      # need this many replies before judging "short"
    min_baseline_words: float = 6.0  # ignore drops for users whose baseline is already short
    short_ratio: float = 0.35        # reply is "short" if <= 35% of baseline words
    short_drop_needed: int = 2       # ... in this many of the last 3 replies
    calm_turns_to_deescalate: int = 2


@dataclass
class SessionState:
    user_id: str = "demo"
    age_group: str = "adult"            # "teen" (13-17) or "adult"
    carer_alert_consent: bool = False   # adults choose; teens get alerts by design
    level: Level = Level.SAFE
    crisis_hold: bool = False
    crisis_count: int = 0
    calm_streak: int = 0
    lengths: deque = field(default_factory=lambda: deque(maxlen=30))
    short_flags: deque = field(default_factory=lambda: deque(maxlen=3))
    history: list = field(default_factory=list)     # (message_level, session_level, action)
    playbook_index: int = 0


class SessionTracker:
    def __init__(self, cfg: Optional[StateConfig] = None):
        self.cfg = cfg or StateConfig()

    def _short_drop(self, s: SessionState, n_words: int) -> tuple[bool, Optional[float]]:
        prev = list(s.lengths)[-self.cfg.baseline_window:]
        if len(prev) < self.cfg.min_baseline_turns:
            return False, None
        base = float(median(prev))
        if base < self.cfg.min_baseline_words:
            return False, base
        return n_words <= max(2.0, self.cfg.short_ratio * base), base

    def update(self, s: SessionState, message_level: Level, n_words: int) -> tuple[Level, list[str]]:
        notes: list[str] = []
        short, base = self._short_drop(s, n_words)
        s.short_flags.append(short)
        s.lengths.append(n_words)
        if short:
            notes.append(f"short reply: {n_words} words vs personal baseline {base:.0f}")

        # Effective level of this turn: the message itself, plus a tension signal
        # from repeated short-reply drops.
        eff = message_level
        if short and sum(s.short_flags) >= self.cfg.short_drop_needed and eff < Level.TENSE:
            eff = Level.TENSE
            notes.append(f"short-reply drop in {sum(s.short_flags)} of last {len(s.short_flags)} turns -> tense")

        if eff == Level.CRISIS:
            s.crisis_hold = True
            s.crisis_count += 1
            s.level = Level.CRISIS
            s.calm_streak = 0
        elif s.crisis_hold:
            s.level = Level.CRISIS          # stays in support mode until the user releases it
            notes.append("crisis hold active")
        elif eff > Level.SAFE:
            s.calm_streak = 0
            if eff > s.level:
                notes.append(f"escalate {int(s.level)} -> {int(eff)}")
            s.level = max(s.level, eff)
        else:
            s.calm_streak += 1
            if s.level > Level.SAFE and s.calm_streak >= self.cfg.calm_turns_to_deescalate:
                notes.append(f"de-escalate {int(s.level)} -> {int(s.level) - 1} after {s.calm_streak} calm turns")
                s.level = Level(int(s.level) - 1)
                s.calm_streak = 0
        return s.level, notes

    @staticmethod
    def release_crisis_hold(s: SessionState) -> None:
        """Called when the user picks 'I am safe now. Go back to practice'.

        The session restarts gently at Level 1 and needs calm turns to reach Level 0.
        """
        s.crisis_hold = False
        s.level = Level.TENSE
        s.calm_streak = 0

    @staticmethod
    def user_chose_continue(s: SessionState) -> None:
        """Called when the user picks 'continue' after a Level 2 pause."""
        if not s.crisis_hold and s.level == Level.OVERWHELMED:
            s.level = Level.TENSE
            s.calm_streak = 0
