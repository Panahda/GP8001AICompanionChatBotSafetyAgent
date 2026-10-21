"""Maps (message level, session level) to what the orchestrator should do.

Fixed texts (helplines, playbook prompts) come from YAML files so the LLM never
writes phone numbers or regulation instructions itself.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from .schema import Action, LayerResult, Level, SafetyDecision
from .state import SessionState

RESOURCES = Path(__file__).parent / "resources"


class Policy:
    def __init__(self, resources_dir: Path = RESOURCES):
        self.help = yaml.safe_load((resources_dir / "helplines.yaml").read_text(encoding="utf-8"))
        self.play = yaml.safe_load((resources_dir / "playbook.yaml").read_text(encoding="utf-8"))
        self.support_line = self.help["support_line"]

    # ------------------------------------------------------------------ texts
    def crisis_text(self, category: Optional[str]) -> str:
        key = self.help["category_to_template"].get(category or "unspecified", "self")
        return self.help["templates"][key].format(**self.help["lines"])

    def playbook_prompt(self, s: SessionState) -> dict:
        prompts = self.play["level2_prompts"]
        p = prompts[s.playbook_index % len(prompts)]
        s.playbook_index += 1
        return p

    # ------------------------------------------------------------------ decision
    def decide(self, s: SessionState, message_level: Level, prev_level: Level,
               category: Optional[str], layers: list[LayerResult], llm_called: bool,
               notes: list[str]) -> SafetyDecision:
        d = SafetyDecision(message_level=message_level, session_level=s.level, action=Action.CONTINUE,
                           category=category, layers=layers, llm_called=llm_called, notes=list(notes))

        if message_level == Level.CRISIS:
            d.action = Action.CRISIS_PAUSE
            d.fixed_reply = self.crisis_text(category)
            d.choices = list(self.help["choices"])
            d.pause_roleplay = d.suppress_feedback = True
            d.carer_alert = s.age_group == "teen" or s.carer_alert_consent
            if d.carer_alert:
                d.notes.append("carer alert: would send 'a crisis was detected' only, no transcript (demo: logged, not sent)")
        elif s.crisis_hold:
            d.action = Action.CRISIS_SUPPORT
            d.companion_directives = [x.format(support_line=self.support_line)
                                      for x in self.play["crisis_support_directives"]]
            d.choices = list(self.help["choices"])
            d.pause_roleplay = d.suppress_feedback = True
        elif s.level == Level.OVERWHELMED:
            d.action = Action.PAUSE_REGULATE
            d.pause_roleplay = d.suppress_feedback = True
            if message_level == Level.OVERWHELMED or prev_level < Level.OVERWHELMED:
                p = self.playbook_prompt(s)
                d.fixed_reply, d.choices = p["text"], list(p["choices"])
                d.notes.append(f"playbook prompt: {p['id']}")
            else:
                d.companion_directives = [
                    "The role-play is paused. Do not continue it and do not give feedback.",
                    "Reply kindly to what the user said in 1 or 2 short sentences.",
                    "Ask if they want to go back to practice or stop for today.",
                ]
                d.choices = ["Go back to practice", "Stop for today"]
        elif s.level == Level.TENSE:
            d.action = Action.SLOW_DOWN
            d.companion_directives = list(self.play["level1_directives"])
            d.choices = list(self.play["level1_choices"])
            d.suppress_feedback = True
        return d
