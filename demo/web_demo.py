"""Local safety test page: no login, no hosting, nothing extra to install.

    python demo/web_demo.py                  # LLM from config.yaml (LM Studio by default)
    python demo/web_demo.py --provider mock  # offline stand-in, no LLM needed

Opens http://localhost:8765 in your browser. The page (demo/web/index.html) is plain HTML;
the browser can't run Python, so this script serves the page and runs the safety agent
behind a small JSON API. It uses only the standard library plus the agent itself, and it
only listens on this computer (127.0.0.1). With LM Studio, messages never leave the machine.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
import uuid
import webbrowser
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from safety.agent import MODES, SafetyAgent  # noqa: E402
from safety.companion import CompanionAgent  # noqa: E402
from safety.feedback import FeedbackAgent  # noqa: E402
from safety.orchestrator import handle_turn  # noqa: E402
from safety.scenarios import get_scenario, list_scenarios  # noqa: E402

PAGE = Path(__file__).parent / "web" / "index.html"
PROVIDERS = ("lmstudio", "mock", "openai", "gemini")
CONTINUE_BUTTONS = {"Go back to practice", "Keep going", "I did it", "I am back", "Keep talking"}
MAX_SESSIONS = 50
ALLOWED_HOSTS = {"localhost", "127.0.0.1"}


def lmstudio_problem(base_url: str, model: str | None) -> str | None:
    """None if LM Studio answers and lists the model, else what is wrong."""
    no_proxy = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # LM Studio is local
    try:
        with no_proxy.open(base_url.rstrip("/") + "/models", timeout=3) as r:
            models = [m["id"] for m in json.load(r).get("data", [])]
    except Exception:
        return (f"LM Studio is not reachable at {base_url}. Open LM Studio, load a model, then "
                "Developer -> Start Server. Until then, unclear risky messages are treated as a crisis.")
    if not models:
        return "LM Studio is running but no model is loaded."
    if model and model not in models:
        return f"LM Studio doesn't list the model '{model}' (it has: {', '.join(models)}). Load it or change llm.model in config.yaml."
    return None


def status(agent: SafetyAgent) -> dict:
    llm = agent.llm.provider if agent.llm else None
    out = {"mode": agent.mode, "provider": getattr(llm, "name", None), "model": getattr(llm, "model", None),
           "classifier": agent.clf.clf.name if agent.clf else None, "llm_problem": None}
    if llm is None:
        out["llm_problem"] = "No LLM is connected, so unclear risky messages are treated as a crisis."
    elif out["provider"] == "mock":
        out["llm_problem"] = "Offline test mode: the LLM check is a simple stand-in, so results are rough."
    elif out["provider"] == "lmstudio":
        out["llm_problem"] = lmstudio_problem(str(llm.client.base_url), llm.model)
    return out


class Demo:
    """Agents (one per provider + mode) and open conversations, all in memory."""

    def __init__(self, provider: str, mode: str):
        self.provider, self.mode = provider, mode
        self.lock = threading.Lock()
        self.agents: dict[tuple[str, str], SafetyAgent] = {}
        self.companion_agents: dict[tuple[str, str], CompanionAgent] = {}
        self.feedback_agents: dict[tuple[str, str], FeedbackAgent] = {}
        self.sessions: dict[str, dict] = {}

    def agent(self, provider: str | None = None, mode: str | None = None) -> SafetyAgent:
        key = (provider or self.provider, mode or self.mode)
        if key[0] not in PROVIDERS or key[1] not in MODES:
            raise ValueError(f"unknown provider or mode: {key}")
        if key not in self.agents:
            self.agents[key] = SafetyAgent.from_config(ROOT / "config.yaml", provider=key[0], mode=key[1])
        return self.agents[key]

    def companion(self, provider: str | None = None, mode: str | None = None) -> CompanionAgent:
        key = (provider or self.provider, mode or self.mode)
        if key not in self.companion_agents:
            sa = self.agent(key[0], key[1])
            llm_prov = sa.llm.provider if sa.llm else None
            self.companion_agents[key] = CompanionAgent(provider=llm_prov, safety_agent=sa)
        return self.companion_agents[key]

    def feedback(self, provider: str | None = None, mode: str | None = None) -> FeedbackAgent:
        key = (provider or self.provider, mode or self.mode)
        if key not in self.feedback_agents:
            sa = self.agent(key[0], key[1])
            llm_prov = sa.llm.provider if sa.llm else None
            self.feedback_agents[key] = FeedbackAgent(provider=llm_prov)
        return self.feedback_agents[key]

    def conv(self, body: dict) -> dict:
        sid = body.get("session_id")
        if sid not in self.sessions:
            raise LookupError("This conversation is gone (the script was restarted). Start a new conversation.")
        return self.sessions[sid]

    # ------------------------------------------------------------------ API routes
    def scenarios(self, b: dict | None = None) -> dict:
        return {"scenarios": [s.to_dict() for s in list_scenarios()]}

    def new(self, b: dict) -> dict:
        provider = b.get("provider") or self.provider
        mode = b.get("mode") or self.mode
        agent = self.agent(provider, mode)
        sid = uuid.uuid4().hex

        sc_id = b.get("scenario") or "hawker_centre"
        try:
            sc = get_scenario(sc_id)
        except KeyError:
            sc = get_scenario("hawker_centre")

        s = agent.new_session(age_group=b.get("age_group", "teen"), carer_alert_consent=bool(b.get("carer_consent")))
        self.sessions[sid] = {
            "agent": agent,
            "state": s,
            "scenario": sc,
            "history": [{"role": "companion", "text": sc.opening_line}],
            "last": None,
            "last_companion_line": sc.opening_line,
            "provider": provider,
            "mode": mode,
        }
        while len(self.sessions) > MAX_SESSIONS:
            self.sessions.pop(next(iter(self.sessions)))

        example_replies = [
            {"text": ex.text, "explanation": ex.explanation} if hasattr(ex, "text") else ex
            for ex in sc.example_replies
        ]
        return {
            "session_id": sid,
            "status": status(agent),
            "scenario": sc.to_dict(),
            "opening_line": sc.opening_line,
            "user_goal": sc.user_goal,
            "setting": sc.setting,
            "example_replies": example_replies,
            "real_life_prompt": sc.real_life_prompt,
        }

    def check(self, b: dict) -> dict:
        c = self.conv(b)
        text = (b.get("text") or "").strip()
        if not text:
            raise ValueError("Empty message.")
        s, agent = c["state"], c["agent"]
        if "age_group" in b:
            s.age_group = b.get("age_group", s.age_group)
        if "carer_consent" in b:
            s.carer_alert_consent = bool(b.get("carer_consent"))

        sc_id = b.get("scenario") or getattr(c.get("scenario"), "id", None) or "hawker_centre"
        try:
            sc = get_scenario(sc_id)
        except KeyError:
            sc = c.get("scenario") or get_scenario("hawker_centre")
        c["scenario"] = sc

        provider = c.get("provider", self.provider)
        mode = c.get("mode", self.mode)
        comp_agent = self.companion(provider, mode)
        fb_agent = self.feedback(provider, mode)

        t0 = time.perf_counter()
        last_comp_line = b.get("context") or c.get("last_companion_line") or sc.opening_line

        turn = handle_turn(
            agent,
            session=s,
            user_text=text,
            last_companion_line=last_comp_line,
            scenario=sc,
            companion_agent=comp_agent,
            feedback_agent=fb_agent,
            history=c.get("history", []),
        )
        c["last"] = turn.decision
        c["last_companion_line"] = turn.reply
        c.setdefault("history", []).append({"role": "user", "text": text})
        c["history"].append({"role": "companion", "text": turn.reply})

        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        d_dict = turn.decision.to_dict()
        return {
            **d_dict,
            "reply": turn.reply,
            "feedback": turn.feedback,
            "buttons": list(turn.buttons),
            "decision": d_dict,
            "scenario_id": sc.id,
            "end_of_session_prompt": sc.real_life_prompt,
            "crisis_hold": s.crisis_hold,
            "ms": elapsed_ms,
        }

    def button(self, b: dict) -> dict:
        c = self.conv(b)
        s, agent = c["state"], c["agent"]
        label = b.get("label") or b.get("action") or ""
        if label.startswith("I am safe now"):
            agent.release_crisis_hold(s)
            effect = "Crisis hold released. The app goes back to practice."
        elif label in CONTINUE_BUTTONS or label in ("Keep talking", "I did it", "Something else"):
            agent.user_chose_continue(s)
            effect = "The app goes back to practice."
        elif label == "Stop for today":
            effect = "The app would end the session here."
        else:
            effect = "Noted. The app stays at the current level."
        return {"effect": effect, "session_level": int(s.level), "crisis_hold": s.crisis_hold, "action": label}

    def reply(self, b: dict) -> dict:
        c = self.sessions.get(b.get("session_id"))
        agent = c["agent"] if c else self.agent()
        d = c["last"] if c else None
        candidate = b.get("reply") if "reply" in b else b.get("text", "")
        # Fixed replies (helplines, playbook) are never checked, so test against a normal turn then.
        g = agent.guard.check(candidate, d if d is not None and d.fixed_reply is None else None)
        return {"ok": g.ok, "action": g.action, "text": g.text, "violations": g.violations,
                "warnings": g.warnings, "hint": g.regenerate_hint}


ROUTES = {
    "/api/new": Demo.new,
    "/api/check": Demo.check,
    "/api/button": Demo.button,
    "/api/action": Demo.button,
    "/api/reply": Demo.reply,
    "/api/guard_test": Demo.reply,
    "/api/scenarios": Demo.scenarios,
}


def make_handler(demo: Demo):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        def _send(self, code: int, body, ctype: str = "application/json; charset=utf-8") -> None:
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _local(self) -> bool:
            # Only answer pages opened on this computer (blocks other sites reaching the API).
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            return host in ALLOWED_HOSTS

        def do_GET(self):
            if not self._local():
                return self._send(403, {"error": "Open the page as http://localhost:<port>."})
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if url.path == "/api/status":
                sid = (parse_qs(url.query).get("session_id") or [None])[0]
                c = demo.sessions.get(sid)
                return self._send(200, status(c["agent"] if c else demo.agent()))
            if url.path == "/api/scenarios":
                return self._send(200, demo.scenarios())
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._local():
                return self._send(403, {"error": "Open the page as http://localhost:<port>."})
            route = ROUTES.get(urlparse(self.path).path)
            if route is None:
                return self._send(404, {"error": "not found"})
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                return self._send(415, {"error": "JSON only"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                with demo.lock:  # one message at a time; the demo has one user
                    out = route(demo, body)
                self._send(200, out)
            except (LookupError, ValueError) as e:
                self._send(400, {"error": str(e)})
            except Exception as e:  # show the problem on the page instead of a silent failure
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def make_server(demo: Demo, port: int) -> ThreadingHTTPServer:
    for p in range(port, port + 10):  # next free port if this one is taken
        try:
            return ThreadingHTTPServer(("127.0.0.1", p), make_handler(demo))
        except OSError:
            continue
    raise OSError(f"No free port between {port} and {port + 9}")


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    ap = argparse.ArgumentParser(description="Local safety test page")
    ap.add_argument("--provider", choices=PROVIDERS, default=(cfg.get("llm") or {}).get("provider") or "mock")
    ap.add_argument("--mode", choices=MODES, default=cfg.get("mode", "full"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="don't open the browser")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # show the messages below straight away

    demo = Demo(args.provider, args.mode)
    print("Loading the safety agent ...")
    st = status(demo.agent())
    print(f"  LLM: {st['provider']}/{st['model']}   classifier: {st['classifier'] or 'not loaded'}   mode: {st['mode']}")
    if st["llm_problem"]:
        print(f"  Note: {st['llm_problem']}")

    server = make_server(demo, args.port)
    url = f"http://localhost:{server.server_address[1]}"
    print(f"\nSafety test page: {url}\nLeave this window open while you use the page. Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
