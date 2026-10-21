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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from safety.agent import MODES, SafetyAgent  # noqa: E402

PAGE = Path(__file__).parent / "web" / "index.html"
PROVIDERS = ("lmstudio", "mock", "openai", "gemini")
CONTINUE_BUTTONS = {"Go back to practice", "Keep going", "I did it", "I am back"}
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
        self.sessions: dict[str, dict] = {}

    def agent(self, provider: str | None = None, mode: str | None = None) -> SafetyAgent:
        key = (provider or self.provider, mode or self.mode)
        if key[0] not in PROVIDERS or key[1] not in MODES:
            raise ValueError(f"unknown provider or mode: {key}")
        if key not in self.agents:
            self.agents[key] = SafetyAgent.from_config(ROOT / "config.yaml", provider=key[0], mode=key[1])
        return self.agents[key]

    def conv(self, body: dict) -> dict:
        sid = body.get("session_id")
        if sid not in self.sessions:
            raise LookupError("This conversation is gone (the script was restarted). Start a new conversation.")
        return self.sessions[sid]

    # ------------------------------------------------------------------ API routes
    def new(self, b: dict) -> dict:
        agent = self.agent(b.get("provider"), b.get("mode"))
        sid = uuid.uuid4().hex
        s = agent.new_session(age_group=b.get("age_group", "teen"), carer_alert_consent=bool(b.get("carer_consent")))
        self.sessions[sid] = {"agent": agent, "state": s, "last": None}
        while len(self.sessions) > MAX_SESSIONS:
            self.sessions.pop(next(iter(self.sessions)))
        return {"session_id": sid, "status": status(agent)}

    def check(self, b: dict) -> dict:
        c = self.conv(b)
        text = (b.get("text") or "").strip()
        if not text:
            raise ValueError("Empty message.")
        s, agent = c["state"], c["agent"]
        s.age_group = b.get("age_group", s.age_group)
        s.carer_alert_consent = bool(b.get("carer_consent"))
        t0 = time.perf_counter()
        d = agent.check_user_message(s, text, context=b.get("context") or None, scenario=b.get("scenario") or None)
        c["last"] = d
        return {**d.to_dict(), "crisis_hold": s.crisis_hold, "ms": round((time.perf_counter() - t0) * 1000)}

    def button(self, b: dict) -> dict:
        c = self.conv(b)
        s, agent, label = c["state"], c["agent"], b.get("label", "")
        if label.startswith("I am safe now"):
            agent.release_crisis_hold(s)
            effect = "Crisis hold released. The app goes back to practice."
        elif label in CONTINUE_BUTTONS:
            agent.user_chose_continue(s)
            effect = "The app goes back to practice."
        elif label == "Stop for today":
            effect = "The app would end the session here."
        else:
            effect = "Noted. The app stays at the current level."
        return {"effect": effect, "session_level": int(s.level), "crisis_hold": s.crisis_hold}

    def reply(self, b: dict) -> dict:
        c = self.sessions.get(b.get("session_id"))
        agent = c["agent"] if c else self.agent()
        d = c["last"] if c else None
        # Fixed replies (helplines, playbook) are never checked, so test against a normal turn then.
        g = agent.guard.check(b.get("reply", ""), d if d is not None and d.fixed_reply is None else None)
        return {"ok": g.ok, "action": g.action, "text": g.text, "violations": g.violations,
                "warnings": g.warnings, "hint": g.regenerate_hint}


ROUTES = {"/api/new": Demo.new, "/api/check": Demo.check, "/api/button": Demo.button, "/api/reply": Demo.reply}


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
