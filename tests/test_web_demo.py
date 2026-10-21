"""The local test page's API (demo/web_demo.py), run with the offline mock LLM."""
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))
import web_demo  # noqa: E402

OPEN = urllib.request.build_opener(urllib.request.ProxyHandler({})).open  # never send localhost through a proxy


@pytest.fixture(scope="module")
def base():
    server = web_demo.make_server(web_demo.Demo("mock", "rules+llm"), 0)  # port 0 = any free port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://localhost:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with OPEN(req, timeout=10) as r:
        return json.load(r)


def test_page_and_status(base):
    with OPEN(base + "/", timeout=10) as r:
        assert b"CompanionAI safety check" in r.read()
    with OPEN(base + "/api/status", timeout=10) as r:
        assert json.load(r)["provider"] == "mock"


def test_crisis_then_release(base):
    sid = post(base, "/api/new", {"age_group": "teen"})["session_id"]
    d = post(base, "/api/check", {"session_id": sid, "text": "One chicken rice please.", "scenario": "hawker_centre"})
    assert d["message_level"] == 0 and d["action"] == "continue"
    d = post(base, "/api/check", {"session_id": sid, "text": "i want to die"})
    assert d["message_level"] == 3 and "1767" in d["fixed_reply"] and d["crisis_hold"] and d["carer_alert"]
    j = post(base, "/api/button", {"session_id": sid, "label": "I am safe now. Go back to practice"})
    assert j["crisis_hold"] is False


def test_reply_check_strips_scores(base):
    g = post(base, "/api/reply", {"session_id": None, "reply": "Good job! That was 8/10."})
    assert "8/10" not in g["text"]


def test_refuses_other_hosts_and_unknown_sessions(base):
    req = urllib.request.Request(base + "/api/status", headers={"Host": "evil.example"})
    with pytest.raises(urllib.error.HTTPError) as e:
        OPEN(req, timeout=10)
    assert e.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as e:
        post(base, "/api/check", {"session_id": "nope", "text": "hi"})
    assert e.value.code == 400
