import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from flux_server.api import REQUEST_ID_HEADER, APIError, JobRegistry, create_app, run_cancellable
from flux_server.config import Settings
from flux_server.engine import GenerationCancelled
from tests.conftest import FakeEngine

BODY = {"prompt": "x", "size": "64x64"}


class BlockingEngine(FakeEngine):
    """Spins until released or cancelled, standing in for a slow generation."""

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def generate(self, job):
        self.started.set()
        while not self.release.is_set():
            if job.cancel is not None and job.cancel.is_set():
                raise GenerationCancelled()
            time.sleep(0.005)
        return super().generate(job)


@pytest.fixture
def blocking():
    engine = BlockingEngine()
    app = create_app(Settings(_env_file=None, ui=False), engine=engine)
    with TestClient(app) as tc:
        yield tc, engine
        engine.release.set()


def _post_in_thread(client, headers):
    box: dict = {}

    def worker():
        box["response"] = client.post("/v1/images/generations", json=BODY, headers=headers)

    thread = threading.Thread(target=worker)
    thread.start()
    return thread, box


def test_registry():
    registry = JobRegistry()
    event = registry.register("a")
    assert len(registry) == 1 and not event.is_set()
    with pytest.raises(APIError) as info:
        registry.register("a")
    assert info.value.status_code == 409 and info.value.code == "duplicate_request_id"
    assert registry.cancel("a") is True and event.is_set()
    assert registry.cancel("missing") is False
    registry.unregister("a")
    registry.unregister("a")
    assert len(registry) == 0


def test_explicit_cancel_returns_409(blocking):
    client, engine = blocking
    thread, box = _post_in_thread(client, {REQUEST_ID_HEADER: "job-1"})
    assert engine.started.wait(5)
    assert client.get("/health").json()["in_flight"] == 1

    r = client.post("/v1/images/job-1/cancel")
    assert r.status_code == 200
    assert r.json() == {"request_id": "job-1", "cancelled": True}

    thread.join(5)
    resp = box["response"]
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "cancelled"
    assert resp.headers[REQUEST_ID_HEADER] == "job-1"  # echoed on errors too
    assert client.get("/health").json()["in_flight"] == 0
    assert client.post("/v1/images/job-1/cancel").status_code == 404


def test_cancel_unknown_id(client):
    r = client.post("/v1/images/nope/cancel")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


def test_duplicate_request_id_rejected_while_in_flight(blocking):
    client, engine = blocking
    thread, box = _post_in_thread(client, {REQUEST_ID_HEADER: "dup"})
    assert engine.started.wait(5)

    r = client.post("/v1/images/generations", json=BODY, headers={REQUEST_ID_HEADER: "dup"})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "duplicate_request_id"

    engine.release.set()
    thread.join(5)
    assert box["response"].status_code == 200
    assert box["response"].headers[REQUEST_ID_HEADER] == "dup"


def test_request_id_generated_when_absent(client):
    r = client.post("/v1/images/generations", json=BODY)
    assert r.status_code == 200
    assert len(r.headers[REQUEST_ID_HEADER]) == 32


def test_request_id_too_long(client):
    r = client.post("/v1/images/generations", json=BODY, headers={REQUEST_ID_HEADER: "a" * 129})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == REQUEST_ID_HEADER


def test_request_id_invalid_chars(client):
    r = client.post("/v1/images/generations", json=BODY, headers={REQUEST_ID_HEADER: "job 1/x"})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == REQUEST_ID_HEADER


def test_max_in_flight_returns_503_overloaded():
    engine = BlockingEngine()
    app = create_app(Settings(_env_file=None, ui=False, max_in_flight=1), engine=engine)
    with TestClient(app) as client:
        thread, box = _post_in_thread(client, {REQUEST_ID_HEADER: "first"})
        assert engine.started.wait(5)
        r = client.post("/v1/images/generations", json=BODY)
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "overloaded"
        engine.release.set()
        thread.join(5)
        assert box["response"].status_code == 200


def test_edits_are_cancellable_too(blocking):
    from tests.conftest import png_bytes

    client, engine = blocking
    box: dict = {}

    def worker():
        box["response"] = client.post(
            "/v1/images/edits",
            files=[("image", ("a.png", png_bytes(), "image/png"))],
            data={"prompt": "x"},
            headers={REQUEST_ID_HEADER: "edit-1"},
        )

    thread = threading.Thread(target=worker)
    thread.start()
    assert engine.started.wait(5)
    assert client.post("/v1/images/edit-1/cancel").status_code == 200
    thread.join(5)
    assert box["response"].status_code == 409


class FakeRequest:
    def __init__(self, disconnect_after: int) -> None:
        self.calls = 0
        self.disconnect_after = disconnect_after
        self.url = SimpleNamespace(path="/v1/images/generations")

    async def is_disconnected(self) -> bool:
        self.calls += 1
        return self.calls > self.disconnect_after


def test_run_cancellable_returns_result_while_connected():
    cancel = threading.Event()
    request = FakeRequest(disconnect_after=10**9)
    result = asyncio.run(run_cancellable(request, lambda: 42, cancel, poll_seconds=0.01))
    assert result == 42
    assert not cancel.is_set()


def test_run_cancellable_sets_cancel_on_disconnect():
    cancel = threading.Event()

    def slow():
        assert cancel.wait(5)
        raise GenerationCancelled()

    request = FakeRequest(disconnect_after=1)
    with pytest.raises(GenerationCancelled):
        asyncio.run(run_cancellable(request, slow, cancel, poll_seconds=0.01))
    assert cancel.is_set()


def test_disconnected_client_gets_499(monkeypatch):
    async def disconnected(self) -> bool:
        return True

    monkeypatch.setattr(Request, "is_disconnected", disconnected)
    engine = BlockingEngine()
    app = create_app(Settings(_env_file=None, ui=False), engine=engine)
    with TestClient(app) as client:
        r = client.post("/v1/images/generations", json=BODY)
    assert r.status_code == 499
    assert r.json()["error"]["code"] == "cancelled"
    assert "disconnected" in r.json()["error"]["message"]
    assert engine.jobs == []  # BlockingEngine raised before recording the job
