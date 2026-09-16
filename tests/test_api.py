"""Integration tests for the FastAPI layer (src.api.app) via FastAPI's TestClient.

Each test drives the HTTP surface only (no direct calls into src.graph/
src.policy/src.execution) to check the API's request/response contract:
POST /tasks for an approved-and-completed run and a policy-denied run,
GET /runs/{run_id} for a run just created and for an unknown run_id, and
GET /health.

The real-provider tests use the same duck-typed fake Anthropic client
technique as tests/test_graph_evaluation_real_provider.py — injected via
`app.dependency_overrides[_get_anthropic_client]` — so none of them make a
network call, require an API key, or even require `anthropic` to be
installed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.api.app import _get_anthropic_client, app

client = TestClient(app)


class FakeMessages:
    def __init__(self, response) -> None:
        self._response = response
        self.create_calls: list[dict] = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return self._response


class FakeAnthropicClient:
    def __init__(self, response) -> None:
        self.messages = FakeMessages(response)


def make_tool_use_response(tool_name: str, arguments: dict):
    return SimpleNamespace(content=[SimpleNamespace(type="tool_use", name=tool_name, input=arguments)])


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    yield
    app.dependency_overrides.clear()


def make_ticket_payload(**overrides) -> dict:
    defaults = dict(
        id="T-1",
        status="open",
        customer_ref="cust-123",
        subject="Package arrived damaged",
        order_ref="ORD-456",
        refund_eligible_amount="50.00",
        currency="USD",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
        updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc).isoformat(),
    )
    defaults.update(overrides)
    return defaults


class TestHealth:
    def test_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestCreateTaskApprovedAndCompleted:
    def test_returns_completed_run(self):
        payload = {
            "ticket": make_ticket_payload(),
            "task_description": "Customer wants a refund for a damaged item.",
            "proposed_action": {
                "tool_name": "issue_refund",
                "arguments": {"ticket_id": "T-1", "amount": "20.00", "currency": "USD", "reason": "damaged item"},
            },
        }
        response = client.post("/tasks", json=payload)
        assert response.status_code == 200

        body = response.json()
        assert "run_id" in body and body["run_id"]
        assert body["run_status"] == "completed"
        assert body["authorization_result"]["decision"] == "approved"
        assert body["execution_result"]["status"] == "success"
        assert body["execution_result"]["verified"] is True


class TestCreateTaskPolicyDenied:
    def test_returns_denied_run(self):
        payload = {
            "ticket": make_ticket_payload(refund_eligible_amount="50.00"),
            "task_description": "Process a refund well above the eligible amount.",
            "proposed_action": {
                "tool_name": "issue_refund",
                "arguments": {"ticket_id": "T-1", "amount": "999.00", "currency": "USD", "reason": "damaged item"},
            },
        }
        response = client.post("/tasks", json=payload)
        assert response.status_code == 200

        body = response.json()
        assert body["run_status"] == "denied"
        assert body["authorization_result"]["decision"] == "denied"
        assert body["authorization_result"]["rule_id"] == "refund-amount-limit"
        assert body["execution_result"] is None


class TestGetRunEvents:
    def test_returns_traced_events_after_post(self):
        payload = {
            "ticket": make_ticket_payload(),
            "task_description": "Add a note to this ticket.",
            "proposed_action": {
                "tool_name": "add_ticket_note",
                "arguments": {"ticket_id": "T-1", "note": "Looked into this."},
            },
        }
        create_response = client.post("/tasks", json=payload)
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        events_response = client.get(f"/runs/{run_id}")
        assert events_response.status_code == 200

        body = events_response.json()
        assert body["run_id"] == run_id
        event_types = [event["event_type"] for event in body["events"]]
        assert event_types == ["proposed", "authorized", "executed", "run_finished"]
        assert all(event["run_id"] == run_id for event in body["events"])

    def test_unknown_run_id_returns_404(self):
        response = client.get("/runs/does-not-exist")
        assert response.status_code == 404


class TestCreateTaskWithRealProviderPath:
    def test_omitted_proposed_action_uses_injected_fake_anthropic_client(self):
        fake_client = FakeAnthropicClient(
            make_tool_use_response(
                "issue_refund",
                {"ticket_id": "T-1", "amount": "20.00", "currency": "USD", "reason": "damaged item"},
            )
        )
        app.dependency_overrides[_get_anthropic_client] = lambda: fake_client

        payload = {
            "ticket": make_ticket_payload(),
            "task_description": "Customer wants a refund for a damaged item.",
        }
        response = client.post("/tasks", json=payload)
        assert response.status_code == 200

        body = response.json()
        assert body["run_status"] == "completed"
        assert body["authorization_result"]["decision"] == "approved"
        assert body["execution_result"]["status"] == "success"
        assert len(fake_client.messages.create_calls) == 1

    def test_omitted_proposed_action_and_mismatched_model_decision_is_denied_not_crashed(self):
        """The 'model' proposing something outside policy is a normal DENIED outcome, not a server error."""
        fake_client = FakeAnthropicClient(
            make_tool_use_response(
                "issue_refund",
                {"ticket_id": "T-1", "amount": "999.00", "currency": "USD", "reason": "damaged item"},
            )
        )
        app.dependency_overrides[_get_anthropic_client] = lambda: fake_client

        payload = {
            "ticket": make_ticket_payload(refund_eligible_amount="50.00"),
            "task_description": "Handle this ticket.",
        }
        response = client.post("/tasks", json=payload)
        assert response.status_code == 200
        assert response.json()["run_status"] == "denied"

    def test_omitted_proposed_action_without_client_or_api_key_returns_400_not_500(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        payload = {
            "ticket": make_ticket_payload(),
            "task_description": "Handle this ticket.",
        }
        response = client.post("/tasks", json=payload)

        assert response.status_code == 400
        assert "ANTHROPIC_API_KEY" in response.json()["detail"]

    def test_never_touches_real_anthropic_sdk_when_client_injected(self):
        import sys

        was_already_imported = "anthropic" in sys.modules
        fake_client = FakeAnthropicClient(
            make_tool_use_response("add_ticket_note", {"ticket_id": "T-1", "note": "Handled."})
        )
        app.dependency_overrides[_get_anthropic_client] = lambda: fake_client

        payload = {"ticket": make_ticket_payload(), "task_description": "Handle this ticket."}
        response = client.post("/tasks", json=payload)

        assert response.status_code == 200
        if not was_already_imported:
            assert "anthropic" not in sys.modules

    def test_does_not_import_anthropic_when_key_missing(self, monkeypatch):
        import sys

        was_already_imported = "anthropic" in sys.modules
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        payload = {"ticket": make_ticket_payload(), "task_description": "Handle this ticket."}
        client.post("/tasks", json=payload)

        if not was_already_imported:
            assert "anthropic" not in sys.modules
