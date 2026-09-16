"""A minimal in-memory evaluation harness for the support-ticket ops agent.

Given a small set of `EvalScenario` fixtures — a `Ticket`, a
`task_description`, a scripted sequence of `ProposedAction`s (fed through
`ScriptedProposeProvider`), and an expected final `RunStatus` (plus,
optionally, an expected `AuthorizationResult.rule_id` or
`ExecutionResult.status`) — `run_scenarios()` runs each one through
`build_agent_graph()` with a fresh `Tracer` per scenario, and reports
whether the actual outcome matched what was expected, along with the full
traced event sequence for any scenario that didn't match.

This is a plain Python utility meant to be called from a test or a
throwaway script — there is no CLI and no persistence layer.

`run_scenario()`/`run_scenarios()` default to `ScriptedProposeProvider`
(no real LLM call), but also accept an explicit `propose_fn` to swap in
any other `ProposeFn`. `run_scenarios_with_real_provider()` and
`run_default_scenarios_against_real_provider()` use this to run the same
scenarios against the real `AnthropicProposeProvider` instead — see their
docstrings for the real-network-call, real-API-cost, and
`ANTHROPIC_API_KEY` implications before calling either of them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from unittest.mock import patch

from src.execution import executor as executor_module
from src.execution.executor import ExecutionWrapper, FailurePredicate
from src.execution.schemas import ExecutionStatus
from src.graph.build import build_agent_graph
from src.graph.nodes import ProposeFn
from src.graph.providers.anthropic_provider import DEFAULT_MODEL as ANTHROPIC_DEFAULT_MODEL, AnthropicProposeProvider
from src.graph.providers.mock import ScriptedProposeProvider
from src.graph.state import AgentState, RunStatus
from src.observability.tracer import Tracer, TraceEvent
from src.tools.schemas import AuthorizedUpdateTicketStatusCommand, ProposedAction, Ticket, TicketStatus


@dataclass
class EvalScenario:
    """One fixed scenario to run through the agent graph and check the outcome of.

    `proposals` is fed to a fresh `ScriptedProposeProvider` per run, so a
    scenario can script a multi-turn sequence, though every scenario in
    this phase's default set only needs one proposal.

    `tool_function_overrides`, if given, temporarily replaces entries in
    `src.execution.executor._TOOL_FUNCTIONS` (keyed by concrete
    `AuthorizedCommand` subtype) for the duration of this scenario's run
    only — e.g. to install a "buggy" tool function stub that runs without
    error but doesn't actually apply its effect, forcing a genuine
    `ExecutionStatus.VERIFICATION_FAILED` the same way
    `tests/test_execution_executor.py`/`tests/test_graph_build.py` already
    do, rather than mocking `verify()` itself.

    `should_fail`, if given, is injected as this scenario's
    `ExecutionWrapper.execute()` failure predicate (see
    `src.execution.executor.FailurePredicate`) — e.g. a predicate that
    always returns `True` simulates transient failure on every attempt,
    exhausting retries and forcing a genuine `ExecutionStatus.FAILURE`,
    the same way `tests/test_execution_executor.py` already does via
    `execute(..., should_fail=...)`.
    """

    name: str
    ticket: Ticket
    task_description: str
    proposals: Sequence[ProposedAction]
    expected_run_status: RunStatus
    expected_rule_id: str | None = None
    expected_execution_status: ExecutionStatus | None = None
    tool_function_overrides: Mapping[type, Callable] | None = None
    should_fail: FailurePredicate | None = None


@dataclass
class EvalResult:
    """The outcome of running one `EvalScenario`."""

    scenario_name: str
    passed: bool
    expected_run_status: RunStatus
    actual_run_status: RunStatus
    expected_rule_id: str | None
    actual_rule_id: str | None
    expected_execution_status: ExecutionStatus | None
    actual_execution_status: ExecutionStatus | None
    events: list[TraceEvent] = field(default_factory=list)


def _inject_should_fail(execution_wrapper: ExecutionWrapper, should_fail: FailurePredicate) -> None:
    """Bind `should_fail` as the default failure predicate for this ExecutionWrapper instance's execute() calls.

    `execute_node` (in `src.graph.nodes`) always calls
    `execution_wrapper.execute(command, ticket)` with no `should_fail`
    argument of its own, so this shadows the instance's bound `execute`
    method with a thin wrapper that supplies `should_fail` as a default,
    forwarding everything else unchanged — no change to
    `src.graph.nodes`/`src.graph.build` is needed.

    Unlike `tool_function_overrides` (which patches a shared module-level
    dict and must be reverted after use), this needs no cleanup:
    `execution_wrapper` is a fresh instance created exclusively for one
    `run_scenario()` call and discarded when it returns, so nothing else
    is ever affected by this instance-level monkeypatch.
    """
    original_execute = execution_wrapper.execute

    def execute_with_should_fail(command, ticket, **kwargs):
        kwargs.setdefault("should_fail", should_fail)
        return original_execute(command, ticket, **kwargs)

    execution_wrapper.execute = execute_with_should_fail


def run_scenario(scenario: EvalScenario, propose_fn: ProposeFn | None = None) -> EvalResult:
    """Run one `EvalScenario` through a freshly built agent graph and check its outcome.

    Uses its own fresh `Tracer` and `ExecutionWrapper` per call, so
    scenarios never share idempotency or trace state with each other.

    `propose_fn` defaults to a fresh `ScriptedProposeProvider(scenario.
    proposals)` (the original, deterministic behavior). Pass a different
    `ProposeFn` — e.g. an `AnthropicProposeProvider` — to drive this same
    scenario's `Ticket`/`task_description` through a different proposal
    source instead; `scenario.proposals` is then unused for the proposal
    itself, though `tool_function_overrides`/`should_fail` still apply
    regardless of where the proposal came from.
    """
    tracer = Tracer()
    execution_wrapper = ExecutionWrapper()
    if scenario.should_fail is not None:
        _inject_should_fail(execution_wrapper, scenario.should_fail)
    provider = propose_fn if propose_fn is not None else ScriptedProposeProvider(scenario.proposals)
    graph = build_agent_graph(provider, execution_wrapper=execution_wrapper, tracer=tracer)

    run_id = f"eval-{scenario.name}"
    initial_state = AgentState(run_id=run_id, ticket=scenario.ticket, task_description=scenario.task_description)

    def invoke() -> AgentState:
        raw_result = graph.invoke(initial_state)
        return AgentState.model_validate(raw_result)

    if scenario.tool_function_overrides:
        with patch.dict(executor_module._TOOL_FUNCTIONS, scenario.tool_function_overrides):
            final_state = invoke()
    else:
        final_state = invoke()

    actual_rule_id = final_state.authorization_result.rule_id if final_state.authorization_result else None
    actual_execution_status = final_state.execution_result.status if final_state.execution_result else None

    passed = final_state.run_status == scenario.expected_run_status
    if scenario.expected_rule_id is not None:
        passed = passed and actual_rule_id == scenario.expected_rule_id
    if scenario.expected_execution_status is not None:
        passed = passed and actual_execution_status == scenario.expected_execution_status

    return EvalResult(
        scenario_name=scenario.name,
        passed=passed,
        expected_run_status=scenario.expected_run_status,
        actual_run_status=final_state.run_status,
        expected_rule_id=scenario.expected_rule_id,
        actual_rule_id=actual_rule_id,
        expected_execution_status=scenario.expected_execution_status,
        actual_execution_status=actual_execution_status,
        events=tracer.events_for_run(run_id),
    )


def run_scenarios(scenarios: Sequence[EvalScenario]) -> list[EvalResult]:
    """Run each scenario in `scenarios` in order and return one `EvalResult` per scenario."""
    return [run_scenario(scenario) for scenario in scenarios]


def run_scenarios_with_real_provider(
    scenarios: Sequence[EvalScenario],
    client: Any | None = None,
    model: str = ANTHROPIC_DEFAULT_MODEL,
) -> list[EvalResult]:
    """Run each scenario in `scenarios` against the real `AnthropicProposeProvider` instead of a script.

    Each scenario gets its own fresh `AnthropicProposeProvider(model=model,
    client=client)`. `scenario.proposals` is ignored for the proposal
    itself — the model decides what to propose from `scenario.ticket`/
    `scenario.task_description` — but `tool_function_overrides`/
    `should_fail`, if set on a scenario, still apply during execution
    regardless of where the proposal came from.

    WARNING: unless `client` is an injected fake (for tests), calling this
    makes real Anthropic API calls, incurring real cost, and requires
    `ANTHROPIC_API_KEY` to be set in the environment (the `anthropic` SDK
    reads it lazily on first call — see `AnthropicProposeProvider`).

    Because the model's real decision may legitimately differ from what a
    scenario scripted for testing purposes, a resulting
    `EvalResult.passed == False` here is informative, not necessarily a
    bug: it means the model didn't do what the scenario's `expected_*`
    fields assumed a scripted proposal would do — `run_scenario()` reports
    that mismatch exactly the same way it reports any other, no special
    handling.
    """
    results = []
    for scenario in scenarios:
        provider = AnthropicProposeProvider(model=model, client=client)
        results.append(run_scenario(scenario, propose_fn=provider))
    return results


def _eval_result_to_json_dict(result: EvalResult) -> dict:
    """Convert one `EvalResult` into a plain, JSON-serializable dict.

    `events` holds pydantic `TraceEvent`s, which is why this isn't just
    `dataclasses.asdict()` — that recurses into dataclass fields but does
    not know how to serialize a nested pydantic model (or the `Decimal`/
    enum values inside one), so each event is explicitly `model_dump(mode=
    "json")`-ed instead.
    """
    return {
        "scenario_name": result.scenario_name,
        "passed": result.passed,
        "expected_run_status": result.expected_run_status.value,
        "actual_run_status": result.actual_run_status.value,
        "expected_rule_id": result.expected_rule_id,
        "actual_rule_id": result.actual_rule_id,
        "expected_execution_status": (
            result.expected_execution_status.value if result.expected_execution_status else None
        ),
        "actual_execution_status": (
            result.actual_execution_status.value if result.actual_execution_status else None
        ),
        "events": [event.model_dump(mode="json") for event in result.events],
    }


def persist_results(
    results: Sequence[EvalResult],
    output_dir: str | Path = "evals",
    model: str | None = None,
    timestamp: datetime | None = None,
) -> Path:
    """Write `results` to a timestamped JSON file under `output_dir`, and return its path.

    The filename uses a Windows-safe ISO-8601 "basic format" timestamp
    (`YYYYMMDDTHHMMSSZ`, no colons — a literal ISO-8601 timestamp like
    `2026-09-16T07:30:00Z` is not a valid Windows filename) so successive
    real-provider runs never collide and sort chronologically by name.
    Creates `output_dir` if it does not already exist. No database, no
    other persistence layer — this is the entire mechanism.
    """
    timestamp = timestamp or datetime.now(timezone.utc)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"run-{timestamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    output_path = output_dir / filename

    payload = {
        "timestamp": timestamp.isoformat(),
        "model": model,
        "results": [_eval_result_to_json_dict(result) for result in results],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


@dataclass
class RealProviderRunOutcome:
    """Outcome of attempting `run_default_scenarios_against_real_provider()`.

    `skipped=True` means no API call was ever attempted (e.g. no API key
    was configured) — `reason` explains why, and `results`/`output_path`
    are empty/`None`. `skipped=False` means the real run happened and was
    persisted; `results` and `output_path` are populated.
    """

    skipped: bool
    reason: str | None
    results: list[EvalResult]
    output_path: Path | None


def run_default_scenarios_against_real_provider(
    output_dir: str | Path = "evals",
    model: str = ANTHROPIC_DEFAULT_MODEL,
) -> RealProviderRunOutcome:
    """Run `build_default_scenarios()` against the real Anthropic API and persist the results.

    Reads `ANTHROPIC_API_KEY` from the environment BEFORE attempting
    anything: if it is not set, this returns immediately with
    `skipped=True` and a clear `reason` — it does not raise, and it does
    not attempt any network call or even import the `anthropic` SDK. If
    the key is set, this makes real Anthropic API calls (one per default
    scenario) that cost real API credits, then writes the results via
    `persist_results()` into `output_dir` (default: the repo's `evals/`
    directory) and returns them with `skipped=False`.

    This function makes real network calls when the API key is present.
    It is not called by any test in this codebase, and nothing in this
    codebase calls it automatically — running it against the live API is
    a deliberate, separate action for a human to take.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return RealProviderRunOutcome(
            skipped=True,
            reason="ANTHROPIC_API_KEY is not set in the environment; skipping the real-provider run.",
            results=[],
            output_path=None,
        )

    scenarios = build_default_scenarios()
    results = run_scenarios_with_real_provider(scenarios, model=model)
    output_path = persist_results(results, output_dir=output_dir, model=model)
    return RealProviderRunOutcome(skipped=False, reason=None, results=results, output_path=output_path)


def format_report(results: Sequence[EvalResult]) -> str:
    """Render a human-readable pass/fail report, with full traced events for any failing scenario."""
    lines: list[str] = []
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(
            f"[{status}] {result.scenario_name}: expected run_status={result.expected_run_status.value}, "
            f"got {result.actual_run_status.value}"
        )
        if result.expected_rule_id is not None:
            lines.append(f"         expected rule_id={result.expected_rule_id!r}, got {result.actual_rule_id!r}")
        if result.expected_execution_status is not None:
            lines.append(
                f"         expected execution_status={result.expected_execution_status.value}, "
                f"got {result.actual_execution_status.value if result.actual_execution_status else None}"
            )
        if not result.passed:
            lines.append("         traced events:")
            for event in result.events:
                lines.append(f"           {event.sequence}: {event.event_type.value}")
    return "\n".join(lines)


def _buggy_update_ticket_status(command, ticket: Ticket) -> Ticket:
    """A tool function stub that runs without error but leaves the ticket's status untouched.

    Used only by the verification-failure default scenario to force a
    genuine `ExecutionStatus.VERIFICATION_FAILED`, matching the pattern
    already used in `tests/test_execution_executor.py` and
    `tests/test_graph_build.py`.
    """
    return ticket.model_copy()


def _always_transiently_fail(attempt: int) -> bool:
    """A FailurePredicate that treats every attempt as a transient failure, exhausting retries.

    Used only by the execution-failure default scenario to force a
    genuine `ExecutionStatus.FAILURE` (retries exhausted), matching the
    pattern already used in `tests/test_execution_executor.py`.
    """
    return True


def build_default_scenarios() -> list[EvalScenario]:
    """Build the fixed set of default evaluation scenarios.

    A function rather than a module-level constant so each call gets
    fresh `Ticket`/`ProposedAction` objects (Pydantic models are mutable,
    and scenarios should not accidentally share state across runs).
    """

    def make_ticket(**overrides) -> Ticket:
        defaults = dict(
            id="T-1",
            status=TicketStatus.OPEN,
            customer_ref="cust-123",
            subject="Package arrived damaged",
            order_ref="ORD-456",
            refund_eligible_amount=Decimal("50.00"),
            currency="USD",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )
        defaults.update(overrides)
        return Ticket(**defaults)

    happy_path = EvalScenario(
        name="approved-and-verified-happy-path",
        ticket=make_ticket(refund_eligible_amount=Decimal("50.00")),
        task_description="Customer wants a refund for a damaged item.",
        proposals=[
            ProposedAction(
                tool_name="issue_refund",
                arguments={"ticket_id": "T-1", "amount": "20.00", "currency": "USD", "reason": "damaged item"},
            )
        ],
        expected_run_status=RunStatus.COMPLETED,
        expected_execution_status=ExecutionStatus.SUCCESS,
    )

    policy_denied = EvalScenario(
        name="policy-denied-over-limit-refund",
        ticket=make_ticket(refund_eligible_amount=Decimal("50.00")),
        task_description=(
            "The customer is requesting a $999 refund for this issue, well above what "
            "they're entitled to — process it."
        ),
        proposals=[
            ProposedAction(
                tool_name="issue_refund",
                arguments={"ticket_id": "T-1", "amount": "999.00", "currency": "USD", "reason": "damaged item"},
            )
        ],
        expected_run_status=RunStatus.DENIED,
        expected_rule_id="refund-amount-limit",
    )

    verification_failed = EvalScenario(
        name="verification-failure-on-status-update",
        ticket=make_ticket(status=TicketStatus.OPEN),
        task_description="Move this ticket to in-progress.",
        proposals=[
            ProposedAction(
                tool_name="update_ticket_status",
                arguments={"ticket_id": "T-1", "new_status": "in_progress", "reason": "starting work"},
            )
        ],
        expected_run_status=RunStatus.FAILED,
        expected_execution_status=ExecutionStatus.VERIFICATION_FAILED,
        tool_function_overrides={AuthorizedUpdateTicketStatusCommand: _buggy_update_ticket_status},
    )

    execution_failure_retries_exhausted = EvalScenario(
        name="execution-failure-retries-exhausted",
        ticket=make_ticket(),
        task_description="Add a note to this ticket.",
        proposals=[
            ProposedAction(
                tool_name="add_ticket_note", arguments={"ticket_id": "T-1", "note": "Attempting to log this."}
            )
        ],
        expected_run_status=RunStatus.FAILED,
        expected_execution_status=ExecutionStatus.FAILURE,
        should_fail=_always_transiently_fail,
    )

    return [happy_path, policy_denied, verification_failed, execution_failure_retries_exhausted]
