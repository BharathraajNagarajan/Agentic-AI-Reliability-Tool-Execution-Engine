"""A scripted-sequence ProposeFn for tests.

No external SDK, no network — safe to import and call with zero API key
and zero network access. Replaces the ad-hoc inline lambdas tests
previously wrote for `propose_node`.
"""

from __future__ import annotations

from typing import Sequence

from src.graph.state import AgentState
from src.tools.schemas import ProposedAction


class ScriptedProposeProvider:
    """A `ProposeFn` that returns proposals from a fixed script, one per call, in order.

    Construct once with the exact sequence of `ProposedAction`s a test
    wants the "LLM" to produce, then pass the instance wherever a
    `ProposeFn` is expected, e.g.
    `build_agent_graph(ScriptedProposeProvider([...]))`.

    Calling it more times than there are scripted proposals raises
    `IndexError` — a test asking for more turns than it scripted is a test
    bug and should fail loudly, not silently repeat or return `None`.
    """

    def __init__(self, proposals: Sequence[ProposedAction]) -> None:
        self._proposals = list(proposals)
        self._next_index = 0

    def __call__(self, state: AgentState) -> ProposedAction:
        if self._next_index >= len(self._proposals):
            raise IndexError(
                f"ScriptedProposeProvider called {self._next_index + 1} time(s) but only "
                f"{len(self._proposals)} proposal(s) were scripted."
            )
        proposal = self._proposals[self._next_index]
        self._next_index += 1
        return proposal

    @property
    def calls_made(self) -> int:
        """How many scripted proposals have been consumed so far, for test assertions."""
        return self._next_index
