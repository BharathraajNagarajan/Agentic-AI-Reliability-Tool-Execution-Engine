"""ProposeFn implementations: things that turn an AgentState into a ProposedAction.

Both providers here conform to `src.graph.nodes.ProposeFn`
(`Callable[[AgentState], ProposedAction]`) and are meant to be handed to
`make_propose_node()` / `build_agent_graph()` — this package does not
change that signature or anything in `src.graph.nodes`/`src.graph.build`.

- `mock.ScriptedProposeProvider`: a fixed sequence of proposals for tests.
  Pure Python, no external SDK, no network — always importable and usable
  with zero API key and zero network access.
- `anthropic_provider.AnthropicProposeProvider`: calls the real Anthropic
  API. The `anthropic` SDK is an optional runtime dependency (see
  pyproject.toml) but is imported lazily inside this provider, not at
  module import time — importing this package, or even constructing an
  `AnthropicProposeProvider`, never requires `anthropic` to be installed;
  only actually calling it without an injected fake client does.
"""

from src.graph.providers.anthropic_provider import AnthropicProposeProvider
from src.graph.providers.mock import ScriptedProposeProvider

__all__ = [
    "AnthropicProposeProvider",
    "ScriptedProposeProvider",
]
