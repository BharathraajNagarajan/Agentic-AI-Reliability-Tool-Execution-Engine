"""Top-level package for the Agentic AI Reliability & Tool Execution Engine.

Architecture principle: an LLM proposes actions; a deterministic layer
(policy, execution, state, observability) authorizes, validates, executes,
and verifies them. The LLM's output is never trusted directly — it is only
a proposal that the deterministic layer decides whether to act on.
"""
