# Agentic AI Reliability & Tool Execution Engine

This project is a support-ticket ops agent built on a single thesis: the
LLM's output is never trusted directly. An LLM proposes actions to take
against support tickets, but a separate, deterministic layer is
responsible for authorizing, validating, executing, and verifying those
actions. The LLM proposal and the system-approved command are treated as
two different things, with the deterministic layer acting as the trust
boundary between them.

## Status

- Architecture scaffolded only.
- No business logic implemented yet.
- No tests passing yet.
- No evaluation run yet.
