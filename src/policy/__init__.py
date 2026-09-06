"""Policy layer.

Intended responsibility: the deterministic authorization layer that
decides whether a proposed action is allowed to run, independent of and
downstream from the LLM. This is the trust boundary between LLM proposal
and system-approved command. No policy rules or authorization logic exist
yet.
"""
