"""API layer.

Exposes the agent over FastAPI: `POST /tasks` submits a support-ticket
task and runs it through the existing graph, `GET /runs/{run_id}`
retrieves a run's traced event history, and `GET /health` reports
service liveness. See `src.api.app` for the FastAPI application and
`src.api.models` for its request/response schemas.
"""

from src.api.app import app

__all__ = ["app"]
