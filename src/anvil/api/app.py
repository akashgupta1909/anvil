"""FastAPI application: run the agent (JSON or SSE), manage sessions, health check."""

import hmac
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sse_starlette import EventSourceResponse, ServerSentEvent

from anvil import __version__
from anvil.api.schemas import (
    ErrorBody,
    ErrorResponse,
    HealthResponse,
    RunRequest,
    SessionResponse,
)
from anvil.config import API_KEY_ENV
from anvil.container import Container
from anvil.llm.client import FallbackLLM
from anvil.memory.sessions import SessionNotFoundError
from anvil.models import AgentEvent, AgentResult, RunStatus
from anvil.service import SessionBusyError
from anvil.tools.approval import AllowAll, DenyAll
from anvil.tools.base import Approver
from anvil.utils.logging import request_id_var

logger = logging.getLogger(__name__)

_ERRORS: dict[type[Exception], tuple[int, str]] = {
    SessionNotFoundError: (status.HTTP_404_NOT_FOUND, "session_not_found"),
    SessionBusyError: (status.HTTP_409_CONFLICT, "session_busy"),
}


def create_app(container: Container) -> FastAPI:
    api_key = os.environ.get(API_KEY_ENV)
    if container.settings.api.require_auth and not api_key:
        raise RuntimeError(f"api.require_auth is on but {API_KEY_ENV} is not set")

    app = FastAPI(title="anvil", version=__version__)
    service = container.service
    bearer = HTTPBearer(auto_error=False)

    def authorize(
        creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> None:
        if not container.settings.api.require_auth:
            return
        if creds is None or not hmac.compare_digest(creds.credentials, api_key or ""):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing API key")

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("unhandled error")
            response = error(
                status.HTTP_500_INTERNAL_SERVER_ERROR, "internal", "internal server error"
            )
        finally:
            request_id_var.reset(token)
        response.headers["x-request-id"] = request_id
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - start) * 1000, 1),
            },
        )
        return response

    def error(status_code: int, code: str, message: str) -> JSONResponse:
        body = ErrorResponse(
            error=ErrorBody(code=code, message=message, request_id=request_id_var.get())
        )
        return JSONResponse(body.model_dump(), status_code=status_code)

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        codes = {401: "unauthorized", 404: "not_found", 405: "method_not_allowed"}
        return error(exc.status_code, codes.get(exc.status_code, "http_error"), str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return error(status.HTTP_422_UNPROCESSABLE_CONTENT, "invalid_request", problems)

    for exc_type, (status_code, code) in _ERRORS.items():

        async def domain_error(
            _: Request, exc: Exception, status_code: int = status_code, code: str = code
        ) -> JSONResponse:
            return error(status_code, code, f"{code.replace('_', ' ')}: {exc}")

        app.add_exception_handler(exc_type, domain_error)

    def approver_for(body: RunRequest) -> Approver:
        return AllowAll() if body.allow_writes else DenyAll()

    @app.get("/health")
    async def health() -> HealthResponse:
        endpoints = {
            role: [f"{e.provider}/{e.model}" for e in llm.endpoints]
            for role, llm in container.llms.items()
            if isinstance(llm, FallbackLLM)
        }
        return HealthResponse(status="ok", version=__version__, llm_endpoints=endpoints)

    protected = [Depends(authorize)]

    @app.post("/v1/sessions", status_code=status.HTTP_201_CREATED, dependencies=protected)
    async def create_session() -> SessionResponse:
        return SessionResponse(session_id=service.sessions.create().id)

    @app.delete(
        "/v1/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=protected
    )
    async def delete_session(session_id: str) -> None:
        service.sessions.delete(session_id)

    @app.post(
        "/v1/runs",
        responses={503: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
        dependencies=protected,
    )
    async def run(body: RunRequest) -> AgentResult:
        result = await service.run(
            body.task,
            approver_for(body),
            session_id=body.session_id,
            request_id=request_id_var.get(),
        )
        if result.status is RunStatus.ERROR:
            raise _LLMUnavailable(result.error or "LLM unavailable")
        return result

    @app.exception_handler(_LLMUnavailable)
    async def llm_unavailable(_: Request, exc: _LLMUnavailable) -> JSONResponse:
        logger.error("llm unavailable", extra={"err": str(exc)})
        return error(status.HTTP_503_SERVICE_UNAVAILABLE, "llm_unavailable", "LLM unavailable")

    @app.post("/v1/runs/stream", response_class=EventSourceResponse, dependencies=protected)
    async def run_stream(body: RunRequest) -> EventSourceResponse:
        """Server-sent events: ``plan``, ``text_delta``, ``tool_call``, ``tool_result``
        and finally one ``final`` event carrying the full result."""
        # run_stream validates the session now, so 404/409 are HTTP errors, not SSE events.
        events = service.run_stream(
            body.task,
            approver_for(body),
            session_id=body.session_id,
            request_id=request_id_var.get(),
        )

        async def sse() -> AsyncIterator[ServerSentEvent]:
            async for event in events:
                yield _to_sse(event)

        return EventSourceResponse(sse(), ping=15)

    return app


class _LLMUnavailable(RuntimeError):
    pass


def _to_sse(event: AgentEvent) -> ServerSentEvent:
    return ServerSentEvent(data=event.model_dump_json(), event=event.type)
