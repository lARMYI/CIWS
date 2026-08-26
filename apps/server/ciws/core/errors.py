"""Typed errors that map cleanly onto HTTP responses and UI messages."""

from __future__ import annotations

from typing import Any


class CIWSError(Exception):
    """Base for everything CIWS raises deliberately."""

    status_code = 500
    code = "ciws_error"

    def __init__(self, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        return {"error": self.code, "message": self.message, "detail": self.detail}


class ConfigError(CIWSError):
    status_code = 400
    code = "config_error"


class NotFound(CIWSError):
    status_code = 404
    code = "not_found"


class ValidationFailed(CIWSError):
    status_code = 422
    code = "validation_failed"


class Unauthorized(CIWSError):
    status_code = 401
    code = "unauthorized"


class Forbidden(CIWSError):
    status_code = 403
    code = "forbidden"


class ProviderError(CIWSError):
    """A model or media provider refused, failed, or timed out."""

    status_code = 502
    code = "provider_error"

    def __init__(self, provider: str, message: str, *, retryable: bool = False, **detail: Any) -> None:
        super().__init__(message, provider=provider, retryable=retryable, **detail)
        self.provider = provider
        self.retryable = retryable


class MissingCredential(ProviderError):
    status_code = 400
    code = "missing_credential"

    def __init__(self, provider: str, env_hint: str = "") -> None:
        msg = f"No API key configured for '{provider}'."
        if env_hint:
            msg += f" Set {env_hint} or add it in Settings -> Credentials."
        super().__init__(provider, msg, retryable=False)


class RateLimited(ProviderError):
    status_code = 429
    code = "rate_limited"

    def __init__(self, provider: str, message: str = "Rate limited", retry_after: float | None = None) -> None:
        super().__init__(provider, message, retryable=True, retry_after=retry_after)
        self.retry_after = retry_after


class ToolError(CIWSError):
    status_code = 400
    code = "tool_error"

    def __init__(self, tool: str, message: str, **detail: Any) -> None:
        super().__init__(message, tool=tool, **detail)
        self.tool = tool


class ApprovalRequired(CIWSError):
    """A tool wanted to do something the security policy gates behind a human."""

    status_code = 403
    code = "approval_required"


class Cancelled(CIWSError):
    status_code = 499
    code = "cancelled"
