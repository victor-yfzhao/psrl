"""
Terminate model retries when the gateway deliberately aborts a request.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import TypeVar

import httpx
from psrl.utils.common.http_utils import RequestAbortedByGatewayError

_T = TypeVar("_T")


def _is_gateway_abort(exc: Exception) -> bool:
    """
    Require HTTP 400 and an exact structured gateway error code.
    """
    if getattr(exc, "status_code", None) != 400:
        return False
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if isinstance(headers, Mapping):
        for key, value in headers.items():
            if isinstance(key, str) and key.lower() == "x-smg-error-code":
                return value == "request_aborted"
    body = getattr(exc, "body", None)
    if not isinstance(body, Mapping) and isinstance(response, httpx.Response):
        try:
            body = response.json()
        except (ValueError, httpx.ResponseNotRead):
            return False
    if not isinstance(body, Mapping):
        return False
    error = body.get("error", body)
    return isinstance(error, Mapping) and error.get("code") == "request_aborted"


def ensure_gateway_abort_handling(model: _T) -> _T:
    """
    Translate structured gateway aborts before the model client's retry loop.
    """
    if RequestAbortedByGatewayError in model.abort_exceptions:
        return model
    original_query = model._query

    @functools.wraps(original_query)
    def query(*args, **kwargs):
        try:
            return original_query(*args, **kwargs)
        except RequestAbortedByGatewayError:
            raise
        except Exception as exc:
            if _is_gateway_abort(exc):
                raise RequestAbortedByGatewayError(request_id="", message=str(exc)) from exc
            raise

    model._query = query
    model.abort_exceptions = [*model.abort_exceptions, RequestAbortedByGatewayError]
    return model
