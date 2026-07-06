"""
Retry, Backoff, and Error Classification
==========================================
Production-grade retry infrastructure:
  - Exponential backoff with jitter
  - Error classification (auth/timeout/network/transient/permanent)
  - Retry decorator with configurable strategy
  - HTTP error classification for API calls
"""

from __future__ import annotations

import functools
import logging
import random
import socket
import time
import urllib.error
from enum import Enum
from typing import (
    Any,
    Callable,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


# 鈹€鈹€ Error Classification 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

class ErrorKind(Enum):
    """Classification of errors for retry decisions.

    Migrated from: upstream/src/services/api/errorUtils.ts :: AxiosErrorKind
    """
    AUTH = 'auth'           # 401, 403 鈥?do NOT retry
    TIMEOUT = 'timeout'     # Socket timeout, read timeout 鈥?RETRY
    NETWORK = 'network'     # DNS, connection refused 鈥?RETRY with backoff
    RATE_LIMIT = 'rate_limit'  # 429 鈥?RETRY with longer backoff
    SERVER = 'server'       # 500, 502, 503, 504 鈥?RETRY with backoff
    PERMANENT = 'permanent'  # 400, 404, 405 鈥?do NOT retry
    OTHER = 'other'         # Unknown 鈥?do NOT retry by default

    @property
    def retryable(self) -> bool:
        return self in (
            ErrorKind.TIMEOUT,
            ErrorKind.NETWORK,
            ErrorKind.RATE_LIMIT,
            ErrorKind.SERVER,
        )


# HTTP status codes that are retryable
_RETRYABLE_STATUS_CODES: Set[int] = {408, 429, 500, 502, 503, 504, 522, 524}

# Exception types that indicate network/transient issues
_NETWORK_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    ConnectionError,
    ConnectionRefusedError,
    ConnectionResetError,
    ConnectionAbortedError,
    socket.timeout,
    socket.gaierror,
    TimeoutError,
    BrokenPipeError,
    OSError,  # includes various errno-based errors
)


def classify_error(error: Exception) -> ErrorKind:
    """
    Classify an exception into an ErrorKind for retry decisions.

    Handles urllib.error.HTTPError, socket errors, and generic exceptions.

    Migrated from: upstream/src/services/api/errorUtils.ts :: classifyAxiosError
    """
    # HTTP errors
    if isinstance(error, urllib.error.HTTPError):
        code = error.code
        if code in (401, 403):
            return ErrorKind.AUTH
        if code == 429:
            return ErrorKind.RATE_LIMIT
        if code in _RETRYABLE_STATUS_CODES:
            return ErrorKind.SERVER
        return ErrorKind.PERMANENT

    # URL errors (wraps socket errors)
    if isinstance(error, urllib.error.URLError):
        reason = getattr(error, 'reason', None)
        if isinstance(reason, socket.timeout):
            return ErrorKind.TIMEOUT
        if isinstance(reason, (socket.gaierror, ConnectionRefusedError)):
            return ErrorKind.NETWORK
        if isinstance(reason, OSError):
            return ErrorKind.NETWORK
        return ErrorKind.NETWORK

    # Direct socket/connection errors
    if isinstance(error, (socket.timeout, TimeoutError)):
        return ErrorKind.TIMEOUT
    if isinstance(error, _NETWORK_EXCEPTIONS):
        return ErrorKind.NETWORK

    return ErrorKind.OTHER


def is_retryable(error: Exception) -> bool:
    """Check if an error is worth retrying."""
    return classify_error(error).retryable


# 鈹€鈹€ Backoff Calculation 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def exponential_backoff(
    attempt: int,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    jitter: bool = True,
) -> float:
    """
    Calculate exponential backoff delay with optional jitter.

    Args:
        attempt: Zero-based attempt number (0 = first retry)
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap
        jitter: Add random jitter to prevent thundering herd

    Returns:
        Delay in seconds
    """
    delay = min(base_delay * (2 ** attempt), max_delay)
    if jitter:
        delay = delay * (0.5 + random.random() * 0.5)
    return delay


def rate_limit_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> float:
    """
    Longer backoff for rate-limited requests (429).

    Uses more aggressive backing off than standard exponential.
    """
    return exponential_backoff(
        attempt, base_delay=base_delay, max_delay=max_delay, jitter=True)


# 鈹€鈹€ Retry Decorator 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def retry(
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    retryable_exceptions: Optional[Sequence[Type[Exception]]] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    classify: bool = True,
) -> Callable:
    """
    Decorator for retrying functions with exponential backoff.

    Supports both error-classification-based and exception-type-based retry.

    Args:
        max_attempts: Maximum number of attempts (including first try)
        base_delay: Base delay between retries
        max_delay: Maximum delay cap
        retryable_exceptions: If set, only retry these exception types
                              (overrides classify_error logic)
        on_retry: Optional callback(error, attempt) called before each retry
        classify: If True, use classify_error() to decide retry eligibility

    Usage:
        @retry(max_attempts=3, base_delay=1.0)
        def fetch_data(url):
            return urllib.request.urlopen(url).read()

        @retry(retryable_exceptions=[ConnectionError, TimeoutError])
        def connect():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_error: Optional[Exception] = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    # Check if we should retry
                    if attempt >= max_attempts - 1:
                        break  # Last attempt, don't retry

                    should_retry = False
                    if retryable_exceptions:
                        should_retry = isinstance(e, tuple(retryable_exceptions))
                    elif classify:
                        should_retry = is_retryable(e)
                    else:
                        should_retry = True

                    if not should_retry:
                        break

                    # Calculate delay
                    kind = classify_error(e)
                    if kind == ErrorKind.RATE_LIMIT:
                        delay = rate_limit_backoff(attempt)
                    else:
                        delay = exponential_backoff(
                            attempt, base_delay=base_delay, max_delay=max_delay)

                    # Callback
                    if on_retry:
                        on_retry(e, attempt)
                    else:
                        logger.warning(
                            "%s attempt %d/%d failed (%s: %s), "
                            "retrying in %.1fs",
                            func.__name__, attempt + 1, max_attempts,
                            kind.value, str(e)[:200], delay)

                    time.sleep(delay)

            raise last_error  # type: ignore[misc]

        return wrapper
    return decorator


# 鈹€鈹€ Standalone Retry Helper 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def retry_call(
    func: Callable[..., T],
    args: tuple = (),
    kwargs: Optional[dict] = None,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    retryable_exceptions: Optional[Sequence[Type[Exception]]] = None,
) -> T:
    """
    Call a function with retry logic (non-decorator form).

    Usage:
        data = retry_call(fetch_url, args=(url,), max_attempts=3)
    """
    kwargs = kwargs or {}

    @retry(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        retryable_exceptions=retryable_exceptions,
    )
    def _inner():
        return func(*args, **kwargs)

    return _inner()

