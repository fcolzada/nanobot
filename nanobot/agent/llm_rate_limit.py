"""Sliding-window requests-per-minute (RPM) limiter for LLM provider calls.

A single limiter instance is owned by the :class:`~nanobot.agent.loop.AgentLoop`
and shared across all sessions.  Before each provider request the runner calls
:meth:`acquire`, which blocks until a slot is available in the sliding
60-second window.  The limit value is resolved per request through a
callable so per-session overrides (``/rpm``) take effect immediately.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable

_WINDOW_SECONDS = 60.0

# Metadata key used to persist a per-session RPM override in session storage.
RPM_LIMIT_META_KEY = "_rpm_limit"


@dataclass(slots=True)
class RateLimitUsage:
    """Snapshot of the current window usage."""

    used: int
    """Number of requests recorded in the last 60 seconds."""

    limit: int | None
    """Configured limit, or ``None`` when limiting is disabled."""


class RateLimiter:
    """Sliding-window rate limiter with FIFO serialization and abort support.

    All callers pass through an :class:`asyncio.Condition` so concurrent
    requests are served in arrival order.  When the window is full, a caller
    waits until the oldest tracked timestamp ages out or the limit changes.

    The effective limit is resolved by calling ``get_limit(session_key)`` on
    every iteration of the wait loop, so a per-session override changed via
    ``/rpm`` takes effect immediately, including for requests already waiting.
    """

    def __init__(
        self,
        *,
        get_limit: Callable[[str | None], int | None],
        clock: Callable[[], float] | None = None,
        window: list[float] | None = None,
    ) -> None:
        self._get_limit = get_limit
        self._clock = clock or time.monotonic
        # Shared timestamp array; survives reload when passed in from
        # global/persistent state.
        self._window: list[float] = window if window is not None else []
        self._cond = asyncio.Condition(asyncio.Lock())
        self._cooldown_until: float = 0.0
        self._disposed = False

    # ------------------------------------------------------------------ config

    async def set_cooldown(self, ms: float) -> None:
        """Block acquires until ``clock() + ms`` (provider 429 handling).

        A longer cooldown extends an existing one; a shorter one does not
        shorten it.
        """
        async with self._cond:
            until = self._clock() + ms / 1000.0
            if until > self._cooldown_until:
                self._cooldown_until = until
            self._cond.notify_all()

    def notify_limit_changed(self) -> None:
        """Wake all waiters so they re-evaluate the limit (called from /rpm).

        This is intentionally synchronous and non-async.  Waiters that are
        mid-``asyncio.wait_for`` will wake on their next timeout and re-check
        the limit.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        async def _wake() -> None:
            async with self._cond:
                self._cond.notify_all()

        # Schedule the wake on the event loop; it will acquire the condition
        # lock and notify all waiters as soon as it is available.
        asyncio.ensure_future(_wake())

    # ------------------------------------------------------------------ acquire

    async def acquire(
        self,
        session_key: str | None,
        on_wait: Callable[[float], None] | None = None,
    ) -> None:
        """Wait until a slot is available, then record the timestamp.

        *on_wait* is called with the estimated seconds until the next slot
        becomes available, whenever the limiter is about to block.  It is
        invoked once per wait iteration (not continuously); callers that need
        periodic UI updates should re-arm their own timer from the callback.

        Raises :exc:`asyncio.CancelledError` if the waiting task is
        cancelled (e.g. user pressed Esc).
        """
        if self._disposed:
            return

        async with self._cond:
            while not self._disposed:
                limit = self._get_limit(session_key)
                if limit is None:
                    return

                now = self._clock()
                self._prune(now)

                # Respect a provider-signalled cooldown first.
                if now < self._cooldown_until:
                    wait_seconds = max(0.0, self._cooldown_until - now)
                    # Wait in 1s slices so on_wait can refresh the UI countdown.
                    slice_for = min(wait_seconds, 1.0) if wait_seconds > 0 else 0.0
                    if on_wait is not None:
                        on_wait(wait_seconds)
                    try:
                        await asyncio.wait_for(
                            self._cond.wait(), timeout=slice_for
                        )
                    except asyncio.TimeoutError:
                        pass
                    continue

                if len(self._window) < limit:
                    self._window.append(now)
                    return

                # Window is full: wait for the oldest tracked entry to age out.
                wait_until = self._window[len(self._window) - limit] + _WINDOW_SECONDS
                wait_seconds = max(0.0, wait_until - now)
                # Wait in 1s slices so on_wait can refresh the UI countdown.
                slice_for = min(wait_seconds, 1.0) if wait_seconds > 0 else 0.0
                if on_wait is not None:
                    on_wait(wait_seconds)
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=slice_for)
                except asyncio.TimeoutError:
                    pass
                # Loop re-reads the limit (it may have changed) and re-checks.

    # ------------------------------------------------------------------ queries

    def _prune(self, now: float) -> None:
        """Remove timestamps older than the 60-second window (must hold lock)."""
        cutoff = now - _WINDOW_SECONDS
        while self._window and self._window[0] <= cutoff:
            self._window.pop(0)

    def usage(self, session_key: str | None) -> RateLimitUsage:
        """Return current window usage (approximate; no lock held)."""
        now = self._clock()
        cutoff = now - _WINDOW_SECONDS
        used = sum(1 for ts in self._window if ts > cutoff)
        return RateLimitUsage(used=used, limit=self._get_limit(session_key))

    # ------------------------------------------------------------------ lifecycle

    async def dispose(self) -> None:
        """Cancel all waiters and clear shared state."""
        async with self._cond:
            self._disposed = True
            self._window.clear()
            self._cooldown_until = 0.0
            self._cond.notify_all()
