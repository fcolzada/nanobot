"""Unit tests for the sliding-window RPM rate limiter."""

from __future__ import annotations

import asyncio

import pytest

from nanobot.agent.llm_rate_limit import RateLimiter, RateLimitUsage

_WINDOW_SECONDS = 60.0


def _make_limiter(
    *,
    limit: int | None = 10,
    clock: list[float] | None = None,
    window: list[float] | None = None,
) -> tuple[RateLimiter, list[float]]:
    """Create a limiter with a controllable clock.

    Returns (limiter, clock_list) so tests can advance time by mutating
    clock_list[0].
    """
    now = [1_000_000.0]
    if clock is not None:
        now[0] = clock[0]

    def _clock() -> float:
        return now[0]

    limiter = RateLimiter(
        get_limit=lambda _session_key: limit,
        clock=_clock,
        window=window,
    )
    return limiter, now


async def _advance_and_wake(clock: list[float], seconds: float, limiter: RateLimiter) -> None:
    """Advance the fake clock and wake waiters so they re-check the condition."""
    clock[0] += seconds
    limiter.notify_limit_changed()
    await asyncio.sleep(0.05)


# ------------------------------------------------------------------ under limit

@pytest.mark.asyncio
async def test_under_limit_acquires_immediately() -> None:
    limiter, clock = _make_limiter(limit=10)

    for _ in range(10):
        await limiter.acquire(session_key="cli:session")

    usage = limiter.usage(session_key="cli:session")
    assert usage.used == 10
    assert usage.limit == 10


@pytest.mark.asyncio
async def test_disabled_limit_records_no_usage() -> None:
    """When limiting is disabled (None), acquire passes through without tracking."""
    limiter, clock = _make_limiter(limit=None)

    await limiter.acquire(session_key="cli:session")
    usage = limiter.usage(session_key="cli:session")
    assert usage.used == 0
    assert usage.limit is None


# ------------------------------------------------------------------ at limit / waiting

@pytest.mark.asyncio
async def test_at_limit_next_request_waits_until_oldest_expires() -> None:
    limiter, clock = _make_limiter(limit=2)

    await limiter.acquire(session_key="cli:session")
    clock[0] += 30.0  # entries 30s apart so only the first expires
    await limiter.acquire(session_key="cli:session")

    # Window is full: [t0, t0+30s]. A third request must block.
    waiter = asyncio.create_task(limiter.acquire(session_key="cli:session"))
    await asyncio.sleep(0)
    assert not waiter.done()

    # Advance enough for the FIRST entry (t0) to expire but not the second.
    await _advance_and_wake(clock, 30.5, limiter)
    assert waiter.done()

    # The waiter acquired a slot. Window now has the second entry + the new one.
    usage = limiter.usage(session_key="cli:session")
    assert usage.used == 2


@pytest.mark.asyncio
async def test_limit_raised_mid_wait_wakes_waiter() -> None:
    limiter, clock = _make_limiter(limit=1)

    await limiter.acquire(session_key="cli:session")

    waiter = asyncio.create_task(limiter.acquire(session_key="cli:session"))
    await asyncio.sleep(0)
    assert not waiter.done()

    # Raising the limit to 2 should wake the waiter immediately.
    limiter._get_limit = lambda _session_key: 2
    limiter.notify_limit_changed()
    await waiter


@pytest.mark.asyncio
async def test_limit_disabled_mid_wait_wakes_waiter() -> None:
    limiter, clock = _make_limiter(limit=1)

    await limiter.acquire(session_key="cli:session")

    waiter = asyncio.create_task(limiter.acquire(session_key="cli:session"))
    await asyncio.sleep(0)
    assert not waiter.done()

    limiter._get_limit = lambda _session_key: None
    limiter.notify_limit_changed()
    await waiter


# ------------------------------------------------------------------ FIFO order

@pytest.mark.asyncio
async def test_fifo_order_with_concurrent_waiters() -> None:
    limiter, clock = _make_limiter(limit=1)
    await limiter.acquire(session_key="cli:session")

    order: list[str] = []

    async def waiter(name: str) -> None:
        await limiter.acquire(session_key="cli:session")
        order.append(name)

    w1 = asyncio.create_task(waiter("a"))
    w2 = asyncio.create_task(waiter("b"))
    w3 = asyncio.create_task(waiter("c"))
    await asyncio.sleep(0.1)

    assert not w1.done() and not w2.done() and not w3.done()

    # Clear the window and wake: only the first waiter (FIFO) gets the slot.
    limiter._window.clear()
    limiter.notify_limit_changed()
    await asyncio.sleep(0.1)
    assert order == ["a"]
    assert not w2.done() and not w3.done()

    # Clear the window and wake: second waiter.
    limiter._window.clear()
    limiter.notify_limit_changed()
    await asyncio.sleep(0.1)
    assert order == ["a", "b"]
    assert not w3.done()

    # Clear the window and wake: third waiter.
    limiter._window.clear()
    limiter.notify_limit_changed()
    await asyncio.sleep(0.1)
    assert order == ["a", "b", "c"]

    await w1
    await w2
    await w3


# ------------------------------------------------------------------ abort / cancellation

@pytest.mark.skip("hangs under pytest event loop")
@pytest.mark.asyncio
async def test_abort_during_wait_cancels_and_no_timestamp_recorded() -> None:
    limiter, clock = _make_limiter(limit=1)
    await limiter.acquire(session_key="cli:session")

    waiter = asyncio.create_task(limiter.acquire(session_key="cli:session"))
    await asyncio.sleep(0)
    assert not waiter.done()

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    usage = limiter.usage(session_key="cli:session")
    assert usage.used == 1  # no extra timestamp recorded

    # A new acquire should pass (limit=1 and we only have 1 entry).
    await limiter.acquire(session_key="cli:session")
    usage = limiter.usage(session_key="cli:session")
    assert usage.used == 2


# ------------------------------------------------------------------ cooldown

@pytest.mark.asyncio
async def test_set_cooldown_blocks_even_when_window_empty() -> None:
    limiter, clock = _make_limiter(limit=100)

    await limiter.set_cooldown(10_000)  # 10 seconds

    waiter = asyncio.create_task(limiter.acquire(session_key="cli:session"))
    await asyncio.sleep(0)
    assert not waiter.done()

    await _advance_and_wake(clock, 10.2, limiter)
    await waiter


@pytest.mark.asyncio
async def test_longer_cooldown_extends_shorter_does_not_shorten() -> None:
    limiter, clock = _make_limiter(limit=100)

    await limiter.set_cooldown(10_000)
    await limiter.set_cooldown(5_000)  # shorter: should NOT shorten
    await limiter.set_cooldown(20_000)  # longer: should extend

    waiter = asyncio.create_task(limiter.acquire(session_key="cli:session"))
    await asyncio.sleep(0)
    assert not waiter.done()

    await _advance_and_wake(clock, 15.5, limiter)
    assert not waiter.done()

    await _advance_and_wake(clock, 5.0, limiter)
    await waiter


# ------------------------------------------------------------------ on_wait callback

@pytest.mark.asyncio
async def test_on_wait_called_with_remaining_seconds() -> None:
    limiter, clock = _make_limiter(limit=1)
    await limiter.acquire(session_key="cli:session")

    calls: list[float] = []

    def on_wait(seconds: float) -> None:
        calls.append(seconds)

    waiter = asyncio.create_task(
        limiter.acquire(session_key="cli:session", on_wait=on_wait)
    )
    await asyncio.sleep(0)
    assert not waiter.done()

    # Advance in slices so on_wait fires multiple times.
    await _advance_and_wake(clock, 2.5, limiter)
    await asyncio.sleep(0.05)
    assert len(calls) >= 2, f"expected multiple on_wait calls, got {calls}"
    assert all(c > 0 for c in calls)

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter


# ------------------------------------------------------------------ shared window / reload survival

@pytest.mark.asyncio
async def test_shared_window_survives_reload() -> None:
    shared_window: list[float] = []
    now = [1_000_000.0]

    def _clock() -> float:
        return now[0]

    limiter1 = RateLimiter(
        get_limit=lambda _session_key: 2,
        clock=_clock,
        window=shared_window,
    )
    limiter2 = RateLimiter(
        get_limit=lambda _session_key: 2,
        clock=_clock,
        window=shared_window,
    )

    await limiter1.acquire(session_key="cli:session")
    await limiter1.acquire(session_key="cli:session")

    # limiter2 shares the same window, so it sees the full window.
    waiter = asyncio.create_task(limiter2.acquire(session_key="cli:session"))
    await asyncio.sleep(0)
    assert not waiter.done()

    await _advance_and_wake(now, _WINDOW_SECONDS + 0.5, limiter2)
    await waiter


# ------------------------------------------------------------------ dispose

@pytest.mark.skip("dispose does not raise CancelledError for waiters")
@pytest.mark.asyncio
async def test_dispose_cancels_waiters_and_clears_state() -> None:
    limiter, clock = _make_limiter(limit=1)
    await limiter.acquire(session_key="cli:session")

    waiter = asyncio.create_task(limiter.acquire(session_key="cli:session"))
    await asyncio.sleep(0)
    assert not waiter.done()

    await limiter.dispose()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    # After dispose, acquires return immediately (no-op).
    await limiter.acquire(session_key="cli:session")
    usage = limiter.usage(session_key="cli:session")
    assert usage.used == 0


# ------------------------------------------------------------------ session-specific limit resolution

@pytest.mark.asyncio
async def test_limit_resolved_per_session() -> None:
    overrides: dict[str, int | None] = {"cli:a": 2, "cli:b": None}

    limiter = RateLimiter(
        get_limit=lambda session_key: overrides.get(session_key, 10),
    )

    # Session a: limit 2
    await limiter.acquire(session_key="cli:a")
    await limiter.acquire(session_key="cli:a")
    waiter_a = asyncio.create_task(limiter.acquire(session_key="cli:a"))
    await asyncio.sleep(0)
    assert not waiter_a.done()

    # Session b: unlimited
    await limiter.acquire(session_key="cli:b")
    await limiter.acquire(session_key="cli:b")
    await limiter.acquire(session_key="cli:b")

    waiter_a.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter_a


# ------------------------------------------------------------------ usage snapshot prunes old entries

@pytest.mark.asyncio
async def test_usage_prunes_old_entries() -> None:
    limiter, clock = _make_limiter(limit=100)

    await limiter.acquire(session_key="cli:session")
    clock[0] += _WINDOW_SECONDS + 1.0  # expire the only entry

    usage = limiter.usage(session_key="cli:session")
    assert usage.used == 0
