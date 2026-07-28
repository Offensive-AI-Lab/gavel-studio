"""Unit tests for app.utils.rate_limit — the in-process sliding-window limiter,
the per-account login throttle, and the X-Forwarded-For client-IP resolution.

These guards are the central server's only protection against login floods,
credential stuffing, and argon2-hash CPU exhaustion, so the boundaries (exactly
at the limit, the window sliding, a forged XFF header) are tested directly.

The limiter uses time.monotonic() internally; we monkeypatch it with a
controllable clock so the window behaviour is deterministic instead of relying
on wall-clock sleeps.
"""
import types

import pytest
from fastapi import HTTPException

from app.utils import rate_limit as rl


# ---------------------------------------------------------------------------
# A controllable monotonic clock
# ---------------------------------------------------------------------------


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(rl.time, "monotonic", c)
    return c


@pytest.fixture(autouse=True)
def fresh_limiter(monkeypatch):
    """Reset the module singleton so tests don't bleed into each other."""
    monkeypatch.setattr(rl, "_limiter", rl._SlidingWindow())


# ---------------------------------------------------------------------------
# _SlidingWindow core
# ---------------------------------------------------------------------------


class TestSlidingWindow:
    def test_allows_up_to_the_limit(self, clock):
        w = rl._SlidingWindow()
        for _ in range(3):
            allowed, retry = w.hit("k", limit=3, window=60)
            assert allowed is True
            assert retry == 0

    def test_blocks_the_request_over_the_limit(self, clock):
        w = rl._SlidingWindow()
        for _ in range(3):
            w.hit("k", limit=3, window=60)
        allowed, retry = w.hit("k", limit=3, window=60)
        assert allowed is False
        assert retry >= 1  # a Retry-After hint is always at least one second

    def test_separate_keys_are_independent(self, clock):
        w = rl._SlidingWindow()
        w.hit("a", limit=1, window=60)
        # 'a' is now full, but 'b' is untouched.
        allowed, _ = w.hit("b", limit=1, window=60)
        assert allowed is True

    def test_window_slides_so_old_hits_expire(self, clock):
        w = rl._SlidingWindow()
        w.hit("k", limit=1, window=10)
        # Immediately blocked within the window.
        assert w.hit("k", limit=1, window=10)[0] is False
        # Move past the window; the old hit drops out and we're allowed again.
        clock.advance(11)
        assert w.hit("k", limit=1, window=10)[0] is True

    def test_count_reflects_only_in_window_hits(self, clock):
        w = rl._SlidingWindow()
        w.add("k")
        w.add("k")
        assert w.count("k", window=10) == 2
        clock.advance(11)
        assert w.count("k", window=10) == 0

    def test_count_of_unknown_key_is_zero(self, clock):
        w = rl._SlidingWindow()
        assert w.count("never-seen", window=10) == 0


# ---------------------------------------------------------------------------
# rate_limit() FastAPI dependency
#
# The per-account login throttle that used to be tested here is gone along with
# login itself. The only rate limit left guards /hf/commit (per IP).
# ---------------------------------------------------------------------------


def _request(xff=None, peer="9.9.9.9"):
    """Minimal stand-in for starlette.Request: only .headers.get and
    .client.host are touched by the code under test."""
    headers = {}
    if xff is not None:
        headers["x-forwarded-for"] = xff
    client = types.SimpleNamespace(host=peer) if peer is not None else None
    return types.SimpleNamespace(
        headers=types.SimpleNamespace(get=lambda k, d=None: headers.get(k, d)),
        client=client,
    )


class TestRateLimitDependency:
    def test_allows_then_blocks_with_429(self, clock):
        dep = rl.rate_limit("login", limit=2, window_seconds=60)
        req = _request(peer="1.2.3.4")
        dep(req)  # 1st — ok
        dep(req)  # 2nd — ok
        with pytest.raises(HTTPException) as ei:
            dep(req)  # 3rd — blocked
        assert ei.value.status_code == 429
        assert "Retry-After" in ei.value.headers

    def test_limit_is_per_ip(self, clock):
        dep = rl.rate_limit("login", limit=1, window_seconds=60)
        dep(_request(peer="1.1.1.1"))
        # A different IP has its own budget.
        dep(_request(peer="2.2.2.2"))  # must not raise


# ---------------------------------------------------------------------------
# _client_ip — must resist a forged X-Forwarded-For
# ---------------------------------------------------------------------------


class TestClientIp:
    def test_uses_peer_when_no_xff(self, monkeypatch):
        monkeypatch.setattr(rl, "_TRUSTED_HOPS", 1)
        assert rl._client_ip(_request(peer="5.5.5.5")) == "5.5.5.5"

    def test_one_trusted_hop_takes_rightmost_appended_ip(self, monkeypatch):
        # The trusted proxy APPENDS the real client IP, so with one hop the
        # trustworthy value is the rightmost entry.
        monkeypatch.setattr(rl, "_TRUSTED_HOPS", 1)
        ip = rl._client_ip(_request(xff="1.1.1.1, 2.2.2.2, 3.3.3.3"))
        assert ip == "3.3.3.3"

    def test_two_trusted_hops_reads_one_further_left(self, monkeypatch):
        monkeypatch.setattr(rl, "_TRUSTED_HOPS", 2)
        ip = rl._client_ip(_request(xff="1.1.1.1, 2.2.2.2, 3.3.3.3"))
        assert ip == "2.2.2.2"

    def test_forged_header_cannot_reach_past_trusted_hops(self, monkeypatch):
        # An attacker stuffs extra IPs on the LEFT. With one trusted hop we only
        # ever read the rightmost, so the spoofed entries are ignored.
        monkeypatch.setattr(rl, "_TRUSTED_HOPS", 1)
        ip = rl._client_ip(_request(xff="evil-spoof, 8.8.8.8"))
        assert ip == "8.8.8.8"

    def test_hops_larger_than_header_falls_back_to_peer(self, monkeypatch):
        monkeypatch.setattr(rl, "_TRUSTED_HOPS", 5)
        assert rl._client_ip(_request(xff="1.1.1.1", peer="7.7.7.7")) == "7.7.7.7"

    def test_zero_hops_ignores_xff_entirely(self, monkeypatch):
        # No proxy in front (local dev): never trust the header.
        monkeypatch.setattr(rl, "_TRUSTED_HOPS", 0)
        assert rl._client_ip(_request(xff="1.2.3.4", peer="7.7.7.7")) == "7.7.7.7"

    def test_no_client_returns_unknown(self, monkeypatch):
        monkeypatch.setattr(rl, "_TRUSTED_HOPS", 1)
        assert rl._client_ip(_request(xff=None, peer=None)) == "unknown"
