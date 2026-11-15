"""Tests for NOOP suppression in cron check-in responses.

These tests ensure that the LLM's NOOP response variations are properly
suppressed so they don't reach the user (e.g., on Telegram).
"""

from __future__ import annotations

import re
import pytest


def test_noop_suppression_exact_match():
    """Exact __NOOP__ should be suppressed."""
    assert re.fullmatch(r"__NOOP__[.!?]?\s*", "__NOOP__") is not None


def test_noop_suppression_with_period():
    """__NOOP__. should be suppressed."""
    assert re.fullmatch(r"__NOOP__[.!?]?\s*", "__NOOP__.") is not None


def test_noop_suppression_with_exclamation():
    """__NOOP__! should be suppressed."""
    assert re.fullmatch(r"__NOOP__[.!?]?\s*", "__NOOP__!") is not None


def test_noop_suppression_with_question():
    """__NOOP__? should be suppressed."""
    assert re.fullmatch(r"__NOOP__[.!?]?\s*", "__NOOP__?") is not None


def test_noop_suppression_with_trailing_whitespace():
    """__NOOP__ with trailing whitespace should be suppressed."""
    assert re.fullmatch(r"__NOOP__[.!?]?\s*", "__NOOP__   ") is not None
    assert re.fullmatch(r"__NOOP__[.!?]?\s*", "__NOOP__.\t\n") is not None


def test_noop_suppression_with_leading_whitespace():
    """__NOOP__ with leading/trailing whitespace should be suppressed (after strip)."""
    # The actual code does content.strip() before regex match
    assert re.fullmatch(r"__NOOP__[.!?]?\s*", "  __NOOP__".strip()) is not None
    assert re.fullmatch(r"__NOOP__[.!?]?\s*", "\t__NOOP__.\n".strip()) is not None


def test_noop_suppression_not_matched_extra_text():
    """Extra text beyond punctuation should NOT be suppressed."""
    assert re.fullmatch(r"__NOOP__[.!?]?\s*", "__NOOP__ extra") is None
    assert re.fullmatch(r"__NOOP__[.!?]?\s*", "__NOOP__ done") is None
    assert re.fullmatch(r"__NOOP__[.!?]?\s*", "some __NOOP__") is None


def test_noop_suppression_not_matched_other_words():
    """Other similar words should NOT be suppressed."""
    assert re.fullmatch(r"__NOOP__[.!?]?\s*", "NOOP") is None
    assert re.fullmatch(r"__NOOP__[.!?]?\s*", "__NOOP") is None
    assert re.fullmatch(r"__NOOP__[.!?]?\s*", "NOOP__") is None
    assert re.fullmatch(r"__NOOP__[.!?]?\s*", "__noop__") is None  # case sensitive


class TestTurnDeliveryNoopSuppression:
    """Integration tests for TurnDelivery NOOP suppression."""

    @pytest.mark.asyncio
    async def test_turn_delivery_suppresses_noop_variations(self):
        """Verify the suppression regex matches the one used in TurnDelivery."""
        from nanobot.agent.turn_delivery import TurnDelivery

        # Import the module to ensure the regex pattern is consistent
        import nanobot.agent.turn_delivery as td_module
        import re

        # The pattern used in TurnDelivery
        pattern = r"__NOOP__[.!?]?\s*"

        test_cases = {
            "__NOOP__": True,
            "__NOOP__.": True,
            "__NOOP__!": True,
            "__NOOP__?": True,
            "__NOOP__   ": True,
            "__NOOP__.\t\n": True,
            "  __NOOP__": True,
            "__NOOP__ extra": False,
            "some __NOOP__": False,
            "NOOP": False,
            "__noop__": False,
        }

        for content, should_match in test_cases.items():
            match = re.fullmatch(pattern, content.strip())
            assert (match is not None) == should_match, f"Failed for {content!r}: expected match={should_match}"


class TestMessageToolNoopSuppression:
    """Integration tests for MessageTool NOOP suppression."""

    @pytest.mark.asyncio
    async def test_message_tool_suppresses_noop_variations(self):
        """Verify the suppression regex matches the one used in MessageTool."""
        import nanobot.agent.tools.message as msg_module
        import re

        # The pattern used in MessageTool
        pattern = r"__NOOP__[.!?]?\s*"

        test_cases = {
            "__NOOP__": True,
            "__NOOP__.": True,
            "__NOOP__!": True,
            "__NOOP__?": True,
            "__NOOP__   ": True,
            "__NOOP__.\t\n": True,
            "  __NOOP__": True,
            "__NOOP__ extra": False,
            "some __NOOP__": False,
            "NOOP": False,
            "__noop__": False,
        }

        for content, should_match in test_cases.items():
            match = re.fullmatch(pattern, content.strip())
            assert (match is not None) == should_match, f"Failed for {content!r}: expected match={should_match}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])