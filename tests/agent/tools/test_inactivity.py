from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.tools.inactivity import InactivityTool, register_inactivity_tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.session.manager import SessionManager
from nanobot.agent.tools.base import ToolResult
from nanobot.cron.session_turns import is_cron_turn


@pytest.fixture
async def session_manager(tmp_path):
    """Create a SessionManager with a test workspace."""
    manager = SessionManager(tmp_path)
    try:
        yield manager
    finally:
        await manager.state.aclose()


@pytest.fixture
def inactivity_tool(session_manager):
    """Create an InactivityTool with a 1-minute threshold."""
    return InactivityTool(session_manager, threshold_minutes=1)


class TestInactivityToolThreshold:
    """Tests for threshold configuration and behavior."""

    @pytest.mark.asyncio
    async def test_configured_threshold_is_used(self, session_manager, inactivity_tool):
        """The tool should use the configured threshold, not an LLM-supplied value."""
        session_key = "test:session"
        # Create session via manager to ensure it's in the store
        session = session_manager.get_or_create(session_key)
        # Set last_user_activity to 5 minutes ago (past the 1-minute threshold)
        session.last_user_activity = datetime.now() - timedelta(minutes=5)
        session_manager.save(session)

        result = await inactivity_tool.execute(action="check", session_key=session_key)
        data = json.loads(str(result))

        # Should use configured threshold (1 minute), not default 30
        assert data["threshold_minutes"] == 1
        assert data["is_inactive"] is True
        assert data["inactive_minutes"] >= 5

    @pytest.mark.asyncio
    async def test_llm_threshold_parameter_is_ignored(self, session_manager):
        """LLM cannot override the configured threshold."""
        # Register tool with custom threshold
        tool = InactivityTool(session_manager, threshold_minutes=5)

        session_key = "test:session2"
        # Create session via manager to ensure it's in the store
        session = session_manager.get_or_create(session_key)
        # 3 minutes ago - past default 30? No, past custom 5? No, under 5
        session.last_user_activity = datetime.now() - timedelta(minutes=3)
        session_manager.save(session)

        result = await tool.execute(action="check", session_key=session_key)
        data = json.loads(str(result))

        # Should use configured 5-minute threshold
        assert data["threshold_minutes"] == 5
        assert data["is_inactive"] is False  # 3 < 5

    @pytest.mark.asyncio
    async def test_different_thresholds_produce_different_results(self, session_manager):
        """Different configured thresholds produce different is_inactive results."""
        session_key = "test:session3"
        # Create session via manager to ensure it's in the store
        session = session_manager.get_or_create(session_key)
        session.last_user_activity = datetime.now() - timedelta(minutes=10)
        session_manager.save(session)

        # Tool with 5-minute threshold
        tool_5min = InactivityTool(session_manager, threshold_minutes=5)
        result_5 = await tool_5min.execute(action="check", session_key=session_key)
        data_5 = json.loads(str(result_5))

        # Tool with 15-minute threshold
        tool_15min = InactivityTool(session_manager, threshold_minutes=15)
        result_15 = await tool_15min.execute(action="check", session_key=session_key)
        data_15 = json.loads(str(result_15))

        assert data_5["threshold_minutes"] == 5
        assert data_5["is_inactive"] is True   # 10 >= 5

        assert data_15["threshold_minutes"] == 15
        assert data_15["is_inactive"] is False  # 10 < 15


class TestInactivityToolActiveVsInactive:
    """Tests for active vs inactive detection."""

    @pytest.mark.asyncio
    async def test_recently_active_returns_not_inactive(self, session_manager, inactivity_tool):
        """Session updated within threshold should return is_inactive=false."""
        session_key = "test:active"
        # Create session via manager to ensure it's in the store
        session = session_manager.get_or_create(session_key)
        session.last_user_activity = datetime.now() - timedelta(seconds=30)  # 30 seconds ago
        session_manager.save(session)

        result = await inactivity_tool.execute(action="check", session_key=session_key)
        data = json.loads(str(result))

        assert data["is_inactive"] is False
        assert data["inactive_minutes"] == 0

    @pytest.mark.asyncio
    async def test_exactly_at_threshold_is_inactive(self, session_manager):
        """Session at exactly the threshold should be considered inactive."""
        tool = InactivityTool(session_manager, threshold_minutes=10)
        session_key = "test:at_threshold"
        # Create session via manager to ensure it's in the store
        session = session_manager.get_or_create(session_key)
        session.last_user_activity = datetime.now() - timedelta(minutes=10)
        session_manager.save(session)

        result = await tool.execute(action="check", session_key=session_key)
        data = json.loads(str(result))

        assert data["is_inactive"] is True
        assert data["inactive_minutes"] >= 10

    @pytest.mark.asyncio
    async def test_just_under_threshold_not_inactive(self, session_manager):
        """Session just under threshold should not be inactive."""
        tool = InactivityTool(session_manager, threshold_minutes=10)
        session_key = "test:under_threshold"
        # Create session via manager to ensure it's in the store
        session = session_manager.get_or_create(session_key)
        session.last_user_activity = datetime.now() - timedelta(minutes=9, seconds=59)
        session_manager.save(session)

        result = await tool.execute(action="check", session_key=session_key)
        data = json.loads(str(result))

        assert data["is_inactive"] is False
        assert data["inactive_minutes"] == 9

    @pytest.mark.asyncio
    async def test_just_over_threshold_is_inactive(self, session_manager):
        """Session just over threshold should be inactive."""
        tool = InactivityTool(session_manager, threshold_minutes=10)
        session_key = "test:over_threshold"
        # Create session via manager to ensure it's in the store
        session = session_manager.get_or_create(session_key)
        session.last_user_activity = datetime.now() - timedelta(minutes=10, seconds=1)
        session_manager.save(session)

        result = await tool.execute(action="check", session_key=session_key)
        data = json.loads(str(result))

        assert data["is_inactive"] is True
        assert data["inactive_minutes"] >= 10


class TestInactivityToolResponseFormat:
    """Tests for response format and content."""

    @pytest.mark.asyncio
    async def test_response_contains_required_fields(self, session_manager, inactivity_tool):
        """Response should contain all expected fields."""
        session_key = "test:fields"
        # Create session via manager to ensure it's in the store
        session = session_manager.get_or_create(session_key)
        session.last_user_activity = datetime.now() - timedelta(minutes=5)
        session_manager.save(session)

        result = await inactivity_tool.execute(action="check", session_key=session_key)
        data = json.loads(str(result))

        assert "inactive_minutes" in data
        assert "is_inactive" in data
        assert "threshold_minutes" in data
        assert "last_active" in data
        assert "checked_at" in data

        assert isinstance(data["inactive_minutes"], int)
        assert isinstance(data["is_inactive"], bool)
        assert isinstance(data["threshold_minutes"], int)
        assert isinstance(data["last_active"], str)
        assert isinstance(data["checked_at"], str)

    @pytest.mark.asyncio
    async def test_threshold_in_response_matches_configured(self, session_manager):
        """Response threshold_minutes should match the configured threshold."""
        # Create all sessions first (before state gets bound)
        for threshold in [1, 5, 15, 30, 60, 120]:
            session_key = f"test:threshold_{threshold}"
            session = session_manager.get_or_create(session_key)
            session.last_user_activity = datetime.now() - timedelta(minutes=1)
            session_manager.save(session)

        # Now test each threshold
        for threshold in [1, 5, 15, 30, 60, 120]:
            tool = InactivityTool(session_manager, threshold_minutes=threshold)
            session_key = f"test:threshold_{threshold}"
            result = await tool.execute(action="check", session_key=session_key)
            data = json.loads(str(result))

            assert data["threshold_minutes"] == threshold

    @pytest.mark.asyncio
    async def test_session_key_from_context(self, session_manager):
        """Tool should use session key from request context when not provided."""
        from nanobot.agent.tools.context import RequestContext, request_context

        tool = InactivityTool(session_manager, threshold_minutes=5)

        with request_context(RequestContext(
            channel="telegram",
            chat_id="12345",
            session_key="telegram:12345",
        )):
            session = session_manager.get_or_create("telegram:12345")
            session.last_user_activity = datetime.now() - timedelta(minutes=10)
            session_manager.save(session)

            result = await tool.execute(action="check")
            data = json.loads(str(result))

            assert data["is_inactive"] is True


class TestInactivityToolErrors:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_unknown_action_returns_error(self, inactivity_tool):
        """Unknown action should return an error."""
        result = await inactivity_tool.execute(action="invalid")
        assert result.is_error
        assert "Unknown action" in str(result)

    @pytest.mark.asyncio
    async def test_missing_session_creates_new_session(self, session_manager, inactivity_tool):
        """Non-existent session gets created with current timestamp (treated as active)."""
        result = await inactivity_tool.execute(action="check", session_key="nonexistent:session")
        # Tool creates a new session, so it returns is_inactive=false
        data = json.loads(str(result))
        assert data["is_inactive"] is False
        assert data["inactive_minutes"] == 0
        # The session should now exist
        session = await session_manager.state.get("nonexistent:session")
        assert session is not None

    @pytest.mark.asyncio
    async def test_no_session_key_available_returns_error(self, session_manager):
        """No session key in context should return error."""
        tool = InactivityTool(session_manager, threshold_minutes=5)

        # Don't set any request context
        result = await tool.execute(action="check", session_key=None)
        assert result.is_error
        assert "No session key available" in str(result)


class TestInactivityToolRegistration:
    """Tests for tool registration."""

    def test_register_inactivity_tool_adds_tool(self):
        """register_inactivity_tool should register the tool."""
        tools = ToolRegistry()
        session_manager = MagicMock(spec=SessionManager)

        register_inactivity_tool(tools, session_manager, threshold_minutes=7)

        definitions = tools.get_definitions()
        names = {d["function"]["name"] for d in definitions}
        assert "inactivity" in names

    def test_registered_tool_has_correct_schema(self):
        """Registered tool should not have threshold_minutes in parameters."""
        tools = ToolRegistry()
        session_manager = MagicMock(spec=SessionManager)

        register_inactivity_tool(tools, session_manager, threshold_minutes=7)

        definitions = tools.get_definitions()
        inactivity_def = next(d for d in definitions if d["function"]["name"] == "inactivity")
        params = inactivity_def["function"]["parameters"]
        properties = params.get("properties", {})

        # threshold_minutes should NOT be in the schema (LLM cannot override)
        assert "threshold_minutes" not in properties
        assert "action" in properties
        assert "session_key" in properties
        assert params.get("required") == ["action"]


class TestCronTurnDetection:
    """Tests for cron turn detection (used to prevent updating updated_at)."""

    def test_is_cron_turn_detects_cron_metadata(self):
        """is_cron_turn should return True for cron trigger metadata."""
        metadata = {
            "_cron_trigger": {
                "job_id": "test-job",
                "job_name": "test-job",
                "run_id": "test-run",
                "prompt_ref": {},
                "persist_content": "test",
            }
        }
        assert is_cron_turn(metadata) is True

    def test_is_cron_turn_false_for_normal_metadata(self):
        """is_cron_turn should return False for normal message metadata."""
        metadata = {
            "message_id": "123",
            "timestamp": "2024-01-01T00:00:00",
        }
        assert is_cron_turn(metadata) is False

    def test_is_cron_turn_false_for_none(self):
        """is_cron_turn should return False for None metadata."""
        assert is_cron_turn(None) is False


class TestInactivityToolNoopBehavior:
    """Tests related to NOOP vs actual response behavior for cron jobs."""

    @pytest.mark.asyncio
    async def test_active_session_returns_is_inactive_false(self, session_manager, inactivity_tool):
        """Active session should allow cron to return NOOP."""
        session_key = "test:noop"
        # Create session via manager to ensure it's in the store
        session = session_manager.get_or_create(session_key)
        session.last_user_activity = datetime.now() - timedelta(seconds=30)
        session_manager.save(session)

        result = await inactivity_tool.execute(action="check", session_key=session_key)
        data = json.loads(str(result))

        # Cron job checks is_inactive: if false, it should return __NOOP__
        assert data["is_inactive"] is False

    @pytest.mark.asyncio
    async def test_inactive_session_returns_is_inactive_true(self, session_manager, inactivity_tool):
        """Inactive session should trigger cron check-in."""
        session_key = "test:checkin"
        # Create session via manager to ensure it's in the store
        session = session_manager.get_or_create(session_key)
        session.last_user_activity = datetime.now() - timedelta(minutes=5)
        session_manager.save(session)

        result = await inactivity_tool.execute(action="check", session_key=session_key)
        data = json.loads(str(result))

        # Cron job checks is_inactive: if true, it should do a check-in
        assert data["is_inactive"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])