"""Inactivity check tool for nanobot personal assistant.

Allows cron jobs to check if the user has been inactive for a specified period.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from nanobot.agent.tools.base import Tool, ToolResult, tool_parameters
from nanobot.agent.tools.context import ToolContext
from nanobot.agent.tools.schema import (
    BooleanSchema,
    IntegerSchema,
    StringSchema,
    tool_parameters_schema,
)

if TYPE_CHECKING:
    from nanobot.agent.tools.registry import ToolRegistry
    from nanobot.session.manager import SessionManager


_INACTIVITY_PARAMETERS = tool_parameters_schema(
    action=StringSchema(
        "Action to perform",
        enum=("check",),
    ),
    session_key=StringSchema(
        "Session key to check (default: current session)"
    ),
    required=["action"],
    description=(
        "Check if the user has been inactive for more than the configured threshold. "
        "Returns the inactivity duration and whether it exceeds the threshold. "
        "Useful for cron jobs that should only engage when the user is away."
    ),
)


@tool_parameters(_INACTIVITY_PARAMETERS)
class InactivityTool(Tool):
    NAME = "inactivity"
    DESCRIPTION = (
        "Check user inactivity duration. "
        "Action: check (returns inactive_minutes, is_inactive boolean). "
        "Use in cron jobs to conditionally engage only when user is away."
    )

    def __init__(self, session_manager: SessionManager, threshold_minutes: int = 30):
        self.session_manager = session_manager
        self.threshold_minutes = threshold_minutes

    @classmethod
    def create(cls, ctx: ToolContext) -> InactivityTool:
        session_manager = ctx.sessions
        if session_manager is None:
            raise RuntimeError("InactivityTool requires a session manager")
        return cls(session_manager=session_manager)

    @property
    def name(self) -> str:
        return "inactivity"

    @property
    def description(self) -> str:
        return (
            "Check user inactivity duration. "
            "Action: check (returns inactive_minutes, is_inactive boolean). "
            "Use in cron jobs to conditionally engage only when user is away."
        )

    async def execute(
        self,
        *,
        action: str,
        session_key: str | None = None,
    ) -> ToolResult:
        if action != "check":
            return ToolResult.error(f"Unknown action: {action}. Use: check")

        # Get session key from request context if not provided
        if session_key is None:
            from nanobot.agent.tools.context import current_request_context
            ctx = current_request_context()
            session_key = ctx.session_key if ctx else None

        if not session_key:
            return ToolResult.error("No session key available")

        # Get session
        try:
            session = await self.session_manager.state.get(session_key)
        except KeyError:
            return ToolResult.error(f"Session not found: {session_key}")

        # Calculate inactivity based on last user activity (not cron activity)
        # Session.last_user_activity uses datetime.now() (local time), so compare in local timezone
        now = datetime.now().astimezone()
        last_active = session.last_user_activity
        if last_active.tzinfo is None:
            # Assume session.updated_at is local time
            last_active = last_active.replace(tzinfo=now.tzinfo)
        else:
            last_active = last_active.astimezone(now.tzinfo)

        inactive_seconds = (now - last_active).total_seconds()
        inactive_minutes = int(inactive_seconds / 60)
        effective_threshold = self.threshold_minutes
        is_inactive = inactive_minutes >= effective_threshold

        import json
        return ToolResult(json.dumps({
            "inactive_minutes": inactive_minutes,
            "is_inactive": is_inactive,
            "threshold_minutes": effective_threshold,
            "last_active": last_active.isoformat(),
            "checked_at": now.isoformat(),
        }))


def register_inactivity_tool(
    tools: ToolRegistry, session_manager: SessionManager, threshold_minutes: int = 30
) -> None:
    """Register the inactivity tool with the given registry.

    The configured threshold always wins; the LLM-supplied value is ignored.
    """
    tools.register(InactivityTool(session_manager, threshold_minutes=threshold_minutes))