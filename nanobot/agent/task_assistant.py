"""Task Assistant extension for nanobot.

Registers the task tool, inactivity tool, and optional cron jobs for periodic task review.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.tasks import register_task_tool
from nanobot.agent.tools.inactivity import register_inactivity_tool
from nanobot.config.schema import Config
from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule
from nanobot.session.manager import SessionManager


CONFIG_FILE = "task_assistant.json"

DEFAULT_MORNING_BRIEFING_PROMPT = """Review my TASKS.md and give me a morning briefing:

1. What's due today or overdue?
2. What are my high-priority active tasks?
3. What should I focus on first?
4. Any tasks that should be moved to backlog or broken down?
5. Any tasks I can mark complete?

Be concise and actionable. Use the task tool to read and update tasks as needed."""


DEFAULT_EVENING_REVIEW_PROMPT = """Review my TASKS.md and give me an evening summary:

1. What did I complete today?
2. What's still in progress?
3. What should be prioritized for tomorrow?
4. Any new tasks to add?

Be concise. Update task statuses/priorities as appropriate."""


DEFAULT_HOURLY_CHECKIN_PROMPT = """Check if I've been inactive using the inactivity tool with the configured threshold.

If I have been inactive (is_inactive: true):
- Give me a gentle check-in about my current task progress
- Ask what I'm working on or if I need help
- Suggest breaking down large tasks if I seem stuck
- Be brief and supportive

If I have NOT been inactive (is_inactive: false):
- Return exactly: __NOOP__
- Do not add any other text
"""


def _load_config(workspace: Path) -> dict[str, Any]:
    """Load task assistant config from workspace/task_assistant.json."""
    config_path = workspace / CONFIG_FILE
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _get_prompt(config: dict[str, Any], key: str, default: str) -> str:
    """Get prompt from config or return default."""
    return config.get("prompts", {}).get(key, default)


def _upsert_cron_job(cron: CronService, name: str, schedule: CronSchedule, message: str,
                          session_key: str, origin_channel: str, origin_chat_id: str) -> None:
    """Create or update cron job to match desired configuration."""
    for job in cron.list_jobs(include_disabled=True):
        if job.name == name:
            # Check if config changed
            schedule_changed = (job.schedule.kind != schedule.kind
                                or job.schedule.expr != schedule.expr
                                or job.schedule.tz != schedule.tz
                                or job.schedule.at_ms != schedule.at_ms
                                or job.schedule.every_ms != schedule.every_ms)
            message_changed = job.payload.message != message
            session_changed = (job.payload.session_key != session_key
                               or job.payload.origin_channel != origin_channel
                               or job.payload.origin_chat_id != origin_chat_id)
            
            if schedule_changed or message_changed or session_changed:
                logger.info("Cron job '{}' config changed, updating", name)
                # If session/origin changed, must recreate (update_job doesn't support those)
                if session_changed:
                    cron.remove_job(job.id)
                    cron.add_job(
                        name=name,
                        schedule=schedule,
                        message=message,
                        session_key=session_key,
                        origin_channel=origin_channel,
                        origin_chat_id=origin_chat_id,
                    )
                else:
                    cron.update_job(
                        job.id,
                        schedule=schedule,
                        message=message,
                    )
            else:
                logger.debug("Cron job '{}' up to date", name)
            return
    
    # Not found - create new
    logger.info("Cron job '{}' created", name)
    cron.add_job(
        name=name,
        schedule=schedule,
        message=message,
        session_key=session_key,
        origin_channel=origin_channel,
        origin_chat_id=origin_chat_id,
    )


def setup_task_assistant(
    tools: ToolRegistry,
    cron: CronService,
    workspace: Path,
    session_manager: SessionManager,
    session_key: str = "cli:direct",
    *,
    enable_morning_briefing: bool = True,
    enable_evening_review: bool = True,
    enable_hourly_checkin: bool = True,
    morning_cron: str = "0 9 * * *",      # 9 AM
    evening_cron: str = "0 18 * * *",      # 6 PM
    checkin_cron: str = "0 * * * 1-5",   # Every hour, Mon-Fri
    timezone: str = "Europe/Rome",
    threshold_minutes: int | None = None,
) -> None:
    """Register task tool, inactivity tool, and optional cron jobs.

    Args:
        tools: ToolRegistry to register the task tool
        cron: CronService to register scheduled jobs
        workspace: Workspace root path
        session_manager: SessionManager for inactivity tool
        session_key: Session key to bind cron jobs to (default: "cli:direct")
        enable_morning_briefing: Add 9 AM daily briefing cron job
        enable_evening_review: Add 6 PM daily review cron job
        enable_hourly_checkin: Add hourly inactivity check-in cron job
        morning_cron: Cron expression for morning briefing
        evening_cron: Cron expression for evening review
        checkin_cron: Cron expression for hourly check-in
        timezone: Timezone for cron expressions

    Prompts can be customized via workspace/task_assistant.json:
    {
        "prompts": {
            "morning_briefing": "Your custom morning prompt...",
            "evening_review": "Your custom evening prompt...",
            "hourly_checkin": "Your custom checkin prompt..."
        },
        "schedule": {
            "morning_cron": "0 8 * * *",
            "evening_cron": "0 19 * * *",
            "checkin_cron": "0 * * * *"
        },
        "timezone": "America/New_York",
        "session_key": "telegram:27748529"
    }
    """
    # Load config file for prompt/schedule overrides
    config = _load_config(workspace)
    prompts = config.get("prompts", {})
    schedule = config.get("schedule", {})

    # Get prompts (config overrides defaults)
    morning_prompt = prompts.get("morning_briefing", DEFAULT_MORNING_BRIEFING_PROMPT)
    evening_prompt = prompts.get("evening_review", DEFAULT_EVENING_REVIEW_PROMPT)
    checkin_prompt = prompts.get("hourly_checkin", DEFAULT_HOURLY_CHECKIN_PROMPT)

    # Get schedule (config overrides defaults)
    morning_cron = schedule.get("morning_cron", morning_cron)
    evening_cron = schedule.get("evening_cron", evening_cron)
    checkin_cron = schedule.get("checkin_cron", checkin_cron)
    timezone = config.get("timezone", timezone)
    session_key = config.get("session_key", session_key)
    configured_threshold = config.get("threshold_minutes")
    if configured_threshold is not None:
        threshold_minutes = configured_threshold

    memory_dir = workspace / "memory"

    # Register the task tool
    register_task_tool(tools, memory_dir)

    # Register the inactivity tool (config always wins; falls back to 30 if unset)
    effective_threshold = threshold_minutes if threshold_minutes is not None else 30
    register_inactivity_tool(tools, session_manager, threshold_minutes=effective_threshold)

    # Register cron jobs
    # Parse session_key into channel and chat_id for proper binding
    if ":" in session_key:
        origin_channel, origin_chat_id = session_key.split(":", 1)
    else:
        origin_channel, origin_chat_id = "cli", session_key

    if enable_morning_briefing:
        _upsert_cron_job(
            cron, "morning-task-briefing",
            CronSchedule(kind="cron", expr=morning_cron, tz=timezone),
            morning_prompt, session_key, origin_channel, origin_chat_id
        )

    if enable_evening_review:
        _upsert_cron_job(
            cron, "evening-task-review",
            CronSchedule(kind="cron", expr=evening_cron, tz=timezone),
            evening_prompt, session_key, origin_channel, origin_chat_id
        )

    if enable_hourly_checkin:
        _upsert_cron_job(
            cron, "hourly-inactivity-checkin",
            CronSchedule(kind="cron", expr=checkin_cron, tz=timezone),
            checkin_prompt, session_key, origin_channel, origin_chat_id
        )


def setup_task_assistant_from_config(
    tools: ToolRegistry,
    cron: CronService,
    session_manager: SessionManager,
    config: Config,
    session_key: str = "cli:direct",
) -> None:
    """Setup from nanobot Config object (reads workspace/task_assistant.json if present)."""
    setup_task_assistant(
        tools=tools,
        cron=cron,
        workspace=config.workspace_path,
        session_manager=session_manager,
        session_key=session_key,
        enable_morning_briefing=True,
        enable_evening_review=True,
        enable_hourly_checkin=True,
        morning_cron="0 9 * * *",
        evening_cron="0 18 * * *",
        checkin_cron="0 * * * 1-5",
        timezone=config.agents.defaults.timezone or "UTC",
    )