"""Task management tool for nanobot personal assistant.

Stores tasks in a human-readable markdown file (TASKS.md) in the workspace memory directory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
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


_TASKS_FILE = "TASKS.md"
_TEMPLATE = """# Tasks

## Active
<!-- Active tasks you're currently working on -->

## Backlog
<!-- Tasks waiting to be started -->

## Completed
<!-- Recently completed tasks (auto-cleaned periodically) -->

"""


_TASK_PARAMETERS = tool_parameters_schema(
    action=StringSchema(
        "Action to perform",
        enum=("list", "add", "update", "complete", "delete", "prioritize"),
    ),
    task_id=StringSchema(
        "Task ID (from list output). Required for update, complete, delete, prioritize."
    ),
    title=StringSchema("Task title. Required for add."),
    priority=StringSchema(
        "Priority level", enum=("high", "medium", "low")
    ),
    due_date=StringSchema("Due date in YYYY-MM-DD format"),
    notes=StringSchema("Optional notes or context for the task"),
    section=StringSchema(
        "Section to place the task in", enum=("active", "backlog", "completed")
    ),
    required=["action"],
    description=(
        "Manage personal tasks in TASKS.md. "
        "Actions: list (show all), add (create new), update (modify), "
        "complete (mark done), delete (remove), prioritize (set priority). "
        "Tasks have: id (auto), title, priority, due_date, notes, section (active/backlog/completed)."
    ),
)


@tool_parameters(_TASK_PARAMETERS)
class TaskTool(Tool):

    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir
        self.tasks_file = memory_dir / _TASKS_FILE

    @classmethod
    def create(cls, ctx: ToolContext) -> TaskTool:
        memory_dir = Path(ctx.workspace) / "memory"
        return cls(memory_dir=memory_dir)

    @property
    def name(self) -> str:
        return "task"

    @property
    def description(self) -> str:
        return (
            "Manage personal tasks and memos in TASKS.md. "
            "Actions: list, add, update, complete, delete, prioritize. "
            "Tasks have: id (auto), title, priority (high/medium/low), "
            "due_date (YYYY-MM-DD), notes, section (active/backlog/completed)."
        )

    def _ensure_file(self) -> str:
        if not self.tasks_file.exists():
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            self.tasks_file.write_text(_TEMPLATE, encoding="utf-8")
        return self.tasks_file.read_text(encoding="utf-8")

    def _write_file(self, content: str) -> None:
        self.tasks_file.write_text(content, encoding="utf-8")

    def _parse_tasks(self, content: str) -> list[dict]:
        """Parse tasks from markdown content."""
        tasks = []
        current_section = "active"
        task_id_counter = 1

        for line in content.split("\n"):
            # Track section headers
            if line.startswith("## "):
                header = line[3:].strip().lower()
                if header in ("active", "backlog", "completed"):
                    current_section = header
                continue

            # Match task lines: - [ ] **Title** (priority: high, due: 2025-01-15)
            match = re.match(r"^-\s*\[([ x])\]\s+\*\*(.+?)\*\s*(?:\((.+?)\))?", line)
            if match:
                done = match.group(1) == "x"
                title = match.group(2).strip()
                metadata_str = match.group(3) or ""

                # Parse metadata
                priority = "medium"
                due_date = None
                notes = None

                for part in metadata_str.split(","):
                    part = part.strip()
                    if part.startswith("priority:"):
                        priority = part.split(":", 1)[1].strip()
                    elif part.startswith("due:"):
                        due_date = part.split(":", 1)[1].strip()
                    elif part.startswith("notes:"):
                        notes = part.split(":", 1)[1].strip()

                tasks.append({
                    "id": str(task_id_counter),
                    "title": title,
                    "done": done,
                    "priority": priority,
                    "due_date": due_date,
                    "notes": notes,
                    "section": current_section,
                })
                task_id_counter += 1

        return tasks

    def _format_task(self, task: dict) -> str:
        """Format a single task as markdown line."""
        checkbox = "x" if task["done"] else " "
        meta_parts = []
        if task["priority"] != "medium":
            meta_parts.append(f"priority: {task['priority']}")
        if task["due_date"]:
            meta_parts.append(f"due: {task['due_date']}")
        if task["notes"]:
            meta_parts.append(f"notes: {task['notes']}")
        meta = f" ({', '.join(meta_parts)})" if meta_parts else ""
        return f"- [{checkbox}] **{task['title']}**{meta}"

    def _rebuild_content(self, tasks: list[dict]) -> str:
        """Rebuild the full TASKS.md content from parsed tasks."""
        sections = {"active": [], "backlog": [], "completed": []}
        for task in tasks:
            sections.setdefault(task["section"], []).append(task)

        lines = ["# Tasks", "", "## Active", ""]
        if sections["active"]:
            lines.extend(self._format_task(t) for t in sections["active"])
        else:
            lines.append("<!-- Active tasks you're currently working on -->")

        lines.extend(["", "## Backlog", ""])
        if sections["backlog"]:
            lines.extend(self._format_task(t) for t in sections["backlog"])
        else:
            lines.append("<!-- Tasks waiting to be started -->")

        lines.extend(["", "## Completed", ""])
        if sections["completed"]:
            lines.extend(self._format_task(t) for t in sections["completed"])
        else:
            lines.append("<!-- Recently completed tasks (auto-cleaned periodically) -->")

        lines.append("")
        return "\n".join(lines)

    def _find_task(self, tasks: list[dict], task_id: str) -> tuple[int, dict] | None:
        for i, task in enumerate(tasks):
            if task["id"] == task_id:
                return i, task
        return None

    async def execute(
        self,
        *,
        action: str,
        task_id: str | None = None,
        title: str | None = None,
        priority: str | None = None,
        due_date: str | None = None,
        notes: str | None = None,
        section: str | None = None,
    ) -> ToolResult:
        content = self._ensure_file()
        tasks = self._parse_tasks(content)

        if action == "list":
            if not tasks:
                return ToolResult("(no tasks)")
            lines = []
            for task in tasks:
                status = "✓" if task["done"] else "○"
                section_label = task["section"][0].upper()
                priority_marker = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(task["priority"], "")
                due = f" 📅{task['due_date']}" if task["due_date"] else ""
                notes_text = f" — {task['notes']}" if task["notes"] else ""
                lines.append(f"{task['id']}. [{status}] [{section_label}] {priority_marker} {task['title']}{due}{notes_text}")
            return ToolResult("\n".join(lines))

        elif action == "add":
            if not title:
                return ToolResult.error("title is required for add")
            new_task = {
                "id": str(len(tasks) + 1),
                "title": title,
                "done": False,
                "priority": priority or "medium",
                "due_date": due_date,
                "notes": notes,
                "section": section or "active",
            }
            tasks.append(new_task)
            self._write_file(self._rebuild_content(tasks))
            return ToolResult(f"Added task #{new_task['id']}: {new_task['title']}")

        elif action == "update":
            if not task_id:
                return ToolResult.error("task_id is required for update")
            found = self._find_task(tasks, task_id)
            if not found:
                return ToolResult.error(f"Task #{task_id} not found")
            i, task = found
            if title is not None:
                task["title"] = title
            if priority is not None:
                task["priority"] = priority
            if due_date is not None:
                task["due_date"] = due_date
            if notes is not None:
                task["notes"] = notes
            if section is not None:
                task["section"] = section
            tasks[i] = task
            self._write_file(self._rebuild_content(tasks))
            return ToolResult(f"Updated task #{task['id']}: {task['title']}")

        elif action == "complete":
            if not task_id:
                return ToolResult.error("task_id is required for complete")
            found = self._find_task(tasks, task_id)
            if not found:
                return ToolResult.error(f"Task #{task_id} not found")
            i, task = found
            task["done"] = True
            task["section"] = "completed"
            tasks[i] = task
            self._write_file(self._rebuild_content(tasks))
            return ToolResult(f"Completed task #{task['id']}: {task['title']}")

        elif action == "delete":
            if not task_id:
                return ToolResult.error("task_id is required for delete")
            found = self._find_task(tasks, task_id)
            if not found:
                return ToolResult.error(f"Task #{task_id} not found")
            i, task = found
            tasks.pop(i)
            # Renumber IDs
            for idx, t in enumerate(tasks):
                t["id"] = str(idx + 1)
            self._write_file(self._rebuild_content(tasks))
            return ToolResult(f"Deleted task #{task_id}: {task['title']}")

        elif action == "prioritize":
            if not task_id:
                return ToolResult.error("task_id is required for prioritize")
            if not priority:
                return ToolResult.error("priority is required for prioritize")
            found = self._find_task(tasks, task_id)
            if not found:
                return ToolResult.error(f"Task #{task_id} not found")
            i, task = found
            task["priority"] = priority
            # Move high priority to active, low to backlog
            if priority == "high":
                task["section"] = "active"
            elif priority == "low":
                task["section"] = "backlog"
            tasks[i] = task
            self._write_file(self._rebuild_content(tasks))
            return ToolResult(f"Prioritized task #{task['id']} to {priority}")

        else:
            return ToolResult.error(f"Unknown action: {action}. Use: list, add, update, complete, delete, prioritize")


def register_task_tool(tools: ToolRegistry, memory_dir: Path) -> None:
    """Register the task tool with the given registry."""
    tools.register(TaskTool(memory_dir))