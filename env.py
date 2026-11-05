import logging
from typing import Any

logger = logging.getLogger(__name__)

_TASK_TICKETS: dict[str, list[dict[str, str | int]]] = {
    "easy": [
        {"id": 1, "text": "My screen is cracked.", "dept": "repair"},
    ],
    "medium": [
        {"id": 1, "text": "My screen is cracked.", "dept": "repair"},
        {"id": 2, "text": "I need a refund for my last invoice.", "dept": "billing"},
        {"id": 3, "text": "How do I upgrade my plan?", "dept": "sales"},
    ],
    "hard": [
        {"id": 1, "text": "Screen cracked.", "dept": "repair"},
        {"id": 2, "text": "Charged twice!", "dept": "billing"},
        {"id": 3, "text": "Want to buy more seats.", "dept": "sales"},
        {"id": 4, "text": "The app keeps crashing on startup.", "dept": "tech"},
        {"id": 5, "text": "I hate this, give me my money back immediately.", "dept": "billing"},
    ],
}


class TicketTriageEnv:
    VALID_TASKS = tuple(_TASK_TICKETS.keys())
    VALID_DEPARTMENTS = ("repair", "billing", "sales", "tech")

    def __init__(self, task: str = "easy") -> None:
        if task not in self.VALID_TASKS:
            raise ValueError(f"Invalid task '{task}'. Must be one of: {', '.join(self.VALID_TASKS)}")
        self._task = task
        self.tickets: list[dict[str, str | int]] = []
        self.current_ticket_idx: int = 0
        self.correct_routing: int = 0
        self.total_steps: int = 0
        self.done: bool = False

    def reset(self) -> dict[str, Any]:
        self.current_ticket_idx = 0
        self.correct_routing = 0
        self.total_steps = 0
        self.done = False
        self.tickets = list(_TASK_TICKETS[self._task])
        return self.state()

    def state(self) -> dict[str, Any]:
        if self.done or self.current_ticket_idx >= len(self.tickets):
            return {"status": "complete", "pending_ticket": None}
        return {
            "status": "in_progress",
            "pending_ticket": self.tickets[self.current_ticket_idx]["text"],
        }

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self.done:
            logger.warning("step() called after environment is done")
            return self.state(), 0.0, True, {"error": "Environment already done"}

        self.total_steps += 1
        current_ticket = self.tickets[self.current_ticket_idx]

        reward = 0.0
        if action.lower().strip() == str(current_ticket["dept"]):
            self.correct_routing += 1
            reward = 1.0 / len(self.tickets)

        self.current_ticket_idx += 1

        if self.current_ticket_idx >= len(self.tickets):
            self.done = True

        return self.state(), round(reward, 2), self.done, {}

    def grader(self) -> float:
        if not self.tickets:
            return 0.0
        return round(self.correct_routing / len(self.tickets), 2)
