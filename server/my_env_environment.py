"""Context-aware task scheduling environment implementation."""

from __future__ import annotations

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import Difficulty, MyAction, MyObservation, Task
except ImportError:
    from models import Difficulty, MyAction, MyObservation, Task


class MyEnvironment(Environment):
    """An RL-style environment for choosing the right task at the right time."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    CONTEXTS: tuple[str, ...] = ("sports", "student", "doctor")
    DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard")
    TASK_LIBRARY: dict[str, tuple[str, str, str]] = {
        "sports": ("injury check", "practice session", "recovery rest"),
        "student": ("exam revision", "assignment work", "take a break"),
        "doctor": ("critical patient review", "follow-up rounds", "paperwork"),
    }
    CONTEXT_KEYWORDS: dict[str, str] = {
        "sports": "injury",
        "student": "exam",
        "doctor": "critical",
    }
    MAX_TIME: int = 5

    def __init__(self):
        self._rng = random.Random()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.context = "student"
        self.difficulty: Difficulty = "easy"
        self.tasks: list[Task] = []
        self.time = 0
        self.reset()

    def reset(self) -> MyObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.context = self._rng.choice(self.CONTEXTS)
        self.difficulty = self._rng.choice(self.DIFFICULTIES)
        task_titles = self.TASK_LIBRARY[self.context]

        self.tasks = [
            Task(
                title=task_titles[0],
                priority=self._rng.randint(4, 5),
                deadline=self._rng.randint(1, 2),
                done=False,
            ),
            Task(
                title=task_titles[1],
                priority=self._rng.randint(2, 4),
                deadline=self._rng.randint(2, 4),
                done=False,
            ),
            Task(
                title=task_titles[2],
                priority=self._rng.randint(1, 2),
                deadline=self._rng.randint(4, 5),
                done=False,
            ),
        ]
        self.time = 0
        return self._build_observation(
            reward=0.0,
            done=False,
            decision_summary="Environment reset. Choose the best unfinished task.",
            chosen_index=None,
        )

    def step(self, action: MyAction) -> MyObservation:  # type: ignore[override]
        self._state.step_count += 1

        chosen_index = action.action
        recommended_index = self._recommended_action()
        reward = 0.0

        if chosen_index < len(self.tasks):
            chosen_task = self.tasks[chosen_index]
            if not chosen_task.done:
                reward = self._score_task_choice(chosen_index, recommended_index)
                chosen_task.done = True
                decision_summary = self._decision_summary(
                    chosen_task=chosen_task,
                    reward=reward,
                    recommended_index=recommended_index,
                )
            else:
                decision_summary = (
                    f"'{chosen_task.title}' was already completed, so the choice earned no reward."
                )
        else:
            reward = 0.1 if recommended_index == 3 else 0.0
            decision_summary = (
                "Skipping is acceptable because all tasks are done."
                if recommended_index == 3
                else "Skipping was not optimal because unfinished tasks still remain."
            )

        self.time += 1
        done = all(task.done for task in self.tasks) or self.time >= self.MAX_TIME
        if done:
            decision_summary += f" | Episode finished in {self.time} steps."
        return self._build_observation(
            reward=reward,
            done=done,
            decision_summary=decision_summary,
            chosen_index=chosen_index,
        )

    def _build_observation(
        self,
        reward: float,
        done: bool,
        decision_summary: str,
        chosen_index: int | None,
    ) -> MyObservation:
        recommended_action = self._best_task()
        return MyObservation(
            context=self.context,
            difficulty=self.difficulty,
            tasks=[task.model_copy(deep=True) for task in self.tasks],
            time=self.time,
            steps_remaining=max(0, self.MAX_TIME - self.time),
            recommended_action=recommended_action,
            score_explanation=self._recommendation_reason(recommended_action),
            decision_summary=decision_summary,
            reward=reward,
            done=done,
            metadata={
                "best_task_index": recommended_action,
                "deadline_pressure": [
                    max(0, task.deadline - self.time) for task in self.tasks
                ],
                "comparison": {
                    "chosen": chosen_index,
                    "recommended": recommended_action,
                },
            },
        )

    def _recommended_action(self) -> int:
        return self._best_task()

    def _best_task(self) -> int:
        available = [
            (index, task) for index, task in enumerate(self.tasks) if not task.done
        ]
        if not available:
            return 3

        if self.difficulty == "easy":
            return available[0][0]

        if self.difficulty == "medium":
            ranked = sorted(
                available,
                key=lambda item: (
                    -self._task_score(item[1], prioritize_deadline=False),
                    item[1].deadline,
                    item[0],
                ),
            )
            return ranked[0][0]

        ranked = sorted(
            available,
            key=lambda item: (
                -self._task_score(item[1], prioritize_deadline=True),
                item[1].deadline - self.time,
                item[0],
            ),
        )
        urgent_index, urgent_task = ranked[0]
        if self.time > urgent_task.deadline:
            fallback = sorted(
                available,
                key=lambda item: (
                    -self._task_score(item[1], prioritize_deadline=False),
                    item[1].deadline,
                    item[0],
                ),
            )
            return fallback[0][0]
        return urgent_index

    def _score_task_choice(self, chosen_index: int, recommended_index: int) -> float:
        chosen_task = self.tasks[chosen_index]
        if chosen_index == recommended_index:
            return 1.0

        if self.difficulty == "easy":
            return 0.5

        if self.difficulty == "medium":
            return (
                0.5
                if self._task_score(chosen_task, prioritize_deadline=False) >= 4
                else 0.0
            )

        return 0.0

    def _decision_summary(
        self, chosen_task: Task, reward: float, recommended_index: int
    ) -> str:
        keyword = self.CONTEXT_KEYWORDS[self.context]
        context_match = keyword in chosen_task.title.lower()
        if reward == 1.0:
            return (
                f"Chose '{chosen_task.title}' because it matches the {self.context} "
                f"context and best balances priority {chosen_task.priority} with "
                f"deadline {chosen_task.deadline}, earning full reward."
            )
        if reward == 0.5:
            return (
                f"Chose '{chosen_task.title}' because it was a reasonable option for "
                f"the {self.context} context, with priority {chosen_task.priority} "
                f"and deadline {chosen_task.deadline}, but it was not the top move."
            )
        best_title = (
            self.tasks[recommended_index].title
            if recommended_index < len(self.tasks)
            else "skip"
        )
        if context_match:
            return (
                f"Chose '{chosen_task.title}' because it fit the {self.context} "
                f"context, but '{best_title}' had the stronger priority/deadline "
                f"tradeoff for this step."
            )
        return (
            f"Chose '{chosen_task.title}', but missed a better option due to lower "
            f"priority or urgency. '{best_title}' was stronger for the {self.context} "
            f"context with a better priority/deadline tradeoff."
        )

    def _task_score(self, task: Task, prioritize_deadline: bool) -> int:
        score = task.priority

        keyword = self.CONTEXT_KEYWORDS[self.context]
        if keyword in task.title.lower():
            score += 2

        if prioritize_deadline:
            score += max(0, 3 - max(0, task.deadline - self.time))

        return score

    def _recommendation_reason(self, recommended_action: int) -> str:
        if recommended_action == 3:
            return "All tasks are complete, so skipping is optimal."

        task = self.tasks[recommended_action]
        keyword = self.CONTEXT_KEYWORDS[self.context]
        context_bonus = keyword in task.title.lower()
        if self.difficulty == "easy":
            return (
                f"Easy mode accepts any unfinished task, and '{task.title}' is the next "
                f"recommended option with priority {task.priority} and deadline {task.deadline}."
            )
        if self.difficulty == "medium":
            reason = (
                f"Medium mode favors stronger priority, so '{task.title}' is recommended "
                f"with priority {task.priority} and deadline {task.deadline}."
            )
            if context_bonus:
                reason += f" It also matches the {self.context} context."
            return reason
        return (
            f"Hard mode balances urgency and importance, so '{task.title}' is best "
            f"with priority {task.priority}, deadline {task.deadline}, and time {self.time}."
            + (f" It also directly fits the {self.context} scenario." if context_bonus else "")
        )

    @property
    def state(self) -> State:
        return self._state
