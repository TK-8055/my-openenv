"""Deterministic graders for the student task scheduling tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod

try:
    from .models import MyObservation
except ImportError:
    from models import MyObservation


class BaseGrader(ABC):
    """Base class for task-specific episode graders."""

    def __init__(self, task_name: str):
        self.task_name = task_name
        self.episode_rewards: list[float] = []

    def reset(self) -> None:
        self.episode_rewards.clear()

    def grade_step(
        self, observation: MyObservation, action: int, reward: float
    ) -> dict[str, float | int]:
        self.episode_rewards.append(reward)
        return {
            "action": action,
            "reward": reward,
            "completed_tasks": sum(1 for task in observation.tasks if task.done),
        }

    @abstractmethod
    def grade_episode(self, final_observation: MyObservation) -> dict[str, float | bool | str]:
        raise NotImplementedError

    def _strict_score(self, value: float) -> float:
        if value <= 0.0:
            return 0.01
        if value >= 1.0:
            return 0.99
        return round(value, 3)


class TaskSchedulingGrader(BaseGrader):
    def __init__(self):
        super().__init__("task-scheduling")

    def grade_episode(self, final_observation: MyObservation) -> dict[str, float | bool | str]:
        completed = sum(1 for task in final_observation.tasks if task.done)
        completion_rate = completed / len(final_observation.tasks)
        avg_reward = (
            sum(self.episode_rewards) / len(self.episode_rewards)
            if self.episode_rewards
            else 0.0
        )
        score = self._strict_score((completion_rate * 0.7) + (avg_reward * 0.3))
        return {
            "task_name": self.task_name,
            "final_score": score,
            "passed": score >= 0.5,
        }


class TaskPriorityGrader(BaseGrader):
    def __init__(self):
        super().__init__("task-priority")

    def grade_episode(self, final_observation: MyObservation) -> dict[str, float | bool | str]:
        total_priority = sum(task.priority for task in final_observation.tasks)
        completed_priority = sum(
            task.priority for task in final_observation.tasks if task.done
        )
        priority_score = completed_priority / total_priority if total_priority else 0.0
        avg_reward = (
            sum(self.episode_rewards) / len(self.episode_rewards)
            if self.episode_rewards
            else 0.0
        )
        score = self._strict_score((priority_score * 0.75) + (avg_reward * 0.25))
        return {
            "task_name": self.task_name,
            "final_score": score,
            "passed": score >= 0.5,
        }


class TaskDeadlineGrader(BaseGrader):
    def __init__(self):
        super().__init__("task-deadline")

    def grade_episode(self, final_observation: MyObservation) -> dict[str, float | bool | str]:
        urgent_tasks = [task for task in final_observation.tasks if task.deadline <= 1]
        urgent_total = len(urgent_tasks)
        urgent_completed = sum(1 for task in urgent_tasks if task.done)
        urgent_score = urgent_completed / urgent_total if urgent_total else 0.0
        avg_reward = (
            sum(self.episode_rewards) / len(self.episode_rewards)
            if self.episode_rewards
            else 0.0
        )
        score = self._strict_score((urgent_score * 0.8) + (avg_reward * 0.2))
        return {
            "task_name": self.task_name,
            "final_score": score,
            "passed": score >= 0.5,
        }


class GraderManager:
    """Simple task-to-grader dispatcher."""

    def __init__(self):
        self.graders = {
            "task-scheduling": TaskSchedulingGrader(),
            "task-priority": TaskPriorityGrader(),
            "task-deadline": TaskDeadlineGrader(),
        }
        self.current_grader: BaseGrader | None = None

    def start_episode(self, task_name: str) -> None:
        if task_name not in self.graders:
            raise ValueError(f"Unknown task: {task_name}")
        self.current_grader = self.graders[task_name]
        self.current_grader.reset()

    def grade_step(self, observation: MyObservation, action: int, reward: float) -> dict[str, float | int]:
        if self.current_grader is None:
            raise RuntimeError("No active grader. Call start_episode first.")
        return self.current_grader.grade_step(observation, action, reward)

    def grade_episode(self, final_observation: MyObservation) -> dict[str, float | bool | str]:
        if self.current_grader is None:
            raise RuntimeError("No active grader. Call start_episode first.")
        return self.current_grader.grade_episode(final_observation)
