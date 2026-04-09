"""Student task scheduling environment implementation."""

from __future__ import annotations

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import Difficulty, MyAction, MyObservation, Task
except ImportError:
    from models import Difficulty, MyAction, MyObservation, Task


class MyEnvironment(Environment):
    """An RL-style environment for planning study work under time pressure."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    TASK_MODES: tuple[str, ...] = (
        "task-scheduling",
        "task-priority",
        "task-deadline",
    )
    MAX_TIME: int = 5
    SCENARIOS: dict[str, dict[str, object]] = {
        "task-scheduling": {
            "context": "coursework",
            "difficulty": "easy",
            "time_budget": 5,
            "objective": (
                "Regular college day: complete core coursework and build momentum "
                "for the rest of the week."
            ),
            "focus_category": "assignment",
            "tasks": (
                {
                    "title": "Finish math homework set",
                    "category": "assignment",
                    "priority": 2,
                    "deadline": 2,
                    "estimated_hours": 2,
                },
                {
                    "title": "Revise biology lecture notes",
                    "category": "revision",
                    "priority": 1,
                    "deadline": 3,
                    "estimated_hours": 1,
                },
                {
                    "title": "Update weekly planner checklist",
                    "category": "planning",
                    "priority": 1,
                    "deadline": 4,
                    "estimated_hours": 1,
                },
            ),
        },
        "task-priority": {
            "context": "exam_week",
            "difficulty": "medium",
            "time_budget": 4,
            "objective": (
                "Exam week pressure: balance revision with near-term coursework "
                "deadlines under limited study hours."
            ),
            "focus_category": "revision",
            "tasks": (
                {
                    "title": "Practice chemistry mock test",
                    "category": "revision",
                    "priority": 3,
                    "deadline": 1,
                    "estimated_hours": 2,
                },
                {
                    "title": "Submit history reflection",
                    "category": "assignment",
                    "priority": 2,
                    "deadline": 1,
                    "estimated_hours": 2,
                },
                {
                    "title": "Prepare tutoring notes",
                    "category": "commitment",
                    "priority": 1,
                    "deadline": 2,
                    "estimated_hours": 1,
                },
            ),
        },
        "task-deadline": {
            "context": "project_crunch",
            "difficulty": "hard",
            "time_budget": 5,
            "objective": (
                "Internship and exams clash: make high-stakes tradeoffs when there "
                "is not enough time to finish every critical task."
            ),
            "focus_category": "project",
            "tasks": (
                {
                    "title": "Capstone project submission",
                    "category": "project",
                    "priority": 3,
                    "deadline": 2,
                    "estimated_hours": 3,
                },
                {
                    "title": "Final exam preparation",
                    "category": "revision",
                    "priority": 3,
                    "deadline": 1,
                    "estimated_hours": 3,
                },
                {
                    "title": "Client meeting preparation",
                    "category": "commitment",
                    "priority": 2,
                    "deadline": 1,
                    "estimated_hours": 2,
                },
                {
                    "title": "Assignment work",
                    "category": "assignment",
                    "priority": 1,
                    "deadline": 2,
                    "estimated_hours": 2,
                },
            ),
        },
    }

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.task_mode = "task-scheduling"
        self.context = "coursework"
        self.difficulty: Difficulty = "easy"
        self.objective = ""
        self.focus_category = "assignment"
        self.tasks: list[Task] = []
        self.time = 0
        self.time_budget = 0
        self.last_action: int | None = None
        self.reset()

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task: str | None = None,
        **kwargs,
    ) -> MyObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        if episode_id is not None:
            self._state.episode_id = episode_id
        selected_task = task or kwargs.get("task_name")
        if isinstance(selected_task, str) and selected_task in self.TASK_MODES:
            self.task_mode = selected_task
        else:
            self.task_mode = "task-scheduling"
        scenario = self.SCENARIOS[self.task_mode]
        self.context = str(scenario["context"])
        self.difficulty = scenario["difficulty"]  # type: ignore[assignment]
        self.objective = str(scenario["objective"])
        self.focus_category = str(scenario["focus_category"])
        self.tasks = [Task(**task_data) for task_data in scenario["tasks"]]  # type: ignore[arg-type]
        self.time = 0
        self.time_budget = int(scenario["time_budget"])
        self.last_action = None
        return self._build_observation(
            reward=None,
            done=False,
            decision_summary="Episode ready. Choose the most valuable task first.",
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
                if self._can_finish(chosen_task):
                    reward = self._score_task_choice(chosen_index, recommended_index)
                    chosen_task.done = True
                    self.time_budget = max(
                        0, self.time_budget - chosen_task.estimated_hours
                    )
                    decision_summary = self._decision_summary(
                        chosen_task=chosen_task,
                        reward=reward,
                        recommended_index=recommended_index,
                    )
                else:
                    decision_summary = (
                        f"'{chosen_task.title}' could not be completed because it "
                        f"needs {chosen_task.estimated_hours} hours but only "
                        f"{self.time_budget} remain."
                    )
            else:
                decision_summary = (
                    f"'{chosen_task.title}' was already completed, so the choice earned no reward."
                )
        else:
            reward = 0.35 if recommended_index == 3 else 0.0
            decision_summary = (
                "Skipping is acceptable because no unfinished task fits the remaining time budget."
                if recommended_index == 3
                else "Skipping was not optimal because unfinished tasks still remain."
            )

        if self.last_action is not None and chosen_index == self.last_action:
            reward -= 0.2
            decision_summary += " Repeating the same action reduced reward."

        reward = self._clamp_reward(reward)
        self.last_action = chosen_index
        self.time += 1
        done = (
            all(task.done for task in self.tasks)
            or self.time >= self.MAX_TIME
            or self.time_budget <= 0
        )
        if done:
            decision_summary += (
                f" | Episode finished in {self.time} steps with "
                f"{self.time_budget} study hours left."
            )
        return self._build_observation(
            reward=reward,
            done=done,
            decision_summary=decision_summary,
            chosen_index=chosen_index,
        )

    def _build_observation(
        self,
        reward: float | None,
        done: bool,
        decision_summary: str,
        chosen_index: int | None,
    ) -> MyObservation:
        recommended_action = self._best_task()
        return MyObservation(
            context=self.context,
            difficulty=self.difficulty,
            task_mode=self.task_mode,
            tasks=self.tasks,
            time=self.time,
            time_budget=self.time_budget,
            steps_remaining=max(0, self.MAX_TIME - self.time),
            recommended_action=recommended_action,
            objective=self.objective,
            conflict_level=self._conflict_level(),
            score_explanation=self._recommendation_reason(recommended_action),
            decision_summary=decision_summary,
            reward=reward,
            done=done,
            metadata={
                "task_mode": self.task_mode,
                "objective": self.objective,
                "best_task_index": recommended_action,
                "time_budget": self.time_budget,
                "completed_tasks": sum(1 for task in self.tasks if task.done),
                "deadline_pressure": [max(0, task.deadline - self.time) for task in self.tasks],
                "comparison": {
                    "chosen": chosen_index if chosen_index is not None else -1,
                    "recommended": recommended_action,
                },
            },
        )

    def _recommended_action(self) -> int:
        return self._best_task()

    def _best_task(self) -> int:
        available = [
            (index, task)
            for index, task in enumerate(self.tasks)
            if not task.done and self._can_finish(task)
        ]
        if not available:
            return 3
        ranked = sorted(
            available,
            key=lambda item: (
                -self._task_score(item[1]),
                item[1].deadline,
                item[1].estimated_hours,
                item[0],
            ),
        )
        return ranked[0][0]

    def _score_task_choice(self, chosen_index: int, recommended_index: int) -> float:
        chosen_task = self.tasks[chosen_index]
        if not self._can_finish(chosen_task):
            return 0.0

        reward = 0.4
        reward += 0.2 * (chosen_task.priority / 3)

        deadline_gap = max(0, chosen_task.deadline - self.time)
        urgency = max(0.0, 0.25 - (0.08 * deadline_gap))
        reward += urgency

        if chosen_task.category == self.focus_category:
            reward += 0.05

        if chosen_index == recommended_index:
            reward += 0.2
        else:
            reward -= 0.15

        return reward

    def _decision_summary(
        self, chosen_task: Task, reward: float, recommended_index: int
    ) -> str:
        context_match = chosen_task.category == self.focus_category
        if reward == 1.0:
            return (
                f"Chose '{chosen_task.title}' because it best supports the "
                f"'{self.objective}' objective, balancing priority "
                f"{chosen_task.priority}, deadline {chosen_task.deadline}, and "
                f"{chosen_task.estimated_hours} study hours."
            )
        if reward >= 0.5:
            return (
                f"Chose '{chosen_task.title}' because it was a reasonable option in "
                f"the {self.context} scenario, but a better tradeoff between urgency "
                f"and workload was still available."
            )
        best_title = (
            self.tasks[recommended_index].title
            if recommended_index < len(self.tasks)
            else "skip"
        )
        if context_match:
            return (
                f"Chose '{chosen_task.title}' because it matches the focus area, "
                f"but '{best_title}' had the stronger deadline or workload fit for "
                f"this step."
            )
        return (
            f"Chose '{chosen_task.title}', but '{best_title}' was the safer study "
            f"choice because it delivered more value before the next deadline."
        )

    def _can_finish(self, task: Task) -> bool:
        return task.estimated_hours <= self.time_budget

    def _clamp_reward(self, reward: float) -> float:
        return round(max(0.0, min(1.0, reward)), 2)

    def _task_score(self, task: Task) -> int:
        urgency = max(0, 4 - max(0, task.deadline - self.time))
        category_bonus = 2 if task.category == self.focus_category else 0
        workload_bonus = max(0, 3 - task.estimated_hours)
        score = (task.priority * 3) + (urgency * 2) + category_bonus + workload_bonus
        return score

    def _conflict_level(self) -> int:
        overdue_risk = sum(1 for task in self.tasks if not task.done and task.deadline <= 2)
        tight_budget = sum(
            1
            for task in self.tasks
            if not task.done and task.estimated_hours >= max(1, self.time_budget)
        )
        return overdue_risk + tight_budget

    def _recommendation_reason(self, recommended_action: int) -> str:
        if recommended_action == 3:
            return (
                "No unfinished task fits the remaining study budget, so skipping is "
                "currently optimal."
            )

        task = self.tasks[recommended_action]
        context_bonus = task.category == self.focus_category
        if self.difficulty == "easy":
            return (
                f"Easy mode rewards steady progress, and '{task.title}' is the best "
                f"next step because it fits within {task.estimated_hours} hours and "
                f"is due in {task.deadline} turn(s)."
            )
        if self.difficulty == "medium":
            reason = (
                f"Medium mode rewards better prioritization, so '{task.title}' is "
                f"recommended with priority {task.priority}, deadline {task.deadline}, "
                f"and workload {task.estimated_hours}."
            )
            if context_bonus:
                reason += " It also matches the current study focus."
            return reason
        return (
            f"Hard mode forces tradeoffs, so '{task.title}' is best because it offers "
            f"the strongest combination of urgency, impact, and workload fit before "
            f"the remaining {self.time_budget} hours run out."
            + (" It also fits the primary scenario objective." if context_bonus else "")
        )

    @property
    def state(self) -> State:
        return self._state
