from typing import Literal
from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation

Difficulty = Literal["easy", "medium", "hard"]
TaskMode = Literal["task-scheduling", "task-priority", "task-deadline"]
ScenarioContext = Literal["coursework", "exam_week", "project_crunch"]

class Task(BaseModel):
    title: str
    category: str
    priority: int
    deadline: int
    estimated_hours: int
    done: bool = False

class MyAction(Action):
    action: int = Field(..., ge=0, le=3)

class MyObservation(Observation):
    context: ScenarioContext
    difficulty: Difficulty
    task_mode: TaskMode
    tasks: list[Task]
    time: int
    time_budget: int
    steps_remaining: int
    recommended_action: int
    objective: str
    conflict_level: int
    score_explanation: str
    decision_summary: str
