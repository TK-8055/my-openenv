from typing import Literal
from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation

Difficulty = Literal["easy", "medium", "hard"]

class Task(BaseModel):
    title: str
    priority: int
    deadline: int
    done: bool = False

class MyAction(Action):
    action: int = Field(..., ge=0, le=3)

class MyObservation(Observation):
    context: Literal["sports", "student", "doctor"]
    difficulty: Difficulty
    tasks: list[Task]
    time: int
    steps_remaining: int
    recommended_action: int
    score_explanation: str
    decision_summary: str