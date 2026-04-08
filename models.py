"""Data models for the context-aware task scheduling environment."""

from typing import Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


Difficulty = Literal["easy", "medium", "hard"]


class Task(BaseModel):
    """A task the agent can choose to complete."""

    title: str = Field(..., description="Human-readable task name")
    priority: int = Field(..., ge=1, le=5, description="Importance of the task")
    deadline: int = Field(..., ge=1, le=5, description="Latest preferred time step")
    done: bool = Field(default=False, description="Whether the task is complete")


class MyAction(Action):
    """Agent action for the environment."""

    action: int = Field(..., ge=0, le=3, description="0-2 selects a task, 3 skips")


class MyObservation(Observation):
    """Current environment observation."""

    context: Literal["sports", "student", "doctor"] = Field(
        ..., description="Current scenario type"
    )
    difficulty: Difficulty = Field(..., description="Difficulty level for this episode")
    tasks: list[Task] = Field(..., description="Current task list")
    time: int = Field(..., ge=0, description="Current environment time step")
    steps_remaining: int = Field(..., ge=0, description="Remaining steps before timeout")
    recommended_action: int = Field(
        ..., ge=0, le=3, description="Best action under the current scoring rule"
    )
    score_explanation: str = Field(
        ..., description="Short explanation of why the best action is recommended"
    )
    decision_summary: str = Field(
        ..., description="Short explanation of the most recent action outcome"
    )
