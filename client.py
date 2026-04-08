"""Client for the context-aware task scheduling environment."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import MyAction, MyObservation, Task


class MyEnv(EnvClient[MyAction, MyObservation, State]):
    """Client wrapper for the scheduling environment."""

    def _step_payload(self, action: MyAction) -> Dict:
        return {"action": action.action}

    def _parse_result(self, payload: Dict) -> StepResult[MyObservation]:
        obs_data = payload.get("observation", {})
        task_items = obs_data.get("tasks", [])

        observation = MyObservation(
            context=obs_data["context"],
            difficulty=obs_data["difficulty"],
            tasks=[Task(**task) for task in task_items],
            time=obs_data.get("time", 0),
            steps_remaining=obs_data.get("steps_remaining", 0),
            recommended_action=obs_data.get("recommended_action", 3),
            score_explanation=obs_data.get("score_explanation", ""),
            decision_summary=obs_data.get(
                "decision_summary",
                "No decision summary provided by server.",
            ),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
