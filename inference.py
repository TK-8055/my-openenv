#!/usr/bin/env python3

import asyncio
import os
import re
from typing import List, Optional

from openai import OpenAI
from openenv.core.containers.runtime.uv_provider import UVProvider

try:
    from .client import MyEnv
    from .models import MyAction
    from .graders import GraderManager
except ImportError:
    from client import MyEnv
    from models import MyAction
    from graders import GraderManager


API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = os.getenv("BENCHMARK", "my_env")
TASKS = ["task-scheduling", "task-priority", "task-deadline"]
MAX_STEPS = 10
SUCCESS_SCORE_THRESHOLD = 0.5
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_prompt(observation) -> str:
    task_lines = []
    for index, task in enumerate(observation.tasks):
        task_lines.append(
            f"{index}: title={task.title}, category={task.category}, "
            f"priority={task.priority}, deadline={task.deadline}, "
            f"estimated_hours={task.estimated_hours}, done={str(task.done).lower()}"
        )

    return (
        "Choose the best next action for this student planning environment.\n"
        "Return exactly one integer: 0, 1, 2, or 3.\n"
        "Pick the strongest unfinished task using the study objective, priority, "
        "deadline pressure, remaining hours, and workload fit.\n\n"
        f"task_mode={observation.task_mode}\n"
        f"context={observation.context}\n"
        f"difficulty={observation.difficulty}\n"
        f"objective={observation.objective}\n"
        f"time={observation.time}\n"
        f"time_budget={observation.time_budget}\n"
        f"steps_remaining={observation.steps_remaining}\n"
        f"conflict_level={observation.conflict_level}\n"
        f"recommended_action={observation.recommended_action}\n"
        f"score_explanation={observation.score_explanation}\n"
        "tasks:\n"
        + "\n".join(task_lines)
    )


def create_openai_client() -> OpenAI:
    if not API_KEY:
        raise RuntimeError("Missing API_KEY or HF_TOKEN")
    if not API_BASE_URL:
        raise RuntimeError("Missing API_BASE_URL")
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


async def create_env():
    if LOCAL_IMAGE_NAME:
        return await MyEnv.from_docker_image(LOCAL_IMAGE_NAME)

    provider = UVProvider(project_path=PROJECT_PATH)
    base_url = provider.start()
    provider.wait_for_ready()
    env = MyEnv(base_url=base_url, provider=provider)
    await env.connect()
    return env


def smoke_test_llm(client: OpenAI) -> None:
    client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Reply with 0 only."}],
        max_tokens=5,
    )


def get_model_action(client: OpenAI, observation) -> int:
    prompt = build_prompt(observation)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful student-planning assistant. Return exactly "
                    "one integer between 0 and 3 and nothing else."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=5,
    )

    text = (completion.choices[0].message.content or "").strip()
    match = re.search(r"\b([0-3])\b", text)
    if match:
        return int(match.group(1))
    return int(observation.recommended_action)


def strict_score(value: float) -> float:
    if value <= 0.0:
        return 0.01
    if value >= 1.0:
        return 0.99
    return round(value, 3)


async def run_task(client: OpenAI, env, grader_manager: GraderManager, task_name: str, seed: int) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.01
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    grader_manager.start_episode(task_name)

    try:
        result = await env.reset(task=task_name, seed=seed)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_str = "null"
            error: Optional[str] = None
            done = bool(result.done)

            try:
                action_value = get_model_action(client, result.observation)
                action_str = str(action_value)
                result = await env.step(MyAction(action=action_value))
                reward = float(result.reward or 0.0)
                grader_manager.grade_step(result.observation, action_value, reward)
                done = bool(result.done)
            except Exception as exc:
                reward = 0.0
                error = str(exc)
                done = True

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        if steps_taken > 0:
            episode_grade = grader_manager.grade_episode(result.observation)
            score = strict_score(float(episode_grade["final_score"]))
            success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception:
        success = False
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    client = None
    env = None
    grader_manager = GraderManager()

    try:
        client = create_openai_client()
        smoke_test_llm(client)
        env = await create_env()

        for index, task_name in enumerate(TASKS, start=1):
            await run_task(client, env, grader_manager, task_name, index)
    except Exception:
        for task_name in TASKS:
            log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.010, rewards=[])
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        for task_name in TASKS:
            log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.010, rewards=[])
