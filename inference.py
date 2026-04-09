import asyncio
import os
import re
from typing import Optional

from openai import OpenAI
from openenv.core.containers.runtime.uv_provider import UVProvider

try:
    from .client import MyEnv
    from .models import MyAction
except ImportError:
    from client import MyEnv
    from models import MyAction


API_KEY = os.environ["API_KEY"]
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
TASK_NAME = os.getenv("TASK_NAME", "task-scheduling")
BENCHMARK = os.getenv("BENCHMARK", "my_env")
MAX_STEPS = 10
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_value = error if error is not None else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def build_prompt(observation) -> str:
    task_lines = []
    for index, task in enumerate(observation.tasks):
        task_lines.append(
            f"{index}: title={task.title}, priority={task.priority}, "
            f"deadline={task.deadline}, done={str(task.done).lower()}"
        )

    return (
        "Choose the best next action for this task scheduling environment.\n"
        "Return exactly one integer: 0, 1, 2, or 3.\n"
        "Pick the strongest unfinished task using context, difficulty, priority, "
        "deadline, and time remaining.\n\n"
        f"context={observation.context}\n"
        f"difficulty={observation.difficulty}\n"
        f"time={observation.time}\n"
        f"steps_remaining={observation.steps_remaining}\n"
        f"recommended_action={observation.recommended_action}\n"
        f"score_explanation={observation.score_explanation}\n"
        "tasks:\n"
        + "\n".join(task_lines)
    )


def create_openai_client() -> OpenAI:
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
    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Reply with 0 only."}],
            max_tokens=5,
        )
    except Exception as exc:
        raise RuntimeError(f"LLM call failed: {exc}") from exc


def choose_action(client: OpenAI, observation) -> int:
    prompt = build_prompt(observation)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful scheduler. Return exactly one integer "
                        "between 0 and 3 and nothing else."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=5,
        )
    except Exception as exc:
        raise RuntimeError(f"LLM call failed: {exc}") from exc

    content = (response.choices[0].message.content or "").strip()
    match = re.search(r"\b([0-3])\b", content)
    if match is None:
        raise RuntimeError(f"LLM returned invalid action: {content!r}")
    return int(match.group(1))


async def main() -> None:
    client = create_openai_client()
    env = None
    rewards: list[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        smoke_test_llm(client)
        env = await create_env()
        result = await env.reset()

        while steps_taken < MAX_STEPS:
            if result.done:
                break

            step_number = steps_taken + 1
            action_value = choose_action(client, result.observation)
            result = await env.step(MyAction(action=action_value))

            reward = float(result.reward or 0.0)
            done = bool(result.done)
            rewards.append(reward)
            steps_taken = step_number

            log_step(
                step=step_number,
                action=str(action_value),
                reward=reward,
                done=done,
                error=None,
            )

            if done:
                break

        if rewards:
            score = sum(rewards) / len(rewards)
        score = max(0.0, min(1.0, score))
        success = bool(rewards) and score > 0.0
    except Exception:
        success = False
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
