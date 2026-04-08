# Context-Aware Task Scheduling Environment

A small OpenEnv environment where an agent chooses the best next task based on context, priority, and urgency.

## Overview

This project simulates real-world task selection in three scenarios:

- `student`
- `doctor`
- `sports`

At each step, the agent chooses one of three tasks or skips. The environment returns:

- the updated observation
- a reward
- an explanation of why the action was good or bad

We use a composite scoring heuristic combining priority, context relevance, and urgency. We explicitly model deadline pressure to simulate urgency. We don't just reward actions, we explain them, making the system interpretable.

## Action Space

- `0`, `1`, `2`: choose one of the available tasks
- `3`: skip

API example:

```json
{
  "action": 1
}
```

## Observation

Each observation includes:

- current `context`
- current `difficulty`
- `tasks` with title, priority, deadline, and completion status
- current `time`
- `steps_remaining`
- `recommended_action`
- `score_explanation`
- `decision_summary`
- `metadata.deadline_pressure`
- `metadata.comparison` with chosen vs recommended action

## Reward Logic

Reward is always in `[0.0, 1.0]`.

- `1.0`: optimal action
- `0.5`: acceptable but not best action in easier settings
- `0.1`: valid skip when all tasks are complete
- `0.0`: poor action

Difficulty affects strictness:

- `easy`: any unfinished task is partly acceptable
- `medium`: better task quality matters more
- `hard`: only the best action earns full reward

## RL Loop

```text
State -> Agent -> Action -> Reward -> Next State
```

## Example

```text
context = "doctor"
tasks = [
  {"title": "critical patient review", "priority": 5, "deadline": 1, "done": false},
  {"title": "follow-up rounds", "priority": 3, "deadline": 3, "done": false},
  {"title": "paperwork", "priority": 1, "deadline": 5, "done": false}
]

recommended_action = 0
action = 0
reward = 1.0
```

Why: the selected task matches the doctor context and has the strongest priority/deadline tradeoff.

## Project Structure

```text
my_env/
├── client.py
├── inference.py
├── models.py
├── pyproject.toml
├── server/
│   ├── app.py
│   ├── my_env_environment.py
│   └── requirements.txt
└── tests/
    └── test_environment.py
```

## Run Locally

If `uv` is installed:

```bash
cd /home/tk/Desktop/hack/my_env
uv sync --extra dev
uv run --project . server
```

Then open:

```text
http://localhost:8000/docs
```

If `uv` is not installed, use the existing virtual environment:

```bash
cd /home/tk/Desktop/hack/my_env
.venv/bin/python -m server.app
```

## Run Inference

In a second terminal:

```bash
cd /home/tk/Desktop/hack/my_env
.venv/bin/python inference.py
```

## Run Tests

With `uv`:

```bash
cd /home/tk/Desktop/hack/my_env
uv sync --extra dev
uv run --project . pytest
```

Without `uv`:

```bash
cd /home/tk/Desktop/hack/my_env
.venv/bin/python -m pip install pytest pytest-cov
.venv/bin/python -m pytest
```

## API Notes

- `POST /reset` starts a new episode
- `POST /step` expects a plain integer field like `{ "action": 0 }`
- `GET /docs` provides the Swagger UI

## Current Strengths

- interpretable reward explanations
- context-aware scheduling logic
- deadline pressure modeling
- recommended vs chosen action comparison
- baseline inference policy aligned with environment scoring

## Limitations

- rule-based baseline, not a learned policy
- fixed task count per episode
- no long-term memory across episodes
