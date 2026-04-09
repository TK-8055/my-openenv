---
title: My Env
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
app_file: server/app.py
pinned: false
---

# Student Task Scheduling Environment

A small OpenEnv environment where an agent helps a student decide what to work on next when coursework, exams, and project deadlines compete for limited study time.
This environment evaluates decision-making under resource constraints, where agents must choose which tasks to sacrifice when full completion is impossible.

## Overview

This project simulates real student planning across three benchmark tasks:

- `task-scheduling`: easy regular-college-day planning with low conflict
- `task-priority`: medium exam-week tradeoffs under limited study time
- `task-deadline`: hard internship-plus-exam clash where sacrifice is required
The hard scenario is intentionally unsolvable, forcing agents to prioritize high-impact tasks rather than attempting full completion.

At each step, the agent chooses one of three tasks or skips. The environment returns:

- the updated observation
- a reward
- an explanation of why the action was good or bad

We score decisions using deterministic reward shaping based on priority, deadline pressure, workload fit, and task relevance to the scenario objective. The environment exposes both a recommended action and a natural-language explanation so the grader stays interpretable.

## Action Space

- `0`, `1`, `2`: choose one of the available tasks
- `3`: skip

API example:

```json
{
  "action": {
    "action": 1
  }
}
```

Compatibility note: the server also accepts a simplified payload `{ "action": 1 }` and normalizes it internally.

## Observation

Each observation includes:

- current `context`
- current `difficulty`
- current `task_mode`
- scenario `objective`
- `tasks` with title, category, priority, deadline, estimated hours, and completion status
- current `time`
- remaining `time_budget`
- `conflict_level`
- `steps_remaining`
- `recommended_action`
- `score_explanation`
- `decision_summary`
- `metadata.deadline_pressure`
- `metadata.comparison` with chosen vs recommended action

## Reward Logic

Reward is always clamped to `[0.0, 1.0]` and reflects partial progress instead of only terminal success.

- completion value (+0.4 when a feasible task is completed)
- priority contribution (+0.2 * priority/3)
- urgency bonus (higher reward for near-deadline work)
- recommendation bonus/penalty (+0.2 for recommended action, `-0.15` otherwise)
- anti-loop penalty (`-0.2` for repeating the same action on consecutive steps)

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
context = "exam_week"
tasks = [
  {"title": "Practice chemistry mock test", "priority": 3, "deadline": 1, "estimated_hours": 2, "done": false},
  {"title": "Submit history reflection", "priority": 2, "deadline": 1, "estimated_hours": 2, "done": false},
  {"title": "Prepare tutoring notes", "priority": 1, "deadline": 2, "estimated_hours": 1, "done": false}
]

recommended_action = 0
action = 0
reward = 1.0
```

Why: the selected task is urgent, high-priority, and aligned with the scenario objective.

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
http://localhost:7860/docs
```

If `uv` is not installed, use the existing virtual environment:

```bash
cd /home/tk/Desktop/hack/my_env
.venv/bin/python -m server.app --port 7860
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

- `GET /` serves the interactive reviewer-friendly UI
- `GET /status` returns a lightweight health payload: `{"status":"ok"}`
- `POST /reset` starts a new episode
- `POST /step` accepts OpenEnv shape `{ "action": {"action": 0} }` and also supports `{ "action": 0 }`
- `GET /docs` provides the Swagger UI

## Current Strengths

- real-world student planning domain
- three deterministic benchmark tasks with increasing real-world pressure
- interpretable reward explanations and scenario objectives
- deadline pressure plus anti-loop behavior modeling
- baseline inference policy aligned with environment scoring

## Limitations

- rule-based baseline, not a learned policy
- fixed task count per episode
- no long-term memory across episodes
