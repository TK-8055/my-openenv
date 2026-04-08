---
title: Context-Aware Task Scheduling Environment
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - scheduling
---

# Context-Aware Task Scheduling Environment

## What This Project Does

This project builds a small RL-style environment where an agent must decide which task to do next.
The key challenge is that tasks have different priorities and deadlines, and the "best" choice depends on context (`student`, `doctor`, or `sports`).

I designed it to mimic everyday planning decisions: do the urgent important item first, avoid wasting steps, and adapt behavior based on difficulty.

## Problem Framing

At each step, the agent answers one question:
**"Which action gives the best outcome right now?"**

Actions:
- `0`, `1`, `2`: choose one of the three tasks
- `3`: skip

## Observation and Reward

Each observation contains:
- current context
- difficulty level (`easy`, `medium`, `hard`)
- three tasks (`title`, `priority`, `deadline`, `done`)
- current time and steps remaining
- recommended action with a short explanation
- decision summary from the previous step

Reward range is always `[0.0, 1.0]`:
- `1.0`: optimal choice
- `0.5`: reasonable but not best
- `0.1`: valid skip (only when skipping is actually appropriate)
- `0.0`: poor choice

## Difficulty Logic

- `easy`: any unfinished task is acceptable
- `medium`: reward mostly follows priority
- `hard`: reward depends on both urgency (deadline pressure) and importance (priority + context match)

## Local Run

From this folder:

```bash
uv sync
uv run --project . server
```

Server docs:

```text
http://localhost:8000/docs
```

Run baseline inference in another terminal:

```bash
uv run --project . python inference.py
```

Run tests:

```bash
uv run --project . pytest
```

## Deployment

```bash
openenv push --repo-id your-username/my-env
```

Then submit the generated Space URL.

## File Guide

- `server/my_env_environment.py`: environment dynamics and scoring logic
- `server/app.py`: FastAPI/OpenEnv app entrypoint
- `models.py`: action/observation schemas
- `client.py`: client parser/wrapper
- `inference.py`: simple baseline policy script
- `tests/test_environment.py`: basic validation tests

## Current Limitations

- Baseline policy is rule-based, not learned.
- Task templates are fixed-size (3 tasks per episode).
- No long-horizon memory across episodes.
