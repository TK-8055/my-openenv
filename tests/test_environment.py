from fastapi.testclient import TestClient

from my_env.models import MyAction
from my_env.server.my_env_environment import MyEnvironment
from server.app import app


def test_reset_observation_shape():
    env = MyEnvironment()
    obs = env.reset()

    assert obs.context in {"coursework", "exam_week", "project_crunch"}
    assert obs.difficulty in {"easy", "medium", "hard"}
    assert obs.task_mode in {"task-scheduling", "task-priority", "task-deadline"}
    assert len(obs.tasks) == 3
    assert obs.time == 0
    assert obs.time_budget > 0
    assert 0 <= obs.steps_remaining <= env.MAX_TIME
    assert obs.recommended_action in {0, 1, 2, 3}
    assert obs.objective
    assert obs.conflict_level >= 0


def test_step_reward_is_bounded():
    env = MyEnvironment()
    env.reset()

    for action in [0, 1, 2, 3]:
        obs = env.step(MyAction(action=action))
        assert 0.0 <= (obs.reward or 0.0) <= 1.0


def test_recommended_action_valid():
    env = MyEnvironment()
    obs = env.reset()

    assert 0 <= obs.recommended_action <= 3


def test_task_modes_map_to_distinct_scenarios():
    env = MyEnvironment()

    scheduling = env.reset(task="task-scheduling")
    assert scheduling.context == "coursework"
    assert scheduling.difficulty == "easy"

    priority = env.reset(task="task-priority")
    assert priority.context == "exam_week"
    assert priority.difficulty == "medium"

    deadline = env.reset(task="task-deadline")
    assert deadline.context == "project_crunch"
    assert deadline.difficulty == "hard"


def test_done_flag():
    env = MyEnvironment()
    env.reset()

    for _ in range(10):
        obs = env.step(MyAction(action=0))

    assert obs.done is True


def test_skip_behavior():
    env = MyEnvironment()
    env.reset()

    obs = env.step(MyAction(action=3))

    assert 0.0 <= (obs.reward or 0.0) <= 1.0


def test_tasks_mark_done():
    env = MyEnvironment()
    env.reset()

    obs = env.step(MyAction(action=0))

    assert any(t.done for t in obs.tasks)


def test_optimal_action_gives_high_reward():
    env = MyEnvironment()
    obs = env.reset()

    best = obs.recommended_action
    obs = env.step(MyAction(action=best))

    assert (obs.reward or 0.0) >= 0.7


def test_repeating_same_action_gets_penalized():
    env = MyEnvironment()
    obs = env.reset(task="task-priority")

    first = env.step(MyAction(action=0))
    second = env.step(MyAction(action=0))

    assert (second.reward or 0.0) <= (first.reward or 0.0)


def test_completed_task_reduces_time_budget():
    env = MyEnvironment()
    obs = env.reset(task="task-scheduling")
    starting_budget = obs.time_budget

    obs = env.step(MyAction(action=obs.recommended_action))

    assert obs.time_budget < starting_budget


def test_full_episode_runs():
    env = MyEnvironment()
    obs = env.reset()

    for _ in range(10):
        obs = env.step(MyAction(action=obs.recommended_action))
        if obs.done:
            break

    assert obs.done is True


def test_http_reset_accepts_empty_body():
    client = TestClient(app)
    response = client.post("/reset", json={})

    assert response.status_code == 200
    assert "observation" in response.json()
