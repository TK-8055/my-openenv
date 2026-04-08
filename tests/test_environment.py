from my_env.models import MyAction
from my_env.server.my_env_environment import MyEnvironment


def test_reset_observation_shape():
    env = MyEnvironment()
    obs = env.reset()

    assert obs.context in {"sports", "student", "doctor"}
    assert obs.difficulty in {"easy", "medium", "hard"}
    assert len(obs.tasks) == 3
    assert obs.time == 0
    assert 0 <= obs.steps_remaining <= env.MAX_TIME
    assert obs.recommended_action in {0, 1, 2, 3}


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

    assert obs.reward == 1.0


def test_full_episode_runs():
    env = MyEnvironment()
    obs = env.reset()

    for _ in range(10):
        obs = env.step(MyAction(action=obs.recommended_action))
        if obs.done:
            break

    assert obs.done is True
