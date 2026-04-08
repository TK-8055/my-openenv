from models import MyAction
from server.my_env_environment import MyEnvironment


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
