from my_env.graders import (
    GraderManager,
    TaskDeadlineGrader,
    TaskPriorityGrader,
    TaskSchedulingGrader,
)
from my_env.models import MyObservation, Task


def make_observation(task_mode: str) -> MyObservation:
    return MyObservation(
        context="coursework" if task_mode == "task-scheduling" else (
            "exam_week" if task_mode == "task-priority" else "project_crunch"
        ),
        difficulty="easy" if task_mode == "task-scheduling" else (
            "medium" if task_mode == "task-priority" else "hard"
        ),
        task_mode=task_mode,
        tasks=[
            Task(
                title="Task A",
                category="assignment",
                priority=3,
                deadline=1,
                estimated_hours=2,
                done=True,
            ),
            Task(
                title="Task B",
                category="revision",
                priority=2,
                deadline=1,
                estimated_hours=2,
                done=False,
            ),
            Task(
                title="Task C",
                category="writing",
                priority=1,
                deadline=2,
                estimated_hours=1,
                done=False,
            ),
        ],
        time=1,
        time_budget=3,
        steps_remaining=4,
        recommended_action=1,
        objective="Test objective",
        conflict_level=1,
        score_explanation="Test explanation",
        decision_summary="Test summary",
        reward=0.8,
        done=False,
        metadata={},
    )


def test_task_scheduling_grader_returns_valid_score():
    grader = TaskSchedulingGrader()
    obs = make_observation("task-scheduling")
    grader.grade_step(obs, action=0, reward=0.8)
    result = grader.grade_episode(obs)
    assert result["task_name"] == "task-scheduling"
    assert 0.0 < result["final_score"] < 1.0


def test_task_priority_grader_returns_valid_score():
    grader = TaskPriorityGrader()
    obs = make_observation("task-priority")
    grader.grade_step(obs, action=0, reward=0.8)
    result = grader.grade_episode(obs)
    assert result["task_name"] == "task-priority"
    assert 0.0 < result["final_score"] < 1.0


def test_task_deadline_grader_returns_valid_score():
    grader = TaskDeadlineGrader()
    obs = make_observation("task-deadline")
    grader.grade_step(obs, action=0, reward=0.8)
    result = grader.grade_episode(obs)
    assert result["task_name"] == "task-deadline"
    assert 0.0 < result["final_score"] < 1.0


def test_grader_manager_dispatches_to_task_graders():
    manager = GraderManager()
    obs = make_observation("task-priority")
    manager.start_episode("task-priority")
    manager.grade_step(obs, action=1, reward=0.6)
    result = manager.grade_episode(obs)
    assert result["task_name"] == "task-priority"
    assert 0.0 < result["final_score"] < 1.0
