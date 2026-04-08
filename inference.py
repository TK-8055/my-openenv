import requests


BASE_URL = "http://localhost:8000"


def choose_action(state: dict) -> int:
    """Pick the unfinished task with the best priority-vs-urgency tradeoff."""
    best_action = 3
    best_score = float("-inf")

    for i, task in enumerate(state.get("tasks", [])):
        if task.get("done", False):
            continue
        score = task.get("priority", 0) - task.get("deadline", 0)
        if score > best_score:
            best_score = score
            best_action = i

    return best_action


def main():
    print("[START]", flush=True)

    try:
        reset_payload = requests.post(f"{BASE_URL}/reset", timeout=10).json()
    except Exception as e:
        print(f"[END] error=reset_failed {e}", flush=True)
        return

    state = reset_payload.get("observation", reset_payload)
    step_count = 0

    while step_count < 10:
        action = choose_action(state)

        try:
            response = requests.post(
                f"{BASE_URL}/step",
                json={"action": action},
                timeout=10,
            ).json()
        except Exception as e:
            print(f"[STEP] error=step_failed {e}", flush=True)
            break

        reward = response.get("reward", 0.0)
        note = response.get("observation", {}).get("decision_summary", "")
        print(
            f"[STEP] step={step_count} action={action} reward={reward} note={note}",
            flush=True,
        )

        state = response.get("observation", response.get("state", {}))

        if response.get("done", False):
            break

        step_count += 1

    print("[END]", flush=True)


if __name__ == "__main__":
    main()
