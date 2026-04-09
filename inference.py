import requests
import os
from openai import OpenAI

API_KEY = os.environ["API_KEY"]
API_BASE_URL = os.environ["API_BASE_URL"]
BASE_URL = API_BASE_URL

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def choose_action(state):
    best = 3
    best_score = float("-inf")

    for i, task in enumerate(state.get("tasks", [])):
        if task.get("done"):
            continue

        score = task.get("priority", 0) - task.get("deadline", 0)
        if score > best_score:
            best_score = score
            best = i

    return best

def main():
    print("[START]", flush=True)

    try:
        res = requests.post(f"{BASE_URL}/reset", timeout=10)
        res.raise_for_status()
        data = res.json()
    except Exception as e:
        print(f"[END] error=reset_failed {e}", flush=True)
        return

    state = data.get("observation", {})
    step = 0

    while step < 10:
        action = choose_action(state)

        try:
            res = requests.post(
                f"{BASE_URL}/step",
                json={"action": action},
                timeout=10
            )
            res.raise_for_status()
            data = res.json()
        except Exception as e:
            print(f"[STEP] error=step_failed {e}", flush=True)
            break

        reward = data.get("reward", 0.0)
        note = data.get("observation", {}).get("decision_summary", "")
        print(
            f"[STEP] step={step} action={action} reward={reward} note={note}",
            flush=True
        )

        state = data.get("observation", {})
        if data.get("done"):
            break

        step += 1

    _ = (client, MODEL_NAME, BASE_URL)
    print("[END]", flush=True)

if __name__ == "__main__":
    main()
