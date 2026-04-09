import os
import requests

BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://tk8055-my-env.hf.space"
)

def main():
    print("[START]")

    try:
        response = requests.post(f"{BASE_URL}/reset", timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"[END] error=reset_failed {e}", flush=True)
        return

    state = data.get("observation", {})

    for i in range(5):
        tasks = state.get("tasks", [])

        action = 0
        best = -999

        for i, t in enumerate(tasks):
            if not t.get("done"):
                score = t["priority"] - t["deadline"]
                if score > best:
                    best = score
                    action = i

        try:
            response = requests.post(
                f"{BASE_URL}/step",
                json={"action": action},
                timeout=10
            )
            response.raise_for_status()
            res = response.json()
        except Exception as e:
            print(f"[END] error=step_failed {e}", flush=True)
            return

        print(f"[STEP] action={action} reward={res.get('reward')}")

        state = res.get("observation", {})

        if res.get("done"):
            break

    print("[END]")

if __name__ == "__main__":
    main()
