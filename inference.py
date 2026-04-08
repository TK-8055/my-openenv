import requests

BASE_URL = "http://localhost:8000"

def main():
    print("[START]")

    state = requests.post(f"{BASE_URL}/reset").json().get("observation", {})

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

        res = requests.post(f"{BASE_URL}/step", json={"action": action}).json()

        print(f"[STEP] action={action} reward={res.get('reward')}")

        state = res.get("observation", {})

        if res.get("done"):
            break

    print("[END]")

if __name__ == "__main__":
    main()