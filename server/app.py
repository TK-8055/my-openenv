"""FastAPI application for the My Env environment."""

from collections import defaultdict, deque
import json
from time import monotonic

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies first."
    ) from e

try:
    from ..models import MyAction, MyObservation
    from .my_env_environment import MyEnvironment
except ImportError:
    from models import MyAction, MyObservation
    from server.my_env_environment import MyEnvironment


app = create_app(
    MyEnvironment,
    MyAction,
    MyObservation,
    env_name="my_env",
    max_concurrent_envs=1,
)

RATE_LIMIT_WINDOW_SECONDS = 10.0
RATE_LIMIT_MAX_REQUESTS = 20
_request_windows: dict[str, deque[float]] = defaultdict(deque)


@app.middleware("http")
async def limit_reset_and_step_requests(request: Request, call_next):
    """Apply lightweight per-IP throttling for expensive environment calls."""
    if request.method == "POST" and request.url.path == "/reset":
        body = await request.body()
        payload: dict | None = None
        if body:
            try:
                decoded = json.loads(body)
                if isinstance(decoded, dict):
                    payload = decoded
            except json.JSONDecodeError:
                payload = None
        if payload is None:
            payload = {}

        query_task = request.query_params.get("task")
        query_task_name = request.query_params.get("task_name")
        selected_task = query_task_name or query_task
        if selected_task:
            payload["task"] = selected_task
            payload["task_name"] = selected_task

        normalized_body = json.dumps(payload).encode("utf-8")

        async def receive_reset() -> dict:
            return {
                "type": "http.request",
                "body": normalized_body,
                "more_body": False,
            }

        request._body = normalized_body
        request._receive = receive_reset

    if request.method == "POST" and request.url.path == "/step":
        body = await request.body()
        if body:
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                payload = None

            if isinstance(payload, dict):
                action = payload.get("action")
                if isinstance(action, int) and not isinstance(action, bool):
                    payload["action"] = {"action": action}
                    normalized_body = json.dumps(payload).encode("utf-8")

                    async def receive() -> dict:
                        return {
                            "type": "http.request",
                            "body": normalized_body,
                            "more_body": False,
                        }

                    request._body = normalized_body
                    request._receive = receive

    if request.method == "POST" and request.url.path in {"/reset", "/step"}:
        now = monotonic()
        key = request.client.host if request.client else "unknown"
        window = _request_windows[key]

        while window and now - window[0] > RATE_LIMIT_WINDOW_SECONDS:
            window.popleft()

        if len(window) >= RATE_LIMIT_MAX_REQUESTS:
            return JSONResponse(
                {
                    "error": "rate_limited",
                    "message": "Too many requests. Please slow down and try again.",
                },
                status_code=429,
            )

        window.append(now)

    return await call_next(request)


def _homepage_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Student Task Scheduler AI</title>
  <style>
    :root {
      --bg-top: #f4f8ff;
      --bg-bottom: #e6f4eb;
      --card: #ffffff;
      --ink: #17202a;
      --muted: #4f5d6b;
      --accent: #0e7490;
      --accent-soft: #d9f2f7;
      --good: #166534;
      --warn: #92400e;
      --ring: rgba(14, 116, 144, 0.25);
      --radius: 16px;
      --shadow: 0 18px 40px rgba(21, 56, 83, 0.14);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Avenir Next", "Trebuchet MS", "Segoe UI", sans-serif;
      color: var(--ink);
      background: linear-gradient(160deg, var(--bg-top), var(--bg-bottom));
      min-height: 100vh;
    }
    .shell {
      max-width: 1040px;
      margin: 0 auto;
      padding: 28px 18px 34px;
    }
    .hero {
      background: radial-gradient(circle at top right, #bae6fd 0, #e0f2fe 32%, #f0f9ff 56%, #fff 100%);
      border: 1px solid #d9edf6;
      border-radius: 24px;
      box-shadow: var(--shadow);
      padding: 24px 22px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: clamp(1.5rem, 2.8vw, 2.15rem);
      letter-spacing: 0.01em;
    }
    .subtitle {
      color: var(--muted);
      margin: 0;
      line-height: 1.45;
    }
    .controls, .board {
      margin-top: 18px;
      background: var(--card);
      border: 1px solid #dfebef;
      border-radius: var(--radius);
      box-shadow: 0 8px 26px rgba(31, 41, 55, 0.08);
      padding: 16px;
    }
    .guide {
      border: 1px solid #cde3ea;
      border-radius: 12px;
      background: #f4fbfd;
      padding: 12px;
      margin-bottom: 12px;
      line-height: 1.45;
    }
    .guide h2 {
      margin: 0 0 8px;
      font-size: 1rem;
    }
    .guide p {
      margin: 6px 0;
      color: #244354;
      font-size: 0.92rem;
    }
    .status-line {
      margin-top: 10px;
      font-size: 0.86rem;
      color: #27566b;
      font-weight: 700;
    }
    .grid {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      align-items: end;
    }
    label {
      display: block;
      font-weight: 700;
      font-size: 0.9rem;
      margin-bottom: 6px;
      color: #2b3a46;
    }
    select, input, button {
      width: 100%;
      border-radius: 11px;
      border: 1px solid #b6d6de;
      padding: 10px 12px;
      font-size: 0.96rem;
      background: #fff;
      color: #0f1720;
    }
    select:focus, input:focus, button:focus {
      outline: none;
      box-shadow: 0 0 0 4px var(--ring);
      border-color: var(--accent);
    }
    button {
      cursor: pointer;
      font-weight: 700;
      background: linear-gradient(180deg, #1aa0c2, #0e7490);
      color: #fff;
      border: none;
      transition: transform 0.14s ease, filter 0.14s ease;
    }
    button:hover { filter: brightness(1.04); transform: translateY(-1px); }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.5;
      transform: none;
    }
    .actions {
      margin-top: 12px;
      display: grid;
      gap: 9px;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    }
    .metrics {
      margin-top: 10px;
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    }
    .pill {
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 0.78rem;
      font-weight: 700;
      background: var(--accent-soft);
      color: #0d5567;
      width: fit-content;
    }
    .metric {
      border: 1px solid #d9e8ea;
      border-radius: 12px;
      padding: 10px;
      background: #f9fcfd;
    }
    .metric .k {
      display: block;
      font-size: 0.78rem;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .metric .v {
      font-size: 1.15rem;
      font-weight: 800;
      color: #11202d;
    }
    .tasks {
      margin-top: 12px;
      display: grid;
      gap: 10px;
    }
    .task {
      border: 1px solid #d5e7ea;
      border-radius: 12px;
      background: #fff;
      padding: 12px;
    }
    .task.done {
      opacity: 0.72;
      background: #f4fff8;
      border-color: #b9e6c8;
    }
    .task.recommended {
      border-color: #0e7490;
      box-shadow: 0 0 0 3px rgba(14, 116, 144, 0.13);
      background: #f0fbff;
    }
    .task-title {
      font-weight: 700;
      margin-bottom: 4px;
    }
    .task-meta {
      color: var(--muted);
      font-size: 0.88rem;
      line-height: 1.4;
    }
    .log {
      margin-top: 12px;
      border-radius: 12px;
      border: 1px solid #d8e4e7;
      background: #f8fbfc;
      padding: 12px;
      white-space: pre-wrap;
      line-height: 1.4;
      min-height: 80px;
      color: #1f2933;
      font-size: 0.9rem;
    }
    .good { color: var(--good); font-weight: 700; }
    .warn { color: var(--warn); font-weight: 700; }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <h1>Student Task Scheduler AI</h1>
      <p class="subtitle">Run an episode, choose actions, and inspect reward feedback from your OpenEnv FastAPI backend in real time.</p>
    </section>

    <section class="controls">
      <div class="guide">
        <h2>How to give input</h2>
        <p>1. Select a <strong>Scenario</strong> (easy, medium, or hard), then click <strong>Start Episode</strong>.</p>
        <p>Optional: set a <strong>Seed</strong> (number) for reproducible runs.</p>
        <p>2. Click one action button each step: <strong>Action 0/1/2</strong> = choose that task index, <strong>Action 3</strong> = Skip.</p>
        <p>3. Repeat actions until status shows <strong>Done</strong>. Use <strong>Recommended</strong>, <strong>Reward</strong>, and the decision log to understand good choices.</p>
      </div>
      <div class="grid">
        <div>
          <label for="difficulty">Scenario</label>
          <select id="difficulty">
            <option value="task-scheduling">Easy: Regular College Day</option>
            <option value="task-priority">Medium: Exam Week Pressure</option>
            <option value="task-deadline">Hard: Internship + Exams Clash</option>
          </select>
        </div>
        <div>
          <label for="seedInput">Seed (optional)</label>
          <input id="seedInput" type="number" min="0" step="1" placeholder="e.g. 42" />
        </div>
        <div>
          <label>&nbsp;</label>
          <button id="resetBtn">Start Episode</button>
        </div>
      </div>
      <div class="status-line" id="scenarioStatus">Scenario not loaded yet.</div>
      <div class="actions">
        <button class="actionBtn" data-action="0">Action 0</button>
        <button class="actionBtn" data-action="1">Action 1</button>
        <button class="actionBtn" data-action="2">Action 2</button>
        <button class="actionBtn" data-action="3">Action 3 (Skip)</button>
      </div>
    </section>

    <section class="board">
      <span class="pill" id="modePill">No episode yet</span>
      <div class="metrics">
        <div class="metric"><span class="k">Context</span><span class="v" id="contextV">-</span></div>
        <div class="metric"><span class="k">Difficulty</span><span class="v" id="difficultyV">-</span></div>
        <div class="metric"><span class="k">Time</span><span class="v" id="timeV">-</span></div>
        <div class="metric"><span class="k">Time Budget</span><span class="v" id="budgetV">-</span></div>
        <div class="metric"><span class="k">Recommended</span><span class="v" id="recV">-</span></div>
        <div class="metric"><span class="k">Reward</span><span class="v" id="rewardV">-</span></div>
        <div class="metric"><span class="k">Progress</span><span class="v" id="progressV">-</span></div>
        <div class="metric"><span class="k">Live Score</span><span class="v" id="scoreV">-</span></div>
        <div class="metric"><span class="k">Status</span><span class="v" id="doneV">-</span></div>
      </div>
      <div class="tasks" id="tasksBox"></div>
      <div class="log" id="logBox">Press "Start Episode" to load state.</div>
    </section>
  </main>

  <script>
    const resetBtn = document.getElementById("resetBtn");
    const actionButtons = [...document.querySelectorAll(".actionBtn")];
    const difficultySel = document.getElementById("difficulty");
    const seedInput = document.getElementById("seedInput");
    const modePill = document.getElementById("modePill");
    const scenarioStatus = document.getElementById("scenarioStatus");
    const tasksBox = document.getElementById("tasksBox");
    const logBox = document.getElementById("logBox");
    const scenarioLabels = {
      "task-scheduling": "Easy (Regular College Day)",
      "task-priority": "Medium (Exam Week Pressure)",
      "task-deadline": "Hard (Internship + Exams Clash)"
    };
    const fields = {
      context: document.getElementById("contextV"),
      difficulty: document.getElementById("difficultyV"),
      time: document.getElementById("timeV"),
      budget: document.getElementById("budgetV"),
      recommended: document.getElementById("recV"),
      reward: document.getElementById("rewardV"),
      progress: document.getElementById("progressV"),
      score: document.getElementById("scoreV"),
      done: document.getElementById("doneV")
    };

    const state = { done: true, lastReward: null, rewards: [], pendingSelection: false };

    function setLoading(active) {
      resetBtn.disabled = active;
      actionButtons.forEach((btn) => {
        btn.disabled = active || state.done || state.pendingSelection;
      });
    }

    function parsePayload(payload) {
      const observation = payload?.observation ?? payload ?? {};
      const rewardValue = payload?.reward ?? observation?.reward ?? null;
      const doneValue = payload?.done ?? observation?.done ?? false;
      return { observation, reward: rewardValue, done: !!doneValue };
    }

    function toSafe(value, fallback = "-") {
      return value === undefined || value === null || value === "" ? fallback : value;
    }

    function render(payload) {
      const { observation, reward, done } = parsePayload(payload);
      state.done = done;
      state.lastReward = reward;
      state.pendingSelection = false;

      modePill.textContent = `Mode: ${toSafe(observation.task_mode)} | Context: ${toSafe(observation.context)}`;
      scenarioStatus.textContent = `Scenario Loaded: ${scenarioLabels[observation.task_mode] || observation.task_mode}`;
      fields.context.textContent = toSafe(observation.context);
      fields.difficulty.textContent = toSafe(observation.difficulty);
      fields.time.textContent = toSafe(observation.time);
      fields.budget.textContent = toSafe(observation.time_budget);
      fields.recommended.textContent = toSafe(observation.recommended_action);
      fields.reward.textContent = reward === null ? "-" : Number(reward).toFixed(2);
      fields.done.innerHTML = done ? '<span class="good">Done</span>' : '<span class="warn">In Progress</span>';

      const tasks = Array.isArray(observation.tasks) ? observation.tasks : [];
      const completed = tasks.filter((task) => task.done).length;
      fields.progress.textContent = `${completed}/${tasks.length || 0}`;
      if (reward !== null && !Number.isNaN(Number(reward))) {
        state.rewards.push(Number(reward));
      }
      const avgScore = state.rewards.length
        ? state.rewards.reduce((acc, value) => acc + value, 0) / state.rewards.length
        : null;
      fields.score.textContent = avgScore === null ? "-" : avgScore.toFixed(2);

      tasksBox.innerHTML = tasks.map((task, idx) => {
        const doneClass = task.done ? "done" : "";
        const recommendedClass = idx === observation.recommended_action && !task.done ? "recommended" : "";
        const doneText = task.done ? "complete" : "pending";
        return `
          <article class="task ${doneClass} ${recommendedClass}">
            <div class="task-title">${idx}. ${task.title}${recommendedClass ? " (recommended)" : ""}</div>
            <div class="task-meta">
              category=${task.category} | priority=${task.priority} | deadline=${task.deadline} | hours=${task.estimated_hours} | ${doneText}
            </div>
          </article>
        `;
      }).join("");

      const decision = toSafe(observation.decision_summary, "No decision summary.");
      const explain = toSafe(observation.score_explanation, "No explanation.");
      logBox.textContent = `Decision: ${decision}\nWhy: ${explain}`;
      setLoading(false);
    }

    const REQUEST_TIMEOUT_MS = 12000;

    async function callApi(path, body) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
      const response = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal
      }).finally(() => clearTimeout(timer));
      if (!response.ok) {
        const text = await response.text();
        throw new Error(`HTTP ${response.status}: ${text}`);
      }
      return response.json();
    }

    function markScenarioChanged() {
      state.pendingSelection = true;
      state.done = true;
      scenarioStatus.textContent = `Scenario selected: ${scenarioLabels[difficultySel.value] || difficultySel.value}. Click Start Episode to load it.`;
      logBox.textContent = "Scenario changed. Click Start Episode to begin a new episode.";
      setLoading(false);
    }

    async function startEpisode() {
      setLoading(true);
      state.rewards = [];
      logBox.textContent = "Resetting environment...";
      try {
        const body = {};
        const seedText = seedInput.value.trim();
        if (seedText !== "") {
          const seedValue = Number(seedText);
          if (!Number.isInteger(seedValue) || seedValue < 0) {
            throw new Error("Seed must be a non-negative integer.");
          }
          body.seed = seedValue;
        }
        const selected = encodeURIComponent(difficultySel.value);
        const payload = await callApi(
          `/reset?task=${selected}&task_name=${selected}`,
          body
        );
        const { observation } = parsePayload(payload);
        if (observation.task_mode !== difficultySel.value) {
          state.pendingSelection = true;
          state.done = true;
          setLoading(false);
          scenarioStatus.textContent = "Scenario mismatch detected. Please click Start Episode again.";
          logBox.textContent = `Mismatch detected: selected=${difficultySel.value}, loaded=${observation.task_mode || "unknown"}`;
          return;
        }
        render(payload);
      } catch (err) {
        state.done = true;
        setLoading(false);
        logBox.textContent = `Reset failed: ${err.message}`;
      }
    }

    async function sendAction(action) {
      if (state.done) return;
      setLoading(true);
      logBox.textContent = `Submitting action ${action}...`;
      try {
        const payload = await callApi("/step", {
          action: { action: Number(action) }
        });
        render(payload);
      } catch (err) {
        setLoading(false);
        logBox.textContent = `Step failed: ${err.message}`;
      }
    }

    resetBtn.addEventListener("click", startEpisode);
    difficultySel.addEventListener("change", markScenarioChanged);
    actionButtons.forEach((btn) => {
      btn.addEventListener("click", () => sendAction(btn.dataset.action));
      btn.disabled = true;
    });
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    return HTMLResponse(_homepage_html())


@app.get("/status", response_class=JSONResponse)
def status() -> JSONResponse:
    return JSONResponse({"status": "ok"})


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Run the local development server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if args.port == 7860:
        main()
    else:
        main(port=args.port)
