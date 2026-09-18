"""Deploy Core V2 to an isolated Vercel project via REST API.

Reads secrets ONLY from agent-core/.env.local (gitignored, never printed,
never committed). Required names:
  VERCEL_DEPLOY_TOKEN  new-account Vercel token (REQUIRED to deploy)
  OPENROUTER_API_KEY   real Luna calls (REQUIRED for live tests)
  TURSO_DATABASE_URL   durable V2 state (else ephemeral /tmp fallback)
  TURSO_AUTH_TOKEN     durable V2 state (else ephemeral /tmp fallback)
Optional:
  AGENT_MODEL          default: openai/gpt-5.6-luna
  VERCEL_PROJECT_NAME  default: scaliffy-agent-core-v2

Usage:  python tools/deploy_vercel.py [--skip-deploy]
Prints only: project id/name, deployment id/url, health JSON.
NEVER prints secret values.

Needs: network access to api.vercel.com.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(ROOT, ".env.local")
PROJECT_DEFAULT = "scaliffy-agent-core-v2"
MODEL_DEFAULT = "openai/gpt-5.6-luna"
API = "https://api.vercel.com"

RUNTIME_FILES = ("api", "src", "requirements.txt", "vercel.json")
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".vercel", ".git"}
SKIP_EXT = {".pyc", ".pyo"}


def fail_missing(name: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"MISSING_ENV_NAME={name}", flush=True)
    raise SystemExit(2)


def load_dotenv(path: str) -> dict:
    vals: dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value:
                    vals[key] = value
    except FileNotFoundError:
        pass
    return vals


def api(method: str, path: str, token: str, payload: object = None) -> tuple[int, object]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(API + path, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8")[:500]
        except Exception:
            body = ""
        return exc.code, {"http_error": exc.code, "body": body}


def collect_files() -> list[dict]:
    out: list[dict] = []
    for top in RUNTIME_FILES:
        full = os.path.join(ROOT, top)
        if os.path.isfile(full):
            with open(full, "rb") as fh:
                out.append({
                    "file": top.replace(os.sep, "/"),
                    "data": base64.b64encode(fh.read()).decode("ascii"),
                    "encoding": "base64",
                })
        elif os.path.isdir(full):
            for dirpath, dirnames, filenames in os.walk(full):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
                for name in filenames:
                    if any(name.endswith(ext) for ext in SKIP_EXT):
                        continue
                    if name.startswith(".env"):
                        continue
                    path = os.path.join(dirpath, name)
                    rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
                    with open(path, "rb") as fh:
                        out.append({
                            "file": rel,
                            "data": base64.b64encode(fh.read()).decode("ascii"),
                            "encoding": "base64",
                        })
    return out


def logs_mode(dep_id: str) -> None:
    secrets = load_dotenv(ENV_FILE)
    token = secrets.get("VERCEL_DEPLOY_TOKEN", "")
    if not token:
        fail_missing("VERCEL_DEPLOY_TOKEN")
    status, events = api(
        "GET", f"/v2/deployments/{dep_id}/events?direction=forward&limit=1000",
        token,
    )
    print(f"EVENTS_STATUS={status}", flush=True)
    if not isinstance(events, list):
        print(f"EVENTS_BODY={str(events)[:500]}", flush=True)
        return
    for item in events:
        if not isinstance(item, dict):
            continue
        payload = item.get("payload", {})
        text = payload.get("text", "") if isinstance(payload, dict) else ""
        if text:
            for line in str(text).splitlines():
                print(f"LOG|{line[:400]}", flush=True)


def main() -> None:
    if "--logs" in sys.argv:
        idx = sys.argv.index("--logs")
        if idx + 1 >= len(sys.argv):
            print("LOGS_USAGE=deploy_vercel.py --logs DEPLOYMENT_ID", flush=True)
            raise SystemExit(2)
        logs_mode(sys.argv[idx + 1])
        return
    if "--unprotect" in sys.argv:
        secrets = load_dotenv(ENV_FILE)
        token = secrets.get("VERCEL_DEPLOY_TOKEN", "")
        if not token:
            fail_missing("VERCEL_DEPLOY_TOKEN")
        project_name = secrets.get("VERCEL_PROJECT_NAME", "") or PROJECT_DEFAULT
        st, proj = api("GET", f"/v9/projects/{project_name}", token)
        pid = str(proj.get("id")) if isinstance(proj, dict) else ""
        st2, body = api("PATCH", f"/v9/projects/{pid}", token, {"ssoProtection": None})
        print(f"UNPROTECT status={st2}", flush=True)
        return
    skip_deploy = "--skip-deploy" in sys.argv
    secrets = load_dotenv(ENV_FILE)
    token = secrets.get("VERCEL_DEPLOY_TOKEN", "")
    if not token:
        fail_missing("VERCEL_DEPLOY_TOKEN")
    project_name = secrets.get("VERCEL_PROJECT_NAME", "") or PROJECT_DEFAULT

    # 1. Find or create project.
    status, proj = api("GET", f"/v9/projects/{project_name}", token)
    if status == 200 and isinstance(proj, dict) and proj.get("id"):
        project_id = str(proj["id"])
        print(f"PROJECT=exists id={project_id} name={proj.get('name')}", flush=True)
    elif status == 404:
        status, proj = api("POST", "/v10/projects", token, {"name": project_name})
        if status not in (200, 201) or not isinstance(proj, dict) or not proj.get("id"):
            print(f"PROJECT_CREATE_FAILED status={status}", flush=True)
            raise SystemExit(1)
        project_id = str(proj["id"])
        print(f"PROJECT=created id={project_id} name={proj.get('name')}", flush=True)
    else:
        print(f"PROJECT_LOOKUP_FAILED status={status}", flush=True)
        raise SystemExit(1)

    # 1b. Pin framework on the project (API-created projects stay null).
    st, _ = api("PATCH", f"/v9/projects/{project_id}", token, {"framework": "python"})
    print(f"FRAMEWORK_PIN status={st}", flush=True)

    # 2. Upsert env vars (names only in output; values never printed).
    desired: dict[str, str] = {}
    for key in ("OPENROUTER_API_KEY", "TURSO_DATABASE_URL", "TURSO_AUTH_TOKEN"):
        if secrets.get(key):
            desired[key] = secrets[key]
    desired["AGENT_MODEL"] = secrets.get("AGENT_MODEL", "") or MODEL_DEFAULT
    desired["CORE_V2_DB_PATH"] = "/tmp/scaliffy_core_v2.db"
    if "OPENROUTER_API_KEY" not in desired:
        fail_missing("OPENROUTER_API_KEY")
    for extra in ("TURSO_DATABASE_URL", "TURSO_AUTH_TOKEN"):
        if extra not in desired:
            print(f"MISSING_ENV_NAME={extra} (durable state falls back to ephemeral /tmp)", flush=True)

    status, existing = api("GET", f"/v9/projects/{project_id}/env", token)
    by_key: dict[str, str] = {}
    if status == 200 and isinstance(existing, dict):
        items = existing.get("envs", existing.get("env", []))
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and item.get("key") and item.get("id"):
                    by_key[str(item["key"])] = str(item["id"])
    for key in sorted(desired):
        if key in by_key:
            st, _ = api(
                "PATCH", f"/v9/projects/{project_id}/env/{by_key[key]}", token,
                {"value": desired[key], "type": "encrypted",
                 "target": ["production", "preview", "development"]},
            )
            print(f"ENV={key} updated status={st}", flush=True)
        else:
            st, _ = api(
                "POST", f"/v9/projects/{project_id}/env", token,
                {"key": key, "value": desired[key], "type": "encrypted",
                 "target": ["production", "preview", "development"]},
            )
            print(f"ENV={key} created status={st}", flush=True)

    if skip_deploy:
        print("DEPLOY=skipped", flush=True)
        return

    # 3. Deploy.
    files = collect_files()
    print(f"FILES={len(files)}", flush=True)
    status, dep = api("POST", "/v13/deployments?skipAutoDetectionConfirmation=1", token, {
        "name": project_name,
        "project": project_id,
        "target": "production",
        "projectSettings": {"framework": "python"},
        "files": files,
    })
    if status not in (200, 201) or not isinstance(dep, dict) or not dep.get("id"):
        print(f"DEPLOY_CREATE_FAILED status={status} detail={str(dep)[:500]}", flush=True)
        raise SystemExit(1)
    dep_id = str(dep["id"])
    url = str(dep.get("url") or "")
    print(f"DEPLOYMENT_ID={dep_id}", flush=True)
    print(f"PUBLIC_V2_URL=https://{url}", flush=True)

    # 4. Wait for READY.
    state = ""
    for _ in range(60):
        time.sleep(10)
        st, info = api("GET", f"/v13/deployments/{dep_id}", token)
        if st == 200 and isinstance(info, dict):
            state = str(info.get("readyState") or info.get("state") or "")
            if state in ("READY", "ERROR", "CANCELED"):
                break
    print(f"DEPLOY_STATE={state or 'unknown'}", flush=True)
    if state != "READY":
        raise SystemExit(1)

    # 5. Health.
    time.sleep(5)
    try:
        with urllib.request.urlopen(f"https://{url}/health", timeout=30) as resp:
            print(f"HEALTH_STATUS={resp.status}", flush=True)
            print(f"HEALTH_BODY={resp.read().decode('utf-8')[:800]}", flush=True)
    except Exception as exc:
        print(f"HEALTH_FAILED={type(exc).__name__}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
