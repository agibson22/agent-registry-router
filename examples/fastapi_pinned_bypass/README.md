# FastAPI example: pinned session bypasses classifier

This example demonstrates two routing paths:

- **Pinned session**: if a session has `pinned_agent` set, the classifier is skipped.
- **Unpinned session**: classifier → validate (fallback) → dispatch.

It uses **toy agents** (no API keys) that implement a PydanticAI-style `.run(message, deps=...)` method.

## Run

From the repo root:

```bash
cd /Users/ag/Sites/agent-registry-router
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
uv pip install fastapi uvicorn
cd examples/fastapi_pinned_bypass
PYTHONPATH=../../src python -m uvicorn main:app --reload
```

Notes:
- Using `python -m uvicorn` ensures you run the `uvicorn` from the active virtualenv.
- `PYTHONPATH=../../src` makes the local package importable even if you skip the editable install.

## Try it

Create a session:

```bash
curl -sS -X POST localhost:8000/sessions | python -m json.tool
```

Send an unpinned message (classifier runs):

```bash
curl -sS -X POST localhost:8000/sessions/{session_id}/messages \\
  -H 'content-type: application/json' \\
  -d '{\"content\":\"hello\"}' | python -m json.tool
```

Pin the session to `internal_tool` (bypasses classifier):

```bash
curl -sS -X POST localhost:8000/sessions/{session_id}/pin \\
  -H 'content-type: application/json' \\
  -d '{\"pinned_agent\":\"internal_tool\"}' | python -m json.tool
```

Send a message again (should bypass classifier):

```bash
curl -sS -X POST localhost:8000/sessions/{session_id}/messages \\
  -H 'content-type: application/json' \\
  -d '{\"content\":\"hello again\"}' | python -m json.tool
```


