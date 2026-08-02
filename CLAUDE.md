# GAVEL Studio — project notes for AI assistants

## Where work is tracked
Product bugs/features live in GitHub issues (`gh issue list`). Before starting
on an issue, check nobody is assigned and leave a comment claiming it. Small
fixes go straight to main with "Fixes #N" in the commit; structural changes go
through a PR.

## Writing user-facing text
For user-facing prose (READMEs, UI copy, help text, docs), the team drafts
with ChatGPT Sol on medium — we prefer its plainer, simpler writing for this.
If you (Claude) are editing user-facing copy directly, match that register:
short sentences, no jargon, no implementation details. Talk to the operator
using the app, never the developer.

## Running
- Users run Docker: `docker compose --env-file backend/.env up --build`.
- Development on a Mac is faster native (MPS): backend `uv venv && uv pip install
  -r requirements.txt -r requirements-dev.txt && uv run uvicorn main:app --port
  8000 --reload --timeout-graceful-shutdown 3` (in backend/, venv activated);
  frontend `npm install && npm run dev`. Both modes share backend/db/gavel.sqlite3.

## Testing
- Backend: `pytest tests/unit` and `tests/integration` (needs requirements-dev.txt).
  Integration tests run against the live DB with snapshot/restore — or set
  DB_PATH=/tmp/test.sqlite3 to isolate.
- Frontend: `npm test` in frontend/.

## Hard rules
- `backend/classifier_engine/reference/` is a VERBATIM copy of the published
  paper code (github.com/Offensive-AI-Lab/gavel) — do not edit files there;
  adaptations go in `backend/evaluation/adapter.py` (see reference/README.md).
  This freeze is slated for removal (issue #16) — until that lands, the rule
  stands.
- The public library syncs read-only from the gavel-rules GitHub repo;
  Studio never writes to it. Contributions go by PR to gavel-rules.
- Schema changes: bump SCHEMA_VERSION in backend/utils/DButils.py and add an
  explicit migration step — the DDL block is CREATE IF NOT EXISTS only and
  never alters existing tables.
