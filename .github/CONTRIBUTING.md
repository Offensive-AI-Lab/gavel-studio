# Contributing to GAVEL Studio

Thanks for taking the time. This page covers changes to Studio itself — the
application in this repository.

**Rules and Cognitive Elements are not contributed here.** The public library
lives in [gavel-rules](https://github.com/Offensive-AI-Lab/gavel-rules), and
contributions to it go by pull request to that repository. Studio reads the
library; it never writes to it.

## Before you start

Work is tracked in [GitHub issues](https://github.com/Offensive-AI-Lab/gavel-studio/issues).
Check that nobody is already assigned, then leave a comment saying you are
taking it. That saves two people writing the same patch.

For anything larger than a fix, open an issue first and describe the approach.
It is easier to agree on a direction before the code exists.

## Running Studio while you work

Docker is the shortest path:

```sh
docker compose --env-file backend/.env up --build
```

To run the two halves natively instead, see [`frontend/README.md`](../frontend/README.md)
for the interface and the repository [README](../README.md) for the backend.
Both modes share the same SQLite database at `backend/db/gavel.sqlite3`.

## Tests

Every code change should come with the tests that cover it.

```sh
cd backend && pytest tests/unit          # fast, no database needed
cd backend && pytest tests/integration   # runs against the local database
cd frontend && npm test
```

The backend tests need the development dependencies
(`pip install -r requirements.txt -r requirements-dev.txt`). To keep your own
database out of the way, set `DB_PATH` to a temporary file.

The unit tests and the frontend tests also run automatically on every pull
request. A red check means the change is not ready to merge.

## Sending your change

Fork the repository, make your change on a branch, and open a pull request
against `main`. Put `Fixes #N` in the description so the issue closes when the
change is merged.

Keep each pull request to one subject, and match the style of the file you are
editing rather than reformatting it. The tests run automatically once the pull
request is open.

For anything structural — a new subsystem, a schema change, a change to how
components talk to each other — agree the approach in the issue before you
write the code. It saves rewriting a finished pull request.

## Two rules that are easy to break by accident

1. **Do not edit `backend/classifier_engine/reference/`.** It is a verbatim copy
   of the published paper code, and Studio's evaluation numbers depend on it
   staying identical. Changes that belong near it go in
   `backend/evaluation/adapter.py`. See that directory's
   [README](../backend/classifier_engine/reference/README.md).
2. **Changing a database table takes two edits, not one.** Studio creates its
   tables with `CREATE TABLE IF NOT EXISTS` in `backend/utils/DButils.py`. On a
   database that already exists, SQLite skips those statements entirely — so
   adding a column to one of them does nothing for anyone who has run Studio
   before. It works on your machine only because you can delete your database
   and start over; everyone else keeps the old table and hits "no such column".
   So when you change a table, also bump `SCHEMA_VERSION` at the top of the
   same file and add the SQL that upgrades an older database — for a new column
   that is `ALTER TABLE rules ADD COLUMN notes TEXT`. It goes in
   `init_database()`, guarded so it runs only when the version stored in the
   database is older than yours. `init_database()` runs on every backend start,
   so users get the upgrade by updating the code and starting Studio. The
   `CREATE TABLE` edit covers people starting fresh; the `ALTER TABLE` covers
   everyone who already has data. Note there is no upgrade step in the file
   yet: `init_database()` currently only detects that the versions differ, so
   the next schema change writes the first one.

## Reporting a security problem

Do not open a public issue. See [SECURITY.md](SECURITY.md).
