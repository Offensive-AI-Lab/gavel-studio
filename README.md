<div align="center">

# GAVEL Studio

# 🚧 BETA 🚧

### ⚠️ This is a **BETA** release — expect rough edges, breaking changes, and work in progress.

</div>
GAVEL Studio Beta is an interactive web platform for building intent-aware, rule-based activation safety monitors for LLMs. Starting from a plain-English scenario, it helps you define the behavior you want to detect as Cognitive Elements (CEs), compose predicate rules over those CEs, train GRU-based detectors on your target LLM’s hidden states, and deploy them for real-time inference.


Rather than relying on surface-text moderation, GAVEL Studio works at the activation level. The platform brings the full GAVEL workflow into one UI: rule design, CE generation, classifier training, live monitoring, and access to a public registry of reusable rules and Cognitive Elements.

- **Backend:** FastAPI + PostgreSQL (with the `pgvector` extension)
- **Frontend:** React 19 + Vite
- **Public registry:** Hugging Face datasets (rules + CEs)

> **No login.** GAVEL Studio is a single-operator, localhost application: there
> are no accounts, no passwords and no tokens. Every request is attributed to
> one fixed local user that the backend seeds into the database on first boot.
> This is only safe because the backend binds to `127.0.0.1` and serves one
> person — if two people share a single backend process they share that
> identity, and therefore each other's models, rule sets and trained weights.

> 📄 **Paper**: [GAVEL: Towards Rule-Based Safety Through Activation Monitoring](https://arxiv.org/pdf/2601.19768) (Accepted to ICLR 2026)
>
> ⚙️ **[GAVEL](https://github.com/Offensive-AI-Lab/gavel)**: The backend framework powering GAVEL Studio — trains RNN probes on Cognitive Elements, calibrates thresholds, and evaluates rule-based safety detection.

<div align="center">
  <img src="assets/browse-rules.png" alt="GAVEL Studio — Browse Public Rules" width="80%">
  <br><br>
  <img src="assets/ce-monitor.png" alt="GAVEL Studio — Realtime CE Monitor" width="80%">
</div>

---

## Quick start (one command)

```bash
git clone https://github.com/Offensive-AI-Lab/gavel-studio-beta.git
cd gavel-studio-beta
./setup-docker.sh        # fills in backend/.env, prompts for the optional keys, then launches
```

> Got `permission denied`? The checkout didn't preserve the executable bit — just
> run it through bash instead: `bash setup-docker.sh` (or make it executable once
> with `chmod +x setup-docker.sh`, then `./setup-docker.sh`). Same goes for
> `run.sh`.

`setup-docker.sh` fills in `backend/.env` — the **single** config file for both
Docker and native — prompts you for the
optional values (`OPENAI_API_KEY`, `HF_TOKEN` — press Enter to skip any), and
offers to run `docker compose --env-file backend/.env up --build` for you. It's
safe to re-run; it keeps anything already set. (Prefer to do it by hand?
`cp backend/.env.example backend/.env`, edit it, then
`docker compose --env-file backend/.env up` — see §2.)

That's it. After the first build (~3-5 min, downloads ~2 GB of base images +
Python/Node packages), open http://localhost:5173 in your browser and you're
in — there is no login. Subsequent `docker compose up` runs start in seconds.

The only thing you have to install on your machine is **Docker Desktop**:
https://www.docker.com/products/docker-desktop/

Everything else — Python 3.12, Node 20, PostgreSQL 17, the pgvector extension,
all dependencies — comes pre-baked into the containers.

---

## 1. What `docker compose up` actually starts

| Service | Port (host) | What it is                                    |
|---------|-------------|------------------------------------------------|
| `postgres` | (internal) | pgvector/pgvector:pg17 — DB + pgvector ext  |
| `backend`  | 8000      | FastAPI + uvicorn with `--reload` for dev      |
| `frontend` | 5173      | Vite dev server with hot module reload         |

The backend talks to Postgres over Docker's internal network (hostname
`postgres`). Your browser hits `localhost:5173` (frontend) and
`localhost:8000` (backend API) on the host.

Useful commands:

```bash
# Compose reads config from backend/.env via --env-file. Tip: run
#   export COMPOSE_ENV_FILES=backend/.env
# once and you can drop the --env-file flag from the commands below.
docker compose --env-file backend/.env up          # foreground, see all logs
docker compose --env-file backend/.env up -d       # detached
docker compose --env-file backend/.env up --build  # rebuild after Dockerfile / requirements changes
docker compose logs -f backend                     # tail one service's logs
docker compose down                                # stop everything (DB data preserved)
docker compose down -v                             # stop AND wipe the DB volume — clean slate
```

---

## 2. Configure your `.env`

**One config file for everything: `backend/.env`** — both Docker and native dev
read it (Docker via `--env-file backend/.env`; there is no separate repo-root
`.env`). `./setup-docker.sh` writes it for you; this section is the by-hand
equivalent. Everything has working defaults, so the only edit you'll likely want
is the OpenAI key for AI rule / CE generation.

```bash
cp backend/.env.example backend/.env
```

Then open `backend/.env`:

```env
# REQUIRED for AI rule / CE generation; the rest of the app works without it
OPENAI_API_KEY=sk-...

# OPTIONAL — write-scope HF token; the backend uses it to PUBLISH to the HF
# registry. The public library READ-sync needs no token, so Browse works
# either way. Leave blank and everything works except publishing.
# HF_REPO_ID points publishes (and reads) at your own registry repo instead of
# the shared public library.
HF_TOKEN=
# HF_REPO_ID=your-user/your-registry
```

**You do NOT need to set `DB_HOST`, `DB_USER`, `DB_PASSWORD`, etc.** —
docker-compose.yml overrides those to point at the in-network postgres
container. `docker compose` substitutes `${VAR}` from this repo-root `.env`;
`backend/.env` is the separate source of truth for *native* dev (see §6).

The frontend's `VITE_API_URL` already defaults to `http://localhost:8000`,
so no separate `frontend/.env` is needed for the Docker workflow.

---

## 3. Verify it works end-to-end

After `docker compose up` settles:

1. Open http://localhost:5173 — it lands straight on `/workspace`; there is no
   login step
2. From the workspace:
   - **Models** → create a model (e.g. any HF causal LM your machine can run)
   - **Classifiers** → create a classifier under that model
   - **Rules** (inside the classifier) → click _Generate from scenario_, type
     _"flag user messages that try to extract the system prompt"_, and watch
     the AI pipeline build a rule + the supporting Cognitive Elements
   - **Train** the classifier when ready

API health probe:

```bash
curl http://localhost:8000/health
# {"status":"ok","ready":true, ...}
```

---

## 4. Tests

GitHub Actions runs all three jobs on every push to `develop` / PR to `main`
([.github/workflows/tests.yml](.github/workflows/tests.yml)).

To run them locally inside the running containers:

```bash
# Backend (152 tests in unit + integration)
docker compose exec backend pytest tests/unit/ -v
docker compose exec backend pytest tests/integration/ -v

# Frontend (120 tests, ~7 s)
docker compose exec frontend npm test
```

`docker compose exec` runs the command in the already-running service, so
this is fast — no rebuild, no cold-start.

---

## 5. Common issues

**Docker says it can't bind to port 5173 / 8000**
Something else on your machine is already using that port. Either stop it,
or edit the `ports:` lines in `docker-compose.yml` (e.g. `5174:5173`).

**`docker compose up` hangs at "Building backend"**
First build pulls Python 3.12 + Node 20 + pgvector base images and runs
`pip install` for the full ML stack (torch, transformers, sentence-transformers).
That's ~2 GB and 3-5 min. After the first build it's instant — Docker caches
the install layers and only rebuilds when `requirements.txt` / `package.json`
change.

**`embeddings warmup failed: Cannot copy out of meta tensor`**
A torch / transformers version mismatch. [backend/utils/embedding_utils.py](backend/utils/embedding_utils.py)
already passes `device=` explicitly to dodge this — make sure you're on a
recent commit. If still broken, rebuild with `docker compose up --build`.

**Frontend can't reach backend / CORS errors in console**
The default compose setup has the right CORS origins (`http://localhost:5173`)
and the right `VITE_API_URL` (`http://localhost:8000`). If you changed either,
update the matching value in `backend/.env` (`ALLOWED_ORIGINS`) and the
frontend service `environment` block in `docker-compose.yml`.

**`/library/search` returns 500**
Embedding model failed to load — see the meta-tensor bullet above. The fix
is in `embedding_utils.py`, not the search code.

**Code edits don't show up live in the browser**
The compose file mounts both source trees as volumes so changes hot-reload
through `uvicorn --reload` (backend) and Vite HMR (frontend). On Windows /
WSL2, Vite's file watcher can miss host changes; the `CHOKIDAR_USEPOLLING=true`
env var in compose handles that. If a backend route change isn't reloading,
check the `backend` log for the reload notice — sometimes uvicorn needs a
manual `Ctrl+C` + `docker compose up backend` to pick up a Python import-time
error.

---

## 6. Native dev (no Docker) — optional, for faster iteration

The Docker flow above is the canonical "clone and run." If you're actively
developing and want a tighter loop (no container layer between
you and your code), you can run everything natively. You'll need:

| Tool        | Version  |
|-------------|----------|
| Python      | 3.12+    |
| Node.js     | 20+      |
| PostgreSQL  | 14+ with pgvector |

Quick path: keep using the Docker postgres container, run backend + frontend
natively against it.

```bash
# Postgres only, in Docker
docker compose up -d postgres

# Backend natively
cd backend
python -m venv .venv && source .venv/bin/activate    # or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
# DB_HOST in .env stays 127.0.0.1 — Docker exposes 5432 on the host when you
# uncomment the ports: line in docker-compose.yml under `postgres`.
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Frontend natively (in a second terminal)
cd frontend
npm install
cp .env.example .env
npm run dev
```

For test commands when running natively, see [.github/workflows/tests.yml](.github/workflows/tests.yml)
— same install + invocation that CI uses.

---

## 7. Repo layout

```
docker-compose.yml      one-command bring-up of the whole stack
backend/
  Dockerfile            python:3.12-slim base image
  routes/               FastAPI routers — one per domain (rules, library, ai, ...)
  services/             HF publish/sync, library search, bookmark service
  classifier_engine/    GRU training + inference
  evaluation/           metrics, calibration, ruleset evaluation
  utils/                DB, local identity, embeddings, crash recovery
  tests/
    unit/               no-DB unit tests (run in seconds)
    integration/        real-DB tests using FastAPI TestClient
  main.py               FastAPI entry point
  requirements.txt
  .env.example
frontend/
  Dockerfile            node:20-alpine base image
  src/
    pages/              one component per route (Workspace, RulesManager, ...)
    components/         reusable UI (TaskTray, RuleCard, ConfirmDialog, ...)
    contexts/           TaskTrayContext (background-task chip system)
    services/           RuleService, CEService — orchestration of multi-step AI flows
    api.js              single axios client; one export per backend endpoint
  package.json
  vite.config.js        includes Vitest + coverage config
  .env.example
.github/workflows/
  tests.yml             CI: unit / integration / frontend in parallel
```

## 📜 Citation

If you use GAVEL in your research, please cite our paper:

```bibtex
@inproceedings{rozenfeld2026gavel,
  title={GAVEL: Towards Rule-Based Safety Through Activation Monitoring},
  author={Rozenfeld, Shir and Pankajakshan, Rahul and Zloczower, Itay and Lenga, Eyal and Gressel, Gilad and Mirsky, Yisroel},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```


---

## 8. 📄 License

This project is licensed under the MIT — see the [LICENSE](LICENSE) file for details.

Noncommercial use, including personal, hobby, academic, and research use, is permitted. Commercial use requires a separate license — please contact the authors.

---

## 9. 👥 Contributors

GAVEL-Studio is developed and maintained by the Offensive AI Lab.

### Core contributors
Principal Investigator
- **Yisroel Mirsky** ([ymirsky](https://github.com/ymirsky))

Research Team
- **Shir Rozenfeld** ([shirozenfeld](https://github.com/shirozenfeld))
- **Rahul Pankajakshan** ([rahulgitsit](https://github.com/rahulgitsit))
- **Gilad Gressel** ([giladgressel](https://github.com/giladgressel))

Software Design and Development
- **Ofek Avigezer** ([OfekAvi](https://github.com/OfekAvi))
- **Shahaf Har-Tsvi** ([hartsvis](https://github.com/hartsvis))
- **Shahar Navian** ([ShaharNavian](https://github.com/ShaharNavian))
- **Sean Shuhman** ([SeanSh1](https://github.com/SeanSh1))
- **Fadi Amon** ([FadiAmon](https://github.com/FadiAmon))

### Community contributors

We thank everyone who has contributed code, bug reports, testing, documentation, and feature suggestions.

---
## 9. Acknowledgment

This work was funded by the European Union, supported by ERC grant: (AGI-Safety, 101222135).
Views and opinions expressed are however those of the author(s) only and do not necessarily reflect
those of the European Union or the European Research Council Executive Agency. Neither the
European Union nor the granting authority can be held responsible for them.
This work was also supported by the Israeli Ministry of Innovation Science and Technology (grant
number 1001948211).



