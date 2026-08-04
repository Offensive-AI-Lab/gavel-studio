#!/usr/bin/env bash
###############################################################################
# One-time environment setup for the DOCKER run mode.
#
#   ./setup-docker.sh          # create/populate backend/.env, then optionally launch
#
# Config lives in ONE place: backend/.env — the SAME file native dev uses. The
# compose file mounts it into the backend container and overrides only the few
# container-only values (CORS origins, the mounted SSH-key path). This script
# fills in backend/.env so a fresh clone can run the whole stack without
# hand-editing:
#   * creates backend/.env (from backend/.env.example, or fresh) if missing,
#   * prompts for the optional parameters (OpenAI / GitHub token),
#     explaining what each unlocks,
#   * offers to run `docker compose --env-file backend/.env up --build`.
#
# There is no login. This stack runs backend + frontend; the database is a
# SQLite file (backend/db/gavel.sqlite3) the backend creates itself. No auth is
# configured anywhere — GAVEL Studio is a single-operator, localhost application.
#
# Re-runnable: existing values are kept; you can press Enter to skip any prompt.
###############################################################################
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ -t 1 ]; then
  C_B=$'\033[34m'; C_G=$'\033[32m'; C_Y=$'\033[33m'; C_D=$'\033[2m'; C_O=$'\033[0m'
else C_B=""; C_G=""; C_Y=""; C_D=""; C_O=""; fi
step() { echo "${C_B}▶ $*${C_O}"; }
ok()   { echo "${C_G}✓ $*${C_O}"; }
warn() { echo "${C_Y}! $*${C_O}"; }

# Single source of truth for app config: backend/.env. Docker reads it via
# `docker compose --env-file backend/.env`, so compose's ${VAR} interpolation
# (GPU_WORKER_URL, GPU_WORKER_TOKEN) comes from this SAME file —
# there is no separate repo-root .env.
ENV="backend/.env"

# --- set KEY=value in $ENV: replace in place if present, else append ---------
set_kv() {
  local key="$1" val="$2"
  if grep -qE "^${key}=" "$ENV" 2>/dev/null; then
    awk -v k="$key" -v v="$val" '
      $0 ~ "^"k"=" { print k"="v; found=1; next } { print }
      END { if (!found) print k"="v }' "$ENV" > "$ENV.tmp" && mv "$ENV.tmp" "$ENV"
  else
    printf '%s=%s\n' "$key" "$val" >> "$ENV"
  fi
}
get_kv() { sed -n "s/^$1=//p" "$ENV" 2>/dev/null | head -1; }

# --- 1. ensure backend/.env exists -------------------------------------------
step "Preparing $ROOT/$ENV"
if [ ! -f "$ENV" ]; then
  if [ -f "backend/.env.example" ]; then cp backend/.env.example "$ENV"; ok "created $ENV from backend/.env.example"; else : > "$ENV"; ok "created an empty $ENV"; fi
else
  ok "$ENV already exists — updating it"
fi

# No auth step: GAVEL Studio has no login. Every request is attributed to one
# fixed local user, seeded into the database at boot.

# --- 2. optional parameters (prompt only if interactive + not already set) ---
prompt_kv() {   # key, human label
  local key="$1" label="$2" cur; cur="$(get_kv "$key")"
  case "$cur" in ghp_xxxx*|github_pat_xxxx*|sk-xxxx*|changeme*) cur="";; esac
  if [ -n "$cur" ]; then ok "$key already set — keeping it"; return; fi
  if [ -t 0 ]; then
    printf "  %s%s (Enter to skip): %s" "$C_D" "$label" "$C_O"
    local val; read -r val || val=""
    set_kv "$key" "$val"
  else
    warn "$key not set — add it to .env: $label"
  fi
}
step "Optional parameters (press Enter to skip)"
# OpenAI drives AI generation in the backend — always relevant.
prompt_kv OPENAI_API_KEY     "OpenAI API key      — enables AI rule/CE generation"
# GITHUB_TOKEN is READ-access and only needed while the gavel-rules library
# repo is private (it also lifts GitHub API rate limits). Once the repo is
# public, leave it blank — the library syncs anonymously. Contributions to
# the library go by pull request to gavel-rules, not through the app.
prompt_kv GITHUB_TOKEN       "GitHub token        — read access to the gavel-rules library repo, only needed while it is private"

ok "$ENV is ready"
echo "    ${C_D}Edit by hand in backend/.env: OPENAI_API_KEY (AI generation), GITHUB_TOKEN (private gavel-rules repo access).${C_O}"

# --- 3. offer to launch ------------------------------------------------------
_DC="docker compose --env-file backend/.env up --build"
if [ -t 0 ] && command -v docker >/dev/null 2>&1; then
  printf "\n  %sRun '%s' now? [Y/n] %s" "$C_D" "$_DC" "$C_O"
  read -r yn || yn="y"
  case "$yn" in [Nn]*) echo "  Skipped. Run it yourself when ready: $_DC" ;;
    *) exec docker compose --env-file backend/.env up --build ;;
  esac
else
  echo "  Next: $_DC"
fi
