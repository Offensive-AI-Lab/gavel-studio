#!/usr/bin/env bash
###############################################################################
# GAVEL Studio — one-shot bootstrap & launch (Linux / macOS / Windows)
#
#   ./run.sh
#
# From a fresh clone to a running app. Idempotent — safe to re-run.
#
#   1. Ensures Python 3.12+ and Node 20+ (auto-installs the missing ones via
#      apt/dnf/pacman/zypper/brew; on Windows it prints install links and
#      stops). There is no database server to install — storage is a SQLite
#      file (backend/db/gavel.sqlite3) the backend creates on first boot.
#   2. Asks for an optional OpenAI key (AI rule/CE generation). Publishing to
#      the HF registry is optional too — add a write-scope HF_TOKEN to
#      backend/.env by hand to enable it; everything else works without one.
#   3. Installs Python deps for the backend and Node deps for the frontend
#      ("downloads everything he needs").
#   4. Writes backend/.env, then launches the backend + frontend and streams
#      their logs. Ctrl+C stops them.
#
# Open http://localhost:5173 when it's up. There is no login — the app serves
# one local operator.
#
# Clean slate: delete backend/db/gavel.sqlite3 — the public library re-syncs
# from HuggingFace on the next boot.
#
# Needs: an internet connection. The FIRST run downloads the ML stack
# (torch, transformers, ...), so it can take several minutes.
###############################################################################
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# --- pretty logging ---------------------------------------------------------
if [ -t 1 ]; then
  C_BLUE=$'\033[34m'; C_GREEN=$'\033[32m'; C_YEL=$'\033[33m'; C_RED=$'\033[31m'; C_DIM=$'\033[2m'; C_OFF=$'\033[0m'
else
  C_BLUE=""; C_GREEN=""; C_YEL=""; C_RED=""; C_DIM=""; C_OFF=""
fi
step() { echo "${C_BLUE}▶ $*${C_OFF}"; }
ok()   { echo "${C_GREEN}✓ $*${C_OFF}"; }
warn() { echo "${C_YEL}! $*${C_OFF}"; }
die()  { echo "${C_RED}✗ $*${C_OFF}" >&2; exit 1; }

# --- config (override via env: e.g. BACKEND_PORT=8001 ./run.sh) --------------
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
LOG_DIR="$ROOT/logs"; mkdir -p "$LOG_DIR"

# --- generic helpers --------------------------------------------------------
have() { command -v "$1" >/dev/null 2>&1; }

case "$(uname -s 2>/dev/null || echo unknown)" in
  Linux*)  OS=linux ;;
  Darwin*) OS=macos ;;
  MINGW*|MSYS*|CYGWIN*) OS=windows ;;
  *) OS=unknown ;;
esac

PKG=""
if   have brew;    then PKG=brew
elif have apt-get; then PKG=apt
elif have dnf;     then PKG=dnf
elif have yum;     then PKG=yum
elif have pacman;  then PKG=pacman
elif have zypper;  then PKG=zypper
fi
SUDO=""
if [ "$(id -u 2>/dev/null || echo 0)" != "0" ] && [ "$PKG" != "brew" ] && have sudo; then SUDO="sudo"; fi

install_hint() {
  case "$OS" in
    linux)   echo "  sudo apt-get install -y $1   (or your distro's package manager)" ;;
    macos)   echo "  brew install $1   (Homebrew: https://brew.sh)" ;;
    windows) echo "  Install $1 and reopen Git Bash (python.org / nodejs.org)." ;;
    *)       echo "  Install $1 and re-run." ;;
  esac
}

pkg_install() { # generic: python | node
  local tool="$1" p=""
  case "$PKG" in
    brew)
      case "$tool" in python) p="python@3.12";; node) p="node";; esac
      brew install $p ;;
    apt)
      case "$tool" in python) p="python3 python3-venv python3-pip";; node) p="nodejs npm";; esac
      $SUDO apt-get update -y && $SUDO apt-get install -y $p ;;
    dnf|yum)
      case "$tool" in python) p="python3 python3-pip";; node) p="nodejs npm";; esac
      $SUDO "$PKG" install -y $p ;;
    pacman)
      case "$tool" in python) p="python python-pip";; node) p="nodejs npm";; esac
      $SUDO pacman -Sy --noconfirm $p ;;
    zypper)
      case "$tool" in python) p="python3 python3-pip";; node) p="nodejs npm";; esac
      $SUDO zypper install -y $p ;;
    *) return 1 ;;
  esac
}

ensure_tool() { # cmd, generic-tool, human-name  -> installs if missing
  local cmd="$1" tool="$2" name="$3"
  have "$cmd" && return 0
  if [ -z "$PKG" ]; then echo "$name not found."; echo "$(install_hint "$cmd")"; return 1; fi
  warn "$name not found."
  local yn="y"
  [ -t 0 ] && { printf "  Install %s now via %s? [Y/n] " "$name" "$PKG"; read -r yn || yn="y"; }
  case "$yn" in [Nn]*) return 1 ;; esac
  step "installing $name via $PKG…"
  pkg_install "$tool" || true
  have "$cmd"
}

pick_python() {
  for c in python3 python py; do
    have "$c" && "$c" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,12) else 1)' 2>/dev/null \
      && { echo "$c"; return 0; }
  done
  return 1
}
venv_python() { if [ -x "$1/Scripts/python.exe" ]; then echo "$1/Scripts/python.exe"; else echo "$1/bin/python"; fi; }

get_env() { [ -f "$1" ] && sed -n "s/^$2=//p" "$1" | head -1 || true; }
set_env() {
  "$PY" - "$1" "$2" "$3" <<'PYEOF'
import sys, os
path, key, val = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(path, encoding="utf-8").read().splitlines() if os.path.exists(path) else []
out, found = [], False
for ln in lines:
    if ln.lstrip().startswith(key + "="):
        out.append(f"{key}={val}"); found = True
    else:
        out.append(ln)
if not found:
    out.append(f"{key}={val}")
open(path, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
PYEOF
}

###############################################################################
# 1. Prerequisites: Python and Node (no database server — SQLite is built in)
###############################################################################
step "Checking prerequisites (OS: $OS${PKG:+, package manager: $PKG})"

PY="$(pick_python || true)"
if [ -z "$PY" ]; then
  ensure_tool python3 python "Python 3.12+" || { echo "$(install_hint python3)"; die "Python 3.12+ required"; }
  PY="$(pick_python || true)"; [ -n "$PY" ] || die "Installed Python is < 3.12 — install 3.12+ and re-run."
fi
ok "python: $("$PY" --version 2>&1) ($PY)"

ensure_tool node node "Node.js 20+" || { echo "$(install_hint nodejs)"; die "Node.js required"; }
have npm || die "npm not found (ships with Node.js)"
NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)"
[ "$NODE_MAJOR" -ge 18 ] 2>/dev/null || warn "Node $(node --version) is old; the frontend wants 20+. If 'npm run dev' fails, upgrade Node (e.g. via nvm)."
ok "node: $(node --version)   npm: $(npm --version)"

ok "database: SQLite file (backend/db/gavel.sqlite3) — created automatically, nothing to install"

###############################################################################
# 2. Optional credentials (so the rest runs unattended)
###############################################################################
step "Optional credentials"
# Reuse values already saved in .env from a previous run; ignore placeholders.
OPENAI_VAL="$(get_env backend/.env OPENAI_API_KEY)"
if [ -t 0 ]; then
  [ -z "$OPENAI_VAL" ] && { printf "  %sOpenAI API key — AI rule & CE generation (Enter to skip): %s" "$C_DIM" "$C_OFF"; read -r OPENAI_VAL || OPENAI_VAL=""; }
fi

###############################################################################
# 3. Dependencies (Python venvs + Node modules)
###############################################################################
step "Installing backend Python deps (first run pulls the ML stack — be patient)"
[ -d backend/.venv ] || "$PY" -m venv backend/.venv
# Absolute path: the launch step does `cd backend` before exec-ing this
# interpreter, so a relative path would wrongly resolve to backend/backend/...
BACKEND_PY="$ROOT/$(venv_python backend/.venv)"
"$BACKEND_PY" -m pip install --quiet --upgrade pip
"$BACKEND_PY" -m pip install -r backend/requirements.txt
ok "backend deps installed"

step "Installing frontend Node deps"
( cd frontend && npm install --no-fund --no-audit )
ok "frontend deps installed"

###############################################################################
# 4. Environment file
###############################################################################
step "Configuring environment files"
[ -f backend/.env ]  || { cp backend/.env.example backend/.env; ok "created backend/.env"; }
[ -f frontend/.env ] || { cp frontend/.env.example frontend/.env 2>/dev/null || :; }

set_env backend/.env ALLOWED_ORIGINS "http://localhost:${FRONTEND_PORT}"
set_env backend/.env FRONTEND_URL "http://localhost:${FRONTEND_PORT}"
set_env backend/.env OPENAI_API_KEY "$OPENAI_VAL"

[ -f frontend/.env ] && set_env frontend/.env VITE_API_URL "http://localhost:${BACKEND_PORT}"

[ -z "$OPENAI_VAL" ] && warn "OPENAI_API_KEY not set — AI rule/CE generation disabled (everything else works)."
[ -z "$(get_env backend/.env HF_TOKEN)" ] && warn "HF_TOKEN not set — publishing to the HF registry disabled (everything else works)."
ok "environment configured"

###############################################################################
# 5. Launch (backend → frontend), streaming logs; Ctrl+C stops all
###############################################################################
wait_http() { # url, label, max_seconds
  local url="$1" label="$2" max="${3:-60}" i=0
  if ! have curl; then sleep 5; return 0; fi
  while [ "$i" -lt "$max" ]; do
    curl -fsS "$url" >/dev/null 2>&1 && { ok "$label is up"; return 0; }
    sleep 1; i=$((i+1))
  done
  warn "$label did not answer at $url within ${max}s — see $LOG_DIR"
}

PIDS=()
cleanup() { echo; step "Shutting down…"; for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; ok "stopped"; }
trap cleanup EXIT INT TERM

step "Starting backend on :$BACKEND_PORT"
( cd backend && exec "$BACKEND_PY" -m uvicorn main:app --host 127.0.0.1 --port "$BACKEND_PORT" --reload ) > "$LOG_DIR/backend.log" 2>&1 &
PIDS+=($!)
wait_http "http://127.0.0.1:${BACKEND_PORT}/docs" "backend" 120

step "Starting frontend on :$FRONTEND_PORT"
( cd frontend && exec npm run dev -- --port "$FRONTEND_PORT" ) > "$LOG_DIR/frontend.log" 2>&1 &
PIDS+=($!)
wait_http "http://127.0.0.1:${FRONTEND_PORT}" "frontend" 60

echo
ok "GAVEL is up!"
echo "    ${C_GREEN}Open  http://localhost:${FRONTEND_PORT}${C_OFF}"
echo "    ${C_DIM}backend  http://localhost:${BACKEND_PORT}/docs${C_OFF}"
echo "    ${C_DIM}logs     $LOG_DIR/{backend,frontend}.log${C_OFF}"
echo "    ${C_YEL}Press Ctrl+C to stop all services.${C_OFF}"
echo
tail -n +1 -f "$LOG_DIR/backend.log" "$LOG_DIR/frontend.log" &
PIDS+=($!)
wait
