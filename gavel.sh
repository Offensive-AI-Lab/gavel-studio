#!/usr/bin/env bash
###############################################################################
# gavel.sh — bring the whole GAVEL Studio stack up (or down) with one command.
#
#   ./gavel.sh          start everything, stream status, Ctrl+C to stop
#   ./gavel.sh stop     stop both servers
#   ./gavel.sh status   show what is currently up
#   ./gavel.sh logs     tail both server logs together
#
# Portable copy: everything is resolved relative to THIS script's directory,
# which must be the repo root (backend/ and frontend/ next to it). Copy or
# clone the folder anywhere, on any machine, and it works the same.
#
# Two processes, no database server: the backend keeps all data in a single
# SQLite file (backend/db/gavel.sqlite3) that it creates on first boot.
#
# Safe to re-run: anything already running is left alone, anything missing is
# started.
#
# Environment knobs (all optional):
#   GAVEL_ENV     conda env to activate (default: gavel_env). If conda or the
#                 env is missing, the script falls back to whatever uvicorn/npm
#                 are already on PATH instead of dying.
#   DB_PATH       where the SQLite file lives (absolute, or relative to
#                 backend/) — e.g. node-local disk, since SQLite's WAL mode
#                 does not work on NFS.
#   GAVEL_RELOAD  set to 1 to run uvicorn with --reload (dev only). Off by
#                 default: the reloader parent binds the port BEFORE the app
#                 loads and keeps it bound even after the app worker crashes,
#                 so "port open" stops meaning "backend works" — the exact
#                 zombie state that used to wedge startup.
###############################################################################
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDIO="$ROOT"
LOGS="$ROOT/logs"
RUN="$ROOT/.run"
ENV_NAME="${GAVEL_ENV:-gavel_env}"

DB_FILE="${DB_PATH:-$STUDIO/backend/db/gavel.sqlite3}"
BACKEND_PORT=8000
FRONTEND_PORT=5173

[ -d "$STUDIO/backend" ] && [ -d "$STUDIO/frontend" ] \
  || { echo "gavel.sh must sit in the repo root (backend/ and frontend/ next to it) — found neither under $STUDIO" >&2; exit 1; }

mkdir -p "$LOGS" "$RUN"

# ---------- pretty output ----------
if [ -t 1 ]; then
  B=$'\033[1m'; DIM=$'\033[2m'; GRN=$'\033[32m'; YEL=$'\033[33m'; RED=$'\033[31m'; CYN=$'\033[36m'; OFF=$'\033[0m'
else
  B=""; DIM=""; GRN=""; YEL=""; RED=""; CYN=""; OFF=""
fi
step() { printf "%s▶ %s%s\n" "$CYN" "$*" "$OFF"; }
ok()   { printf "%s  ✓ %s%s\n" "$GRN" "$*" "$OFF"; }
warn() { printf "%s  ! %s%s\n" "$YEL" "$*" "$OFF"; }
err()  { printf "%s  ✗ %s%s\n" "$RED" "$*" "$OFF" >&2; }
die()  { err "$*"; exit 1; }

# ---------- helpers ----------
# Bounded connect probe. A bare /dev/tcp connect can block for MINUTES when a
# wedged server's accept backlog is full (Linux drops the SYN and the kernel
# retransmits) — that silent hang was one way startup appeared frozen. A
# listener that cannot even accept within 2s is treated as down.
port_open() {
  local port=$1
  if command -v timeout >/dev/null 2>&1; then
    timeout 2 bash -c "exec 3<>/dev/tcp/127.0.0.1/$port" 2>/dev/null
  else
    (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null
  fi
}

# True when SOMETHING holds the port, even a listener too wedged to accept
# (full backlog: the connect neither succeeds nor is refused, so the bounded
# probe times out with rc 124). port_open answers "can I talk to it?";
# port_occupied answers "is the port free to bind?" — the cleanup paths must
# use the latter or a wedged listener is mistaken for a free port and never
# cleared.
port_occupied() {
  local port=$1 rc
  if command -v timeout >/dev/null 2>&1; then
    timeout 2 bash -c "exec 3<>/dev/tcp/127.0.0.1/$port" 2>/dev/null; rc=$?
    [ "$rc" -eq 0 ] || [ "$rc" -eq 124 ]
  else
    (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null
  fi
}

wait_for_port() {   # wait_for_port <port> <seconds> <label> [pid]  →  0 up / 1 timeout / 2 process died
  local port=$1 limit=$2 label=$3 pid=${4:-} start=$SECONDS
  while [ $((SECONDS - start)) -lt "$limit" ]; do
    port_open "$port" && return 0
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      return 2
    fi
    sleep 1
  done
  return 1
}

# True when the backend actually answers HTTP on /health. A raw TCP accept is
# not enough: a uvicorn reloader (or a half-dead process) can hold the port
# open while nothing serves. Falls back to the port probe if curl is missing.
backend_http_ok() {
  if command -v curl >/dev/null 2>&1; then
    curl -fsS -m 3 -o /dev/null "http://127.0.0.1:$BACKEND_PORT/health" 2>/dev/null
  else
    port_open "$BACKEND_PORT"
  fi
}

wait_for_backend() {   # wait_for_backend <seconds> [pid]  →  0 up / 1 timeout / 2 process died
  # Wall-clock deadline, not an iteration count: each probe can itself take a
  # few seconds (curl -m 3) against a half-dead server, and that must not
  # stretch the wait.
  local limit=$1 pid=${2:-} start=$SECONDS
  while [ $((SECONDS - start)) -lt "$limit" ]; do
    backend_http_ok && return 0
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      return 2
    fi
    sleep 1
  done
  return 1
}

# Kill whatever is LISTENING on the port — the fallback for leftovers we have
# no pidfile for (crashed earlier run, hand-started server, stale reloader).
# Tool order: lsof first (portable flags — macOS fuser doesn't understand
# -k/-TERM/-n tcp and would silently no-op while shadowing lsof), psmisc
# fuser next, ss (iproute, always on EL9) as the minimal-install fallback.
kill_port() {       # kill_port <port>  →  0 iff the port ended up free
  local port=$1 pids="" i=0
  if command -v lsof >/dev/null 2>&1; then
    pids=$(lsof -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
    # shellcheck disable=SC2086 — word splitting of the pid list is intended
    [ -n "$pids" ] && kill -TERM $pids 2>/dev/null
  elif command -v fuser >/dev/null 2>&1; then
    fuser -k -TERM -n tcp "$port" >/dev/null 2>&1
  elif command -v ss >/dev/null 2>&1; then
    pids=$(ss -tlnpH "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | sort -u)
    # shellcheck disable=SC2086
    [ -n "$pids" ] && kill -TERM $pids 2>/dev/null
  else
    warn "none of lsof/fuser/ss found — cannot identify the process on :$port"
  fi
  while [ "$i" -lt 8 ] && port_occupied "$port"; do sleep 0.5; i=$((i+1)); done
  if port_occupied "$port"; then
    if [ -n "$pids" ]; then
      # shellcheck disable=SC2086
      kill -KILL $pids 2>/dev/null
    elif command -v fuser >/dev/null 2>&1; then
      fuser -k -KILL -n tcp "$port" >/dev/null 2>&1
    fi
    sleep 1
  fi
  ! port_occupied "$port"
}

# Start a server detached in its OWN process group, so we can kill the whole
# tree later (npm/vite spawn children that would otherwise survive). setsid
# is not everywhere (e.g. macOS) — without it the child stays in our group
# and stop_one falls back to killing the recorded pid only; nohup keeps that
# child alive when the terminal closes (no own session = it would get HUP).
# The command is passed as argv, never re-parsed, so paths with spaces or
# quotes cannot break it. 9>&- keeps the startup-lock fd out of the servers:
# a server inheriting it would hold the flock forever after an aborted start,
# deadlocking every future ./gavel.sh.
spawn() {           # spawn <name> <workdir> <command...>
  local name=$1 dir=$2; shift 2
  if command -v setsid >/dev/null 2>&1; then
    setsid bash -c 'cd "$1" && shift && exec "$@"' _ "$dir" "$@" > "$LOGS/$name.log" 2>&1 9>&- &
  else
    nohup bash -c 'cd "$1" && shift && exec "$@"' _ "$dir" "$@" > "$LOGS/$name.log" 2>&1 9>&- &
  fi
  echo $! > "$RUN/$name.pid"
}

# Refuse to signal a recycled PID: .run/*.pid lives on NFS and survives crashes
# and reboots, so the recorded pid may now belong to an unrelated process of
# this user. Only signal pids whose command line still looks like the server
# we spawned; anything else is treated as stale and merely cleaned up.
pid_is_ours() {     # pid_is_ours <pid> <name>
  local cmd
  cmd=$(tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null) \
    || cmd=$(ps -o args= -p "$1" 2>/dev/null) || return 1
  case "$2:$cmd" in
    backend:*uvicorn*|central:*uvicorn*|central:*python*) return 0 ;;
    frontend:*npm*|frontend:*vite*|frontend:*node*) return 0 ;;
    tail:*tail*) return 0 ;;
    *) return 1 ;;
  esac
}

stop_one() {        # stop_one <name>
  # NB: separate `local` statements — bash expands all args to a single `local`
  # before running it, so "$RUN/$name.pid" on the same line would reference
  # $name before it is assigned (fatal under `set -u`).
  local name=$1
  local pidfile="$RUN/$name.pid"
  local pid
  [ -f "$pidfile" ] || return 0
  pid=$(cat "$pidfile" 2>/dev/null || true)
  if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null && pid_is_ours "$pid" "$name"; then
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    # Give it a real chance to exit cleanly before SIGKILL: a hard-killed
    # backend leaves stale SQLite WAL/lock state behind (much worse on NFS),
    # which is exactly what used to wedge the NEXT boot. Watch the whole
    # process GROUP, not just the leader — a lingering child (compute job,
    # esbuild) must still trigger the escalation after the leader exits.
    local i=0
    while [ "$i" -lt 16 ] && { kill -0 -- "-$pid" 2>/dev/null || kill -0 "$pid" 2>/dev/null; }; do
      sleep 0.5; i=$((i+1))
    done
    if kill -0 -- "-$pid" 2>/dev/null || kill -0 "$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null
      kill -KILL "$pid" 2>/dev/null
    fi
    ok "stopped $name"
  fi
  rm -f "$pidfile"
}

# Best effort: activate the conda env if we can find it, otherwise carry on
# with whatever is already on PATH. Portability beats strictness here — the
# hard requirements (uvicorn, npm) are checked right before starting.
activate_env() {
  if [ -z "${CONDA_EXE:-}" ] && ! command -v conda >/dev/null 2>&1; then
    # ${HOME:-…}: stay best-effort even with no HOME in the environment
    # (systemd unit, env -i) — set -u would otherwise abort the whole script.
    for base in /storage/modules/packages/anaconda \
                "${HOME:-/nonexistent}/anaconda3" "${HOME:-/nonexistent}/miniconda3" \
                "${HOME:-/nonexistent}/miniforge3" "${HOME:-/nonexistent}/mambaforge" \
                /opt/conda /opt/anaconda3 /opt/miniconda3 \
                /usr/local/anaconda3 /usr/local/miniconda3; do
      [ -f "$base/etc/profile.d/conda.sh" ] && { . "$base/etc/profile.d/conda.sh"; break; }
    done
  else
    local base; base="$(conda info --base 2>/dev/null)"
    [ -n "$base" ] && [ -f "$base/etc/profile.d/conda.sh" ] && . "$base/etc/profile.d/conda.sh"
  fi
  if command -v conda >/dev/null 2>&1; then
    conda activate "$ENV_NAME" 2>/dev/null \
      || warn "conda env '$ENV_NAME' not found — using tools already on PATH"
  else
    warn "conda not found — using tools already on PATH"
  fi
}

require_tools() {   # require_tools <tool...>
  local missing=() t
  for t in "$@"; do command -v "$t" >/dev/null 2>&1 || missing+=("$t"); done
  [ "${#missing[@]}" -eq 0 ] \
    || die "missing on PATH: ${missing[*]} — activate/install them (conda env '$ENV_NAME'?) and re-run"
}

# Human-readable state of the SQLite database file.
db_state() {
  if [ -f "$DB_FILE" ]; then
    local size; size=$(du -h "$DB_FILE" 2>/dev/null | cut -f1)
    printf "%sready%s (%s)" "$GRN" "$OFF" "${size:-?}"
  else
    printf "%snot created yet%s (the backend makes it on first boot)" "$YEL" "$OFF"
  fi
}

# ---------- subcommands ----------
do_status() {
  printf "\n%sGAVEL Studio — status%s\n" "$B" "$OFF"
  printf "  %-12s %-22s %s\n" "backend"  ":$BACKEND_PORT"  "$(port_open $BACKEND_PORT  && echo "${GRN}running${OFF}" || echo "${RED}stopped${OFF}")"
  printf "  %-12s %-22s %s\n" "frontend" ":$FRONTEND_PORT" "$(port_open $FRONTEND_PORT && echo "${GRN}running${OFF}" || echo "${RED}stopped${OFF}")"
  printf "  %-12s %-22s %s\n" "database" "sqlite file" "$(db_state)"
  printf "\n  %sdb:   %s%s\n"   "$DIM" "$DB_FILE" "$OFF"
  printf "  %slogs: %s%s\n\n"   "$DIM" "$LOGS" "$OFF"
}

do_stop() {
  # './gavel.sh stop --db' used to also stop Postgres. There is no database
  # server any more, so accept the flag and explain rather than error out on
  # muscle memory.
  if [ "${1:-}" = "--db" ]; then
    warn "--db is obsolete: the database is a plain file now, nothing to stop"
  fi
  step "Stopping servers"
  stop_one frontend; stop_one backend
  stop_one central   # harmless no-op unless a pre-merge central server is still around
  # Our own log tail (recorded in .run/tail.pid), then any tails leaked by
  # older sessions. The pkill pattern is ERE-escaped so a repo path containing
  # regex metacharacters cannot defuse the match.
  stop_one tail
  local esc
  esc=$(printf '%s' "$LOGS" | sed 's/[][\.^$*+?(){}|]/\\&/g')
  pkill -u "$(id -un)" -f "tail -n 0 -F $esc/backend\.log $esc/frontend\.log" 2>/dev/null && ok "cleaned up orphaned log tail(s)"
  # Leftovers we have no pidfile for (crashed run, stale reloader, hand-started
  # server) used to survive `stop` and wedge the next start — clear them by
  # port. port_occupied, not port_open: a wedged listener that no longer
  # accepts must still be found and killed.
  local port
  for port in $FRONTEND_PORT $BACKEND_PORT; do
    if port_occupied "$port"; then
      warn "something without a pidfile still holds :$port — killing it"
      kill_port "$port" && ok "freed :$port" || warn "could not free :$port (find it: ss -tlnp 'sport = :$port' or lsof -iTCP:$port)"
    fi
  done
  echo
}

do_logs() {
  tail -n 40 -F "$LOGS/backend.log" "$LOGS/frontend.log" 2>/dev/null
}

do_start() {
  activate_env
  require_tools uvicorn npm

  # --- startup mutex --------------------------------------------------------
  # The "is it already running?" guard below is a port check, so two copies of
  # this script launched within the startup window would both see closed ports
  # and both spawn servers. uvicorn would just fail loudly on the second bind,
  # but Vite does NOT: with strictPort unset it silently moves to 5174, leaving
  # an invisible duplicate dev server. Holding a lock across the spawn phase
  # makes a concurrent run WAIT, then correctly find everything already up.
  # Released before we start tailing, so `status`/`logs` are never blocked.
  # flock is not everywhere (e.g. macOS) — then we just skip the mutex.
  HAVE_FLOCK=0
  if command -v flock >/dev/null 2>&1; then
    HAVE_FLOCK=1
    exec 9>"$RUN/gavel.lock"
    if ! flock -n 9; then
      printf "%s  … another ./gavel.sh is starting the stack, waiting for it%s\n" "$DIM" "$OFF"
      flock 9
    fi
  fi

  printf "\n%sGAVEL Studio%s  %s%s%s\n\n" "$B" "$OFF" "$DIM" "$STUDIO" "$OFF"

  # --- 1. Database -----------------------------------------------------------
  # Nothing to start: it is a SQLite file the backend opens (and creates, with
  # its schema, on first boot). Reported here so the line still tells you where
  # your data lives.
  step "Database"
  ok "sqlite $(db_state)"
  printf "  %s%s%s\n" "$DIM" "$DB_FILE" "$OFF"
  # --- 2. Backend ------------------------------------------------------------
  # "Up" is judged by HTTP /health, never by the port alone: an open port can
  # be a stale uvicorn holding the socket with nothing behind it (see
  # GAVEL_RELOAD note in the header) — saying "already running" for one of
  # those left the UI stuck on "Backend not responding yet" forever.
  step "Backend (rules, training, GPU)"
  BACKEND_UP=0
  if port_occupied $BACKEND_PORT; then
    if backend_http_ok; then
      ok "already running on :$BACKEND_PORT"
      BACKEND_UP=1
    else
      BOOT_PID=$(cat "$RUN/backend.pid" 2>/dev/null || true)
      if ! port_open $BACKEND_PORT; then
        # Holds the port but cannot even complete a TCP accept: dead listener
        # with a full backlog. Nothing to wait for.
        warn "listener on :$BACKEND_PORT is wedged (accepts nothing) — clearing it"
      elif [ -n "$BOOT_PID" ] && kill -0 "$BOOT_PID" 2>/dev/null && pid_is_ours "$BOOT_PID" backend; then
        # A backend we spawned is still warming up (GAVEL_RELOAD=1 binds the
        # port before the 30-60s torch import) — grant it a real boot budget
        # instead of killing half-warmed progress.
        warn ":$BACKEND_PORT is held by a backend we spawned (pid $BOOT_PID), /health not up yet — giving it 90s"
        if wait_for_backend 90 "$BOOT_PID"; then
          ok "already running on :$BACKEND_PORT"
          BACKEND_UP=1
        fi
      else
        warn "port :$BACKEND_PORT is open but /health is not answering — giving it 20s in case it is still booting"
        if wait_for_backend 20; then
          ok "already running on :$BACKEND_PORT"
          BACKEND_UP=1
        fi
      fi
      if [ "$BACKEND_UP" -eq 0 ]; then
        warn "stale backend holds :$BACKEND_PORT — clearing it"
        stop_one backend
        kill_port $BACKEND_PORT \
          || die "could not free :$BACKEND_PORT — kill the process listening on it and re-run"
      fi
    fi
  fi
  if [ "$BACKEND_UP" -eq 0 ]; then
    RELOAD_FLAG=""
    [ "${GAVEL_RELOAD:-0}" = "1" ] && RELOAD_FLAG="--reload"
    spawn backend "$STUDIO/backend" \
      uvicorn main:app --host 127.0.0.1 --port $BACKEND_PORT $RELOAD_FLAG
    BACKEND_PID=$(cat "$RUN/backend.pid" 2>/dev/null || true)
    printf "  %sfirst boot warms up torch + transformers, ~30-60s…%s\n" "$DIM" "$OFF"
    wait_for_backend 180 "$BACKEND_PID"
    case $? in
      0) ok "started on :$BACKEND_PORT" ;;
      2) err "backend process died during startup — last lines of $LOGS/backend.log:"
         tail -n 25 "$LOGS/backend.log" 2>/dev/null | sed 's/^/    /' >&2
         stop_one backend
         die "backend did not start" ;;
      *) err "backend did not answer /health within 3 minutes — last lines of $LOGS/backend.log:"
         tail -n 25 "$LOGS/backend.log" 2>/dev/null | sed 's/^/    /' >&2
         # Kill the half-booted process: leaving it holding the port is what
         # used to force the stop/start dance on the next attempt.
         stop_one backend
         die "backend did not start" ;;
    esac
  fi

  # --- 3. Frontend -----------------------------------------------------------
  step "Interface"
  if port_open $FRONTEND_PORT; then
    ok "already running on :$FRONTEND_PORT"
  else
    if port_occupied $FRONTEND_PORT; then
      warn "listener on :$FRONTEND_PORT is wedged (accepts nothing) — clearing it"
      stop_one frontend
      kill_port $FRONTEND_PORT \
        || die "could not free :$FRONTEND_PORT — kill the process listening on it and re-run"
    fi
    # --strictPort: if the port is taken after all (race with something else),
    # fail loudly here instead of Vite silently drifting to 5174.
    spawn frontend "$STUDIO/frontend" npm run dev -- --port $FRONTEND_PORT --strictPort
    FRONTEND_PID=$(cat "$RUN/frontend.pid" 2>/dev/null || true)
    wait_for_port $FRONTEND_PORT 90 frontend "$FRONTEND_PID"
    case $? in
      0) ok "started on :$FRONTEND_PORT" ;;
      2) err "frontend process died during startup — last lines of $LOGS/frontend.log:"
         tail -n 15 "$LOGS/frontend.log" 2>/dev/null | sed 's/^/    /' >&2
         stop_one frontend
         die "frontend did not start (npm install missing? Node 20+ required)" ;;
      *) err "frontend did not open :$FRONTEND_PORT within 90s — last lines of $LOGS/frontend.log:"
         tail -n 15 "$LOGS/frontend.log" 2>/dev/null | sed 's/^/    /' >&2
         stop_one frontend
         die "frontend did not start — see $LOGS/frontend.log" ;;
    esac
  fi

  [ "$HAVE_FLOCK" -eq 1 ] && flock -u 9   # startup done — let any waiting invocation proceed

  printf "\n  %sGAVEL Studio is up →%s  %shttp://localhost:%s%s\n" "$B" "$OFF" "$CYN" "$FRONTEND_PORT" "$OFF"
  printf "  %slogs: %s%s\n"        "$DIM" "$LOGS" "$OFF"
  printf "  %sCtrl+C stops both servers%s\n\n" "$DIM" "$OFF"

  # Stream the logs; Ctrl+C tears the servers down. Closing the terminal (HUP)
  # leaves the servers up (they are detached) but must not leak the tail —
  # orphaned `tail -F` processes used to pile up across sessions.
  tail -n 0 -F "$LOGS/backend.log" "$LOGS/frontend.log" 2>/dev/null 9>&- &
  TAILPID=$!
  echo "$TAILPID" > "$RUN/tail.pid"
  trap 'echo; kill "$TAILPID" 2>/dev/null; rm -f "$RUN/tail.pid"; do_stop; exit 0' INT TERM
  trap 'kill "$TAILPID" 2>/dev/null; rm -f "$RUN/tail.pid"; exit 0' HUP
  wait $TAILPID
}

case "${1:-start}" in
  start)  do_start ;;
  stop)   do_stop "${2:-}" ;;
  status) do_status ;;
  logs)   do_logs ;;
  *)      echo "usage: ./gavel.sh [start|stop|status|logs]"; exit 1 ;;
esac
