#!/usr/bin/env bash
# setup_worker.sh — set up and start the GAVEL GPU worker ON a GPU box.
#
# Run this ON the GPU machine (RunPod / AWS / Colab / a lab box), from inside
# the cloned repo's gavel-gpu-worker/ folder:
#
#     git clone https://github.com/Offensive-AI-Lab/gavel-studio.git
#     cd gavel-studio/gavel-gpu-worker
#     ./setup_worker.sh
#
# It stages the backend's ML code, installs everything, generates a token,
# starts the worker, and prints the two lines to paste into your backend's
# backend/.env. Safe to re-run (it restarts the worker cleanly).
#
# Options (all optional):
#   --url <https-url>   Your provider's public HTTPS URL for this box (e.g. the
#                       RunPod proxy URL). Only used to print the GPU_WORKER_URL
#                       line at the end; if omitted you fill it in yourself.
#   --token <token>     Use this WORKER_TOKEN instead of generating one.
#   --port <n>          Port the worker listens on (default 8000).
#   --foreground        Run in this terminal instead of the background.
#   --no-start          Stage + install only; don't start the worker.
set -u

PORT=8000
URL=""
TOKEN="${WORKER_TOKEN:-}"
FOREGROUND=0
START=1

while [ $# -gt 0 ]; do
    case "$1" in
        --url)        URL="$2"; shift 2 ;;
        --token)      TOKEN="$2"; shift 2 ;;
        --port)       PORT="$2"; shift 2 ;;
        --foreground|--fg) FOREGROUND=1; shift ;;
        --no-start)   START=0; shift ;;
        -h|--help)    awk 'NR==1{next} /^#/{sub(/^# ?/,"");print;next} {exit}' "$0"; exit 0 ;;
        *)            echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

if [ -t 1 ]; then
    R=$'\033[31m'; G=$'\033[32m'; Y=$'\033[33m'; B=$'\033[34m'; BO=$'\033[1m'; X=$'\033[0m'
else R=""; G=""; Y=""; B=""; BO=""; X=""; fi
say()  { printf '%s\n' "$*"; }
ok()   { printf '%s %s\n' "${G}OK${X}" "$*"; }
warn() { printf '%s %s\n' "${Y}! ${X}" "$*"; }
err()  { printf '%s %s\n' "${R}x ${X}" "$*" >&2; }
step() { printf '\n%s %s\n' "${B}>${X}" "${BO}$*${X}"; }
abort(){ err "$1"; exit 1; }

cd "$(dirname "$0")" || abort "cannot cd to the script's directory"
WORKER_DIR="$(pwd)"
PIDFILE="$HOME/.gavel_worker.pid"
LOG="$HOME/gavel_worker.log"

launch_cmd() {
    if command -v gavel-gpu-worker >/dev/null 2>&1; then echo "gavel-gpu-worker"
    else echo "$PY -m gavel_gpu_worker.app"; fi
}

print_connect() {
    printf '\n%s================ put these in your backend/.env ================%s\n' "$BO" "$X"
    if [ -n "$URL" ]; then
        printf 'GPU_WORKER_URL=%s\n' "$URL"
    else
        printf 'GPU_WORKER_URL=https://<your-box-public-https-url>%s   # <-- paste your provider URL%s\n' "$Y" "$X"
    fi
    printf 'GPU_WORKER_TOKEN=%s\n' "$TOKEN"
    printf '%s===============================================================%s\n' "$BO" "$X"
    [ -z "$URL" ] && warn "The worker serves plain HTTP on :$PORT — expose it over HTTPS and use that URL above."
    say ""
    say "Logs:  tail -f $LOG       Stop:  kill \$(cat $PIDFILE)"
}

printf '\n%sGAVEL GPU worker setup%s\n' "$BO" "$X"
say "Sets up and starts the worker on this GPU box."

# --- prerequisites ---------------------------------------------------------
step "Checking prerequisites"
PY=""
for c in python3 python; do command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }; done
[ -n "$PY" ] || abort "Python not found. Install Python 3.10+ and re-run."
ok "python: $($PY --version 2>&1)"
$PY -m pip --version >/dev/null 2>&1 || abort "pip is not available for $PY."
if command -v nvidia-smi >/dev/null 2>&1; then
    ok "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
else
    warn "no nvidia-smi found — the worker will run on CPU (slow but works)."
fi
[ -f "scripts/stage_engine.py" ] || abort "Run this from gavel-gpu-worker/ (scripts/stage_engine.py not found)."

# --- stage the backend's ML code ------------------------------------------
step "Staging engine code (copying the backend's ML code in)"
$PY scripts/stage_engine.py || abort "Staging failed — is the full repo present (backend/ next to gavel-gpu-worker/)?"

# --- install ---------------------------------------------------------------
step "Installing dependencies (first run downloads torch etc. — a few minutes)"
[ -f gavel_code/engine-requirements.txt ] || abort "gavel_code/engine-requirements.txt missing after staging."
$PY -m pip install -q -r gavel_code/engine-requirements.txt || abort "ML dependency install failed."
$PY -m pip install -q .                                     || abort "worker install failed."
ok "dependencies installed"

# --- token -----------------------------------------------------------------
step "Worker token"
if [ -z "$TOKEN" ]; then
    TOKEN="$($PY - <<'EOF'
import secrets
print(secrets.token_hex(24))
EOF
)"
    ok "generated a new WORKER_TOKEN"
else
    ok "using the token you provided"
fi

EV="$(GAVEL_CODE_DIR="$WORKER_DIR/gavel_code" $PY -c 'import gavel_gpu_worker.config as c; print(c.engine_version())' 2>/dev/null)"
[ -n "$EV" ] && ok "engine version: $EV (must match your backend)" || warn "could not read engine version"

PY_ENV() { env WORKER_TOKEN="$TOKEN" WORKER_PORT="$PORT" GAVEL_CODE_DIR="$WORKER_DIR/gavel_code" "$@"; }

if [ "$START" = "0" ]; then
    step "Done (worker not started, per --no-start)"
    say "Start it later with:"
    say "  WORKER_TOKEN=$TOKEN WORKER_PORT=$PORT $(launch_cmd)"
    print_connect
    exit 0
fi

# --- start -----------------------------------------------------------------
step "Starting the worker"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
    kill "$(cat "$PIDFILE")" 2>/dev/null; sleep 1
    ok "stopped the previously-started worker"
fi

LCMD="$(launch_cmd)"

if [ "$FOREGROUND" = "1" ]; then
    print_connect
    say ""
    say "Running in the foreground — press Ctrl+C to stop."
    exec env WORKER_TOKEN="$TOKEN" WORKER_PORT="$PORT" GAVEL_CODE_DIR="$WORKER_DIR/gavel_code" $LCMD
fi

PY_ENV nohup $LCMD >"$LOG" 2>&1 &
echo $! > "$PIDFILE"
sleep 3

healthy=0
if command -v curl >/dev/null 2>&1; then
    curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1 && healthy=1
fi
if [ "$healthy" = "0" ]; then
    $PY - "$PORT" <<'EOF' >/dev/null 2>&1 && healthy=1
import sys, urllib.request
urllib.request.urlopen("http://localhost:%s/health" % sys.argv[1], timeout=5).read()
EOF
fi

if [ "$healthy" = "1" ]; then
    ok "worker is up (pid $(cat "$PIDFILE"))"
else
    warn "worker didn't answer /health yet — check the log: $LOG"
fi
print_connect
