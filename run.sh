#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  run.sh — Bangla OCR App launcher
#
#  Usage:
#    ./run.sh docker start   — Run app inside Docker container
#    ./run.sh docker stop    — Stop and remove the container
#    ./run.sh local start    — Run app in local .venv
#    ./run.sh local stop     — Kill the local Streamlit process
#
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# ── Config ───────────────────────────────────────────────────
DOCKER_IMAGE="bangla-ocr-app:0.1"
DOCKER_CONTAINER="bangla-ocr"
PORT=8501
VENV_DIR="$(dirname "$0")/.venv"
APP_FILE="$(dirname "$0")/app.py"
PID_FILE="/tmp/bangla-ocr-local.pid"

# ── Helpers ──────────────────────────────────────────────────
info()    { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
success() { echo -e "\033[1;32m[OK]\033[0m    $*"; }
warn()    { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
error()   { echo -e "\033[1;31m[ERROR]\033[0m $*"; exit 1; }

usage() {
    echo ""
    echo "  Usage: ./run.sh <mode> <action>"
    echo ""
    echo "  Modes:"
    echo "    docker   — Run inside Docker container (model baked in)"
    echo "    local    — Run directly with local .venv"
    echo ""
    echo "  Actions:"
    echo "    start    — Start the app"
    echo "    stop     — Stop the app"
    echo ""
    echo "  Examples:"
    echo "    ./run.sh docker start"
    echo "    ./run.sh docker stop"
    echo "    ./run.sh local start"
    echo "    ./run.sh local stop"
    echo ""
    exit 1
}

# ── Docker mode ───────────────────────────────────────────────
docker_start() {
    info "Checking Docker..."
    command -v docker >/dev/null 2>&1 || error "Docker is not installed or not in PATH."

    # Check image exists
    if ! docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
        error "Docker image '$DOCKER_IMAGE' not found. Build it first:\n  docker build -t $DOCKER_IMAGE ."
    fi

    # Remove stale container if it exists
    if docker ps -a --format '{{.Names}}' | grep -q "^${DOCKER_CONTAINER}$"; then
        warn "Container '$DOCKER_CONTAINER' already exists — removing it first."
        docker rm -f "$DOCKER_CONTAINER" >/dev/null
    fi

    info "Starting Docker container '$DOCKER_CONTAINER' on port $PORT..."
    docker run -d \
        --name "$DOCKER_CONTAINER" \
        -p "${PORT}:8501" \
        "$DOCKER_IMAGE" >/dev/null

    # Wait for health check
    info "Waiting for Streamlit to be ready..."
    for i in {1..20}; do
        if curl -sf "http://localhost:${PORT}/_stcore/health" >/dev/null 2>&1; then
            success "App is running → http://localhost:${PORT}"
            return 0
        fi
        sleep 1
    done
    warn "Health check timed out — container may still be starting."
    info "Check logs with: docker logs $DOCKER_CONTAINER"
}

docker_stop() {
    if docker ps --format '{{.Names}}' | grep -q "^${DOCKER_CONTAINER}$"; then
        info "Stopping container '$DOCKER_CONTAINER'..."
        docker stop "$DOCKER_CONTAINER" >/dev/null
        docker rm "$DOCKER_CONTAINER" >/dev/null
        success "Container stopped and removed."
    else
        warn "No running container named '$DOCKER_CONTAINER' found."
    fi
}

# ── Local mode ────────────────────────────────────────────────
local_start() {
    # Validate venv
    if [ ! -f "$VENV_DIR/bin/python" ]; then
        error "Virtual environment not found at '$VENV_DIR'.\nCreate it first:\n  python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    fi

    # Validate app file
    if [ ! -f "$APP_FILE" ]; then
        error "app.py not found at '$APP_FILE'."
    fi

    # Check if already running
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        warn "Streamlit already running (PID $(cat "$PID_FILE"))."
        info "  Stop it first with: ./run.sh local stop"
        exit 1
    fi

    info "Starting Streamlit with local .venv on port $PORT..."
    cd "$(dirname "$APP_FILE")"
    nohup "$VENV_DIR/bin/streamlit" run app.py \
        --server.port="$PORT" \
        --server.headless=true \
        --browser.gatherUsageStats=false \
        > /tmp/bangla-ocr-local.log 2>&1 &

    echo $! > "$PID_FILE"
    info "Waiting for Streamlit to be ready..."

    for i in {1..20}; do
        if curl -sf "http://localhost:${PORT}/_stcore/health" >/dev/null 2>&1; then
            success "App is running → http://localhost:${PORT}  (PID $(cat "$PID_FILE"))"
            info "  Logs: tail -f /tmp/bangla-ocr-local.log"
            return 0
        fi
        sleep 1
    done
    warn "Health check timed out — check logs: tail -f /tmp/bangla-ocr-local.log"
}

local_stop() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        PID=$(cat "$PID_FILE")
        info "Stopping Streamlit (PID $PID)..."
        kill "$PID"
        rm -f "$PID_FILE"
        success "Streamlit stopped."
    else
        # Try finding it by port as fallback
        PID=$(lsof -ti tcp:"$PORT" 2>/dev/null | head -1 || true)
        if [ -n "$PID" ]; then
            info "Stopping process on port $PORT (PID $PID)..."
            kill "$PID"
            rm -f "$PID_FILE"
            success "Stopped."
        else
            warn "No local Streamlit process found on port $PORT."
        fi
    fi
}

# ── Entrypoint ────────────────────────────────────────────────
MODE="${1:-}"
ACTION="${2:-}"

[ -z "$MODE" ] || [ -z "$ACTION" ] && usage

case "$MODE" in
    docker)
        case "$ACTION" in
            start) docker_start ;;
            stop)  docker_stop  ;;
            *)     usage ;;
        esac
        ;;
    local)
        case "$ACTION" in
            start) local_start ;;
            stop)  local_stop  ;;
            *)     usage ;;
        esac
        ;;
    *)
        usage
        ;;
esac
