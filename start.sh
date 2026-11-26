#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"

BACKEND_URL="http://127.0.0.1:8000"
FRONTEND_URL="http://127.0.0.1:5173"
OLLAMA_URL="http://127.0.0.1:11434"
MODEL="qwen3.5:9b-q4_K_M"

BACKEND_PID=""
FRONTEND_PID=""
OLLAMA_PID=""

cleanup() {
    echo ""
    echo "Shutting down SkinSegAI..."

    for pid in "$FRONTEND_PID" "$BACKEND_PID" "$OLLAMA_PID"; do
        if [ -n "$pid" ]; then
            kill "$pid" 2>/dev/null || true
        fi
    done

    wait 2>/dev/null || true
    echo "Shutdown complete."
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

fail() {
    echo "ERROR: $1" >&2
    exit 1
}

wait_ready() {
    local url="$1"
    local name="$2"
    local pid="$3"

    for ((i=0; i<90; i++)); do
        if [ -n "$pid" ]; then
            kill -0 "$pid" 2>/dev/null || \
                fail "$name stopped unexpectedly. Check its log."
        fi

        if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
            echo "[READY] $name"
            return 0
        fi

        sleep 1
    done

    fail "$name did not become ready."
}

echo "=========================================="
echo "         SkinSegAI Application"
echo "=========================================="

# Verify dependencies.

[ -x ".venv/bin/python" ] || \
    fail "Python virtual environment not found."

command -v node >/dev/null || fail "Node.js not found."
command -v ollama >/dev/null || fail "Ollama not found."

[ -f "frontend/node_modules/vite/bin/vite.js" ] || \
    fail "Frontend dependencies missing. Run npm ci inside frontend."

# Prevent conflicting frontend or backend instances.

for port in 8000 5173; do
    if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
        fail "Port $port is occupied. Stop the existing server first."
    fi
done

LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/skinsegai.XXXXXX")"

echo "Logs: $LOG_DIR"

# Start or reuse Ollama.

if curl -fsS --max-time 2 "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
    echo "[READY] Existing Ollama service"
else
    echo "[STARTING] Ollama..."

    ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
    OLLAMA_PID=$!

    wait_ready "$OLLAMA_URL/api/tags" "Ollama" "$OLLAMA_PID"
fi

if ! ollama list | awk 'NR > 1 {print $1}' | grep -xF "$MODEL" >/dev/null; then
    fail "Required Ollama model is missing: $MODEL"
fi

echo "[READY] Local AI model: $MODEL"

# Start FastAPI.

echo "[STARTING] Python backend..."

"$ROOT/.venv/bin/python" -m uvicorn backend.main:app \
    --host 127.0.0.1 \
    --port 8000 \
    --workers 1 \
    > "$LOG_DIR/backend.log" 2>&1 &

BACKEND_PID=$!

wait_ready "$BACKEND_URL/api/health" "FastAPI" "$BACKEND_PID"

# Start React.

echo "[STARTING] React frontend..."

(
    cd "$ROOT/frontend"
    exec node node_modules/vite/bin/vite.js \
        --host 127.0.0.1 \
        --port 5173 \
        --strictPort
) > "$LOG_DIR/frontend.log" 2>&1 &

FRONTEND_PID=$!

wait_ready "$FRONTEND_URL" "React" "$FRONTEND_PID"

echo ""
echo "=========================================="
echo "        ALL SERVICES ARE READY"
echo "=========================================="
echo "Application: $FRONTEND_URL"
echo "Backend API: $BACKEND_URL"
echo "API Docs:    $BACKEND_URL/docs"
echo "AI Model:    $MODEL"
echo "Logs:        $LOG_DIR"
echo ""
echo "Press Ctrl+C to stop the application."
echo ""

open "$FRONTEND_URL" || true

# Monitor service availability.

while true; do
    kill -0 "$BACKEND_PID" 2>/dev/null || \
        fail "Backend stopped. Check $LOG_DIR/backend.log"

    kill -0 "$FRONTEND_PID" 2>/dev/null || \
        fail "Frontend stopped. Check $LOG_DIR/frontend.log"

    if [ -n "$OLLAMA_PID" ]; then
        kill -0 "$OLLAMA_PID" 2>/dev/null || \
            fail "Ollama stopped. Check $LOG_DIR/ollama.log"
    fi

    sleep 2
done
