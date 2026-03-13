#!/usr/bin/env bash
# deploy.sh — Rebuild, reset optimizer config, and restart the trader stack.
#
# What this does:
#   1. Stop all running containers
#   2. Reset /data/config.json from the project source (wipes optimizer drift)
#   3. Rebuild the collector image with the latest code
#   4. Start all services
#   5. Confirm the config reset inside the running container
#
# Usage:
#   ./deploy.sh              # full rebuild + config reset
#   ./deploy.sh --no-reset   # rebuild only, keep existing optimizer config

set -euo pipefail

# ── Terminal colours ──────────────────────────────────────────────────────────
BOLD='\033[1m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
DIM='\033[2m'
NC='\033[0m'

step()  { echo -e "\n${CYAN}${BOLD}▶  $1${NC}"; }
ok()    { echo -e "   ${GREEN}✓  $1${NC}"; }
warn()  { echo -e "   ${YELLOW}⚠  $1${NC}"; }
fatal() { echo -e "   ${RED}✗  $1${NC}"; exit 1; }
info()  { echo -e "   ${DIM}$1${NC}"; }

# ── Parse flags ───────────────────────────────────────────────────────────────
RESET_CONFIG=true
for arg in "$@"; do
    case $arg in
        --no-reset) RESET_CONFIG=false ;;
        -h|--help)
            echo "Usage: $0 [--no-reset]"
            echo "  --no-reset   Skip config reset — keeps current optimizer-tuned config"
            exit 0
            ;;
        *) fatal "Unknown argument: $arg" ;;
    esac
done

# ── Ensure we run from project root ──────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Header ────────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}╔══════════════════════════════════════════════╗"
echo       "║   Pknwitq Trader  —  Deploy & Reset          ║"
echo -e    "╚══════════════════════════════════════════════╝${NC}"
echo -e "   ${DIM}$(date -u '+%Y-%m-%d %H:%M UTC')${NC}\n"

# ── Preflight checks ─────────────────────────────────────────────────────────
step "Preflight checks"

command -v docker >/dev/null 2>&1 || fatal "docker is not installed or not in PATH"
docker compose version >/dev/null 2>&1 || fatal "docker compose (v2) is not available"

[ -f "docker-compose.yml" ] || fatal "docker-compose.yml not found — run from project root"
[ -f "trader/config.json" ] || fatal "trader/config.json not found"

if [ ! -f ".env" ]; then
    warn ".env file not found — API keys may be missing (trading will fail)"
else
    ok ".env present"
fi

ok "All preflight checks passed"

# ── Step 1: Stop running services ────────────────────────────────────────────
step "Stopping running services"
docker compose down --remove-orphans 2>&1 | sed 's/^/   /'
ok "Services stopped"

# ── Step 2: Reset optimizer config ───────────────────────────────────────────
if [ "$RESET_CONFIG" = true ]; then
    step "Resetting optimizer config from project source"
    mkdir -p ./data
    cp trader/config.json data/config.json
    ok "data/config.json  ←  trader/config.json (optimizer drift cleared)"

    # Show which sections are now active
    info "Active sections: scheduler · market · signal · trading"
    info "Optimizer will re-tune from scratch on the next 15+ resolved trades"
else
    step "Skipping config reset (--no-reset)"
    info "Keeping existing data/config.json (optimizer-tuned values preserved)"
fi

# ── Step 3: Rebuild collector image ──────────────────────────────────────────
step "Rebuilding collector image"
docker compose build collector 2>&1 | grep -E "^(Step|---> |Successfully|#[0-9]|\[)" | sed 's/^/   /' || true
ok "Collector image rebuilt with latest code and config"

# ── Step 4: Start all services ───────────────────────────────────────────────
step "Starting all services"
docker compose up -d 2>&1 | sed 's/^/   /'
ok "Services started"

# ── Step 5: Wait for collector to be healthy ─────────────────────────────────
step "Waiting for collector to initialise"
info "Watching for /data/status.json (max 90s)..."

MAX_WAIT=90
ELAPSED=0
READY=false
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if docker exec pknwitq-collector test -f /data/status.json 2>/dev/null; then
        READY=true
        break
    fi
    printf "   ${DIM}.${NC}"
    sleep 3
    ELAPSED=$((ELAPSED + 3))
done
echo

if [ "$READY" = true ]; then
    ok "Collector is ready"
else
    warn "Collector not ready after ${MAX_WAIT}s — check logs:"
    info "  docker compose logs collector"
fi

# ── Step 6: Verify config is live inside the container ───────────────────────
# No copy needed — data/config.json was written from the project source in step 2
# and the bind-mount makes it immediately visible to the container as /data/config.json.
# Running cp /app/config.json here would overwrite it with the image-baked copy,
# which may be stale if Docker reused a cached COPY layer during the build.
if [ "$RESET_CONFIG" = true ]; then
    step "Verifying config is live inside container"
    if docker exec pknwitq-collector test -f /data/config.json 2>/dev/null; then
        ok "/data/config.json is present and live (sourced from trader/config.json)"
    else
        warn "/data/config.json not visible inside container — check volume mount"
    fi
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo -e "\n${GREEN}${BOLD}✓  Deploy complete${NC}\n"

# Print container status
docker compose ps 2>/dev/null | sed 's/^/   /' || true

echo
echo -e "   ${BOLD}Dashboard:${NC}  http://localhost:5000/live.html"
echo
echo -e "   ${DIM}Useful commands:"
echo -e "     docker compose logs -f collector   # live trader logs"
echo -e "     docker compose logs -f monitor     # dashboard/API logs"
echo -e "     docker compose down                # stop everything"
echo -e "     ./deploy.sh --no-reset             # redeploy, keep optimizer config${NC}"
echo
