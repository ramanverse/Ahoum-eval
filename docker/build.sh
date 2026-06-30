#!/usr/bin/env bash
# =============================================================================
# build.sh — Helper script to build, run, and manage Ahoum Docker containers
# =============================================================================
# Usage:
#   ./docker/build.sh build      Build both UI and API images
#   ./docker/build.sh up         Start all services
#   ./docker/build.sh down       Stop all services
#   ./docker/build.sh test       Run smoke tests against running containers
#   ./docker/build.sh logs       Tail logs from all services
#   ./docker/build.sh clean      Remove containers, images, and volumes
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

log()  { echo -e "\033[1;36m[ahoum]\033[0m $*"; }
ok()   { echo -e "\033[1;32m[✓]\033[0m $*"; }
err()  { echo -e "\033[1;31m[✗]\033[0m $*" >&2; exit 1; }

CMD="${1:-up}"

case "$CMD" in

  build)
    log "Building ahoum-ui and ahoum-api images..."
    docker compose -f "$SCRIPT_DIR/docker-compose.yml" build --parallel
    ok "Images built successfully"
    ;;

  up)
    log "Starting Ahoum services..."
    docker compose -f "$SCRIPT_DIR/docker-compose.yml" up --build -d
    log "Waiting for services to be healthy..."
    sleep 5

    # Poll health endpoints
    for svc in "http://localhost:8501/_stcore/health" "http://localhost:8080/health"; do
      for i in {1..12}; do
        if curl -sf "$svc" > /dev/null 2>&1; then
          ok "$svc is up"
          break
        fi
        [ "$i" -eq 12 ] && err "Service $svc failed to start within 60s"
        sleep 5
      done
    done

    echo ""
    log "═══════════════════════════════════════════"
    log "  Streamlit UI  →  http://localhost:8501"
    log "  FastAPI Docs  →  http://localhost:8080/docs"
    log "  API Health    →  http://localhost:8080/health"
    log "═══════════════════════════════════════════"
    ;;

  down)
    log "Stopping Ahoum services..."
    docker compose -f "$SCRIPT_DIR/docker-compose.yml" down
    ok "Services stopped"
    ;;

  test)
    log "Running smoke tests against running containers..."

    # Health check
    curl -sf http://localhost:8080/health | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='ok', d" \
      && ok "API /health OK"

    # Facets endpoint
    curl -sf "http://localhost:8080/facets?limit=5" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['total']>0, d" \
      && ok "API /facets OK"

    # Evaluate endpoint
    PAYLOAD='{"conversation_id":"smoke-test","domain":"general","mode":"feature","turns":[{"speaker":"user","text":"Hello, can you help me?"},{"speaker":"assistant","text":"Of course! What do you need help with?"}]}'
    curl -sf -X POST http://localhost:8080/evaluate \
      -H "Content-Type: application/json" \
      -d "$PAYLOAD" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['num_facets_evaluated']==300, d" \
      && ok "API /evaluate (300 facets) OK"

    # Streamlit UI
    curl -sf http://localhost:8501/_stcore/health > /dev/null && ok "Streamlit UI OK"

    echo ""
    ok "All smoke tests passed ✓"
    ;;

  logs)
    docker compose -f "$SCRIPT_DIR/docker-compose.yml" logs -f
    ;;

  clean)
    log "Removing containers, images, and volumes..."
    docker compose -f "$SCRIPT_DIR/docker-compose.yml" down -v --rmi local
    ok "Cleanup complete"
    ;;

  *)
    echo "Usage: $0 {build|up|down|test|logs|clean}"
    exit 1
    ;;

esac
