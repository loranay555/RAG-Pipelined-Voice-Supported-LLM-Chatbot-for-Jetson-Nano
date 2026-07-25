#!/usr/bin/env bash
# Writes the Jetson's current LAN IP into .env as SITE_ADDRESS.
# Re-run this whenever DHCP hands the board a different address.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

IP="${1:-}"
if [[ -z "${IP}" ]]; then
    IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}')"
fi

if [[ -z "${IP}" ]]; then
    echo "Could not detect a LAN IP. Pass it explicitly: $0 192.168.1.50" >&2
    exit 1
fi

[[ -f .env ]] || cp .env.example .env

HTTPS_PORT="$(grep -E '^HTTPS_PORT=' .env | cut -d= -f2 || true)"
HTTPS_PORT="${HTTPS_PORT:-8443}"
ADDRESS="https://${IP}:${HTTPS_PORT}"

set_var() {
    local key="$1" value="$2"
    if grep -qE "^${key}=" .env; then
        sed -i "s|^${key}=.*|${key}=${value}|" .env
    else
        echo "${key}=${value}" >> .env
    fi
}

set_var SITE_ADDRESS "${ADDRESS}"
# Caddy needs the bare host separately: clients connecting to an IP send no SNI.
set_var SITE_HOST "${IP}"

echo "SITE_ADDRESS=${ADDRESS}"
echo "SITE_HOST=${IP}"
echo
echo "Open this on your phone (same wifi):  ${ADDRESS}"
echo "The certificate is self-signed, so accept the browser warning once."
echo "Restart the proxy to pick it up:      docker compose up -d --force-recreate caddy"
