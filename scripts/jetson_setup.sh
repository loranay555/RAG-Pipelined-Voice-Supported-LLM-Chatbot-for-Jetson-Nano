#!/usr/bin/env bash
# One-time host preparation for a Jetson Orin Nano/NX 8 GB.
# Run with sudo. Everything here is about making 8 GB of shared CPU+GPU memory
# survive llama.cpp + whisper.cpp + qdrant running at the same time.
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
    echo "Run with sudo: sudo bash $0" >&2
    exit 1
fi

echo "== 1. Docker default runtime =="
# Without this, `docker build` cannot see the CUDA libraries and the llama.cpp
# / whisper.cpp images fail to compile.
DAEMON=/etc/docker/daemon.json
if ! grep -q '"default-runtime"' "${DAEMON}" 2>/dev/null; then
    cp "${DAEMON}" "${DAEMON}.bak.$(date +%s)" 2>/dev/null || true
    python3 - "${DAEMON}" <<'PY'
import json, sys, pathlib
path = pathlib.Path(sys.argv[1])
cfg = json.loads(path.read_text()) if path.exists() and path.read_text().strip() else {}
cfg.setdefault("runtimes", {})["nvidia"] = {
    "path": "nvidia-container-runtime", "runtimeArgs": []
}
cfg["default-runtime"] = "nvidia"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(cfg, indent=2))
print("updated", path)
PY
    systemctl restart docker
else
    echo "  already configured"
fi

echo "== 2. Power mode =="
# Mode IDs are NOT portable: on an Orin NX 8 GB reference kit mode 0 is 15W,
# not MAXN. Pick by name/wattage instead of hardcoding an ID.
CONF=/etc/nvpmodel.conf
if [[ -r "${CONF}" ]] && command -v nvpmodel >/dev/null; then
    echo "  available modes:"
    grep -oP '< POWER_MODEL ID=\K[0-9]+ NAME=\S+' "${CONF}" | sed 's/ NAME=/  /' | sed 's/^/    /'

    # `|| true`: no MAXN entry is normal on some boards, and a bare grep
    # miss would kill the script under `set -e`.
    TARGET="$(grep -oP '< POWER_MODEL ID=\K[0-9]+(?= NAME=MAXN)' "${CONF}" | head -1 || true)"
    if [[ -z "${TARGET}" ]]; then
        # no MAXN on this board -> take the highest advertised wattage
        TARGET="$(grep -oP '< POWER_MODEL ID=\K[0-9]+ NAME=[0-9]+W' "${CONF}" \
                  | sed 's/ NAME=/ /; s/W$//' \
                  | sort -k2 -n | tail -1 | cut -d' ' -f1 || true)"
    fi

    if [[ -n "${TARGET}" ]]; then
        echo "  switching to mode ${TARGET}"
        nvpmodel -m "${TARGET}" || echo "  (nvpmodel -m failed)"
    else
        echo "  !! could not determine the max mode; set it yourself: nvpmodel -m <id>"
    fi
    nvpmodel -q || true
else
    echo "  (nvpmodel unavailable, skipping)"
fi
jetson_clocks || echo "  (jetson_clocks unavailable, skipping)"

echo "== 3. Swap =="
# The board ships with 4 GB of zram, which is compressed *RAM* and therefore
# does not add headroom. A real swapfile on NVMe does, and stops the OOM killer
# from taking out llama.cpp during a long model load.
if ! swapon --show | grep -q /swapfile; then
    fallocate -l 8G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
    echo "  8 GB swapfile enabled"
else
    echo "  already present"
fi

sysctl -w vm.swappiness=10 >/dev/null
grep -q '^vm.swappiness' /etc/sysctl.conf || echo 'vm.swappiness=10' >> /etc/sysctl.conf

echo "== 4. Optional: run headless =="
echo "  The desktop session costs ~800 MB. To reclaim it:"
echo "    sudo systemctl set-default multi-user.target && sudo reboot"

echo
echo "Done. Check free memory with: sudo tegrastats"
