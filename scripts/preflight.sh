#!/usr/bin/env bash
# Read-only environment check. Changes nothing. Run on the Jetson before
# copying the project over:  bash preflight.sh
echo "=============================================================="
echo " 1. BOARD / JETPACK"
echo "=============================================================="
cat /etc/nv_tegra_release 2>/dev/null || echo "!! /etc/nv_tegra_release yok - bu bir Jetson mu?"
[ -f /sys/firmware/devicetree/base/model ] && { tr -d '\0' < /sys/firmware/devicetree/base/model; echo; }
dpkg -l 2>/dev/null | grep -E "nvidia-jetpack " || echo "(nvidia-jetpack paketi listelenemedi)"
echo "arch      : $(uname -m)"
echo "kernel    : $(uname -r)"
echo "os        : $(. /etc/os-release && echo "$PRETTY_NAME")"

echo
echo "=============================================================="
echo " 2. MEMORY / SWAP / POWER"
echo "=============================================================="
free -h
echo "--- swap ---"
swapon --show 2>/dev/null || echo "(swap yok)"
zramctl 2>/dev/null | head -5
echo "--- power mode ---"
nvpmodel -q 2>/dev/null || echo "(nvpmodel calistirilamadi, sudo gerekebilir)"
echo "--- graphical session ---"
systemctl get-default 2>/dev/null

echo
echo "=============================================================="
echo " 3. DISK  (build + modeller icin ~40 GB bos alan gerekli)"
echo "=============================================================="
df -h / /var/lib/docker 2>/dev/null | sort -u
echo "--- blok cihazlar (NVMe mi SD kart mi?) ---"
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT,ROTA 2>/dev/null

echo
echo "=============================================================="
echo " 4. DOCKER"
echo "=============================================================="
docker --version 2>/dev/null || echo "!! docker yok"
docker compose version 2>/dev/null || echo "!! 'docker compose' (v2) yok"
echo "--- kullanici docker grubunda mi? ---"
id -nG | tr ' ' '\n' | grep -qx docker && echo "EVET (sudo gerekmez)" || echo "HAYIR (her komutta sudo gerekecek)"
echo "--- runtimes ---"
docker info 2>/dev/null | grep -iE "runtime|storage driver|docker root dir" || echo "!! docker info calismadi"

echo
echo "=============================================================="
echo " 5. NVIDIA CONTAINER RUNTIME"
echo "=============================================================="
which nvidia-container-runtime nvidia-ctk 2>/dev/null || echo "!! nvidia-container-runtime bulunamadi"
dpkg -l 2>/dev/null | grep -E "nvidia-container|libnvidia-container" | awk '{print $2, $3}'
echo "--- /etc/docker/daemon.json ---"
cat /etc/docker/daemon.json 2>/dev/null || echo "(dosya yok)"

echo
echo "=============================================================="
echo " 6. AG + NGC ERISIMI"
echo "=============================================================="
echo "LAN IP    : $(ip -4 route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}')"
echo "hostname  : $(hostname)"
curl -s -o /dev/null -w "huggingface : HTTP %{http_code}\n" --max-time 15 https://huggingface.co
curl -s -o /dev/null -w "nvcr.io     : HTTP %{http_code}\n" --max-time 15 https://nvcr.io/v2/
curl -s -o /dev/null -w "open-meteo  : HTTP %{http_code}\n" --max-time 15 https://api.open-meteo.com/v1/forecast?latitude=39.9\&longitude=32.8\&current=temperature_2m

echo
echo "=============================================================="
echo " 7. GPU KONTEYNERDEN GORUNUYOR MU?  (asil test)"
echo "=============================================================="
TAG="nvcr.io/nvidia/l4t-cuda:12.6.11-runtime"
echo "cekiliyor: ${TAG}  (birkac dakika surebilir)"
if docker pull -q "${TAG}" >/dev/null 2>&1; then
    echo "pull      : OK"
    docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all \
        "${TAG}" bash -c 'ls /usr/local/cuda*/lib64/libcudart.so* 2>/dev/null | head -3; echo "cuda kutuphaneleri yukaridaki gibi gorunuyorsa GPU baglanmis demektir"' \
        || echo "!! --runtime nvidia ile calistirilamadi"
else
    echo "!! ${TAG} cekilemedi -- JetPack surumunuze uyan etiketi bulmamiz gerekecek"
fi

echo
echo "=============================================================="
echo " BITTI - ciktinin tamamini gonderin"
echo "=============================================================="
