# Jetson Sesli Asistan

Jetson Orin 8 GB üzerinde tamamen yerel çalışan, aynı wifi ağındaki
telefondan/bilgisayardan erişilebilen sesli + yazılı asistan.

```
telefon ─https─► caddy ─► backend (FastAPI)
                            ├─► stt       whisper.cpp  (CUDA)
                            ├─► llm       llama.cpp    (CUDA)  Qwen3-4B
                            ├─► embedder  onnxruntime  (CPU)   multilingual-e5-small
                            ├─► qdrant    vektör DB
                            └─► internet  web araması, hava durumu
```

Ses akışı: tarayıcı mikrofon → 16 kHz PCM → WebSocket → sunucuda VAD ile cümle
kesme → whisper.cpp → metin → araçlar → llama.cpp → token token geri.

Jetson Orin NX 8 GB / JetPack 6.0 üzerinde çalışır durumda doğrulandı.
Ölçülen değerler bölüm 8'de.

---

## 1. Kurulum

### Ön koşullar

JetPack 6.x kurulu bir Orin ve wifi bağlantısı. Ortamı kontrol etmek için:

```bash
bash scripts/preflight.sh
```

Kart, JetPack sürümü, disk, Docker, nvidia runtime ve ağ erişimini raporlar.
Hiçbir şeyi değiştirmez.

### Docker

JetPack Docker'ı kurulu getirmeyebilir. Kuruluysa **sürümüne dikkat edin** —
Docker 28+ L4T çekirdeğinde çalışmıyor (sebebi bölüm 5'te):

```bash
sudo apt-get update && sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=arm64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu jammy stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
sudo apt-get update

V=5:27.5.1-1~ubuntu.22.04~jammy
sudo apt-get install -y docker-ce=$V docker-ce-cli=$V containerd.io \
                        docker-buildx-plugin docker-compose-plugin
sudo apt-mark hold docker-ce docker-ce-cli    # 28'e yükselmesin
sudo usermod -aG docker $USER
```

Ardından **oturumu kapatıp açın** (grup üyeliği için).

### Asistan

```bash
cd jetson-assistant

# 1) host hazırlığı: nvidia default runtime, güç modu, 8 GB swap
make setup

# 2) modelleri indir (~3.1 GB)
make models

# 3) imajları derle — llama.cpp ve whisper.cpp CUDA derlemesi
#    Orin NX 15W'ta ~40 dakika (ikisi paralel), bir kere yapılır
make build 2>&1 | tee build.log

# 4) çalıştır
make up
make health
```

`make setup` LAN IP'nizi bulup `.env`'e yazar. Kart birden fazla ağda görünüyorsa
(ör. Tailscale de varsa) doğru IP'yi elle verin:

```bash
bash scripts/set_lan_ip.sh 192.168.1.68
```

Telefondan açacağınız adres:

```
https://192.168.1.68:8443
```

> **Sertifika uyarısı normaldir.** Caddy kendi iç sertifika otoritesiyle imza
> atıyor. "Gelişmiş → Yine de devam et" dedikten sonra mikrofon çalışır.
> `https://` yazmayı unutmayın; telefon tarayıcıları portlu adreste otomatik
> HTTPS'e geçmez.

IP değişirse: `bash scripts/set_lan_ip.sh <yeni-ip> && docker compose up -d --force-recreate caddy`

---

## 2. Neden bu parçalar?

| Parça | Seçim | Gerekçe |
|---|---|---|
| LLM | Qwen3-4B-Instruct-2507 Q4_K_M | ~2.6 GB. 8 GB paylaşımlı bellekte whisper + qdrant ile birlikte sığıyor. Tool calling şablonu GGUF'un içinde. Düşünme modu olmadığı için sesli kullanımda gecikme düşük. |
| STT | whisper.cpp `small-q5_1` | Türkçe + İngilizce aynı modelde. Streaming Conformer/Zipformer modelleri daha düşük gecikmeli ama Türkçe desteği yok. |
| Embedding | multilingual-e5-small (ONNX, CPU) | 384 boyut, 118M parametre. Türkçe retrieval'ı boyutuna göre çok iyi. Bilerek CPU'da: GPU tamamen LLM + whisper'ın. |
| Vektör DB | Qdrant | Vektörler ve payload disk üzerinde (`on_disk`), RAM'i yemesin diye. |
| Proxy | Caddy | `getUserMedia()` yalnızca "secure context"te çalışır. `http://192.168.x.x` üzerinden telefonun mikrofonu **sessizce** kapalı kalırdı. |
| UI | Vanilla JS + CSS | Build adımı yok, ~30 KB. |

### RAG ne zaman devreye giriyor?

**Kullanıcının sorusu — sesli ya da yazılı — modele hiç dokunulmadan gider.**
Transkript kırpılmaz, yeniden yazılmaz, önüne bir şey eklenmez. Modele giden
mesaj listesi tam olarak şudur:

```
system : sistem promptu
user   : "Kart odasının anahtarı nerede duruyor?"
```

Dokümanlara yalnızca model gerekli görüp `knowledge_search` aracını çağırınca
bakılır; sonuç ayrı bir araç mesajı olarak döner, soruyla karışmaz.

`ALWAYS_ON_RAG=true` yaparsanız her soruda otomatik arama yapılır ve bulunanlar
ek bir sistem mesajı olarak eklenir. Tüm koleksiyon tek bir konuya aitse bir tur
tasarruf ettirir, aksi halde alakasız bağlam taşır.

### İki farklı eşik neden var?

e5 modellerinin kosinüs skorları dar bir banda sıkışır; "0.5 civarı alakasız"
sezgisi burada çalışmaz. Türkçe veriyle ölçüldü:

| | skor aralığı |
|---|---|
| doğru parça | 0.83 – 0.89 |
| alakasız parça | 0.72 – 0.80 |

`RAG_MIN_SCORE=0.82` otomatik enjeksiyon için (varsayılan kapalı): oraya düşen
her yanlış sonuç, dokümanlarla ilgisi olmayan bir sorunun önüne yapışır.

`RAG_TOOL_MIN_SCORE=0.75` ise `knowledge_search` için: model zaten sorunun
kullanıcının dosyalarıyla ilgili olduğuna karar vermiştir, her alıntının yanında
benzerlik skorunu görür, ve "bulunamadı" demek sınırda bir pasaj vermekten daha
kötüdür. Fark somut: doğru PDF'i bulan bir sorgu 0.809 aldı, 0.82 ile elenip boş
dönerdi.

Parça boyutu eşikten daha belirleyici çıktı: üç konuyu barındıran 780
karakterlik tek parça, hem alakalı hem alakasız sorguya **0.820** verdi. 450
karaktere düşürünce alakalılar 0.825–0.884'e çıktı, alakasız 0.801'de kaldı.
`CHUNK_CHARS` değişince saklanan indeks otomatik geçersiz sayılır.

---

## 3. Kullanım

**Mikrofon: basılı tutun.** Tuttuğunuz sürece kaydeder, bıraktığınızda gönderir.
Basılı tutarken sessizlikte otomatik kesme devre dışıdır — cümle ortasında nefes
alsanız bölünmez. 25 saniyelik üst sınır güvenlik için durur.

**Dil seçici:** sağ üstteki `TR / EN / Oto`. Seçim tarayıcıda saklanır ve her ses
paketiyle gönderilir. Varsayılan **TR** — `Oto` varsayılan değil, çünkü whisper
dili sesin ilk saniyelerinden tahmin ediyor ve 2 saniyelik Türkçe cümlelerde
düzenli olarak Farsça/Arapça'ya kayıyor.

**Yazmak:** mikrofon zorunlu değil, alttaki kutuya yazabilirsiniz. HTTPS
olmayan bir adresten girdiyseniz mikrofon kapalı olur ama yazı çalışır.

**Belge yüklemek:** 📎 düğmesi, ya da sayfaya sürükle-bırak. PDF, `.txt`, `.md`,
`.csv`, `.json` kabul edilir; sonuç sohbete düşer (`kilavuz.pdf eklendi — 12
parça`). Toplu iş için `data/docs/` klasörüne koyup `make rescan` deyin; sadece
değişen dosyalar yeniden işlenir.

PDF'ler pypdf'in metin katmanından okunur. Taranmış (görüntü) PDF'lerde metin
katmanı yoktur; bunlar OCR ister, bu yığında OCR yok — böyle bir dosya
yüklenirse 400 ile açıkça reddedilir.

**Sohbet geçmişi** bellekte, WebSocket bağlantısına bağlı. Sayfa yenilenince
sıfırlanır; ⟲ düğmesi de temizler. Son 3 tur modele gönderilir. Belgeler ve
hafıza kalıcıdır (Qdrant volume'unda).

**Komut satırından:**

```bash
curl -N -X POST http://localhost:8080/api/chat \
  -H 'content-type: application/json' \
  -d '{"message":"İstanbul hava durumu nedir?"}'

curl -X POST -F "file=@kilavuz.pdf" http://localhost:8080/api/ingest/upload
```

### Asistanın araçları

| Araç | Kaynak | API anahtarı |
|---|---|---|
| `knowledge_search` | Qdrant (yüklenen dokümanlar + hafıza) | – |
| `web_search` | DuckDuckGo (veya `SEARXNG_URL` verilirse SearxNG) | yok |
| `get_weather` | open-meteo.com | yok |

Sadece üç araç var, bilerek. `fetch_page`, `remember` ve `get_current_time`
kodda duruyor ama modele **tanıtılmıyor** — ölçüm, dördüncü aracın kalan üçünün
doğruluğunu bozduğunu gösterdi (bölüm 5).
Saat sistem promptunda.

### Birden fazla cihaz

Her tarayıcı sekmesi kendi oturumunu alır: sohbet geçmişi, dil seçimi ve
mikrofon akışı bağımsızdır. Ama llama.cpp ve whisper tek istek işler, ikinci
kullanıcı sıraya girer.

İki cihaz düzenli kullanacaksa `.env` içinde `LLM_PARALLEL=2` yapın. Ek bellek
maliyeti yok (`LLM_CTX` iki yuvaya bölünür) ve her kullanıcı kendi prompt
cache'ini korur. Bedeli: kişi başı bağlam yarıya iner. Aynı anda iki kişi
konuşacaksa `INTERIM_TRANSCRIPTION=false` da yapın — canlı önizleme whisper
kuyruğunun yarısını yiyor.

---

## 4. Ayarlar

Hepsi `.env` içinde.

```bash
# --- bellek / hız ---
LLM_CTX=8192              # 4096 yaparsanız ~600 MB kazanırsınız
LLM_CACHE_TYPE=q8_0       # f16 kaliteyi çok az artırır, KV belleğini ikiye katlar
LLM_PARALLEL=1            # 2 = iki eşzamanlı kullanıcı, bağlam bölünür
INTERIM_TRANSCRIPTION=true  # konuşurken canlı önizleme; GPU'yu meşgul eder

# --- ses ---
WHISPER_MODEL_FILE=ggml-small-q5_1.bin  # ggml-small.bin (466 MB) daha doğru
STT_LANGUAGE=tr           # arayüzde seçim yapılmamışsa varsayılan
VAD_SILENCE_MS=900        # basılı tutma modunda kullanılmaz
VAD_AGGRESSIVENESS=2      # sessiz ortamda 1 daha iyi olabilir

# --- model davranışı ---
TOOL_DECISION_TEMPERATURE=0.1  # araç seçimi bir sınıflandırma, 0.6'da kararsız
HISTORY_TURNS=3           # artırmak prompt'u uzatır ve araç çağırmayı bozar

# --- RAG ---
RAG_MIN_SCORE=0.82
RAG_TOOL_MIN_SCORE=0.75
CHUNK_CHARS=450           # değiştirirseniz indeks otomatik geçersizleşir
ALWAYS_ON_RAG=false
```

Model değiştirmek: `models/llm/` içine yeni GGUF'u koyup `LLM_MODEL_FILE`'ı
güncelleyin, `docker compose up -d llm`. İmajı yeniden derlemeye gerek yok.
Aynısı whisper için de geçerli.

---

## 5. Sorun giderme

### `iptables ... table 'raw' does not exist` — container ağı kurulamıyor

L4T çekirdeğinde `iptable_raw.ko` derlenmemiş, Docker 28+ ise "direct access
filtering" için o tabloyu kullanıyor. Docker'ı 27.x'e sabitleyin (bölüm 1). `update-alternatives` ile iptables'ı nft arka ucuna almak
da işe yarar ama sistem geneli bir değişikliktir; makinede Tailscale/VPN varsa
onların kuralları görünmez olur.

### `l4t-cuda:...` çekilemedi / derleme CUDA hatası veriyor

Etiket karttaki CUDA sürümüyle eşleşmeli: `cat /usr/local/cuda/version.json`

| JetPack | L4T | etiket |
|---|---|---|
| 6.0 | r36.3 | `12.2.12-devel` / `12.2.12-runtime` |
| 6.1 / 6.2 | r36.4 | `12.6.11-devel` / `12.6.11-runtime` |
| 5.x | r35 | `11.4.19-devel` / `11.4.19-runtime` |

`.env` içinde `L4T_CUDA_DEVEL` / `L4T_CUDA_RUNTIME`. `CUDA_ARCH` Orin'de 87.
`12.2.12-devel` bulunamazsa alternatif: `nvcr.io/nvidia/l4t-jetpack:r36.3.0`.

### Link hatası: `libcuda.so.1 not found`, `undefined reference to cuMemCreate`

`libcuda.so.1` CUDA *sürücüsüdür*; nvidia container runtime onu yalnızca
**çalışma anında** enjekte eder, `docker build` sırasında yoktur. Çözüm iki
parçalı ve ikisi de gerekli:

```dockerfile
RUN ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
...
-DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs"
```

**`-L` tek başına yetmez.** `ld`, bir kütüphanenin *dolaylı* bağımlılığını
(`libggml-cuda.so` → `DT_NEEDED: libcuda.so.1`) çözerken `-L` yollarına bakmaz;
yalnızca `-rpath-link`, `-rpath`, `LD_LIBRARY_PATH` ve sistem dizinlerine bakar.
Ayrıca birebir `libcuda.so.1` adını arar, stub'ın dosya adı ise `libcuda.so`.

`-rpath` değil **`-rpath-link`**: ikincisi yalnızca link zamanı arama yoludur,
ikiliye gömülmez. `-rpath` olsaydı Jetson'da gerçek sürücü yerine boş stub
yüklenir ve GPU sessizce çalışmazdı.

Derleme dizini BuildKit cache mount'unda tutuluyor; bir hata olursa tekrar
denemede ~440 dosya baştan derlenmez.

### Telefonda `ERR_SSL_PROTOCOL_ERROR`

IP adresine bağlanan istemciler SNI gönderemez (RFC 6066 SNI'da IP'ye izin
vermez), Caddy de hangi sertifikayı sunacağını seçemez. `.env` içinde
`SITE_HOST` dolu olmalı (sadece IP, şemasız/portsuz) — `set_lan_ip.sh` bunu
yazar. Caddyfile'daki `default_sni` bu değeri kullanır.

### Telefonda mikrofon butonu gri

`http://` ile girmişsinizdir. `https://<ip>:8443` kullanın. Sayfa üstünde uyarı
şeridi de çıkar.

### Model araçları hiç çağırmıyor, cevabı uyduruyor

Sistem promptu fazla uzamıştır. Qwen3-4B üzerinde temperature 0 ile ölçüldü:

| sistem promptu | sonuç |
|---|---|
| 104 karakter | `knowledge_search` çağırıyor |
| 249 karakter | çağırıyor |
| 589 karakter | **hiç çağırmıyor** |
| 907 karakter | hiç çağırmıyor |

Kırılmaya sebep olan paragraf modele açıkça "araçları kullan" diyordu — sorun
ifade değil, **uzunluk**. Hata sessiz: model "dosyayı yükleyin" diye cevap verir.
Çalışan bütçe **~400 karakter**.

**Araç şemaları aynı bütçeyi paylaşıyor.** Beş soruluk sınama (hava / haber /
fiyat / belge / sohbet):

| şemalar | toplam | sonuç |
|---|---|---|
| 5 araç, güçlü tarifler | 1470 krktr | haber sorusunu kaçırıyor |
| **3 araç, güçlü tarifler** | **993 krktr** | **beşi de doğru** |
| 3 araç, daha da güçlü tarifler | 1056 krktr | haber sorusunu yine kaçırıyor |

Son satır önemli: tarifi daha ısrarcı yazmak ama uzatmak **kötüleştirdi**.
İkinci kural: tarif, modelin cevabı **bilmediğini** söylemeli. "Weather for a
city" yok sayılıyor çünkü model havayı bildiğini sanıyor; "You do not know
today's weather — always call this" çalışıyor.

Araç veya prompt eklerken `services/backend/app/tools.py` ve `llm.py` içindeki
ölçüm notlarını okuyun, sınamayı tekrarlayın.

### Türkçe konuşma Farsça/Arapça yazıya çevriliyor

Dil algılama kısa kayıtlarda güvenilmez. Arayüzdeki dil seçiciyi `TR` yapın
(varsayılan zaten bu). Kalıcı varsayılan `.env` içindeki `STT_LANGUAGE`.

### Transkripsiyon kalitesi düşük

Sırasıyla: mikrofona yaklaşın (en çok fark eden), `VAD_AGGRESSIVENESS=1`, ve
kuantize olmayan modele geçin — `ggml-small.bin` (466 MB) gürültülü kayıtta ve
nadir kelimelerde belirgin daha iyi.

### `make build` sırasında derleyici öldü / OOM

`services/llm/Dockerfile` ve `services/stt/Dockerfile` içindeki `BUILD_JOBS=4`'ü
2'ye düşürün. `make setup`'ın eklediği 8 GB swap bunu genelde önler.

### `nvpmodel` sadece 15W ve 7W gösteriyor

Normal. Bazı Orin NX kartlarında MAXN/25W profili tanımlı değildir; 15W (mod 0)
o kartta tavandır ve `make setup` onu seçer. 7W moduna **girmeyin**.

### Cevaplar çok yavaş

`sudo tegrastats` bakın. `RAM` doluysa `LLM_CTX`'i 4096'ya düşürün veya
`INTERIM_TRANSCRIPTION=false` yapın. Masaüstü oturumunu kapatmak ~800 MB
kazandırır: `sudo systemctl set-default multi-user.target && sudo reboot`.

### GPU gerçekten kullanılıyor mu?

```bash
sudo tegrastats | grep --line-buffered -o "GR3D_FREQ [0-9]*%"
```

Cevap üretilirken %90-99, boştayken %0 olmalı. Sürekli %0 ise:

```bash
docker compose exec llm /opt/llama/llama-server --list-devices
docker compose exec llm ldd /opt/llama/libggml-cuda.so | grep libcuda
```

`CUDA0: Orin` ve `libcuda.so.1 => /usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1`
görmelisiniz.

---

## 6. Dosya düzeni

```
docker-compose.yml
.env.example
caddy/Caddyfile              TLS + statik dosya + ters proxy
frontend/                    index.html, app.js, worklet.js, styles.css
services/
  backend/app/
    main.py                  FastAPI, websocket döngüsü, REST
    llm.py                   llama.cpp'ye karşı araç çağrı döngüsü + sistem promptu
    tools.py                 araç şemaları ve uygulamaları
    rag.py                   Qdrant koleksiyonları, chunk'lama, PDF, ingest
    audio.py                 WebRTC VAD ile cümle kesme
    clients.py               embedder / stt / llm HTTP istemcileri
    config.py                tüm ayarlar
  embedder/server.py         ONNX embedding servisi
  llm/Dockerfile             llama.cpp CUDA derlemesi
  stt/Dockerfile             whisper.cpp CUDA derlemesi
models/                      GGUF + whisper + onnx (git'e girmez)
data/docs/                   RAG'a girecek dosyalarınız
scripts/
  preflight.sh               ortam kontrolü (hiçbir şeyi değiştirmez)
  jetson_setup.sh            nvidia runtime, güç modu, swap
  download_models.sh         model indirme
  set_lan_ip.sh              SITE_ADDRESS / SITE_HOST
```

---

## 7. Bilerek yapılmayanlar

- **TTS yok.** Cevap yazıyla dönüyor. Piper (ONNX, ~50 MB, Türkçe sesi var)
  yedinci servis olarak eklenebilir; bellek maliyeti ~150 MB.
- **Kimlik doğrulama yok.** Ev ağı varsayıldı. Ağ güvenilmiyorsa Caddy'ye
  `basic_auth` eklemek iki satır. Bu yüzden `tailscale funnel` gibi sayfayı
  internete açan bir şey kullanmayın.
- **Kalıcı sohbet geçmişi yok.** Bellekte tutuluyor. Vektör araması
  gerektirmediği için Qdrant'a değil, SQLite'a yazılması gerekir.
- **`fetch_page` ve `remember` devre dışı.** Kod duruyor, şema listesinde yok —
  dördüncü araç ölçülebilir şekilde diğerlerini bozuyordu. Daha büyük bir
  modele geçerseniz geri açılabilir.
- **OCR yok.** Taranmış PDF'ler reddedilir.

---

## 8. Ölçümler

Jetson Orin NX 8 GB, JetPack 6.0 (L4T r36.3, CUDA 12.2.12), güç modu 15W.

| | değer |
|---|---|
| LLM üretim hızı | **10.8 token/sn** |
| LLM prompt işleme | **359 token/sn** |
| Prompt cache | ikinci istekte 932 token → 1 token yeniden işleniyor |
| Derleme süresi | ~40 dk (llama.cpp + whisper.cpp paralel) |
| Çalışırken RAM | 6.4 / 7.6 GB (mlock ile model sabitlenmiş) |
| GPU (üretim sırasında) | %99 |
| Sıcaklık | derlemede 65°C, çıkarımda 55°C |

Prompt cache sayesinde sistem promptu ve araç şemalarının maliyeti sadece ilk
istekte ödeniyor.

**Hız artırmak için:** speculative decoding (Qwen3-0.6B taslak model, ~1.5-2x,
~600 MB ek bellek) veya daha küçük modele inmek. Güç modunda yapılacak bir şey
yok, 15W bu kartta tavan.

### Neyin doğrulandığı

**Gerçek donanımda çalışırken doğrulandı:** telefondan HTTPS erişimi ve
mikrofon, bas-konuş modu, VAD segmentasyonu, whisper CUDA transkripsiyonu,
Qwen3-4B GPU'da üretim, araç çağrıları (hava durumu, web araması, belge
arama), PDF ve metin yükleme, dil seçici, Qdrant kalıcılığı.

**x86'da sahte LLM/STT ile doğrulandı:** WebSocket protokolü uçtan uca,
deltalara bölünmüş `tool_calls` argümanlarının birleştirilmesi, embedding
retrieval kalitesi, chunk'lama ve yeniden indeksleme, PDF metin çıkarma ve
hata yolları, bas-konuş modunun duraklamada bölmediği.

**Doğrulanmadı:** eşzamanlı çoklu kullanıcı, gerçek bir taranmış PDF ile
"metin katmanı yok" dalı, uzun süreli kararlılık.
