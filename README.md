# Jetson Sesli Asistan

Jetson Orin Nano/NX 8 GB üzerinde tamamen yerel çalışan, aynı wifi ağındaki
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
kesme → whisper.cpp → metin → RAG + araçlar → llama.cpp → token token geri.

---

## 1. Hızlı kurulum

Jetson'da, JetPack 6.x kurulu, Docker kurulu ve wifi bağlıyken:

```bash
cd jetson-assistant

# 1) host hazırlığı: nvidia default runtime, MAXN, 8 GB swap
make setup

# 2) modelleri indir (~3.2 GB)
make models

# 3) imajları derle — llama.cpp ve whisper.cpp CUDA derlemesi
#    Orin Nano'da 45-70 dakika sürer, bir kere yapılır
make build

# 4) çalıştır
make up
make health
```

`make setup` LAN IP'nizi bulup `.env` içindeki `SITE_ADDRESS`'i doldurur.
Adresi ekrana yazar, telefondan onu açarsınız:

```
https://192.168.1.50:8443
```

> **Sertifika uyarısı normaldir.** Caddy kendi iç sertifika otoritesiyle imza
> atıyor. "Gelişmiş → Yine de devam et" dedikten sonra mikrofon çalışır.

IP değişirse: `bash scripts/set_lan_ip.sh && docker compose up -d --force-recreate caddy`

---

## 2. Neden bu parçalar?

| Parça | Seçim | Gerekçe |
|---|---|---|
| LLM | Qwen3-4B-Instruct-2507 Q4_K_M | ~2.6 GB. 8 GB'lik paylaşımlı bellekte whisper + qdrant ile birlikte rahat sığar. Tool calling şablonu GGUF'un içinde geliyor, Türkçesi iyi. Düşünme modu olmadığı için sesli kullanımda gecikme düşük. |
| STT | whisper.cpp `small-q5_1` | Türkçe + İngilizce aynı modelde, dil otomatik algılanıyor. Streaming Conformer/Zipformer modelleri daha düşük gecikmeli ama Türkçe desteği yok. |
| Embedding | multilingual-e5-small (ONNX, CPU) | 384 boyut, 118M parametre. Türkçe retrieval'ı boyutuna göre çok iyi (aşağıda ölçüm var). Bilerek CPU'da: GPU tamamen LLM + whisper'ın. |
| Vektör DB | Qdrant | İstenildiği gibi. Vektörler ve payload disk üzerinde tutuluyor (`on_disk`), RAM'i yemesin diye. |
| Proxy | Caddy | `getUserMedia()` yalnızca "secure context"te çalışır. `http://192.168.x.x` üzerinden telefonun mikrofonu **sessizce** kapalı kalırdı. |
| UI | Vanilla JS + CSS | Build adımı yok, ~25 KB. |

### 8 GB bellek bütçesi

| | RAM |
|---|---|
| llama.cpp (ağırlık + 8k KV, q8_0 cache) | ~3.2 GB |
| whisper.cpp small-q5 | ~0.6 GB |
| qdrant | ~0.3 GB |
| backend + embedder + caddy | ~0.6 GB |
| **toplam** | **~4.7 GB** |

Kalanı JetPack'in kendisi kullanıyor. Masaüstü oturumu kapalıysa (`multi-user.target`)
yaklaşık 800 MB daha kazanırsınız. `sudo tegrastats` ile izleyin.

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

`ALWAYS_ON_RAG=true` yaparsanız eski davranışa dönersiniz: her soruda otomatik
arama yapılır ve bulunanlar ek bir sistem mesajı olarak eklenir. Tüm koleksiyon
tek bir konuya aitse bir tur tasarruf ettirir, aksi halde alakasız bağlam taşır.

### İki farklı eşik neden var?

e5 modellerinin kosinüs skorları dar bir banda sıkışır; "0.5 civarı alakasız"
sezgisi burada çalışmaz. Türkçe örnek veriyle ölçtüm:

| | skor aralığı |
|---|---|
| doğru parça | 0.83 – 0.89 |
| alakasız parça | 0.72 – 0.80 |

`RAG_MIN_SCORE=0.82` otomatik enjeksiyon için (kapalı): oraya düşen her yanlış
sonuç, dokümanlarla ilgisi olmayan bir sorunun önüne yapışır, o yüzden sıkı.

`RAG_TOOL_MIN_SCORE=0.75` ise `knowledge_search` için: model zaten sorunun
kullanıcının dosyalarıyla ilgili olduğuna karar vermiştir, her alıntının yanında
benzerlik skorunu görür, ve "bulunamadı" demek sınırda bir pasaj vermekten daha
kötüdür. Ölçümde fark somut çıktı: doğru PDF'i bulan bir sorgu 0.809 aldı, yani
0.82 ile elenip boş dönerdi.

Ayrıca parça boyutu eşikten daha belirleyici
çıktı: üç ayrı konuyu barındıran 780 karakterlik tek bir parça, hem alakalı hem
alakasız sorguya **0.820** verdi. 450 karaktere düşürünce alakalı sorgular
0.825–0.884'e çıktı, alakasız olan 0.801'de kaldı. `CHUNK_CHARS`'ı
değiştirdiğinizde saklanan indeks otomatik olarak geçersiz sayılır ve dosyalar
yeniden parçalanır.

---

## 3. Kullanım

**Dil seçici:** sağ üstteki `TR / EN / Oto`. Seçim tarayıcıda saklanır ve her
ses paketiyle birlikte gönderilir. Varsayılan **TR** — `Oto` varsayılan değil,
çünkü whisper dili sesin ilk saniyelerinden tahmin ediyor ve 2 saniyelik Türkçe
cümlelerde düzenli olarak Farsça/Arapça'ya kayıyor.

**Telefon/tarayıcı:** mikrofon butonuna basıp konuşun, tekrar basınca durur.
Konuşurken sustuğunuzda VAD cümleyi kendiliğinden kapatır ve işlemeye başlar —
butona tekrar basmayı beklemez. Mikrofon zorunlu değil, alttaki kutuya
yazabilirsiniz.

**Kendi dokümanlarınız:** `data/docs/` içine `.pdf`/`.md`/`.txt`/`.csv`/`.json`
koyun, `make rescan` deyin. Sadece değişen dosyalar yeniden işlenir. Tek dosyayı
ayağa kalkmış sisteme yüklemek için:

```bash
curl -X POST -F "file=@kilavuz.pdf" http://localhost:8080/api/ingest/upload
```

PDF'ler pypdf'in metin katmanından okunur. Taranmış (görüntü) PDF'lerde metin
katmanı yoktur; bunlar OCR ister, bu yığında OCR yok — böyle bir dosya
yüklenirse 400 ile açıkça reddedilir.

**Komut satırından:**

```bash
curl -N -X POST http://localhost:8080/api/chat \
  -H 'content-type: application/json' \
  -d '{"message":"İstanbul hava durumu nedir?"}'
```

### Asistanın araçları

| Araç | Kaynak | API anahtarı |
|---|---|---|
| `knowledge_search` | Qdrant (yüklenen dokümanlar + hafıza) | – |
| `web_search` | DuckDuckGo (veya `SEARXNG_URL` verilirse SearxNG) | yok |
| `get_weather` | open-meteo.com | yok |

Sadece üç araç var, bilerek. `fetch_page`, `remember` ve `get_current_time`
kodda duruyor ama modele **tanıtılmıyor** — ölçüm, dördüncü aracın kalan
üçünün doğruluğunu bozduğunu gösterdi (aşağıda). Saat sistem promptunda.

---

## 4. Ayarlar

Hepsi `.env` içinde. Sık kurcalanacaklar:

```bash
LLM_CTX=8192              # bağlam. 4096 yaparsanız ~600 MB kazanırsınız
LLM_CACHE_TYPE=q8_0       # f16 yaparsanız kalite çok az artar, KV belleği ikiye katlanır
WHISPER_MODEL_FILE=ggml-small-q5_1.bin   # base-q5_1 = 2x hızlı, Türkçesi zayıf
WHISPER_LANGUAGE=auto     # tr veya en ile sabitlerseniz algılama adımı kalkar, biraz hızlanır
VAD_SILENCE_MS=700        # cümle bitti sayılması için gereken sessizlik
INTERIM_TRANSCRIPTION=true  # konuşurken canlı önizleme; GPU'yu meşgul eder, false yapılabilir
RAG_MIN_SCORE=0.82        # aşağıda açıklandı
CHUNK_CHARS=450           # değiştirirseniz indeks otomatik geçersizleşir
```

Model değiştirmek: `models/llm/` içine yeni GGUF'u koyup `LLM_MODEL_FILE`'ı
güncelleyin, `docker compose up -d llm`. İmajı yeniden derlemeye gerek yok.

---

## 5. Sorun giderme

**`make build` sırasında derleyici öldü / OOM**
nvcc bellek canavarı. `services/llm/Dockerfile` ve `services/stt/Dockerfile`
içindeki `BUILD_JOBS=4`'ü 2'ye düşürün.

**`nvidia-container-runtime` bulunamadı / build CUDA görmüyor**
`make setup` Docker'ın default runtime'ını `nvidia` yapar. Yapmadıysanız
imajlar derlenirken CUDA başlıklarını bulamaz.

**Telefonda mikrofon butonu gri**
`http://` ile girmişsinizdir. `https://<ip>:8443` kullanın. Sayfa üstünde bir
uyarı şeridi de çıkar.

**`l4t-cuda:...` çekilemedi / derleme CUDA hatası veriyor**
Etiket karttaki CUDA sürümüyle eşleşmeli. Kontrol: `cat /usr/local/cuda/version.json`

| JetPack | L4T | etiket |
|---|---|---|
| 6.0 | r36.3 | `12.2.12-devel` / `12.2.12-runtime` |
| 6.1 / 6.2 | r36.4 | `12.6.11-devel` / `12.6.11-runtime` |
| 5.x | r35 | `11.4.19-devel` / `11.4.19-runtime` |

`.env` içinde `L4T_CUDA_DEVEL` / `L4T_CUDA_RUNTIME` ile değiştirin.
`CUDA_ARCH` Orin'de her durumda 87.

**Link hatası: `libcuda.so.1 not found`, `undefined reference to cuMemCreate`**
`libcuda.so.1` CUDA *sürücüsüdür*; nvidia container runtime onu yalnızca
**çalışma anında** enjekte eder, `docker build` sırasında ortada yoktur. Bu
yüzden her iki Dockerfile toolkit'in stub kütüphanesini kullanıyor. Çözüm iki
parçalı ve **ikisi de gerekli**:

```dockerfile
RUN ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
...
-DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs"
```

**`-L` tek başına yetmez.** `ld`, bir kütüphanenin *dolaylı* bağımlılığını
(`libggml-cuda.so` → `DT_NEEDED: libcuda.so.1`) çözerken `-L` yollarına bakmaz;
yalnızca `-rpath-link`, `-rpath`, `LD_LIBRARY_PATH` ve sistem dizinlerine bakar.
Ayrıca birebir `libcuda.so.1` adını arar, oysa stub'ın dosya adı `libcuda.so` —
symlink bu yüzden şart.

`-rpath` değil **`-rpath-link`** kullanılıyor: ikincisi yalnızca link zamanı
arama yoludur, ikiliye gömülmez. `-rpath` olsaydı Jetson'da gerçek sürücü yerine
boş stub yüklenir ve GPU sessizce çalışmazdı.

Derleme dizini ayrıca BuildKit cache mount'unda tutuluyor; bir hata olursa
tekrar denemede ~440 dosya baştan derlenmez.

**`iptables ... table 'raw' does not exist` — container ağı kurulamıyor**
L4T çekirdeğinde `iptable_raw.ko` derlenmemiş (`/lib/modules/$(uname -r)/kernel/net/ipv4/netfilter/`
altında yok), Docker 28+ ise "direct access filtering" için o tabloyu kullanıyor.
Çözüm: Docker'ı 27.x'e sabitleyin.

```bash
V=5:27.5.1-1~ubuntu.22.04~jammy
sudo apt-get install -y --allow-downgrades docker-ce=$V docker-ce-cli=$V
sudo apt-mark hold docker-ce docker-ce-cli   # yoksa ilk upgrade'de geri kırılır
sudo systemctl restart docker
```

`update-alternatives` ile iptables'ı nft arka ucuna almak da işe yarar ama
sistem geneli bir değişikliktir; makinede Tailscale/VPN/firewall varsa onların
kuralları görünmez olur. Docker'ı sabitlemek daha dar kapsamlı.

**`nvpmodel` sadece 15W ve 7W gösteriyor**
Normal. Bazı Orin NX kartlarında MAXN/25W profili tanımlı değildir; 15W (mod 0)
o kartta tavandır ve `make setup` onu seçer. 7W moduna **girmeyin**, LLM
kullanılamaz hale gelir.

**`l4t-cuda:12.2.12-devel` bulunamadı**
JetPack 6.0 için alternatif derleme imajı: `.env` içinde
`L4T_CUDA_DEVEL=nvcr.io/nvidia/l4t-jetpack:r36.3.0` yapın. Bu imaj CUDA
toolkit'i içerir, sadece daha büyüktür; `L4T_CUDA_RUNTIME` aynı kalabilir.

**Cevaplar çok yavaş**
`sudo tegrastats` bakın. `RAM` doluysa `LLM_CTX`'i 4096'ya düşürün veya
`INTERIM_TRANSCRIPTION=false` yapın. `GR3D_FREQ` düşükse `sudo nvpmodel -m 0`.

**Model araçları hiç çağırmıyor, cevabı uyduruyor**
Sistem promptu fazla uzamıştır. Qwen3-4B üzerinde temperature 0 ile ölçtüm:

| sistem promptu | sonuç |
|---|---|
| 104 karakter | `knowledge_search` çağırıyor |
| 249 karakter | çağırıyor |
| 589 karakter | **hiç çağırmıyor** |
| 907 karakter | hiç çağırmıyor |

Araç sayısı değil **promptun uzunluğu** belirleyici: kısa promptla 5 araç
sorunsuz, uzun promptla 1 araç bile çalışmıyor. Üstelik kırılmaya sebep olan
paragraf modele açıkça "araçları kullan" diyordu — yani sorun ifade değil,
uzunluk. Hata sessiz: model "dosyayı yükleyin" diye cevap verir.

Çalışan bütçe **~400 karakter**. `services/backend/app/llm.py` içindeki
`SYSTEM_PROMPT`'a bir şey eklerseniz yeniden ölçün.

**Araç şemaları da aynı bütçeyi paylaşıyor.** Beş soruluk bir sınamayla ölçtüm
(hava / haber / fiyat / belge / sohbet):

| şemalar | toplam | sonuç |
|---|---|---|
| 5 araç, güçlü tarifler | 1470 krktr | haber sorusunu kaçırıyor |
| **3 araç, güçlü tarifler** | **993 krktr** | **beşi de doğru** |
| 3 araç, daha da güçlü tarifler | 1056 krktr | haber sorusunu yine kaçırıyor |

Son satır önemli: tarifi daha ısrarcı yazmak ama uzatmak **kötüleştirdi**.
Belirleyici olan ifadenin şiddeti değil, toplam uzunluk.

İkinci kural: tarif, modelin cevabı **bilmediğini** söylemeli. "Weather for a
city" gibi betimleyici bir tarif yok sayılıyor çünkü model havayı bildiğini
sanıyor; "You do not know today's weather — always call this" çalışıyor.

Araç eklemek isterseniz `services/backend/app/tools.py` içindeki ölçüm notunu
okuyun ve sınamayı tekrarlayın.

**Türkçe konuşma Farsça/Arapça yazıya çevriliyor**
Dil algılama kısa kayıtlarda güvenilmez. Arayüzdeki dil seçiciyi `TR` yapın
(varsayılan zaten bu). Kalıcı varsayılanı `.env` içindeki `STT_LANGUAGE` belirler.

**Transkripsiyon kalitesi düşük**
Sırasıyla deneyin: mikrofona yaklaşın (en çok fark eden), `VAD_AGGRESSIVENESS=1`,
ve kuantize olmayan modele geçin — `ggml-small.bin` (466 MB), `ggml-small-q5_1`'e
göre gürültülü kayıtta ve nadir kelimelerde belirgin daha iyi.

**Whisper boş metin döndürüyor**
Mikrofon izni verilmiş ama ses gelmiyor olabilir. `docker compose logs -f backend`
ile `transcript` olaylarını izleyin; `VAD_AGGRESSIVENESS` değerini 1'e düşürmeyi
deneyin (sessiz ortamda 3 fazla agresif kalır).

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
    llm.py                   llama.cpp'ye karşı araç çağrı döngüsü
    tools.py                 web arama, hava durumu, RAG, hafıza
    rag.py                   Qdrant koleksiyonları, chunk'lama, ingest
    audio.py                 WebRTC VAD ile cümle kesme
    clients.py               embedder / stt / llm HTTP istemcileri
    config.py                tüm ayarlar
  embedder/server.py         ONNX embedding servisi
  llm/Dockerfile             llama.cpp CUDA derlemesi
  stt/Dockerfile             whisper.cpp CUDA derlemesi
models/                      GGUF + whisper + onnx (git'e girmez)
data/docs/                   RAG'a girecek dosyalarınız
scripts/                     model indirme, LAN IP, host hazırlığı
```

---

## 7. Bilerek yapılmayanlar

- **TTS yok.** Cevap yazıyla dönüyor. İstenirse Piper (ONNX, ~50 MB, Türkçe sesi
  var) yedinci bir servis olarak eklenebilir; bellek maliyeti ~150 MB.
- **Kimlik doğrulama yok.** Ev ağı varsayıldı. Ağ güvenilmiyorsa Caddy'ye
  `basic_auth` eklemek iki satır.
- **Çoklu kullanıcı yok.** `LLM_PARALLEL=1`; ikinci kullanıcı sırada bekler.
  8 GB'de aynı anda iki dizi KV cache tutmak riskli.

---

## 8. Neyin test edildiği

Proje bir x86 geliştirme makinesinde yazıldı, hedef Jetson (aarch64 + CUDA)
orada değildi. Şeffaf olmak gerekirse:

**Gerçekten çalıştırılıp doğrulandı (x86, sahte llama.cpp/whisper.cpp ile):**
- Embedding servisi: model yükleniyor, 384 boyut, Türkçe ve diller arası
  retrieval doğru sıralıyor.
- Qdrant koleksiyon oluşturma, chunk'lama, ingest, hash ile atlama,
  `CHUNK_CHARS` değişince yeniden indeksleme.
- PDF yükleme: gerçek bir PDF'ten metin çıkarma, indeksleme ve PDF'e özgü
  sorguyla geri bulma. Bozuk PDF'ler 400 döndürüyor. (Gerçek bir *taranmış*
  PDF ile "metin katmanı yok" dalı denenmedi.)
- Sorunun modele değiştirilmeden ulaştığı, mesaj listesi doğrudan okunarak
  doğrulandı: sadece sistem promptu + kullanıcı mesajı.
- WebSocket akışı uçtan uca: metin turu, ses turu (sentetik PCM → WebRTC VAD →
  cümle kesme → STT → cevap), `reset`, `ping`.
- Araç çağrı döngüsü: birden çok delta'ya bölünmüş `tool_calls` argümanlarının
  birleştirilmesi, aracın çalıştırılması, sonucun modele geri verilmesi.
- Gerçek dış servisler: open-meteo (canlı Ankara verisi geldi), DuckDuckGo
  araması, saat, hafızaya yazma + geri okuma, hatalı argüman/bilinmeyen araç
  hataları.
- `docker compose config` ve `caddy validate`.

**Test edilemedi, ilk çalıştırmada görülecek:**
- llama.cpp ve whisper.cpp'nin CUDA/aarch64 derlemesi (`make build`).
- Gerçek Qwen3-4B çıktısı, gerçek whisper doğruluğu ve gerçek token hızları.
- Tarayıcı mikrofon yolu (AudioWorklet + self-signed sertifika) gerçek telefonda.
- README'deki bellek bütçesi tablosu hesap, ölçüm değil.

Modeller `./models/` altına zaten indirildi (~3.1 GB), `make models` tekrar
indirmez.
