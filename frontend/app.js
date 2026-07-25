/* Jetson Assistant — dependency-free front end.
 *
 * Audio path: getUserMedia -> AudioWorklet -> (resample to 16 kHz) -> PCM16
 * -> websocket binary frames. Raw PCM rather than MediaRecorder on purpose:
 * MediaRecorder gives webm/opus on Chrome and mp4/aac on Safari, which would
 * force an ffmpeg decode step on the Jetson. PCM16 is what whisper.cpp wants
 * anyway, and it works identically in every browser.
 */

const TARGET_RATE = 16000;

const el = {
  messages: document.getElementById('messages'),
  empty: document.getElementById('empty-state'),
  status: document.getElementById('status'),
  dot: document.getElementById('conn-dot'),
  mic: document.getElementById('btn-mic'),
  send: document.getElementById('btn-send'),
  reset: document.getElementById('btn-reset'),
  input: document.getElementById('input'),
  lang: document.getElementById('lang'),
  attach: document.getElementById('btn-attach'),
  fileInput: document.getElementById('file-input'),
  banner: document.getElementById('insecure-banner'),
  httpsHint: document.getElementById('https-hint'),
};

// Whisper detects the language from the first seconds of audio, which is
// unreliable on short utterances -- Turkish regularly comes back as Persian.
// Pinning it per session is what makes voice input dependable.
const LANG_KEY = 'assistant.language';

const state = {
  language: localStorage.getItem(LANG_KEY) || 'tr',
  ws: null,
  connected: false,
  recording: false,
  audio: null,          // { ctx, stream, node, source }
  botBubble: null,      // element currently receiving tokens
  interimBubble: null,
  toolRow: null,
  reconnectDelay: 500,
};

/* --------------------------------------------------------------- websocket */

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  state.ws = ws;

  ws.onopen = () => {
    state.connected = true;
    state.reconnectDelay = 500;
    el.dot.className = 'dot online';
    setStatus('hazır');
    // The server starts every connection at its default, so re-assert ours.
    sendJSON({ type: 'set_language', language: state.language });
  };

  ws.onclose = () => {
    state.connected = false;
    el.dot.className = 'dot offline';
    setStatus('bağlantı koptu, yeniden deneniyor…');
    stopRecording(true);
    setTimeout(connect, state.reconnectDelay);
    state.reconnectDelay = Math.min(state.reconnectDelay * 2, 10000);
  };

  ws.onerror = () => ws.close();
  ws.onmessage = (event) => handleEvent(JSON.parse(event.data));
}

function sendJSON(obj) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify(obj));
  }
}

const STATUS_TEXT = {
  idle: 'hazır',
  listening: 'dinliyor…',
  transcribing: 'yazıya çevriliyor…',
  thinking: 'düşünüyor…',
};

function handleEvent(msg) {
  switch (msg.type) {
    case 'status':
      setStatus(STATUS_TEXT[msg.state] || msg.state);
      break;

    case 'transcript':
      if (msg.final) {
        clearInterim();
      } else if (msg.text) {
        showInterim(msg.text);
      }
      break;

    case 'user_message':
      clearInterim();
      addMessage('user', msg.text);
      state.botBubble = null;
      state.toolRow = null;
      break;

    case 'token':
      appendToken(msg.text);
      break;

    case 'tool':
      renderTool(msg);
      break;

    case 'sources':
      renderSources(msg.items);
      break;

    case 'done':
      if (state.botBubble) state.botBubble.classList.remove('cursor');
      state.botBubble = null;
      setControlsBusy(false);
      break;

    case 'error':
      addMessage('error', msg.message);
      state.botBubble = null;
      setControlsBusy(false);
      break;
  }
}

/* ------------------------------------------------------------------ render */

function setStatus(text) {
  el.status.textContent = text;
}

function hideEmpty() {
  if (!el.empty) return;
  el.empty.remove();
  el.empty = null;
}

function atBottom() {
  return el.messages.scrollHeight - el.messages.scrollTop - el.messages.clientHeight < 120;
}

function scroll(force) {
  if (force || atBottom()) el.messages.scrollTop = el.messages.scrollHeight;
}

function addMessage(kind, text) {
  hideEmpty();
  const div = document.createElement('div');
  div.className = `msg ${kind}`;
  div.textContent = text;
  el.messages.appendChild(div);
  scroll(true);
  return div;
}

function appendToken(text) {
  if (!state.botBubble) {
    state.botBubble = addMessage('bot', '');
    state.botBubble.classList.add('cursor');
  }
  const stick = atBottom();
  state.botBubble.textContent += text;
  scroll(stick);
}

function showInterim(text) {
  hideEmpty();
  if (!state.interimBubble) {
    state.interimBubble = document.createElement('div');
    state.interimBubble.className = 'msg interim';
    el.messages.appendChild(state.interimBubble);
  }
  state.interimBubble.textContent = text;
  scroll(true);
}

function clearInterim() {
  if (state.interimBubble) {
    state.interimBubble.remove();
    state.interimBubble = null;
  }
}

const TOOL_LABELS = {
  web_search: 'web araması',
  fetch_page: 'sayfa okunuyor',
  get_weather: 'hava durumu',
  get_current_time: 'saat',
  knowledge_search: 'belgeler',
  remember: 'hafızaya kaydediliyor',
};

function renderTool(msg) {
  hideEmpty();
  if (!state.toolRow) {
    state.toolRow = document.createElement('div');
    state.toolRow.className = 'tools';
    el.messages.appendChild(state.toolRow);
  }

  const id = `tool-${msg.name}`;
  let chip = state.toolRow.querySelector(`[data-tool="${id}"]`);
  if (!chip) {
    chip = document.createElement('span');
    chip.className = 'tool-chip running';
    chip.dataset.tool = id;
    state.toolRow.appendChild(chip);
  }
  chip.textContent = TOOL_LABELS[msg.name] || msg.name;
  chip.className = msg.phase === 'end' ? 'tool-chip done' : 'tool-chip running';
  scroll(true);
}

function renderSources(items) {
  if (!items || !items.length) return;
  hideEmpty();
  const div = document.createElement('div');
  div.className = 'sources';
  div.textContent = 'kaynak: ' + items.map((i) => i.source).join(', ');
  el.messages.appendChild(div);
  scroll(true);
}

function setControlsBusy(busy) {
  el.send.disabled = busy;
}

/* ------------------------------------------------------------------ upload */

/* Uploads go over plain HTTPS, not the websocket: they are one-shot, can be
   several MB, and must keep working while a turn is streaming. */
async function uploadFile(file) {
  const note = addMessage('system', `${file.name} yükleniyor…`);
  const body = new FormData();
  body.append('file', file);

  try {
    const res = await fetch('/api/ingest/upload', { method: 'POST', body });
    let data = {};
    try { data = await res.json(); } catch { /* non-JSON error body */ }

    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);

    const chunks = data.chunks ?? 0;
    note.className = 'msg system ok';
    note.textContent = chunks
      ? `${data.source} eklendi — ${chunks} parça, ${data.characters} karakter`
      : `${data.source} okundu ama içinde metin bulunamadı`;
  } catch (err) {
    note.className = 'msg error';
    note.textContent = `${file.name} yüklenemedi: ${err.message}`;
  }
}

async function handleFiles(files) {
  el.attach.disabled = true;
  try {
    // sequential: the Jetson embeds on 2 CPU threads, parallel uploads would
    // just queue up behind each other anyway
    for (const file of files) await uploadFile(file);
  } finally {
    el.attach.disabled = false;
  }
}

/* ------------------------------------------------------------------- audio */

function floatToPCM16(floats) {
  const out = new Int16Array(floats.length);
  for (let i = 0; i < floats.length; i++) {
    const s = Math.max(-1, Math.min(1, floats[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

/* Linear resampler with carry-over between blocks, so no sample is dropped at
   block boundaries (that would show up as periodic clicks to the VAD). */
function makeResampler(fromRate, toRate) {
  if (fromRate === toRate) return (block) => block;

  const ratio = fromRate / toRate;
  let position = 0;
  let tail = new Float32Array(0);

  return (block) => {
    const input = new Float32Array(tail.length + block.length);
    input.set(tail, 0);
    input.set(block, tail.length);

    const outLength = Math.max(0, Math.floor((input.length - 1 - position) / ratio) + 1);
    const out = new Float32Array(outLength);

    for (let i = 0; i < outLength; i++) {
      const idx = position + i * ratio;
      const low = Math.floor(idx);
      const frac = idx - low;
      out[i] = input[low] * (1 - frac) + (input[low + 1] ?? input[low]) * frac;
    }

    const consumed = Math.floor(position + outLength * ratio);
    tail = input.subarray(Math.min(consumed, input.length - 1));
    position = position + outLength * ratio - consumed;
    return out;
  };
}

async function startRecording() {
  if (state.recording || !state.connected) return;

  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    showInsecureBanner();
    return;
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
  } catch (err) {
    addMessage('error', 'Mikrofona erişilemedi: ' + err.message);
    return;
  }

  // Ask for 16 kHz directly; most browsers honour it and we skip resampling.
  let ctx;
  try {
    ctx = new AudioContext({ sampleRate: TARGET_RATE });
  } catch {
    ctx = new AudioContext();
  }
  await ctx.resume();
  await ctx.audioWorklet.addModule('/worklet.js');

  const source = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, 'capture-processor');
  const resample = makeResampler(ctx.sampleRate, TARGET_RATE);

  node.port.onmessage = (event) => {
    if (!state.recording) return;
    const pcm = floatToPCM16(resample(event.data));
    if (pcm.length && state.ws?.readyState === WebSocket.OPEN) {
      state.ws.send(pcm.buffer);
    }
  };

  source.connect(node);
  // Keep the graph alive without echoing the mic back to the speakers.
  const sink = ctx.createGain();
  sink.gain.value = 0;
  node.connect(sink).connect(ctx.destination);

  state.audio = { ctx, stream, node, source, sink };
  state.recording = true;
  el.mic.classList.add('recording');
  sendJSON({ type: 'audio_start', language: state.language, mode: 'hold' });
}

function stopRecording(silent) {
  if (!state.recording) return;
  state.recording = false;
  el.mic.classList.remove('recording');

  const audio = state.audio;
  state.audio = null;
  if (audio) {
    try { audio.node.port.onmessage = null; } catch {}
    try { audio.source.disconnect(); audio.node.disconnect(); } catch {}
    audio.stream.getTracks().forEach((t) => t.stop());
    audio.ctx.close().catch(() => {});
  }

  if (!silent) sendJSON({ type: 'audio_stop' });
}

function showInsecureBanner() {
  el.banner.hidden = false;
  el.httpsHint.textContent = `https://${location.hostname}:8443`;
  el.mic.disabled = true;
}

/* ------------------------------------------------------------------- input */

function sendText() {
  const text = el.input.value.trim();
  if (!text || !state.connected) return;
  el.input.value = '';
  el.input.style.height = 'auto';
  setControlsBusy(true);
  sendJSON({ type: 'text', text });
}

el.attach.addEventListener('click', () => el.fileInput.click());

el.fileInput.addEventListener('change', () => {
  const files = Array.from(el.fileInput.files || []);
  el.fileInput.value = '';   // so re-picking the same file fires change again
  if (files.length) handleFiles(files);
});

/* Drag and drop anywhere on the page. */
document.addEventListener('dragover', (e) => {
  if (e.dataTransfer?.types?.includes('Files')) e.preventDefault();
});
document.addEventListener('drop', (e) => {
  const files = Array.from(e.dataTransfer?.files || []);
  if (!files.length) return;
  e.preventDefault();
  handleFiles(files);
});

el.lang.value = state.language;
el.lang.addEventListener('change', () => {
  state.language = el.lang.value;
  localStorage.setItem(LANG_KEY, state.language);
  sendJSON({ type: 'set_language', language: state.language });
});

/* Push-to-talk. Pointer events cover mouse, touch and pen in one path, and
   setPointerCapture keeps the release event coming to the button even if the
   finger slides off it -- otherwise the mic would stay open. */
let micHeld = false;

el.mic.addEventListener('pointerdown', async (e) => {
  if (e.button !== 0 && e.pointerType === 'mouse') return;
  e.preventDefault();
  if (micHeld || !state.connected) return;

  micHeld = true;
  try { el.mic.setPointerCapture(e.pointerId); } catch {}
  el.mic.classList.add('recording');

  await startRecording();
  // released again while getUserMedia was still starting up
  if (!micHeld) stopRecording(false);
});

function releaseMic() {
  if (!micHeld) return;
  micHeld = false;
  el.mic.classList.remove('recording');
  stopRecording(false);
}

el.mic.addEventListener('pointerup', releaseMic);
el.mic.addEventListener('pointercancel', releaseMic);
// belt and braces: a pointerup that lands outside the captured element
window.addEventListener('blur', releaseMic);

// stop long-press turning into a text selection / callout on mobile
el.mic.addEventListener('contextmenu', (e) => e.preventDefault());

el.send.addEventListener('click', sendText);

el.input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendText();
  }
});

el.input.addEventListener('input', () => {
  el.input.style.height = 'auto';
  el.input.style.height = Math.min(el.input.scrollHeight, 140) + 'px';
});

el.reset.addEventListener('click', () => {
  sendJSON({ type: 'reset' });
  el.messages.innerHTML = '';
  state.botBubble = null;
  state.toolRow = null;
  state.interimBubble = null;
  setControlsBusy(false);
});

document.addEventListener('click', (e) => {
  const chip = e.target.closest('.chip[data-q]');
  if (!chip) return;
  el.input.value = chip.dataset.q;
  sendText();
});

document.addEventListener('visibilitychange', () => {
  if (document.hidden) releaseMic();
});

if (!window.isSecureContext) showInsecureBanner();
connect();
