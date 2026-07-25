// Mic capture worklet.
//
// The render quantum is 128 frames (~2.7 ms at 48 kHz); posting each one would
// mean ~375 messages/sec across the worklet port. We batch to BATCH frames
// (~43 ms at 48 kHz) before posting, which keeps the main thread quiet while
// staying well under the VAD's 20 ms granularity after downsampling.

const BATCH = 2048;

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(BATCH);
    this.offset = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;

    let read = 0;
    while (read < channel.length) {
      const take = Math.min(BATCH - this.offset, channel.length - read);
      this.buffer.set(channel.subarray(read, read + take), this.offset);
      this.offset += take;
      read += take;

      if (this.offset === BATCH) {
        const out = this.buffer.slice(0);
        this.port.postMessage(out, [out.buffer]);
        this.offset = 0;
      }
    }
    return true;
  }
}

registerProcessor('capture-processor', CaptureProcessor);
