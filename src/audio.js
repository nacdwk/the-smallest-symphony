// Audio engine: a tiny additive synthesizer, a reverb hall and an ambient drone.
// Every method is safe to call before start() or after the user chose silence: it returns
// no-op voices, so movements never need to check whether sound is on.

const NOOP = () => {};
const NULL_VOICE = new Proxy({}, { get: (t, k) => (k === 'then' ? undefined : NOOP) });

// D minor pentatonic around D2. Everything in the piece sits in this key so that
// plucks from different movements always harmonise with the drone.
const ROOT = 73.42;
const PENTA = [0, 3, 5, 7, 10];

export class AudioEngine {
  constructor() {
    this.ctx = null;
    this.enabled = false;
    this.muted = false;
    this.root = ROOT;
    this._drone = null;
    this._listeners = new Set();
  }

  get now() { return this.ctx ? this.ctx.currentTime : 0; }
  get on() { return this.enabled && !this.muted; }

  /** Must be called from a user gesture. */
  async start() {
    if (this.ctx) { if (this.ctx.state !== 'running') await this.ctx.resume(); return; }
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    const ctx = new AC({ latencyHint: 'interactive' });
    this.ctx = ctx;

    this.master = ctx.createGain();
    this.master.gain.value = 0;
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -16; comp.knee.value = 18; comp.ratio.value = 4;
    comp.attack.value = 0.004; comp.release.value = 0.25;
    this.master.connect(comp).connect(ctx.destination);

    // `out` is where voices connect: it feeds a dry path and a long reverb send.
    this.out = ctx.createGain();
    const dry = ctx.createGain(); dry.gain.value = 0.78;
    this.reverb = ctx.createConvolver();
    this.reverb.buffer = this._impulse(4.6, 3.1);
    const wet = ctx.createGain(); wet.gain.value = 0.42;
    this.out.connect(dry).connect(this.master);
    this.out.connect(this.reverb).connect(wet).connect(this.master);

    // A quieter bus for the ambient bed so it never masks the instrument.
    this.bed = ctx.createGain(); this.bed.gain.value = 0.5;
    this.bed.connect(this.out);

    this.analyser = ctx.createAnalyser();
    this.analyser.fftSize = 2048;
    this.master.connect(this.analyser);

    if (ctx.state !== 'running') await ctx.resume();
    this.enabled = true;
    this.master.gain.setTargetAtTime(this.muted ? 0 : 0.9, ctx.currentTime, 0.4);
    this._emit();
  }

  setMuted(m) {
    this.muted = m;
    if (this.ctx) this.master.gain.setTargetAtTime(m ? 0 : 0.9, this.ctx.currentTime, 0.15);
    this._emit();
  }
  onChange(fn) { this._listeners.add(fn); return () => this._listeners.delete(fn); }
  _emit() { this._listeners.forEach((f) => f(this)); }

  // ---- pitch helpers -------------------------------------------------------
  /** Frequency `semis` semitones above the piece's root (D2 = 73.42 Hz). */
  freq(semis = 0) { return this.root * Math.pow(2, semis / 12); }
  /** i-th note of the D minor pentatonic scale, starting at D2 + octave*12. Wraps upward. */
  degree(i, octave = 0) {
    const o = Math.floor(i / PENTA.length);
    const d = ((i % PENTA.length) + PENTA.length) % PENTA.length;
    return this.freq(PENTA[d] + 12 * (o + octave));
  }

  // ---- one-shot instruments ------------------------------------------------
  /**
   * Plucked-string tone built from decaying harmonics (higher partials die faster).
   * opts: gain (0.25), decay seconds (2.4), bright 0..1 (0.5), pan -1..1, partials (10),
   * inharm (0, small positive values make it bell-like), when (seconds from now), out (AudioNode)
   */
  pluck(freq, opts = {}) {
    if (!this.on) return;
    const { gain = 0.25, decay = 2.4, bright = 0.5, pan = 0, partials = 10, inharm = 0, when = 0, out } = opts;
    const ctx = this.ctx, t0 = ctx.currentTime + when;
    const bus = ctx.createGain(); bus.gain.value = gain;
    const p = ctx.createStereoPanner(); p.pan.value = Math.max(-1, Math.min(1, pan));
    bus.connect(p).connect(out || this.out);
    const nyq = ctx.sampleRate * 0.45;
    let longest = 0;
    for (let k = 1; k <= partials; k++) {
      const f = freq * k * (1 + inharm * k * k);
      if (f > nyq) break;
      const amp = Math.pow(k, -(2.1 - 1.6 * bright)) * (k === 1 ? 1 : 0.9);
      const tau = decay / (1 + (k - 1) * (0.55 - 0.35 * bright));
      const o = ctx.createOscillator(); o.type = 'sine';
      o.frequency.value = f * (1 + (Math.random() - 0.5) * 0.0016);
      const g = ctx.createGain();
      g.gain.setValueAtTime(0, t0);
      g.gain.linearRampToValueAtTime(amp, t0 + 0.004);
      g.gain.setTargetAtTime(0, t0 + 0.004, tau / 3.2);
      o.connect(g).connect(bus);
      o.start(t0); o.stop(t0 + tau * 2.2 + 0.1);
      longest = Math.max(longest, tau * 2.2 + 0.1);
    }
    setTimeout(() => { try { bus.disconnect(); p.disconnect(); } catch (e) {} }, (when + longest + 0.5) * 1000);
  }

  /** Glassy bell: inharmonic partials, long decay. Good for "events" (a particle identified, a universe settling). */
  bell(freq, opts = {}) {
    this.pluck(freq, { partials: 7, inharm: 0.0045, bright: 0.75, decay: 4.2, gain: 0.16, ...opts });
  }

  /** Soft swelling tone (sine + octave + fifth), for sustained moments. */
  swell(freq, opts = {}) {
    if (!this.on) return;
    const { gain = 0.12, attack = 0.8, dur = 2.5, pan = 0, when = 0, out } = opts;
    const ctx = this.ctx, t0 = ctx.currentTime + when;
    const bus = ctx.createGain(); bus.gain.value = 0;
    bus.gain.setValueAtTime(0, t0);
    bus.gain.linearRampToValueAtTime(gain, t0 + attack);
    bus.gain.setTargetAtTime(0, t0 + attack + dur, 0.6);
    const p = ctx.createStereoPanner(); p.pan.value = pan;
    bus.connect(p).connect(out || this.out);
    [[1, 1], [2, 0.3], [1.5, 0.18], [3, 0.08]].forEach(([m, a]) => {
      const o = ctx.createOscillator(); o.type = 'sine'; o.frequency.value = freq * m;
      const g = ctx.createGain(); g.gain.value = a;
      o.connect(g).connect(bus); o.start(t0); o.stop(t0 + attack + dur + 4);
    });
    setTimeout(() => { try { bus.disconnect(); p.disconnect(); } catch (e) {} }, (when + attack + dur + 4.5) * 1000);
  }

  /** Filtered noise burst (whoosh, crackle, foam). */
  noise(opts = {}) {
    if (!this.on) return;
    const { gain = 0.12, dur = 0.8, freq = 1200, q = 0.8, type = 'bandpass', attack = 0.02, sweepTo, pan = 0, when = 0, out } = opts;
    const ctx = this.ctx, t0 = ctx.currentTime + when;
    const src = ctx.createBufferSource(); src.buffer = this._noiseBuffer();
    const f = ctx.createBiquadFilter(); f.type = type; f.frequency.value = freq; f.Q.value = q;
    if (sweepTo) f.frequency.exponentialRampToValueAtTime(sweepTo, t0 + dur);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t0); g.gain.linearRampToValueAtTime(gain, t0 + attack);
    g.gain.setTargetAtTime(0, t0 + attack, Math.max(0.02, (dur - attack) / 3));
    const p = ctx.createStereoPanner(); p.pan.value = pan;
    src.connect(f).connect(g).connect(p).connect(out || this.out);
    src.start(t0, Math.random() * 1.5); src.stop(t0 + dur + 0.5);
    setTimeout(() => { try { p.disconnect(); } catch (e) {} }, (when + dur + 1) * 1000);
  }

  // ---- sustained voices ----------------------------------------------------
  /**
   * Additive voice with n harmonic partials whose amplitudes you drive every frame.
   * Returns { setFreq(f, glide), setAmp(i, a, glide), setAmps(array, glide), setGain(g, glide), setPan(p), stop(release) }.
   * Partial i (0-based) sounds at freq * (i+1). Keep sum of amps around 1; gain defaults to 0.18.
   */
  additive(freq, n = 8, opts = {}) {
    if (!this.on) return NULL_VOICE;
    const { gain = 0.18, type = 'sine', out } = opts;
    const ctx = this.ctx;
    const bus = ctx.createGain(); bus.gain.value = 0;
    bus.gain.setTargetAtTime(gain, ctx.currentTime, 0.05);
    const p = ctx.createStereoPanner();
    bus.connect(p).connect(out || this.out);
    const oscs = [], gains = [];
    for (let i = 0; i < n; i++) {
      const o = ctx.createOscillator(); o.type = type; o.frequency.value = freq * (i + 1);
      const g = ctx.createGain(); g.gain.value = 0;
      o.connect(g).connect(bus); o.start();
      oscs.push(o); gains.push(g);
    }
    let stopped = false;
    const v = {
      setFreq(f, glide = 0.03) { if (stopped) return; oscs.forEach((o, i) => o.frequency.setTargetAtTime(f * (i + 1), ctx.currentTime, glide)); },
      setAmp(i, a, glide = 0.03) { if (stopped || !gains[i]) return; gains[i].gain.setTargetAtTime(Math.max(0, a), ctx.currentTime, glide); },
      setAmps(arr, glide = 0.03) { arr.forEach((a, i) => v.setAmp(i, a, glide)); },
      setGain(g, glide = 0.05) { if (!stopped) bus.gain.setTargetAtTime(g, ctx.currentTime, glide); },
      setPan(x) { if (!stopped) p.pan.setTargetAtTime(Math.max(-1, Math.min(1, x)), ctx.currentTime, 0.05); },
      stop(release = 0.4) {
        if (stopped) return; stopped = true;
        bus.gain.setTargetAtTime(0, ctx.currentTime, release / 4);
        const end = ctx.currentTime + release + 0.1;
        oscs.forEach((o) => o.stop(end));
        setTimeout(() => { try { bus.disconnect(); p.disconnect(); } catch (e) {} }, (release + 0.4) * 1000);
      },
    };
    return v;
  }

  /**
   * Ambient bed for a movement. `notes` are semitone offsets from the root (e.g. [0, 7, 15]).
   * Crossfades from whatever bed was playing. opts: gain (0.1), cutoff Hz (900), wobble 0..1 (0.3).
   */
  setDrone(notes, opts = {}) {
    if (!this.enabled) { this._pendingDrone = [notes, opts]; return; }
    const { gain = 0.1, cutoff = 900, wobble = 0.3 } = opts;
    const ctx = this.ctx, t = ctx.currentTime;
    if (this._drone) this._drone.stop(3.5);
    if (!notes || !notes.length) { this._drone = null; return; }
    const bus = ctx.createGain(); bus.gain.value = 0;
    bus.gain.setTargetAtTime(gain, t, 1.4);
    const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = cutoff; lp.Q.value = 0.7;
    const lfo = ctx.createOscillator(); lfo.frequency.value = 0.06;
    const lfoG = ctx.createGain(); lfoG.gain.value = cutoff * 0.45 * wobble;
    lfo.connect(lfoG).connect(lp.frequency); lfo.start();
    lp.connect(bus).connect(this.bed);
    const oscs = [lfo];
    notes.forEach((s, i) => {
      const f = this.freq(s);
      [-1, 1].forEach((side) => {
        const o = ctx.createOscillator(); o.type = i === 0 ? 'triangle' : 'sawtooth';
        o.frequency.value = f; o.detune.value = side * (4 + i * 2);
        const g = ctx.createGain(); g.gain.value = (i === 0 ? 0.5 : 0.22) / notes.length;
        const pn = ctx.createStereoPanner(); pn.pan.value = side * 0.45;
        o.connect(g).connect(pn).connect(lp); o.start(); oscs.push(o);
      });
    });
    this._drone = {
      stop: (rel = 3) => {
        bus.gain.setTargetAtTime(0, ctx.currentTime, rel / 4);
        const end = ctx.currentTime + rel + 0.2;
        oscs.forEach((o) => o.stop(end));
        setTimeout(() => { try { bus.disconnect(); } catch (e) {} }, (rel + 0.6) * 1000);
      },
    };
  }
  resumePendingDrone() {
    if (this._pendingDrone) { const [n, o] = this._pendingDrone; this._pendingDrone = null; this.setDrone(n, o); }
  }

  // ---- internals -----------------------------------------------------------
  _impulse(seconds, decay) {
    const ctx = this.ctx, rate = ctx.sampleRate, len = Math.floor(rate * seconds);
    const buf = ctx.createBuffer(2, len, rate);
    for (let c = 0; c < 2; c++) {
      const d = buf.getChannelData(c);
      for (let i = 0; i < len; i++) {
        const t = i / len;
        d[i] = (Math.random() * 2 - 1) * Math.pow(1 - t, decay) * (i < rate * 0.012 ? i / (rate * 0.012) : 1);
      }
    }
    return buf;
  }
  _noiseBuffer() {
    if (this._nb) return this._nb;
    const ctx = this.ctx, len = ctx.sampleRate * 3;
    const b = ctx.createBuffer(1, len, ctx.sampleRate), d = b.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
    return (this._nb = b);
  }
}
