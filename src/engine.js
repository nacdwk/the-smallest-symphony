// Stage engine: renderer, post-processing, the frame loop, input routing and the
// movement-to-movement transition (every scene collapses into one plucked string, then reopens).
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import * as fx from './fx.js';

export const POST_DEFAULTS = {
  bloomStrength: 1.0, bloomRadius: 0.6, bloomThreshold: 0.18,
  exposure: 1.0, aberration: 0.0018, grain: 0.035, vignette: 0.85,
};

const FinalShader = {
  uniforms: {
    tDiffuse: { value: null }, uRes: { value: new THREE.Vector2(1, 1) }, uTime: { value: 0 },
    uGrain: { value: 0.035 }, uVignette: { value: 0.85 }, uAberration: { value: 0.0018 },
    uCollapse: { value: 0 }, uLine: { value: 0 }, uLineLen: { value: 1 }, uWobble: { value: 0 },
  },
  vertexShader: /* glsl */`varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
  fragmentShader: /* glsl */`
    uniform sampler2D tDiffuse; uniform vec2 uRes; uniform float uTime, uGrain, uVignette, uAberration, uCollapse, uLine, uLineLen, uWobble;
    varying vec2 vUv;
    float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
    void main(){
      vec2 uv = vUv; float c = uCollapse;
      float sy = mix(1.0, 0.0012, c), sx = mix(1.0, 0.9, c);
      vec2 q = vec2(0.5 + (uv.x - 0.5) / sx, 0.5 + (uv.y - 0.5) / sy);
      vec3 col = vec3(0.0);
      if (q.x >= 0.0 && q.x <= 1.0 && q.y >= 0.0 && q.y <= 1.0) {
        vec2 d = q - 0.5; float r2 = dot(d, d);
        vec2 off = d * uAberration * (1.0 + 2.0 * r2);
        col = vec3(texture2D(tDiffuse, q + off).r, texture2D(tDiffuse, q).g, texture2D(tDiffuse, q - off).b);
      }
      col *= 1.0 + c * 2.2;
      // the string the scene collapses into
      float x0 = 0.5 - 0.5 * uLineLen;
      float s = clamp((uv.x - x0) / max(uLineLen, 1e-4), 0.0, 1.0);
      float env = sin(3.14159265 * s);
      float wob = uWobble * env * (0.6 * sin(uv.x * 38.0 - uTime * 52.0) + 0.4 * sin(uv.x * 17.0 + uTime * 29.0)) * 0.014;
      float dy = (uv.y - 0.5 - wob) * uRes.y;
      float inside = step(x0, uv.x) * step(uv.x, x0 + uLineLen);
      float line = (exp(-dy * dy / 1.6) * 1.5 + exp(-abs(dy) / 16.0) * 0.32) * env * inside;
      col += vec3(1.0, 0.86, 0.62) * line * uLine;
      // vignette + grain
      vec2 v = uv - 0.5; v.x *= uRes.x / uRes.y;
      col *= 1.0 - uVignette * smoothstep(0.42, 1.25, length(v));
      col += (hash(uv * uRes + fract(uTime * 7.31) * 113.0) - 0.5) * uGrain;
      gl_FragColor = vec4(max(col, 0.0), 1.0);
    }`,
};

const easeInCubic = (t) => t * t * t;
const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);

export class Engine {
  constructor({ canvas, audio, ui, quality = 'high' }) {
    this.canvas = canvas; this.audio = audio; this.ui = ui; this.quality = quality;
    this.isTouch = matchMedia('(pointer: coarse)').matches;
    this.basePR = Math.min(window.devicePixelRatio || 1, quality === 'low' ? 1.25 : 1.75);
    this.pr = this.basePR;

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance', alpha: false, stencil: false });
    renderer.setPixelRatio(this.pr);
    renderer.setClearColor(fx.palette.void, 1);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    this.renderer = renderer;

    const w = window.innerWidth, h = window.innerHeight;
    renderer.setSize(w, h, false);
    const rt = new THREE.WebGLRenderTarget(w * this.pr, h * this.pr, { type: THREE.HalfFloatType, samples: quality === 'low' ? 0 : 4 });
    this.composer = new EffectComposer(renderer, rt);
    this.composer.setPixelRatio(this.pr);
    this.composer.setSize(w, h);
    this.renderPass = new RenderPass(new THREE.Scene(), new THREE.PerspectiveCamera());
    this.bloom = new UnrealBloomPass(new THREE.Vector2(w, h), POST_DEFAULTS.bloomStrength, POST_DEFAULTS.bloomRadius, POST_DEFAULTS.bloomThreshold);
    this.output = new OutputPass();
    this.final = new ShaderPass(FinalShader);
    this.composer.addPass(this.renderPass);
    this.composer.addPass(this.bloom);
    this.composer.addPass(this.output);
    this.composer.addPass(this.final);

    this.post = { ...POST_DEFAULTS };
    this.postTarget = { ...POST_DEFAULTS };

    this.movements = [];
    this.instances = [];
    this.index = -1;
    this.current = null;
    this.time = 0;
    this.transition = null;
    this.pointer = { x: 0, y: 0, ndc: new THREE.Vector2(), down: false };
    this._frameTimes = [];
    this._lastDegrade = 0;

    this.ctx = this._makeCtx();
    this._bindInput();
    window.addEventListener('resize', () => this._resize());
    document.addEventListener('visibilitychange', () => { this._last = performance.now(); });
    this._resize();
  }

  _makeCtx() {
    const self = this;
    return {
      THREE, fx,
      renderer: this.renderer, canvas: this.canvas, audio: this.audio, ui: this.ui, engine: this,
      get width() { return window.innerWidth; },
      get height() { return window.innerHeight; },
      get aspect() { return window.innerWidth / Math.max(1, window.innerHeight); },
      get time() { return self.time; },
      get pixelRatio() { return self.pr; },
      isTouch: this.isTouch,
      quality: this.quality,
      pointer: this.pointer,
      setPost: (p) => Object.assign(this.postTarget, p),
      setCursor: (c) => { this.canvas.className = c || ''; },
      next: () => this.next(),
    };
  }

  register(defs) {
    this.movements = defs;
    this.instances = defs.map(() => null);
    this.ui.buildScore(defs, (i) => this.goTo(i));
  }

  _instance(i) {
    if (!this.instances[i]) {
      const def = this.movements[i];
      try {
        this.instances[i] = def.create(this.ctx);
      } catch (err) {
        console.error(`Movement ${def.numeral} failed to build`, err);
        const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
        this.instances[i] = { scene, camera, enter() {}, exit() {}, update() {}, _broken: true };
        this.ui.toast(`Movement ${def.numeral} could not start on this device.`);
      }
      const inst = this.instances[i];
      try { this.renderer.compile(inst.scene, inst.camera); } catch (e) { /* compile is an optimisation only */ }
    }
    return this.instances[i];
  }

  _activate(i) {
    const def = this.movements[i];
    if (this.current) { try { this.current.exit && this.current.exit(); } catch (e) { console.error(e); } }
    this.ui.clearPanels();
    this.index = i;
    const inst = this._instance(i);
    this.current = inst;
    this.ui.setMovement(def, i, this.movements.length);
    this.postTarget = { ...POST_DEFAULTS, ...(def.post || {}), ...(inst.post || {}) };
    this.renderPass.scene = inst.scene;
    this.renderPass.camera = inst.camera;
    this._fitCamera(inst);
    if (def.drone) this.audio.setDrone(def.drone.notes, def.drone.opts || {});
    try { inst.enter && inst.enter(); } catch (e) { console.error(e); }
    if (location.hash !== `#m${i + 1}`) { try { history.replaceState(null, '', `#m${i + 1}`); } catch (e) {} }
  }

  /** Jump without transition (used at boot). */
  start(i = 0) {
    this._activate(i);
    Object.assign(this.post, this.postTarget);
    this._last = performance.now();
    requestAnimationFrame((t) => this._frame(t));
  }

  goTo(i) {
    const n = this.movements.length;
    i = ((i % n) + n) % n;
    if (i === this.index || this.transition) return;
    // preload the next instance during the collapse so the swap frame is cheap
    this.transition = { to: i, t: 0, swapped: false };
    this.ui.el.hud.classList.add('swapping');
    this.audio.noise({ freq: 400, sweepTo: 3200, dur: 0.55, gain: 0.05, q: 2.5 });
  }
  next() { this.goTo(this.index + 1); }
  prev() { this.goTo(this.index - 1); }

  _stepTransition(dt) {
    const tr = this.transition; if (!tr) return;
    tr.t += dt;
    const T1 = 0.55, T2 = 0.95, T3 = 1.75;
    const u = this.final.uniforms;
    if (tr.t < T1) {
      const k = easeInCubic(tr.t / T1);
      u.uCollapse.value = k; u.uLine.value = fx.smoothstep(0.35, 1, tr.t / T1); u.uLineLen.value = 1; u.uWobble.value = 0;
    } else {
      if (!tr.swapped) {
        tr.swapped = true;
        this._activate(tr.to);
        const f = this.audio.degree(2 + tr.to, 1);
        this.audio.pluck(f, { gain: 0.32, decay: 3.2, bright: 0.65 });
        this.audio.pluck(f * 1.5, { gain: 0.12, decay: 2.6, bright: 0.4, when: 0.05, pan: 0.3 });
      }
      if (tr.t < T2) {
        const k = (tr.t - T1) / (T2 - T1);
        u.uCollapse.value = 1; u.uLine.value = 1; u.uWobble.value = Math.sin(k * Math.PI) * 1.0 * (1 - k * 0.5);
      } else {
        this.ui.el.hud.classList.remove('swapping');
        const k = Math.min(1, (tr.t - T2) / (T3 - T2));
        u.uCollapse.value = 1 - easeOutCubic(k); u.uLine.value = 1 - fx.smoothstep(0, 0.6, k); u.uWobble.value = 0;
        if (k >= 1) { this.transition = null; u.uCollapse.value = 0; u.uLine.value = 0; }
      }
    }
  }

  _frame(now) {
    requestAnimationFrame((t) => this._frame(t));
    const dt = Math.min(0.05, Math.max(0, (now - (this._last || now)) / 1000));
    this._last = now;
    if (document.hidden) return;
    this.time += dt;
    this._stepTransition(dt);
    this._applyPost(dt);
    const inst = this.current;
    if (inst && inst.update) { try { inst.update(dt, this.time); } catch (e) { if (!inst._warned) { console.error(e); inst._warned = true; } } }
    this.final.uniforms.uTime.value = this.time;
    this.composer.render(dt);
    this._adapt(dt);
    if (this.fpsEl) this._fps(dt);
  }

  _applyPost(dt) {
    const k = fx.damp(3, dt), p = this.post, t = this.postTarget;
    for (const key in p) p[key] += ((t[key] ?? POST_DEFAULTS[key]) - p[key]) * k;
    this.bloom.strength = p.bloomStrength; this.bloom.radius = p.bloomRadius; this.bloom.threshold = p.bloomThreshold;
    this.renderer.toneMappingExposure = p.exposure;
    const u = this.final.uniforms;
    u.uGrain.value = p.grain; u.uVignette.value = p.vignette; u.uAberration.value = p.aberration;
  }

  // Lower the pixel ratio if frames are consistently slow. Never raises it again mid-session.
  _adapt(dt) {
    const ft = this._frameTimes; ft.push(dt); if (ft.length > 90) ft.shift();
    if (ft.length < 90 || this.time - this._lastDegrade < 3 || this.transition) return;
    const avg = ft.reduce((a, b) => a + b, 0) / ft.length;
    if (avg > 0.024 && this.pr > 0.75) {
      this.pr = Math.max(0.75, this.pr * 0.82);
      this._lastDegrade = this.time; ft.length = 0;
      this._resize();
    }
  }

  _fps(dt) {
    this._fpsAcc = (this._fpsAcc || 0) + dt; this._fpsN = (this._fpsN || 0) + 1;
    if (this._fpsAcc > 0.5) { this.fpsEl.textContent = `${Math.round(this._fpsN / this._fpsAcc)} fps · pr ${this.pr.toFixed(2)}`; this._fpsAcc = 0; this._fpsN = 0; }
  }

  _fitCamera(inst) {
    const cam = inst && inst.camera;
    if (cam && cam.isPerspectiveCamera) { cam.aspect = window.innerWidth / Math.max(1, window.innerHeight); cam.updateProjectionMatrix(); }
    if (inst && inst.resize) { try { inst.resize(window.innerWidth, window.innerHeight); } catch (e) { console.error(e); } }
  }

  _resize() {
    const w = window.innerWidth, h = window.innerHeight;
    this.renderer.setPixelRatio(this.pr);
    this.renderer.setSize(w, h, false);
    this.composer.setPixelRatio(this.pr);
    this.composer.setSize(w, h);
    this.final.uniforms.uRes.value.set(w * this.pr, h * this.pr);
    this.instances.forEach((inst) => inst && this._fitCamera(inst));
  }

  // ---- input -------------------------------------------------------------------------
  _ev(e) {
    const r = this.canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    const ndc = new THREE.Vector2((x / r.width) * 2 - 1, -(y / r.height) * 2 + 1);
    const ev = { x, y, ndc, dx: x - this.pointer.x, dy: y - this.pointer.y, button: e.button, buttons: e.buttons, pointerType: e.pointerType || 'mouse', shift: e.shiftKey, alt: e.altKey, meta: e.metaKey || e.ctrlKey, original: e };
    this.pointer.x = x; this.pointer.y = y; this.pointer.ndc.copy(ndc);
    return ev;
  }
  _call(name, ev) {
    const inst = this.current;
    if (!inst || !inst[name] || this.transition) return false;
    try { return inst[name](ev); } catch (e) { console.error(e); return false; }
  }
  _bindInput() {
    const c = this.canvas;
    c.addEventListener('pointerdown', (e) => { this.pointer.down = true; try { c.setPointerCapture(e.pointerId); } catch (_) {} this._call('onPointerDown', this._ev(e)); });
    c.addEventListener('pointermove', (e) => this._call('onPointerMove', this._ev(e)));
    const up = (e) => { if (!this.pointer.down) return; this.pointer.down = false; this._call('onPointerUp', this._ev(e)); };
    c.addEventListener('pointerup', up);
    c.addEventListener('pointercancel', up);
    c.addEventListener('wheel', (e) => { e.preventDefault(); const ev = this._ev(e); ev.deltaY = e.deltaY * (e.deltaMode === 1 ? 16 : 1); ev.deltaX = e.deltaX; this._call('onWheel', ev); }, { passive: false });
    c.addEventListener('contextmenu', (e) => e.preventDefault());
    window.addEventListener('keydown', (e) => {
      const tag = (e.target && e.target.tagName) || '';
      if (/INPUT|TEXTAREA|SELECT/.test(tag) && e.target.type !== 'checkbox') return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (document.getElementById('intro') && !document.getElementById('intro').classList.contains('gone')) return;
      if (this._call('onKey', { key: e.key, code: e.code, shift: e.shiftKey, repeat: e.repeat, type: 'down', original: e })) { e.preventDefault(); return; }
      if (!e.repeat && this.ui.handleKey(e)) { e.preventDefault(); return; }
      if (e.key === ']' || e.key === 'ArrowRight' || e.key === 'PageDown') { e.preventDefault(); this.next(); }
      else if (e.key === '[' || e.key === 'ArrowLeft' || e.key === 'PageUp') { e.preventDefault(); this.prev(); }
    });
    window.addEventListener('keyup', (e) => { this._call('onKeyUp', { key: e.key, code: e.code, type: 'up', original: e }); });
  }
}
