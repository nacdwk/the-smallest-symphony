// Boot: load the six movements, wire the programme chrome, run the intro.
import * as THREE from 'three';
import { Engine } from './engine.js';
import { AudioEngine } from './audio.js';
import { UI } from './ui.js';

// Each movement is imported on its own so one that fails to load cannot take the rest
// of the piece down with it; esbuild still inlines these literal imports into one file.
const LOADERS = [
  () => import('./movements/m1-descent.js'),
  () => import('./movements/m2-instrument.js'),
  () => import('./movements/m3-dimensions.js'),
  () => import('./movements/m4-worldsheet.js'),
  () => import('./movements/m5-branes.js'),
  () => import('./movements/m6-landscape.js'),
];
const FALLBACK = [['I', 'Descent'], ['II', 'The Instrument'], ['III', 'Hidden Dimensions'], ['IV', 'Worldsheets'], ['V', 'Branes'], ['VI', 'The Landscape']];

const params = new URLSearchParams(location.search);
const hashM = /^#m([1-6])$/.exec(location.hash || '');
const startIndex = Math.max(0, Math.min(5, (parseInt(params.get('m') || (hashM && hashM[1]) || '1', 10) || 1) - 1));
const skipIntro = params.has('skip');
const lowQ = params.get('q') === 'low' || (matchMedia('(pointer: coarse)').matches && Math.min(screen.width, screen.height) < 820);

const audio = new AudioEngine();
const ui = new UI();
const canvas = document.getElementById('gl');

let engine;
try {
  engine = new Engine({ canvas, audio, ui, quality: lowQ ? 'low' : 'high' });
} catch (err) {
  console.error(err);
  document.getElementById('go').innerHTML = '<span class="loading">This piece needs WebGL 2, which this browser did not provide.</span>';
  throw err;
}
window.__symphony = engine;

const loaded = await Promise.allSettled(LOADERS.map((f) => f()));
const defs = loaded.map((r, i) => {
  if (r.status === 'fulfilled' && r.value && r.value.default) return r.value.default;
  console.error(`Movement ${FALLBACK[i][0]} failed to load`, r.reason);
  return { id: 'missing-' + i, numeral: FALLBACK[i][0], title: FALLBACK[i][1], tempo: '', caption: 'This movement failed to load.', create: () => { throw new Error('missing module'); } };
});
engine.register(defs);
engine.start(startIndex);

if (params.has('fps')) {
  const el = document.createElement('div'); el.className = 'fps'; document.getElementById('hud').appendChild(el); engine.fpsEl = el;
}

// ---- programme chrome ----------------------------------------------------------------
const hud = document.getElementById('hud');
const intro = document.getElementById('intro');
const sndBtn = document.getElementById('snd');
const sndLbl = document.getElementById('snd-l');
const paintSound = () => {
  const on = audio.enabled && !audio.muted;
  sndBtn.classList.toggle('on', on);
  sndBtn.setAttribute('aria-pressed', String(on));
  sndLbl.textContent = on ? 'Sound on' : 'Sound off';
};
audio.onChange(paintSound);
paintSound();

sndBtn.addEventListener('click', async () => {
  if (!audio.enabled) { await audio.start(); audio.setMuted(false); audio.resumePendingDrone(); }
  else audio.setMuted(!audio.muted);
});
document.getElementById('fs').addEventListener('click', () => {
  const d = document;
  if (d.fullscreenElement) { d.exitFullscreen && d.exitFullscreen().catch(() => {}); return; }
  const el = d.documentElement;
  const req = el.requestFullscreen || el.webkitRequestFullscreen;
  if (req) { try { const p = req.call(el); if (p && p.catch) p.catch(() => ui.toast('Fullscreen is not available here.')); } catch (e) { ui.toast('Fullscreen is not available here.'); } }
  else ui.toast('Fullscreen is not available here.');
});
document.getElementById('prev').addEventListener('click', () => engine.prev());
document.getElementById('next').addEventListener('click', () => engine.next());

function begin(withSound, instant = false) {
  if (instant) { intro.style.transition = 'none'; hud.style.transition = 'none'; }
  intro.classList.add('gone');
  hud.classList.remove('hidden');
  const go = async () => {
    if (withSound) { await audio.start(); audio.resumePendingDrone(); }
    const inst = engine.current;
    if (inst && inst.onBegin) { try { inst.onBegin(); } catch (e) { console.error(e); } }
  };
  go();
  canvas.focus({ preventScroll: true });
}

const goBox = document.getElementById('go');
goBox.innerHTML = '';
const b1 = document.createElement('button'); b1.type = 'button'; b1.className = 'btn primary'; b1.textContent = 'Begin with sound';
const b2 = document.createElement('button'); b2.type = 'button'; b2.className = 'btn'; b2.textContent = 'Begin in silence';
b1.addEventListener('click', () => begin(true));
b2.addEventListener('click', () => begin(false));
goBox.append(b1, b2);

if (skipIntro) begin(false, true);
if (params.has('demo')) setTimeout(() => { const i = engine.current; if (i && i.demo) i.demo(); }, 700);
// Test hook: leave the start movement and come back, to prove exit() and re-enter() are clean.
if (params.has('cycle')) {
  setTimeout(() => engine.goTo(startIndex + 1), 1500);
  setTimeout(() => engine.goTo(startIndex), 4200);
  if (params.has('demo')) setTimeout(() => { const i = engine.current; if (i && i.demo) i.demo(); }, 6400);
}
// Test hook: freeze a frame mid-transition (collapse into the string) for screenshots.
if (params.has('tx')) setTimeout(() => engine.goTo(startIndex + 1), 1000);
