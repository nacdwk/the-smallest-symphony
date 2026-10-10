# The Smallest Symphony — build spec

A playable, cinematic three.js piece that explains string theory in **six movements**. The framing is a
concert programme: each chapter is a *movement* with a numeral, a title and an Italian tempo marking, and
every interaction makes sound in one shared key (D minor pentatonic). The user does not watch string
theory. They **play** it.

The bar is "blow me away": every movement must be visually stunning on its own, deeply interactive, and
physically honest. Mediocre-but-working is a failure. So is gorgeous-but-wrong physics.

---

## 1. Project layout (do not restructure)

```
page.html                 programme chrome (HUD, intro, CSS). Shared. DO NOT EDIT.
src/main.js               boot + intro. Shared. DO NOT EDIT.
src/engine.js             renderer, bloom, final pass, transitions, input routing. Shared. DO NOT EDIT.
src/audio.js              AudioEngine (synth, reverb, drone). Shared. DO NOT EDIT.
src/ui.js                 caption / readout / controls / hint / toast builders. Shared. DO NOT EDIT.
src/fx.js                 palette, glow sprite, starfield, GLSL noise. Shared. DO NOT EDIT.
src/movements/m1-descent.js     I.   Descent
src/movements/m2-instrument.js  II.  The Instrument
src/movements/m3-dimensions.js  III. Hidden Dimensions
src/movements/m4-worldsheet.js  IV.  Worldsheets
src/movements/m5-branes.js      V.   Branes
src/movements/m6-landscape.js   VI.  The Landscape
tools/serve.py            dev server (port 8765) that wraps page.html like the production host does
tools/snap.sh             headless-Chrome screenshot + console capture
```

You own exactly **one** movement file. Everything your movement needs that is not in the shared files
goes inside your file. If you believe a shared file needs a change, do not make it: describe the change in
your final report.

Imports allowed: `three`, `three/addons/...` (three r170 examples/jsm, e.g. `three/addons/controls/OrbitControls.js`,
`three/addons/lines/Line2.js`, `three/addons/math/SimplexNoise.js`), and `../fx.js`. Nothing else (no
network fetches, no textures from URLs; generate textures procedurally with canvas/DataTexture).

## 2. The movement contract

```js
// src/movements/mN-name.js
import * as THREE from 'three';
import { palette, hdr, modeColor, spectrum, glowTexture, makeStarfield, GLSL, clamp, lerp, smoothstep, damp, sup, rng } from '../fx.js';

export default {
  id: 'instrument',
  numeral: 'II',
  title: 'The Instrument',
  tempo: 'Allegro, pizzicato',
  caption: 'Two or three plain sentences (HTML allowed, <em> for key terms). At most ~60 words.',
  fine: '<p>"Look closer" panel: 2–4 short paragraphs of accurate physics, including what the picture simplifies.</p>',
  drone: { notes: [0, 7, 12], opts: { gain: 0.1, cutoff: 900 } },   // semitones above D2; stay in D minor pentatonic: 0 3 5 7 10 12 15 17 19 22 24
  post: { bloomStrength: 1.0, bloomRadius: 0.6, bloomThreshold: 0.18, exposure: 1.0, aberration: 0.0018, grain: 0.035, vignette: 0.85 }, // optional overrides

  create(ctx) {              // called once, lazily, the first time the movement is visited
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, ctx.aspect, 0.01, 500);
    // ... build everything here ...
    return {
      scene, camera,                    // REQUIRED. Engine renders this scene with this camera.
      post: { /* optional, same keys as above */ },
      enter() {},                       // REQUIRED. Movement became active: build readout/controls (panels are cleared before every enter), start voices, show hint.
      exit() {},                        // REQUIRED. Movement left: stop sustained voices, disable OrbitControls, cancel timers. Instance is kept and may be entered again.
      update(dt, t) {},                 // REQUIRED. Per frame. dt seconds (clamped ≤ 0.05), t = engine time.
      resize(w, h) {},                  // optional. Engine already updates PerspectiveCamera.aspect.
      onPointerDown(ev) {}, onPointerMove(ev) {}, onPointerUp(ev) {},  // optional
      onWheel(ev) {},                   // optional. ev.deltaY in pixels.
      onKey(ev) { return false; },      // optional. Return true if you consumed ev.key (then the engine/controls skip it).
      onKeyUp(ev) {},                   // optional.
      onBegin() {},                     // optional. User pressed "Begin" on the intro (only matters for movement I, which sits behind the intro).
      demo() {},                        // REQUIRED. Programmatically perform your signature interaction so a headless screenshot shows the movement "in play".
    };
  },
};
```

Pointer event object `ev`: `{ x, y, ndc: THREE.Vector2, dx, dy, button, buttons, pointerType, shift, alt, meta, original }`
(x/y are CSS pixels relative to the canvas; `ndc` is ready for `Raycaster.setFromCamera`). Pointer capture
is already set on pointerdown. Events are suppressed during transitions. If you use `OrbitControls`, attach
it to `ctx.canvas`, set `enabled = false` in `exit()` and `true` in `enter()`. The engine's handlers run
before OrbitControls' handlers, so you can set `controls.enabled = false` inside `onPointerDown` when the
pointer grabs one of your objects, and re-enable it in `onPointerUp`.

Global keys the engine already uses when you do not consume them: `ArrowLeft`/`ArrowRight`/`[`/`]`/PageUp/PageDown
(previous/next movement). If you use arrow keys, consume them in `onKey` and say so in the hint.

### ctx (passed to create)
| member | what |
|---|---|
| `ctx.THREE`, `ctx.fx` | three namespace, fx module |
| `ctx.renderer`, `ctx.canvas` | the shared WebGLRenderer (WebGL2) and its canvas |
| `ctx.width`, `ctx.height`, `ctx.aspect`, `ctx.pixelRatio` | live getters |
| `ctx.time` | engine time (s) |
| `ctx.isTouch`, `ctx.quality` | `'high'` or `'low'` (phones). On `'low'`: halve particle counts / raymarch steps. |
| `ctx.pointer` | `{ x, y, ndc, down }`, live |
| `ctx.setPost({...})` | animate post-processing toward new values (bloom, exposure, aberration, grain, vignette) |
| `ctx.setCursor('' \| 'grab' \| 'grabbing' \| 'point')` | canvas cursor |
| `ctx.next()` | go to the next movement (e.g. from a "Continue" button) |
| `ctx.audio` | see §4 |
| `ctx.ui` | see §3 |

## 3. UI (`ctx.ui`) — build panels in `enter()`, every time

- `ctx.ui.readout({ title, rows: [{ id, k, v, cls }], note, html })` → `{ set(id, value, cls?), note(html), title(text), el }`
  Top-right live readout. `k` = small uppercase label, `v` = value (tabular mono). `cls` may be `gold`, `cyan`, `mag`
  for coloured values; `big` is available for one hero number via `html`. `set()` is cheap to call every frame (skips no-op writes).
  You can put custom HTML (e.g. a mini spectrum of `<i>` bars you restyle each frame via `el.querySelector`) in `html`.
  Keep it compact: ≤ 7 rows.
- `ctx.ui.controls([...items])` → `{ get(id), set(id, value), flash(id, index) }`. Bottom-right panel. Item types:
  `slider {id,label,min,max,step,value,format,onInput}`, `segmented {id,label,options:[{value,label,kbd}],value,onChange}`,
  `button {id,label,onClick,kbd}` (consecutive buttons share a row), `toggle {id,label,value,onChange,kbd}`,
  `keys {id,label,keys:[{label,sub,color,kbd,onPress}]}` (piano-style pads with a glow in `color`), `html {id,html}`.
  `kbd` is a `KeyboardEvent.key` value; the engine routes it while your movement is active. ≤ 5 controls. Every
  control must have a keyboard shortcut or be reachable by Tab.
- `ctx.ui.hint(text, ms?)` / `ctx.ui.hideHint()` — the breathing instruction at the bottom centre ("Drag the string to pluck it").
  Show one in `enter()`; hide it after the user has clearly understood (e.g. first successful interaction).
- `ctx.ui.toast(html, ms?)` — transient top-centre message for discoveries ("A graviton escaped the brane").
- `ctx.ui.setCaption({ body, fine })` — only if the caption must change mid-movement (rarely).

Screen real estate: the left ~90px holds the movement rail, the bottom-left ~460×260px holds the caption,
the bottom-right ~340×(your controls) the controls, the top-right ~320×(your readout) the readout. Compose your
hero object in the centre, slightly right of centre is fine. On phones (≤760px wide) the controls become a
full-width bottom sheet (max 38vh) and the caption shrinks to the title only, so keep the hero object in the
upper-middle of the frame on portrait screens (adjust camera distance/FOV in `resize()` when `aspect < 1`).

## 4. Audio (`ctx.audio`) — sound is half the piece

Everything is in D minor pentatonic so all movements harmonise with the drone. All calls are safe no-ops
while sound is off (headless tests run with sound off).

- `audio.on` (bool), `audio.ctx` (AudioContext or null), `audio.out` (voice bus with reverb), `audio.now`
- `audio.freq(semitones)` — Hz above D2 (73.42 Hz). `audio.degree(i, octave)` — i-th pentatonic note.
- `audio.pluck(freq, { gain, decay, bright, pan, partials, inharm, when })` — plucked string (decaying harmonics).
- `audio.bell(freq, opts)` — glassy inharmonic bell, for "events".
- `audio.swell(freq, { gain, attack, dur, pan, when })` — soft sustained tone.
- `audio.noise({ gain, dur, freq, q, type, attack, sweepTo, pan, when })` — filtered noise (whoosh, foam, crackle).
- `audio.additive(freq, n, { gain })` → voice `{ setFreq(f, glide), setAmp(i, a, glide), setAmps(arr), setGain(g), setPan(p), stop(release) }`.
  Partial i sounds at `freq*(i+1)`. **Create sustained voices lazily** in `update()` with `if (!voice && ctx.audio.on) voice = ...`
  so they appear if the user turns sound on mid-movement; **stop them in `exit()`** and null the reference.
- For anything custom (e.g. a Shepard tone), build Web Audio nodes on `audio.ctx` and connect to `audio.out`. Guard with `if (audio.on)`.
- Levels: a single pluck gain ≈ 0.15–0.3; sustained voices ≈ 0.05–0.15. Never harsh. Never constant beeping:
  rate-limit event sounds (e.g. ≥ 80 ms apart). The drone is set by the engine from `drone`.

## 5. Visual language

- **World**: the void (`palette.void`, violet-black). Light comes from the subject itself: luminous filaments,
  additive particles, emissive surfaces. Use `hdr(color, k)` (k 1.5–6) on things that should bloom; keep secondary
  elements below 1.0 so the bloom has hierarchy. No grey MeshStandardMaterial plastic, no default-lit look.
  If you use lit materials, use MeshPhysicalMaterial with iridescence/transmission or custom ShaderMaterials.
- **Strings** are always recognisably strings: thin, white-hot core with a coloured halo, smooth (≥ 256
  segments), never jaggy, often with motion trails or a faint afterglow.
- **Palette** (`fx.palette`): `planck` gold (accent; "the string itself"), `harmonic` cyan (low modes, open
  strings, light), `resonance` magenta (high modes, closed strings, gravity), `violet` (hidden dimensions),
  `ember` (tension, heat, infinities), `mint` (resolution, habitable), `ink` (neutral highlights).
  `modeColor(n)` gives the colour of harmonic n; use it whenever you show a mode number.
- **Depth**: every scene has layered depth: a starfield or dust (`makeStarfield`), fog or atmospheric
  falloff, and a slowly breathing camera (tiny sinusoidal drift) even when idle. Things move a little all the time.
- **Motion**: ease everything with `damp(rate, dt)`; no snapping. Respect `matchMedia('(prefers-reduced-motion: reduce)')`
  by reducing idle camera drift.
- **Composition**: one hero subject, generous negative space, cinematic framing. The first still frame of
  your movement (before any interaction) must already look like a film still.
- The post chain (bloom → ACES tone map → chromatic aberration, vignette, film grain) is applied for you.

## 6. Copy

Write for a curious adult with no physics background. Plain, direct sentences. Use real units and terms
(10⁻³⁵ m, spin 2, α′, Calabi–Yau). No em-dash asides, no "not X, but Y" constructions, no stock phrases
("worth noting", "honest caveat", "delve"), no exclamation marks. The `fine` panel is where precision lives:
say what the visual simplifies and what is established versus speculative. String theory has **no
experimental confirmation**; never imply otherwise.

## 7. Performance & robustness

- 60 fps on an M-series laptop at 1440×900, DPR 1.75, MSAA 4×. Budget per movement: ≤ ~300k triangles,
  ≤ ~150k particles, full-screen raymarch only inside a bounding mesh with ≤ 96 steps (halve on `quality === 'low'`).
- **No allocation in `update()`** or pointer-move hot paths: reuse scratch `Vector3`s, typed arrays, attributes
  (`attr.needsUpdate = true`). Never rebuild geometry per frame (rebuilding on a slider change is fine, but debounce).
- `exit()` must leave nothing running: voices stopped, timers cleared, OrbitControls disabled, no window listeners left
  (prefer the engine-routed `onPointer*`/`onKey` handlers over your own listeners).
- `enter()` after `exit()` must restore a fully working state (test it: visit, leave, return).
- No console errors or warnings. No `alert/confirm/prompt`. Works with sound off.
- Touch: pointer events cover touch; every essential interaction must be possible with one finger or a control.

## 8. Testing (mandatory)

The dev server runs at `http://127.0.0.1:8765/`. `tools/snap.sh` renders it in headless Chrome:

```bash
tools/snap.sh "m=2&skip&demo" shots/m2/desktop-demo.png 1440 900 7000    # query, out file, width, height, virtual ms
tools/snap.sh "m=2&skip" shots/m2/idle.png 1440 900 5000
tools/snap.sh "m=2&skip&demo" shots/m2/phone.png 390 844 7000
```

`snap.sh` runs in **real time** over the DevTools protocol: frames run at normal speed and the last argument is
how long to wait after page load before capturing. Widths under 760 emulate a real phone at exactly that size
(touch, coarse pointer, so `ctx.quality === 'low'` and `ctx.isTouch` are true), so 390×844 is a faithful phone shot.
`m=N` starts at movement N, `skip` skips the intro, `demo` calls your `demo()` 0.7 s after start. It prints
page console lines and errors; any `Uncaught`/`Error` line is a bug to fix. Then **look at the PNG with the
Read tool**. Iterate until it is beautiful. You may add your own debug query params (read via
`new URLSearchParams(location.search)`) to stage specific states; they must be harmless when absent.
Put screenshots under `shots/<your movement id>/`. Syntax-check with `node --check` is not enough because of
the bare `three` import; the screenshot run is the real test.

Headless Chrome renders WebGL2 with a real GPU here (expect 60 fps). It does not produce sound, so verify audio code by reading it carefully.
