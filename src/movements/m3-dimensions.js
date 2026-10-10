// III. Hidden Dimensions (Adagio misterioso)
//
// One continuous scene with three camera stations. Nothing hard-cuts:
//   1. "A curled dimension": an infinite line x a circle of radius R (the garden hose). A momentum
//      string and a winding string sit on it; their tones cross at the self-dual radius, and the
//      T-dual view re-describes everything on a circle of radius 1/R without changing a single energy.
//   2. "Six dimensions, folded": the hose collapses into a thread of ordinary space and, at one point
//      of that thread, Hanson's picture of the Fermat curve z1^n + z2^n = 1 dissolves into view, with a
//      closed string wrapped on one of its handles.
//   3. "At every point of space": pull back. The thread becomes one line of a lattice and the shape
//      becomes the copy at its centre point; every other lattice point carries the same shape.
//
// Rendering: glow strings are screen-space ribbons (white core + coloured halo) rebuilt from typed
// arrays each frame. The tube is an analytic cylinder whose radius is a uniform. The Calabi-Yau slice
// stores the 4D point and its two 4D tangents per vertex, so the projection angle alpha is animated on
// the GPU and normals stay exact. It is drawn in three passes: an opaque dark-glass body, a front-only
// additive glow (fresnel + patch grid + contours of the hidden coordinate), and a faint x-ray pass
// that shows the folds behind. The lattice copies are one InstancedMesh rotated in the vertex shader.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { palette, hdr, glowTexture, makeStarfield, GLSL, clamp, lerp, smoothstep, damp, rng } from '../fx.js';

const TAU = Math.PI * 2;
const HALF_PI = Math.PI / 2;
const RW = 0.42;              // world units per sqrt(alpha') at view scale 1 (the tube radius is R * RW)
const TUBE_HALF = 48;         // half length of the tube mesh; fog hides its ends
const CY_A = 1.25;            // Hanson's range y in [-a, a]
const CY_SIZE = 1.18;         // hero size after normalising every degree to the same extent
const INST_SCALE = 0.15;      // lattice copies relative to the hero
const LAT_STEP = 1.6;         // lattice spacing (world units)
const D4 = 293.66;            // tone for energy 1 (D4, so the self-dual tones meet on the tonic)
const CHORD = [12, 19, 22, 27, 31];        // D3 A3 C4 F4 A4: the Calabi-Yau hum (semitones above D2)
const CHORD_PAN = [-0.3, 0.28, -0.12, 0.38, -0.42];
const ARP = [24, 27, 31, 34, 36, 39];      // D4 F4 A4 C5 D5 F5: plucked loop arpeggio
const LN_EMIN = Math.log(0.2), LN_EMAX = Math.log(14);

const POST = {
  1: { bloomStrength: 0.92, bloomRadius: 0.62, bloomThreshold: 0.2, exposure: 1.0, aberration: 0.0018, grain: 0.035, vignette: 0.9 },
  2: { bloomStrength: 0.62, bloomRadius: 0.42, bloomThreshold: 0.32, exposure: 1.0, aberration: 0.002, grain: 0.035, vignette: 0.92 },
  3: { bloomStrength: 0.85, bloomRadius: 0.5, bloomThreshold: 0.28, exposure: 1.0, aberration: 0.0005, grain: 0.035, vignette: 0.88 },
};

const easeInOut = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
const fmt2 = (x) => (x >= 9.995 ? x.toFixed(1) : x.toFixed(2));
const MINUS = '−';
const signed = (v) => (v < 0 ? MINUS + Math.abs(v) : String(v));
const genus = (n) => ((n - 1) * (n - 2)) / 2;
const wrap01 = (x) => x - Math.round(x);   // signed distance on a loop of length 1
// The HUD sets labels in uppercase, which would turn α into A and χ into X. Symbols and variables keep their case.
const sym = (s) => `<span style="text-transform:none;letter-spacing:.02em">${s}</span>`;
const PENTA = [0, 3, 5, 7, 10];
/** Nearest D-minor-pentatonic pitch (in semitones above D2) to a frequency. */
function pentaSemis(f) {
  const s = 12 * Math.log2(Math.max(20, f) / 73.42), o = Math.floor(s / 12);
  let best = 0, bd = 1e9;
  for (let k = -1; k <= 1; k++) for (let j = 0; j < 5; j++) { const c = (o + k) * 12 + PENTA[j], d = Math.abs(c - s); if (d < bd) { bd = d; best = c; } }
  return best;
}

// (n, w) pairs listed in the stage-1 energy ladder
const LAD_N = new Int8Array([1, 0, 1, 2, 0, 2, 1, 2, 3, 0, 3, 1, 3, 2, 3]);
const LAD_W = new Int8Array([0, 1, 1, 0, 2, 1, 2, 2, 0, 3, 1, 3, 2, 3, 3]);

// ------------------------------------------------------------------------------------------
// GLSL
// ------------------------------------------------------------------------------------------
const GLSL_LINE = /* glsl */`
float aaLine(float v, float px) {
  float w = max(fwidth(v), 1e-5);
  float d = abs(fract(v - 0.5) - 0.5) / w;          // distance to the nearest integer, in pixels
  return (1.0 - smoothstep(px, px + 1.0, d)) * (1.0 - smoothstep(0.16, 0.42, w));   // fade when too dense
}`;

// Screen-space ribbon: every curve point is two vertices pushed apart along the screen normal.
const RIBBON_VS = /* glsl */`
attribute vec3 prev;
attribute vec3 next;
attribute float side;
attribute float glow;
uniform vec2 uRes;
uniform float uWidth;
uniform float uDimBack;
uniform float uBackLevel;
// side * w and w, divided per fragment: screen-linear interpolation across the ribbon (GLSL ES has no
// noperspective). Perspective-correct interpolation would bend the core inside every quad whose ends sit
// at different depths, and bloom turns that sawtooth into a row of beads along long lines.
varying float vSideW;
varying float vW;
varying float vGlow;
void main() {
  mat4 mvp = projectionMatrix * modelViewMatrix;
  vec4 c = mvp * vec4(position, 1.0);
  vec4 p = mvp * vec4(prev, 1.0);
  vec4 n = mvp * vec4(next, 1.0);
  vec2 hr = uRes * 0.5;
  vec2 cs = c.xy / c.w * hr;
  vec2 d1 = cs - p.xy / p.w * hr;
  vec2 d2 = n.xy / n.w * hr - cs;
  float l1 = length(d1), l2 = length(d2);
  vec2 dir = (l1 > 1e-3 ? d1 / l1 : vec2(0.0)) + (l2 > 1e-3 ? d2 / l2 : vec2(0.0));
  float ld = length(dir);
  dir = ld > 1e-4 ? dir / ld : vec2(1.0, 0.0);
  cs += vec2(-dir.y, dir.x) * side * uWidth;
  gl_Position = vec4(cs / hr * c.w, c.z, c.w);
  vSideW = side * c.w;
  vW = c.w;
  vGlow = glow;
  if (uDimBack > 0.5) {
    // strings on the tube: dim the stretches that run around its far side
    vec3 wp = (modelMatrix * vec4(position, 1.0)).xyz;
    vec3 radial = normalize(vec3(0.0, wp.y, wp.z) + vec3(0.0, 1e-6, 0.0));
    float f = dot(radial, normalize(cameraPosition - wp));
    vGlow *= mix(uBackLevel, 1.0, smoothstep(-0.25, 0.3, f));
  }
}`;
// Fade under the movement rail (left edge, desktop only), so lines never run through the HUD text.
const GLSL_RAIL = /* glsl */`
uniform vec2 uRes;
uniform float uRail;
float railFade() { return mix(1.0, mix(0.35, 1.0, smoothstep(0.08, 0.2, gl_FragCoord.x / uRes.x)), uRail); }`;
const RIBBON_FS = /* glsl */`
uniform vec3 uColor;
uniform float uIntensity;
uniform float uK;
uniform float uCore;
uniform float uWidth;
varying float vSideW;
varying float vW;
varying float vGlow;
${GLSL_RAIL}
void main() {
  float v = clamp(vSideW / vW, -1.0, 1.0);
  // the white core is sized in pixels (sigma at least 1 px), so a thin far line never collapses to a stepping sub-pixel thread
  float px = v * uWidth, sg = max(1.0, 0.121 * uWidth);
  float core = exp(-0.5 * px * px / (sg * sg));
  float halo = exp(-v * v * 4.5) * (1.0 - v * v);
  vec3 col = uColor * (halo * 0.5 + core * 0.55) + vec3(1.0, 0.95, 0.88) * core * uCore;
  gl_FragColor = vec4(col * vGlow * uIntensity * uK * railFade(), 1.0);
}`;

// The garden hose: a cylinder of radius uRadius along x, drawn as additive glass with a coordinate grid.
const TUBE_VS = /* glsl */`
attribute vec2 aST;
uniform float uRadius;
uniform float uHalf;
varying vec3 vWP;
varying vec3 vN;
varying float vTh;
varying float vX;
void main() {
  float th = aST.y;
  vec3 nr = vec3(0.0, cos(th), sin(th));
  vec3 p = vec3(aST.x * uHalf, 0.0, 0.0) + nr * uRadius;
  vec4 wp = modelMatrix * vec4(p, 1.0);
  vWP = wp.xyz; vN = nr; vTh = th; vX = p.x;
  gl_Position = projectionMatrix * viewMatrix * wp;
}`;
const TUBE_FS = /* glsl */`
uniform vec3 uColor;
uniform vec3 uLine;
uniform vec3 uGold;
uniform float uTime;
uniform float uIntensity;
uniform float uRing;
uniform float uGhost;
uniform float uFogFar;
uniform float uMerge;
varying vec3 vWP;
varying vec3 vN;
varying float vTh;
varying float vX;
${GLSL_LINE}
${GLSL_RAIL}
void main() {
  vec3 V = normalize(cameraPosition - vWP);
  float an = abs(dot(normalize(vN), V));
  float rim = pow(1.0 - an, 4.0) * smoothstep(0.0, 0.07, an);   // peak just inside the silhouette: no aliased edge
  float lon = aaLine(vTh / 6.2831853 * 16.0, 0.55);
  float ring = aaLine(vX / uRing, 0.55);                       // one ring per sqrt(alpha') of the long direction
  float dots = step(0.55, fract(vX / uRing * 3.0));
  lon = mix(lon, lon * dots, uGhost);                           // the dual tube is drawn dotted
  ring = mix(ring, aaLine(vX / uRing * 0.5, 0.45), uGhost);
  float face = gl_FrontFacing ? 1.0 : 0.5;
  float d = length(vWP - cameraPosition);
  float fog = (1.0 - smoothstep(uFogFar * 0.22, uFogFar, d)) * (1.0 - smoothstep(26.0, 46.0, abs(vX)));
  float flow = pow(0.5 + 0.5 * sin(vX * 0.7 - uTime * 1.1), 12.0);
  vec3 body = uColor * (0.012 + rim * 0.5 + flow * rim * 0.35);
  vec3 lines = uLine * (lon * 0.45 + ring * 0.32) * (0.3 + 0.7 * rim);
  vec3 col = mix(body + lines, body * 0.42 + lines * 0.6, uGhost);
  col = mix(col, col * 0.6 + uGold * (rim * 0.22 + (lon + ring) * 0.12), uMerge);   // R = 1: both tubes coincide
  gl_FragColor = vec4(col * face * fog * uIntensity * railFade(), 1.0);
}`;

// Calabi-Yau slice. z4 = (Re z1, Im z1, Re z2, Im z2); t4u, t4v = unit 4D tangents along the patch axes.
// Projection to 3D: (Re z1, Re z2, cos(a) Im z1 + sin(a) Im z2). The dropped coordinate
// (-sin(a) Im z1 + cos(a) Im z2) is the hidden one; its level sets are the moving contour lines.
const CY_VS = /* glsl */`
attribute vec4 z4;
attribute vec4 t4u;
attribute vec4 t4v;
attribute vec4 pinfo;
uniform float uAlpha;
uniform float uNorm;
varying vec3 vN;
varying vec3 vWP;
varying vec4 vInfo;
varying float vHid;
varying vec3 vObj;
#ifdef INST
attribute vec4 iData;
attribute float iScale;
uniform float uGrow;
uniform float uTime;
uniform vec4 uFocus;   // xyz: where the camera is looking into the lattice, w: how far we have zoomed into a point (0..1)
varying float vHi;
varying float vFog;
varying float vDc;
mat3 rotAxis(vec3 a, float t) {
  float c = cos(t), s = sin(t), C = 1.0 - c;
  return mat3(c + a.x * a.x * C, a.y * a.x * C + a.z * s, a.z * a.x * C - a.y * s,
              a.x * a.y * C - a.z * s, c + a.y * a.y * C, a.z * a.y * C + a.x * s,
              a.x * a.z * C + a.y * s, a.y * a.z * C - a.x * s, c + a.z * a.z * C);
}
#else
uniform float uAlphaOff;   // the hero borrows a lattice copy's hidden angle while it stands in for that copy
#endif
vec3 proj(vec4 q, float c, float s) { return vec3(q.x, q.z, c * q.y + s * q.w); }
void main() {
  float al = uAlpha;
#ifdef INST
  al += iData.w;
#else
  al += uAlphaOff;
#endif
  float c = cos(al), s = sin(al);
  vec3 p = proj(z4, c, s) * uNorm;
  vec3 nn = cross(proj(t4u, c, s), proj(t4v, c, s));
  float l = length(nn);
  nn = l > 1e-10 ? nn / l : vec3(0.0, 0.0, 1.0);
  vHid = -s * z4.y + c * z4.w;
  vInfo = pinfo;
  vObj = p;
#ifdef INST
  vec3 ax = normalize(vec3(sin(iData.z) * cos(iData.y * 6.2832), cos(iData.z), sin(iData.z) * sin(iData.y * 6.2832)));
  mat3 R = rotAxis(ax, iData.x * 6.2832 + uTime * (0.1 + 0.18 * iData.y));
  p = R * p;
  nn = R * nn;
  vec3 ctr = instanceMatrix[3].xyz;
  float r0 = length(ctr) * 0.12;
  float ap = smoothstep(r0, r0 + 0.35, uGrow);                // copies appear in a ripple from the centre
  p *= iScale * ap;
  vHi = smoothstep(1.05, 3.0, iScale);
  vec4 wp = modelMatrix * instanceMatrix * vec4(p, 1.0);
  nn = mat3(modelMatrix * instanceMatrix) * nn;
  float dc = length(wp.xyz - cameraPosition);
  vDc = dc;
  float fk = uFocus.w;
  // fog far away, and no blown-out copies at the lens (a wider margin while zoomed into one point, and none at all
  // within a few units, which is where the camera passes through the lattice on its way out)
  vFog = exp(-max(0.0, dc - 13.0) * 0.12) * smoothstep(mix(3.0, 6.5, fk), mix(9.0, 12.5, fk), dc) * smoothstep(4.5, 8.0, dc) * ap;
  // zoomed into one point, the rest of space steps back into a dim field
  vFog *= mix(1.0, 0.12 + 0.24 * smoothstep(3.0, 9.0, length(ctr - uFocus.xyz)), fk);
  // a slow swell of light travelling outward through the lattice
  vFog *= 0.84 + 0.16 * sin(length(ctr) * 0.9 - uTime * 0.8);
#else
  vec4 wp = modelMatrix * vec4(p, 1.0);
  nn = mat3(modelMatrix) * nn;
#endif
  vN = nn;
  vWP = wp.xyz;
  gl_Position = projectionMatrix * viewMatrix * wp;
}`;
const CY_FS = /* glsl */`
uniform float uTime;
uniform float uReveal;
uniform float uIntensity;
uniform float uBreath;
uniform vec3 uDeep;
uniform vec3 uViolet;
uniform vec3 uGold;
uniform vec3 uCyan;
uniform vec3 uMag;
uniform vec3 uKey;
uniform float uLines;
varying vec3 vN;
varying vec3 vWP;
varying vec4 vInfo;
varying float vHid;
varying vec3 vObj;
#ifdef INST
varying float vHi;
varying float vFog;
varying float vDc;
#endif
${GLSL.noise}
${GLSL.iridescence}
${GLSL_LINE}
void main() {
  vec3 N = normalize(vN);
  vec3 V = normalize(cameraPosition - vWP);
  float an = abs(dot(N, V));
  float fres = pow(1.0 - an, 4.0) * smoothstep(0.0, 0.05, an);
  float edgeY = abs(vInfo.w);
#if PASS < 3
  // dissolve: noise threshold, plus the cut edges (|y| -> a) fray first since the surface continues.
  // Fully revealed (the usual case) skips the noise entirely: a uniform branch, so it costs nothing.
  float burn = 0.0;
  if (uReveal < 0.995) {
    float h = snoise(vObj * 2.1 + vec3(3.1, 1.7, 0.0)) * 0.5 + 0.5;
    h = max(h * 0.86, smoothstep(0.78, 1.0, edgeY));
    float thr = uReveal * 1.14 - 0.02;
    if (h > thr) discard;
    burn = 1.0 - smoothstep(0.0, 0.06, thr - h);
  }
#endif
  float gu = aaLine(vInfo.z * 6.0, 0.5);
  float gv = aaLine(vInfo.w * 5.0, 0.5);
  float hid = aaLine(vHid * 5.0, 0.7);
  float ph = vInfo.x * 0.61 + vInfo.y * 0.37;
  vec3 ir = iridescence(fres * 0.8 + ph + vHid * 0.12 + uTime * 0.012);
  vec3 hidCol = mix(uCyan, uMag, 0.5 + 0.5 * sin(vHid * 2.6 + uTime * 0.25));
  float grid = max(gu, gv);
#if PASS == 0
  // dark glass: mostly the void, a thin-film sheen at grazing angles
  vec3 col = uDeep * (0.35 + 0.65 * an) + ir * (0.012 + fres * 0.1);
  col += uGold * burn * 2.0;
  gl_FragColor = vec4(col, 1.0);
#elif PASS == 1
  // glass with volume: a key light from the upper left and a cyan rim from behind, both fixed in the world,
  // so highlights glide over the folds as the shape turns. The hidden-coordinate contours are the only bright lines.
  vec3 Nf = dot(N, V) < 0.0 ? -N : N;
  vec3 L1 = normalize(vec3(-0.5, 0.7, 0.5));
  float spec = pow(max(dot(Nf, normalize(V + L1)), 0.0), 70.0) * 0.3;
  float sheen = pow(max(dot(Nf, normalize(V + L1)), 0.0), 9.0) * 0.035;
  float rimB = pow(1.0 - an, 3.0) * smoothstep(0.0, 0.7, dot(Nf, normalize(vec3(0.55, -0.45, -0.7))));
  vec3 col = ir * fres * 0.42 * uBreath
           + uViolet * grid * (0.02 + 0.06 * fres) * uLines
           + hidCol * hid * (0.15 + 0.25 * fres) * uLines
           + uKey * (spec + sheen) + uCyan * rimB * 0.3
           + ir * 0.07 * (1.0 - uLines)
           + uGold * burn * 3.0;
  gl_FragColor = vec4(col * uIntensity, 1.0);
#elif PASS == 2
  vec3 col = ir * fres * 0.25 + hidCol * hid * 0.25;
  gl_FragColor = vec4(col * uIntensity * 0.06, 1.0);
#else
  // lattice copies: solid iridescent silhouettes (thin lines would alias into sparkle at this size). The contours come
  // in only on a copy that grows under the pointer, so it reads as the same shape as the hero.
  vec3 col = ir * fres * 0.55 + ir * 0.07 + (hidCol * hid * 0.2 + uViolet * grid * 0.1) * vHi;
  col *= 1.0 + vHi * 0.45;
  // the nearest copies stack many additive layers on more pixels: hold them back so they keep their colour instead of blooming white
  col *= mix(0.55, 1.0, smoothstep(6.0, 12.0, vDc));
  // atmospheric depth: far copies sink into a dim violet
  col = mix(col, vec3(0.10, 0.08, 0.25) * (0.35 + 1.1 * fres), smoothstep(12.0, 26.0, vDc));
  gl_FragColor = vec4(col * vFog * uIntensity, 1.0);
#endif
}`;

// Faint lattice of ordinary 3D space.
const GRID_VS = /* glsl */`
varying vec3 vWP;
void main() { vec4 wp = modelMatrix * vec4(position, 1.0); vWP = wp.xyz; gl_Position = projectionMatrix * viewMatrix * wp; }`;
const GRID_FS = /* glsl */`
uniform vec3 uColor;
uniform float uOpacity;
uniform vec3 uExt;
uniform vec4 uFocus;   // xyz: focused lattice point, w: zoom-in amount
varying vec3 vWP;
void main() {
  vec3 q = abs(vWP) / uExt;
  float edge = 1.0 - smoothstep(0.62, 1.0, max(q.x, max(q.y, q.z)));
  float dc = length(vWP - cameraPosition);
  float fog = exp(-max(0.0, dc - 5.0) * 0.07) * smoothstep(1.5, 6.0, dc);   // lines never streak past the lens
  // looking into one point: the three lines through it stay lit (a fragment is on one of them when two of its
  // coordinates match the point's), the rest of space recedes
  vec3 d = abs(vWP - uFocus.xyz);
  float onAxes = step(1.5, step(d.x, 0.02) + step(d.y, 0.02) + step(d.z, 0.02));
  float keep = mix(0.1, 1.0 - 0.7 * smoothstep(3.0, 12.0, length(d)), onAxes);
  fog *= mix(1.0, keep, uFocus.w);
  gl_FragColor = vec4(uColor * uOpacity * edge * fog, 1.0);
}`;

// ------------------------------------------------------------------------------------------
// Hanson's construction (CPU): z1 = e^{2 pi i k1/n} cos(z)^{2/n}, z2 = e^{2 pi i k2/n} sin(z)^{2/n}
// ------------------------------------------------------------------------------------------
const _ha = new Float64Array(4), _hb = new Float64Array(4);
function hanson(n, k1, k2, x, y, out) {
  const cx = Math.cos(x), sx = Math.sin(x), ch = Math.cosh(y), sh = Math.sinh(y), p = 2 / n;
  const cr = cx * ch, ci = -sx * sh;   // cos(x + iy); cr >= 0 on the patch, so the principal branch never jumps
  const sr = sx * ch, si = cx * sh;    // sin(x + iy); sr >= 0 likewise
  const m1 = Math.pow(Math.hypot(cr, ci), p), a1 = p * Math.atan2(ci, cr) + (TAU * k1) / n;
  const m2 = Math.pow(Math.hypot(sr, si), p), a2 = p * Math.atan2(si, sr) + (TAU * k2) / n;
  out[0] = m1 * Math.cos(a1); out[1] = m1 * Math.sin(a1);
  out[2] = m2 * Math.cos(a2); out[3] = m2 * Math.sin(a2);
}
/** Unit 4D tangent by central differences (one-sided at the patch edges, where cos or sin may vanish). */
function tangent(n, k1, k2, x, y, dx, dy, out) {
  const x0 = clamp(x - dx, 0, HALF_PI), x1 = clamp(x + dx, 0, HALF_PI);
  const y0 = clamp(y - dy, -CY_A, CY_A), y1 = clamp(y + dy, -CY_A, CY_A);
  hanson(n, k1, k2, x0, y0, _ha); hanson(n, k1, k2, x1, y1, _hb);
  let l = 0;
  for (let k = 0; k < 4; k++) { out[k] = _hb[k] - _ha[k]; l += out[k] * out[k]; }
  l = Math.sqrt(l) || 1;
  for (let k = 0; k < 4; k++) out[k] /= l;
}
const cyNorm = (n) => CY_SIZE / Math.pow(Math.cosh(CY_A), 2 / n);

/** n^2 patches, each a (G+1)^2 grid over x in [0, pi/2], y in [-a, a], merged into one geometry. */
function buildCY(n, G) {
  const P = n * n, V1 = G + 1, NV = P * V1 * V1;
  const z4 = new Float32Array(NV * 4), tu = new Float32Array(NV * 4), tv = new Float32Array(NV * 4);
  const info = new Float32Array(NV * 4), pos = new Float32Array(NV * 3);
  const idx = new Uint32Array(P * G * G * 6);
  const t = new Float64Array(4);
  const dx = HALF_PI * 0.002, dy = CY_A * 0.002;
  let v = 0, q = 0;
  for (let k1 = 0; k1 < n; k1++) {
    for (let k2 = 0; k2 < n; k2++) {
      const base = v;
      for (let j = 0; j <= G; j++) {
        for (let i = 0; i <= G; i++) {
          const x = (HALF_PI * i) / G, y = -CY_A + (2 * CY_A * j) / G;
          hanson(n, k1, k2, x, y, t); z4.set(t, v * 4);
          pos[v * 3] = t[0]; pos[v * 3 + 1] = t[2]; pos[v * 3 + 2] = Math.SQRT1_2 * (t[1] + t[3]);
          tangent(n, k1, k2, x, y, dx, 0, t); tu.set(t, v * 4);
          tangent(n, k1, k2, x, y, 0, dy, t); tv.set(t, v * 4);
          info[v * 4] = k1 / n; info[v * 4 + 1] = k2 / n; info[v * 4 + 2] = i / G; info[v * 4 + 3] = (2 * j) / G - 1;
          v++;
        }
      }
      for (let j = 0; j < G; j++) {
        for (let i = 0; i < G; i++) {
          const a = base + j * V1 + i, b = a + 1, c = a + V1, d = c + 1;
          idx[q++] = a; idx[q++] = b; idx[q++] = d; idx[q++] = a; idx[q++] = d; idx[q++] = c;
        }
      }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('z4', new THREE.BufferAttribute(z4, 4));
  g.setAttribute('t4u', new THREE.BufferAttribute(tu, 4));
  g.setAttribute('t4v', new THREE.BufferAttribute(tv, 4));
  g.setAttribute('pinfo', new THREE.BufferAttribute(info, 4));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 3);
  return g;
}

// complex helpers for the loop (pairs [re, im]); results go into a shared scratch pair
const _c = new Float64Array(2);
function cpow(re, im, p) {   // principal power
  const r = Math.pow(Math.hypot(re, im), p), a = p * Math.atan2(im, re);
  _c[0] = r * Math.cos(a); _c[1] = r * Math.sin(a);
}
/** Exact tangent plane of z1^n + z2^n = 1 at (z1, z2): dz1 = -(z2/z1)^(n-1) dz2 (or the mirror), for dz = 1 and dz = i. */
function curveTangents(n, a, b, c, d, tu, tv, o) {
  // a + ib = z1, c + id = z2; use the larger coordinate in the denominator
  const useZ2 = Math.hypot(a, b) >= Math.hypot(c, d);
  const nr = useZ2 ? c : a, ni = useZ2 ? d : b, dr = useZ2 ? a : c, di = useZ2 ? b : d;
  const q = dr * dr + di * di;
  cpow((nr * dr + ni * di) / q, (ni * dr - nr * di) / q, n - 1);
  const kr = -_c[0], ki = -_c[1];   // d(other) = k * d(chosen)
  // dz_chosen = 1: d(other) = k; dz_chosen = i: d(other) = i k
  let u0, u1, u2, u3, v0, v1, v2, v3;
  if (useZ2) { u0 = kr; u1 = ki; u2 = 1; u3 = 0; v0 = -ki; v1 = kr; v2 = 0; v3 = 1; }
  else { u0 = 1; u1 = 0; u2 = kr; u3 = ki; v0 = 0; v1 = 1; v2 = -ki; v3 = kr; }
  const lu = Math.hypot(u0, u1, u2, u3), lv = Math.hypot(v0, v1, v2, v3);
  tu[o] = u0 / lu; tu[o + 1] = u1 / lu; tu[o + 2] = u2 / lu; tu[o + 3] = u3 / lu;
  tv[o] = v0 / lv; tv[o + 1] = v1 / lv; tv[o + 2] = v2 / lv; tv[o + 3] = v3 / lv;
}

/**
 * A closed string on a non-contractible cycle of the slice. On the real line y = 0 each patch (k1, k2) is an arc
 * from A_k1 = (w^k1, 0) to B_k2 = (0, w^k2). The chain (0,0) A0->B0, (1,0) B0->A1, (1,1) A1->B1, (0,1) B1->A0
 * closes. Seen as a branched cover of the z2-plane it runs from branch point 1 to branch point w on one sheet and
 * back on the next, the standard construction of a homology cycle on this curve.
 * Where two arcs meet, the chain has a genuine corner of angle 2 pi/n. Near A_k the coordinate z2 is a local chart
 * of the surface (and z1 near B_k), so each corner is replaced inside the disc |z2| < EPS by a smooth fillet in that
 * chart (a quadratic Bezier tangent to both incoming rays), lifted back with z1 = w^k (1 - z2^n)^(1/n). Inside a
 * disc this is a homotopy, so the homology class is unchanged. The whole curve is then resampled to equal 4D arc length.
 */
function buildLoop(n, M) {
  const arcs = [0, 0, 1, 1, 0, -1, 1, 1, 1, 0, 1, -1];   // (k1, k2, direction) x 4
  const EPS = 0.32, DN = 500, DF = 160, N = M * 4;
  const xe = Math.pow(EPS, n / 2);
  const x0 = Math.asin(xe), x1 = Math.acos(xe);   // |z2| = EPS at x0 (near A), |z1| = EPS at x1 (near B)
  const total = 4 * (DN + DF);
  const raw = new Float64Array(total * 4);
  const t4 = new Float64Array(4);
  let o = 0;
  const w = (k) => (TAU * k) / n;
  for (let a = 0; a < 4; a++) {
    const k1 = arcs[a * 3], k2 = arcs[a * 3 + 1], dir = arcs[a * 3 + 2];
    for (let i = 0; i < DN; i++) {
      const s = 0.5 - 0.5 * Math.cos((Math.PI * i) / DN);   // dense near the ends
      const x = dir > 0 ? x0 + (x1 - x0) * s : x1 - (x1 - x0) * s;
      hanson(n, k1, k2, x, 0, t4); raw.set(t4, o * 4); o++;
    }
    // fillet at the corner this arc runs into
    const na = arcs[((a + 1) % 4) * 3], nb = arcs[((a + 1) % 4) * 3 + 1];
    const atB = dir > 0;   // ending at B_k2 (z1 small): the next arc leaves with its own k1
    const k = atB ? k2 : k1;                         // the fixed root of unity at this corner
    const thIn = w(atB ? k1 : k2), thOut = w(atB ? na : nb);
    let dth = thOut - thIn;
    if (dth > Math.PI) dth -= TAU; else if (dth < -Math.PI) dth += TAU;
    const ci = Math.cos(thIn), si = Math.sin(thIn), co = Math.cos(thIn + dth), so = Math.sin(thIn + dth);
    for (let i = 0; i < DF; i++) {
      const s = i / DF, A = (1 - s) * (1 - s), B = s * s;
      const lr = EPS * (A * ci + B * co), li = EPS * (A * si + B * so);   // the local coordinate
      cpow(lr, li, n);
      cpow(1 - _c[0], -_c[1], 1 / n);
      const fr = _c[0] * Math.cos(w(k)) - _c[1] * Math.sin(w(k)), fi = _c[0] * Math.sin(w(k)) + _c[1] * Math.cos(w(k));
      if (atB) { raw[o * 4] = lr; raw[o * 4 + 1] = li; raw[o * 4 + 2] = fr; raw[o * 4 + 3] = fi; }
      else { raw[o * 4] = fr; raw[o * 4 + 1] = fi; raw[o * 4 + 2] = lr; raw[o * 4 + 3] = li; }
      o++;
    }
  }
  // resample the closed polyline at equal 4D arc length
  const cum = new Float64Array(total + 1);
  for (let i = 1; i <= total; i++) {
    const p = ((i - 1) % total) * 4, q = (i % total) * 4;
    cum[i] = cum[i - 1] + Math.hypot(raw[q] - raw[p], raw[q + 1] - raw[p + 1], raw[q + 2] - raw[p + 2], raw[q + 3] - raw[p + 3]);
  }
  const L = cum[total];
  const p4 = new Float32Array(N * 4), u4 = new Float32Array(N * 4), v4 = new Float32Array(N * 4);
  let j = 0;
  for (let m = 0; m < N; m++) {
    const target = (L * m) / N;
    while (j < total - 1 && cum[j + 1] < target) j++;
    const f = (target - cum[j]) / Math.max(1e-12, cum[j + 1] - cum[j]);
    const p = j * 4, q = ((j + 1) % total) * 4;
    for (let c = 0; c < 4; c++) p4[m * 4 + c] = raw[p + c] + (raw[q + c] - raw[p + c]) * f;
    curveTangents(n, p4[m * 4], p4[m * 4 + 1], p4[m * 4 + 2], p4[m * 4 + 3], u4, v4, m * 4);
  }
  return { p4, u4, v4, N };
}

function makeTubeGeo(segT) {
  const segS = 2, nv = (segS + 1) * (segT + 1);
  const st = new Float32Array(nv * 2), pos = new Float32Array(nv * 3);
  let v = 0;
  for (let j = 0; j <= segS; j++) for (let i = 0; i <= segT; i++) { st[v * 2] = -1 + (2 * j) / segS; st[v * 2 + 1] = (TAU * i) / segT; v++; }
  const idx = new Uint16Array(segS * segT * 6);
  let q = 0;
  for (let j = 0; j < segS; j++) {
    for (let i = 0; i < segT; i++) {
      const a = j * (segT + 1) + i, b = a + 1, c = a + segT + 1, d = c + 1;
      idx[q++] = a; idx[q++] = b; idx[q++] = d; idx[q++] = a; idx[q++] = d; idx[q++] = c;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('aST', new THREE.BufferAttribute(st, 2));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  return g;
}

/** Very dim violet nebula painted once on a canvas and wrapped on a far sphere. */
function makeBackdrop() {
  const W = 1024, H = 512, c = document.createElement('canvas');
  c.width = W; c.height = H;
  const g = c.getContext('2d');
  g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
  const R = rng(1303);
  const cols = ['139,108,255', '58,34,140', '255,79,191', '111,243,255', '36,22,92'];
  for (let i = 0; i < 90; i++) {
    const x = R() * W, y = H * 0.5 + (R() - 0.5) * H * (R() < 0.7 ? 0.5 : 0.9);
    const r = 30 + Math.pow(R(), 1.6) * 260;
    const pick = R();
    const ci = pick < 0.45 ? 0 : pick < 0.7 ? 1 : pick < 0.86 ? 4 : pick < 0.94 ? 2 : 3;
    const a = (ci === 2 || ci === 3 ? 0.035 : 0.085) * (0.4 + R() * 0.6);
    for (let k = -1; k <= 1; k++) {
      const xx = x + k * W;
      const grd = g.createRadialGradient(xx, y, 0, xx, y, r);
      grd.addColorStop(0, `rgba(${cols[ci]},${a})`);
      grd.addColorStop(1, `rgba(${cols[ci]},0)`);
      g.fillStyle = grd; g.fillRect(xx - r, y - r, 2 * r, 2 * r);
    }
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const m = new THREE.Mesh(new THREE.SphereGeometry(420, 48, 24), new THREE.MeshBasicMaterial({ map: tex, side: THREE.BackSide, depthWrite: false, color: 0xd8d0ff }));
  m.renderOrder = -10;
  return m;
}

// ------------------------------------------------------------------------------------------
export default {
  id: 'dimensions',
  numeral: 'III',
  title: 'Hidden Dimensions',
  tempo: 'Adagio misterioso',
  caption: 'Superstring theory only works in <em>ten dimensions</em>. We see four, so six must be curled up smaller than we can detect. Their <em>shape</em> decides which particles and forces exist.',
  fine: [
    '<p><em>Why ten.</em> A quantum string is only self-consistent in one particular number of dimensions. The calculation that cancels the anomaly on its worldsheet works in 26 dimensions for the bosonic string and in 10 for the superstring. Nobody picked the number.</p>',
    '<p><em>A curled dimension.</em> In the 1920s Kaluza and Klein showed that an extra dimension rolled into a small circle of radius R would be invisible, and that motion around it comes in steps with energy n/R. The sliding loop is a classical picture. A state of definite momentum is spread evenly round the circle, like the ring with n crests.</p>',
    '<p><em>T-duality.</em> A closed string can also wrap the circle w times, which costs wR/α′. Replace R by α′/R, swap n with w, and the whole list of energies is unchanged. Point particles have no such symmetry. A circle smaller than the string length √α′ behaves exactly like a larger one. (Here α′ = 1. The numbers are the circle\'s contribution to each string\'s mass, and vibrations add more.)</p>',
    '<p><em>The folded shape.</em> In the simplest constructions, keeping some supersymmetry requires the six extra dimensions to form a Calabi–Yau manifold. The surface shown is Hanson\'s rendering of z₁ⁿ + z₂ⁿ = 1: a complex curve, two real dimensions living in four, projected into three. The bright contours are levels of the fourth, hidden coordinate. For n = 5 it is the usual stand-in for a slice of the quintic threefold. The gold string wraps a handle of this slice. In the full quintic every loop can shrink, because the space is simply connected, so the picture shows the idea on the slice only. In the simplest heterotic models the number of particle families is |χ|/2: the quintic has χ = −200 and gives 100, while three families need |χ| = 6.</p>',
    '<p><em>Status.</em> No experiment has found extra dimensions or confirmed string theory. Collider data only require dimensions that ordinary particles move in to be smaller than about 10⁻¹⁹ m. Dimensions that only gravity feels could be as large as about 0.03 mm before tabletop tests of gravity would notice. Hundreds of millions of candidate Calabi–Yau shapes are catalogued, and different choices give different particles and forces. Movement VI returns to that problem.</p>',
  ].join(''),
  drone: { notes: [0, 10, 15, 22], opts: { gain: 0.085, cutoff: 760, wobble: 0.35 } },
  post: POST[1],

  create(ctx) {
    const { audio, ui } = ctx;
    const LOW = ctx.quality === 'low';
    const REDUCED = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
    const Q = new URLSearchParams(location.search);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, ctx.aspect, 0.05, 900);
    const controls = new OrbitControls(camera, ctx.canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;
    controls.enablePan = false;
    controls.rotateSpeed = 0.55;
    controls.zoomSpeed = 0.7;
    controls.enabled = false;
    // its listeners (one of them on the document) are attached only while the movement is active: enter() connects
    { const ta = ctx.canvas.style.touchAction; controls.disconnect(); ctx.canvas.style.touchAction = ta; }

    const SH = { uTime: { value: 0 }, uRes: { value: new THREE.Vector2(ctx.width * ctx.pixelRatio, ctx.height * ctx.pixelRatio) }, uRail: { value: ctx.width > 760 ? 1 : 0 } };
    const ribbons = [];

    // ------------------------------------------------------------ glow ribbon factory
    function makeRibbon(maxPts, o) {
      const closed = o.closed !== false;
      const nPts = maxPts + (closed ? 1 : 0), nV = nPts * 2;
      const pos = new Float32Array(nV * 3), prv = new Float32Array(nV * 3), nxt = new Float32Array(nV * 3);
      const side = new Float32Array(nV), glow = new Float32Array(nV).fill(1);
      for (let i = 0; i < nPts; i++) { side[2 * i] = -1; side[2 * i + 1] = 1; }
      const idx = new Uint16Array((nPts - 1) * 6);
      for (let i = 0; i < nPts - 1; i++) {
        const a = 2 * i, q = i * 6;
        idx[q] = a; idx[q + 1] = a + 1; idx[q + 2] = a + 2; idx[q + 3] = a + 1; idx[q + 4] = a + 3; idx[q + 5] = a + 2;
      }
      const geo = new THREE.BufferGeometry();
      const aPos = new THREE.BufferAttribute(pos, 3).setUsage(THREE.DynamicDrawUsage);
      const aPrv = new THREE.BufferAttribute(prv, 3).setUsage(THREE.DynamicDrawUsage);
      const aNxt = new THREE.BufferAttribute(nxt, 3).setUsage(THREE.DynamicDrawUsage);
      const aGlow = new THREE.BufferAttribute(glow, 1).setUsage(THREE.DynamicDrawUsage);
      geo.setAttribute('position', aPos); geo.setAttribute('prev', aPrv); geo.setAttribute('next', aNxt);
      geo.setAttribute('side', new THREE.BufferAttribute(side, 1)); geo.setAttribute('glow', aGlow);
      geo.setIndex(new THREE.BufferAttribute(idx, 1));
      const shared = {
        uColor: { value: new THREE.Color(o.color) }, uIntensity: { value: 0 }, uCore: { value: o.core ?? 1 },
        uWidth: { value: (o.width ?? 5) * ctx.pixelRatio }, uRes: SH.uRes, uRail: SH.uRail,
        uDimBack: { value: o.dimBack ? 1 : 0 }, uBackLevel: { value: o.backLevel ?? 0.28 },
      };
      const mk = (depthTest, k) => new THREE.ShaderMaterial({
        uniforms: { ...shared, uK: { value: k } }, vertexShader: RIBBON_VS, fragmentShader: RIBBON_FS,
        transparent: true, depthWrite: false, depthTest, blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide,   // screen-space quads wind clockwise; never cull them
      });
      const mesh = new THREE.Mesh(geo, mk(o.depthTest !== false, 1));
      mesh.frustumCulled = false; mesh.renderOrder = o.order ?? 8;
      let through = null;
      if (o.through) {   // the same curve seen through an opaque surface, faint
        through = new THREE.Mesh(geo, mk(false, o.through));
        through.frustumCulled = false; through.renderOrder = (o.order ?? 8) - 1;
      }
      const r = {
        mesh, through, shared, baseWidth: o.width ?? 5, n: maxPts,
        setIntensity(v) {
          shared.uIntensity.value = v;
          mesh.visible = v > 1e-3;
          if (through) through.visible = mesh.visible;
        },
        set(src, glowSrc) {
          const n = maxPts, N = closed ? n + 1 : n;
          for (let k = 0; k < N; k++) {
            const i = closed && k === n ? 0 : k;
            const ip = i === 0 ? (closed ? n - 1 : 0) : i - 1;
            const inx = i === n - 1 ? (closed ? 0 : n - 1) : i + 1;
            const o6 = k * 6, i3 = i * 3, p3 = ip * 3, n3 = inx * 3;
            pos[o6] = pos[o6 + 3] = src[i3]; pos[o6 + 1] = pos[o6 + 4] = src[i3 + 1]; pos[o6 + 2] = pos[o6 + 5] = src[i3 + 2];
            prv[o6] = prv[o6 + 3] = src[p3]; prv[o6 + 1] = prv[o6 + 4] = src[p3 + 1]; prv[o6 + 2] = prv[o6 + 5] = src[p3 + 2];
            nxt[o6] = nxt[o6 + 3] = src[n3]; nxt[o6 + 1] = nxt[o6 + 4] = src[n3 + 1]; nxt[o6 + 2] = nxt[o6 + 5] = src[n3 + 2];
            if (glowSrc) glow[2 * k] = glow[2 * k + 1] = glowSrc[i];
          }
          aPos.needsUpdate = true; aPrv.needsUpdate = true; aNxt.needsUpdate = true;
          if (glowSrc) aGlow.needsUpdate = true;
        },
      };
      ribbons.push(r);
      return r;
    }

    // ------------------------------------------------------------ world: nebula, stars, dust
    scene.add(makeBackdrop());
    const stars = makeStarfield({ count: LOW ? 1300 : 2600, radius: 220, inner: 90, size: 1.5, intensity: 1.05 });
    const dust = makeStarfield({ count: LOW ? 220 : 460, radius: 17, inner: 2.5, size: 1.15, intensity: 0.6, tintAmount: 0.6 });
    scene.add(stars, dust);

    // ------------------------------------------------------------ stage 1: the hose, its dual, the strings
    const tubeGeo = makeTubeGeo(LOW ? 96 : 160);
    const tubeMat = () => new THREE.ShaderMaterial({
      uniforms: {
        uTime: SH.uTime, uRadius: { value: 1 }, uHalf: { value: TUBE_HALF }, uRes: SH.uRes, uRail: SH.uRail,
        uColor: { value: hdr(palette.violet, 1.15) }, uLine: { value: hdr(palette.ink, 0.4) }, uGold: { value: hdr(palette.planck, 1.2) },
        uIntensity: { value: 1 }, uRing: { value: RW }, uGhost: { value: 0 }, uFogFar: { value: 34 }, uMerge: { value: 0 },
      },
      vertexShader: TUBE_VS, fragmentShader: TUBE_FS,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    });
    const tubeA = new THREE.Mesh(tubeGeo, tubeMat());   // radius R
    const tubeB = new THREE.Mesh(tubeGeo, tubeMat());   // radius 1/R (the T-dual description)
    tubeA.frustumCulled = tubeB.frustumCulled = false;
    tubeA.renderOrder = tubeB.renderOrder = 2;
    scene.add(tubeA, tubeB);

    const NS = LOW ? 200 : 320, NRING = 160, NTAIL = 72, NAXIS = 96;
    // the momentum loop is drawn a little wider than the coil; the two strings trade widths in the dual view
    const cyanStr = makeRibbon(NS, { color: hdr(palette.harmonic, 2.0), width: 7, core: 1.25, dimBack: true, order: 9 });
    const magStr = makeRibbon(NS, { color: hdr(palette.resonance, 2.0), width: 5.5, core: 1.25, dimBack: true, order: 9 });
    const cyanRing = makeRibbon(NRING, { color: hdr(palette.harmonic, 1.25), width: 3.6, core: 0.6, dimBack: true, order: 7 });
    const magRing = makeRibbon(NRING, { color: hdr(palette.resonance, 1.25), width: 3.6, core: 0.6, dimBack: true, order: 7 });
    const cyanTail = makeRibbon(NTAIL, { closed: false, color: hdr(palette.harmonic, 1.1), width: 5, core: 0.45, dimBack: true, order: 8 });
    const magTail = makeRibbon(NTAIL, { closed: false, color: hdr(palette.resonance, 1.1), width: 5, core: 0.45, dimBack: true, order: 8 });
    const axis = makeRibbon(NAXIS, { closed: false, color: hdr(palette.violet.clone().lerp(palette.ink, 0.35), 1.3), width: 2.4, core: 0.8, order: 6 });
    for (const r of [cyanStr, magStr, cyanRing, magRing, cyanTail, magTail, axis]) scene.add(r.mesh);

    const bufA = new Float32Array(NS * 3), bufB = new Float32Array(NS * 3), bufOut = new Float32Array(NS * 3);
    const glA = new Float32Array(NS), glB = new Float32Array(NS), glOut = new Float32Array(NS);
    const ringBuf = new Float32Array(NRING * 3), ringGlow = new Float32Array(NRING);
    const tailBuf = new Float32Array(NTAIL * 3), tailGlow = new Float32Array(NTAIL);
    {
      const ap = new Float32Array(NAXIS * 3), ag = new Float32Array(NAXIS);
      for (let i = 0; i < NAXIS; i++) {
        const x = -46 + (92 * i) / (NAXIS - 1);
        ap[i * 3] = x;
        ag[i] = 1 - smoothstep(14, 46, Math.abs(x));
      }
      axis.set(ap, ag);
    }

    // A closed loop lying on the cylinder of radius r, centred at (xc, thc): the momentum string. Its own size is
    // set by the string scale, kept between about half and three quarters of the circle so it always reads.
    function loopShape(out, gl, N, r, zoom, xc, thc, t, seed) {
      const rho = clamp(0.6 * RW * zoom, 0.45 * r, 0.72 * r);
      const rs = Math.max(r, 1e-4);
      for (let i = 0; i < N; i++) {
        const s = (TAU * i) / N;
        const wob = 1 + 0.08 * Math.sin(3 * s + 2.3 * t + seed) + 0.04 * Math.sin(5 * s - 3.1 * t + seed);
        const lx = rho * Math.cos(s) * wob, la = (rho * Math.sin(s) * wob) / rs;
        const rad = r * (1.05 + 0.03 * Math.sin(4 * s + 1.7 * t));
        out[i * 3] = xc + lx; out[i * 3 + 1] = rad * Math.cos(thc + la); out[i * 3 + 2] = rad * Math.sin(thc + la);
        gl[i] = 1;
      }
    }
    // A closed string winding w times around the circle. w = 1 is a clean ring round the tube. For w >= 2, g(u) packs
    // the turns into the outbound pass and the return runs almost straight along the tube at a slightly larger radius
    // (a wire wound on a hose); the return leg is drawn dimmer so the turns read first.
    function coilShape(out, gl, N, r, zoom, xc, w, phi, t) {
      if (w === 1) {
        for (let i = 0; i < N; i++) {
          const u = i / N, th = phi + TAU * u;
          out[i * 3] = xc + 0.03 * r * Math.sin(3 * TAU * u - 2 * t);
          out[i * 3 + 1] = r * 1.05 * Math.cos(th); out[i * 3 + 2] = r * 1.05 * Math.sin(th);
          gl[i] = 1;
        }
        return;
      }
      const D = (0.24 + 0.27 * w) * Math.max(0.55 * r, 0.32 * RW * zoom);
      for (let i = 0; i < N; i++) {
        const u = i / N, cu = Math.cos(TAU * u), su = Math.sin(TAU * u);
        const g = u + (0.9 * (1 - cu)) / TAU;
        const th = phi + TAU * w * g + 0.018 * Math.sin(TAU * u * 5 - 2.4 * t);
        const rad = r * (1.05 + 0.05 * (1 - su) + 0.012 * Math.sin(TAU * 6 * u - 3 * t));
        out[i * 3] = xc - D * cu; out[i * 3 + 1] = rad * Math.cos(th); out[i * 3 + 2] = rad * Math.sin(th);
        gl[i] = 0.4 + 0.6 * smoothstep(-0.55, 0.25, su);
      }
    }
    // The momentum eigenstate's wave: n crests around the circle, co-moving with the loop, drawn as white-hot beads.
    function ringShape(r, xc, k, thc) {
      for (let i = 0; i < NRING; i++) {
        const th = (TAU * i) / NRING, c = Math.cos(k * (th - thc));
        const rad = r * (1.1 + 0.2 * c);
        ringBuf[i * 3] = xc; ringBuf[i * 3 + 1] = rad * Math.cos(th); ringBuf[i * 3 + 2] = rad * Math.sin(th);
        const b = 0.5 + 0.5 * c, b2 = b * b;
        ringGlow[i] = 0.08 + 0.92 * b2 * b2 * b2;
      }
    }
    // Comet tail behind the orbiting loop.
    function tailShape(r, xc, thc, span) {
      for (let i = 0; i < NTAIL; i++) {
        const f = i / (NTAIL - 1), th = thc - span * (1 - f);
        tailBuf[i * 3] = xc; tailBuf[i * 3 + 1] = r * 1.05 * Math.cos(th); tailBuf[i * 3 + 2] = r * 1.05 * Math.sin(th);
        tailGlow[i] = f * f;
      }
    }

    // ------------------------------------------------------------ stage 2: the Calabi-Yau slice
    const CYU = {
      uTime: SH.uTime, uAlpha: { value: Math.PI / 4 }, uBreath: { value: 1 }, uGrow: { value: 0 },
      uDeep: { value: palette.deep.clone().multiplyScalar(0.55) }, uViolet: { value: hdr(palette.violet, 1.0) },
      uGold: { value: hdr(palette.planck, 1.0) }, uCyan: { value: hdr(palette.harmonic, 1.0) }, uMag: { value: hdr(palette.resonance, 1.0) },
      uKey: { value: palette.ink.clone().lerp(palette.violet, 0.3) }, uLines: { value: 1 }, uAlphaOff: { value: 0 },
    };
    const cyMat = (pass, su) => new THREE.ShaderMaterial({
      defines: { PASS: pass }, uniforms: { ...CYU, ...su, uIntensity: { value: 1 } },
      vertexShader: CY_VS, fragmentShader: CY_FS, side: THREE.DoubleSide,
      transparent: pass !== 0, depthWrite: pass === 0, depthTest: pass !== 2,
      blending: pass === 0 ? THREE.NormalBlending : THREE.AdditiveBlending,
      polygonOffset: pass === 0, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
    });

    // patch grid sizes: the hero (full size), a mid level for the hero while it stands in a lattice point, and the copies
    const hiG = { 3: 48, 4: 40, 5: 36, 6: 28 }, loG = { 3: 6, 4: 5, 5: 4, 6: 4 };
    const cyHi = {}, cyMid = {}, cyLo = {}, loops = {};
    const getHi = (n) => cyHi[n] || (cyHi[n] = buildCY(n, LOW ? Math.round(hiG[n] * 0.6) : hiG[n]));
    const getMid = (n) => cyMid[n] || (cyMid[n] = buildCY(n, 12));
    const getLo = (n) => cyLo[n] || (cyLo[n] = buildCY(n, loG[n]));
    const getLoop = (n) => loops[n] || (loops[n] = buildLoop(n, LOW ? 80 : 120));

    // a faint violet aura behind the shape gives the void some air
    const aura = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture(), color: palette.violet.clone().multiplyScalar(0.035), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
    aura.scale.setScalar(6.5);
    aura.renderOrder = -5;
    scene.add(aura);
    const hero = new THREE.Group();
    const heroRot = new THREE.Euler(0.9, 0, 0.6);
    if (Q.has('hrot')) { const r = Q.get('hrot').split(',').map(Number); heroRot.set(r[0] || 0, r[1] || 0, r[2] || 0); }
    hero.rotation.copy(heroRot);
    const SPIN = Q.has('hrot') ? 0 : 0.045;
    scene.add(hero);
    function makeSlot() {
      const su = { uReveal: { value: 0 }, uNorm: { value: 1 } };
      const g0 = getHi(5);
      const body = new THREE.Mesh(g0, cyMat(0, su)), glow = new THREE.Mesh(g0, cyMat(1, su)), xray = new THREE.Mesh(g0, cyMat(2, su));
      body.renderOrder = 0; glow.renderOrder = 3; xray.renderOrder = 4;
      for (const m of [body, glow, xray]) { m.frustumCulled = false; hero.add(m); }
      return {
        su, rev: 0, target: 0, n: 0, lod: 0, xray,
        setDegree(n) { this.n = n; su.uNorm.value = cyNorm(n); this.lod = -1; this.setLod(0); },
        // lod 1: the mid mesh, while the hero is the size of a lattice copy (thousands of triangles instead of 130k)
        setLod(l) {
          if (l === this.lod) return;
          this.lod = l;
          const g = l ? getMid(this.n) : getHi(this.n);
          body.geometry = glow.geometry = xray.geometry = g;
        },
        show(v) { body.visible = glow.visible = v; if (!v) xray.visible = false; },
      };
    }
    let slotCur = makeSlot(), slotOld = makeSlot();
    slotOld.show(false);

    const NLOOP = (LOW ? 80 : 120) * 4;
    const loopRib = makeRibbon(NLOOP, { color: hdr(palette.planck, 1.8), width: 6, core: 1.05, through: 0.12, order: 9 });
    hero.add(loopRib.mesh, loopRib.through);
    const loopOut = new Float32Array(NLOOP * 3), loopGlow = new Float32Array(NLOOP);

    // ------------------------------------------------------------ stage 3: lattice of space
    const NX = 7, NY = LOW ? 3 : 5, NZ = 5, LN = NX * NY * NZ;   // 175 points on high (140k triangles of copies at n = 5)
    const lat = new Float32Array(LN * 3);
    const latI = new Int8Array(LN * 3);
    let centerIdx = 0;
    {
      let i = 0;
      for (let iz = 0; iz < NZ; iz++) for (let iy = 0; iy < NY; iy++) for (let ix = 0; ix < NX; ix++) {
        const a = ix - (NX - 1) / 2, b = iy - (NY - 1) / 2, c = iz - (NZ - 1) / 2;
        lat[i * 3] = a * LAT_STEP; lat[i * 3 + 1] = b * LAT_STEP; lat[i * 3 + 2] = c * LAT_STEP;
        latI[i * 3] = a; latI[i * 3 + 1] = b; latI[i * 3 + 2] = c;
        if (a === 0 && b === 0 && c === 0) centerIdx = i;
        i++;
      }
    }
    const instGeo = new THREE.BufferGeometry();
    const iData = new Float32Array(LN * 4);
    {
      const R = rng(77);
      for (let i = 0; i < LN; i++) { iData[i * 4] = R(); iData[i * 4 + 1] = R(); iData[i * 4 + 2] = R() * Math.PI; iData[i * 4 + 3] = (R() - 0.5) * 1.6; }
    }
    instGeo.setAttribute('iData', new THREE.InstancedBufferAttribute(iData, 4));
    const iScale = new Float32Array(LN).fill(1);
    const iBase = new Float32Array(LN).fill(1);          // damped scale before the sympathetic swell
    const iRing = new Float32Array(LN).fill(-100);       // when each copy's sympathetic swell starts
    const iScaleAttr = new THREE.InstancedBufferAttribute(iScale, 1).setUsage(THREE.DynamicDrawUsage);
    instGeo.setAttribute('iScale', iScaleAttr);
    instGeo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 30);
    function setInstDegree(n) {
      const g = getLo(n);
      instGeo.setAttribute('position', g.getAttribute('position'));
      instGeo.setAttribute('z4', g.getAttribute('z4'));
      instGeo.setAttribute('t4u', g.getAttribute('t4u'));
      instGeo.setAttribute('t4v', g.getAttribute('t4v'));
      instGeo.setAttribute('pinfo', g.getAttribute('pinfo'));
      instGeo.setIndex(g.index);
      instNorm.value = cyNorm(n);
    }
    const instNorm = { value: 1 };
    const instMat = new THREE.ShaderMaterial({
      defines: { PASS: 3, INST: 1 }, uniforms: { ...CYU, uReveal: { value: 1 }, uNorm: instNorm, uIntensity: { value: 1 }, uFocus: { value: new THREE.Vector4() } },
      vertexShader: CY_VS, fragmentShader: CY_FS, side: THREE.DoubleSide,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    const inst = new THREE.InstancedMesh(instGeo, instMat, LN);
    inst.frustumCulled = false;
    inst.renderOrder = 3;
    {
      const m = new THREE.Matrix4();
      for (let i = 0; i < LN; i++) { m.makeScale(INST_SCALE, INST_SCALE, INST_SCALE).setPosition(lat[i * 3], lat[i * 3 + 1], lat[i * 3 + 2]); inst.setMatrixAt(i, m); }
      inst.instanceMatrix.needsUpdate = true;
    }
    scene.add(inst);

    const ext = new THREE.Vector3(((NX - 1) / 2 + 0.6) * LAT_STEP, ((NY - 1) / 2 + 0.6) * LAT_STEP, ((NZ - 1) / 2 + 0.6) * LAT_STEP);
    const gridMat = new THREE.ShaderMaterial({
      uniforms: { uColor: { value: hdr(palette.violet.clone().lerp(palette.ink, 0.4), 0.36) }, uOpacity: { value: 0 }, uExt: { value: ext }, uFocus: { value: new THREE.Vector4() } },
      vertexShader: GRID_VS, fragmentShader: GRID_FS, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    const gridGeo = new THREE.BufferGeometry();
    {
      const seg = [];
      for (let b = 0; b < NY; b++) for (let c = 0; c < NZ; c++) { const y = (b - (NY - 1) / 2) * LAT_STEP, z = (c - (NZ - 1) / 2) * LAT_STEP; seg.push(-ext.x, y, z, ext.x, y, z); }
      for (let a = 0; a < NX; a++) for (let c = 0; c < NZ; c++) { const x = (a - (NX - 1) / 2) * LAT_STEP, z = (c - (NZ - 1) / 2) * LAT_STEP; seg.push(x, -ext.y, z, x, ext.y, z); }
      for (let a = 0; a < NX; a++) for (let b = 0; b < NY; b++) { const x = (a - (NX - 1) / 2) * LAT_STEP, y = (b - (NY - 1) / 2) * LAT_STEP; seg.push(x, y, -ext.z, x, y, ext.z); }
      gridGeo.setAttribute('position', new THREE.Float32BufferAttribute(seg, 3));
    }
    const grid = new THREE.LineSegments(gridGeo, gridMat);
    grid.frustumCulled = false;
    grid.renderOrder = 1;
    scene.add(grid);

    // ------------------------------------------------------------ floating labels (DOM, under the HUD)
    const labelLayer = document.createElement('div');
    labelLayer.style.cssText = 'position:absolute;inset:0;pointer-events:none;overflow:hidden;display:none;';
    if (ctx.canvas.parentNode) ctx.canvas.parentNode.insertBefore(labelLayer, ctx.canvas.nextSibling);
    function makeLabel(color) {
      const el = document.createElement('div');
      el.style.cssText = 'position:absolute;left:0;top:0;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10px;letter-spacing:.14em;'
        + `text-transform:uppercase;white-space:nowrap;color:${color};opacity:0;transition:opacity .5s;text-shadow:0 0 10px rgba(5,4,10,.95);will-change:transform;`;
      labelLayer.appendChild(el);
      return { el, x: -1e4, y: -1e4, op: -1, html: '' };
    }
    const labels = { cyan: makeLabel('#6ff3ff'), mag: makeLabel('#ff4fbf'), loop: makeLabel('#ffc46b'), pt: makeLabel('#ece7f6') };
    function placeLabel(L, wx, wy, wz, dy, op, below = false) {
      if (op > 0.01) {
        V.set(wx, wy, wz).project(camera);
        if (V.z > 1 || V.z < -1) op = 0;
        const x = Math.round(((V.x + 1) / 2) * ctx.width), y = Math.round(((1 - V.y) / 2) * ctx.height + dy);
        if (x !== L.x || y !== L.y) { L.x = x; L.y = y; L.el.style.transform = `translate3d(${x}px,${y}px,0) translate(-50%,${below ? '0' : '-100%'})`; }
      }
      const o = Math.round(op * 20) / 20;
      if (o !== L.op) { L.op = o; L.el.style.opacity = String(o); }
    }
    function labelHTML(L, html) { if (L.html !== html) { L.html = html; L.el.innerHTML = html; } }
    // a hairline leader from a label outside the shape to a point on the string
    const leader = document.createElement('div');
    leader.style.cssText = 'position:absolute;left:0;top:0;height:1px;transform-origin:0 50%;opacity:0;transition:opacity .5s;'
      + 'background:linear-gradient(90deg,rgba(255,196,107,.9),rgba(255,196,107,.25));';
    const leaderDot = document.createElement('div');
    leaderDot.style.cssText = 'position:absolute;left:-3px;top:-3px;width:6px;height:6px;border-radius:50%;border:1px solid #ffc46b;box-shadow:0 0 8px #ffc46b;';
    leader.appendChild(leaderDot);
    labelLayer.appendChild(leader);
    let leaderIdx = 0, leaderOp = -1, leaderLen = -1, leaderX = -1e4, leaderY = -1e4, leaderA = 1e4;
    let labelKey = -1;
    const SP = { x: 0, y: 0 };
    function toScreen(v) { v.project(camera); SP.x = ((v.x + 1) / 2) * ctx.width; SP.y = ((1 - v.y) / 2) * ctx.height; return v.z < 1; }
    function placeLoopLabel(op) {
      if (op <= 0.01) {
        if (leaderOp !== 0) { leaderOp = 0; leader.style.opacity = '0'; }
        placeLabel(labels.loop, 0, 0, 0, 0, 0);
        return;
      }
      hero.updateMatrixWorld();
      V.copy(hero.position); toScreen(V); const cx = SP.x, cy = SP.y;
      // screen radius of the shape
      V.copy(camera.position).sub(hero.position).normalize().cross(camera.up).normalize().multiplyScalar(CY_SIZE * 1.45 * hero.scale.x).add(hero.position);
      toScreen(V); const rs = Math.hypot(SP.x - cx, SP.y - cy);
      // the loop point that sits furthest towards the upper left (away from the readout), with hysteresis
      const nar = narrow();
      let best = leaderIdx, bs = -1e9, cur = -1e9;
      for (let i = 0; i < NLOOP; i += 8) {
        V.set(loopOut[i * 3], loopOut[i * 3 + 1], loopOut[i * 3 + 2]).applyMatrix4(hero.matrixWorld); toScreen(V);
        const sc = nar ? -(SP.y - cy) - Math.abs(SP.x - cx) * 0.5 : -(SP.x - cx) * 0.75 - (SP.y - cy) * 0.66;
        if (sc > bs) { bs = sc; best = i; }
        if (i === leaderIdx) cur = sc;
      }
      if (bs > cur + 18) leaderIdx = best;
      V.set(loopOut[leaderIdx * 3], loopOut[leaderIdx * 3 + 1], loopOut[leaderIdx * 3 + 2]).applyMatrix4(hero.matrixWorld); toScreen(V);
      const px = SP.x, py = SP.y;
      let dx = px - cx, dy = py - cy; const dl = Math.hypot(dx, dy);
      if (dl < 1) { dx = -0.75; dy = -0.66; } else { dx /= dl; dy /= dl; }
      if (nar) { dx = dx * 0.25; dy = -1; const q = Math.hypot(dx, dy); dx /= q; dy /= q; }
      const lx = cx + dx * (rs + 24), ly = cy + dy * (rs + 24);
      // write the leader's style only when it moved by a visible amount (half a pixel, a few hundredths of a degree)
      const len = Math.round(Math.hypot(lx - px, ly - py) * 2) / 2, ang = Math.round(Math.atan2(ly - py, lx - px) * 2000) / 2000;
      const qx = Math.round(px * 2) / 2, qy = Math.round(py * 2) / 2;
      if (len !== leaderLen) { leaderLen = len; leader.style.width = `${len}px`; }
      if (qx !== leaderX || qy !== leaderY || ang !== leaderA) {
        leaderX = qx; leaderY = qy; leaderA = ang;
        leader.style.transform = `translate3d(${qx}px,${qy}px,0) rotate(${ang}rad)`;
      }
      const o = Math.round(op * 20) / 20;
      if (o !== leaderOp) { leaderOp = o; leader.style.opacity = String(o); }
      const L = labels.loop;
      const x = Math.round(nar ? lx : lx + (dx < 0 ? -6 : 6)), y = Math.round(nar ? ly - 4 : ly);
      if (x !== L.x || y !== L.y) { L.x = x; L.y = y; L.el.style.transform = nar ? `translate3d(${x}px,${y}px,0) translate(-50%,-100%)` : `translate3d(${x}px,${y}px,0) translate(${dx < 0 ? '-100%' : '0'},-50%)`; }
      if (o !== L.op) { L.op = o; L.el.style.opacity = String(o); }
    }

    // ------------------------------------------------------------ state
    let active = false;
    let stage = 1;
    const W = [0, 1, 0, 0];             // smoothed stage weights, index 1..3
    let T = 0;
    const s1 = {
      Rlog: 1, RlogD: 1,                // log2 R; R = 2 to start, so the dual tube nests inside
      n: 1, w: 1, dualT: 0, dualD: 0,
      zoomLog: 0, zoomD: 0, intro: -1,  // log of the view scale (line -> tube); intro tween progress
      thM: 0.6, thW: 2.2, phC: 0.3, phM: 1.1,
      touched: false, lastSign: 0, lastBell: -10, lastToast: -10, smallTold: false, lastRS: 0, flashT: -10,
    };
    const s2 = { n: 5, base: Math.PI / 4, auto: 0, hold: 0, loopVis: 0, loopN: 5, pluckA: 0, pluckS: 0, pluckT: 0, lastPluck: -10, plucked: false, alpha: Math.PI / 4 };
    const s3 = { hover: -1, focus: -1, heroIdx: 0, swapTo: -1, focusK: 0, lastHover: -10, lastRing: -10, toasted: false, downX: 0, downY: 0, downT: 0, latK: 0 };
    s3.heroIdx = centerIdx;
    let fitMul = 1;
    const narrow = () => ctx.width <= 760;
    // portrait screens pull the camera back, by a different amount per station (the lattice should still fill the frame)
    const stageMul = (s, mul = fitMul, focused = false) => Math.pow(mul, s === 3 && !focused ? 0.35 : s === 1 ? 0.85 : 1);
    let grabbing = null, grabY = 0, grabLog = 0;
    let panel = null, ro = null, ladder = null, ladderR = -1;
    let readoutT = 0, audioT = 0, sliderT = 0;
    let vMom = null, vWin = null, hum = null, humN = 0;
    let hintUp = 0;                     // which stage's hint is showing (0: none)
    const timers = new Set();
    const later = (fn, ms) => { const id = setTimeout(() => { timers.delete(id); fn(); }, ms); timers.add(id); };

    // scratch
    const V = new THREE.Vector3(), V2 = new THREE.Vector3(), tmpT = new THREE.Vector3(), tmpD = new THREE.Vector3();
    const camLocal = new THREE.Vector3(), invM = new THREE.Matrix4();
    const heroPos = new THREE.Vector3(), heroPosT = new THREE.Vector3();
    let heroScale = 1, heroSpin = 0;
    const qA = new THREE.Quaternion(), qB = new THREE.Quaternion(), eul = new THREE.Euler(), axV = new THREE.Vector3();
    const driftPrev = new THREE.Vector3();
    const raycaster = new THREE.Raycaster();
    const tanHalf = () => Math.tan((camera.fov * Math.PI) / 360);

    // ------------------------------------------------------------ camera rig
    const HOME = {
      1: { target: new THREE.Vector3(0.55, 0.05, 0), dir: new THREE.Vector3(0.5, 0.3, 0.81).normalize(), dist: 8.6 },
      2: { target: new THREE.Vector3(0, 0, 0), dir: new THREE.Vector3(0.38, 0.3, 0.875).normalize(), dist: 6.6 },
      3: { target: new THREE.Vector3(0.3, -0.4, 0), dir: new THREE.Vector3(0.42, 0.42, 0.8).normalize(), dist: 17 },
    };
    const rig = { fly: false, t: 0, dur: 1, lift: 0, t0: new THREE.Vector3(), t1: new THREE.Vector3(), d0: new THREE.Vector3(), d1: new THREE.Vector3(), r0: 1, r1: 1 };
    function flyTo(target, dir, dist, dur, lift = 0.4) {
      rig.t0.copy(controls.target);
      rig.d0.copy(camera.position).sub(driftPrev).sub(controls.target);
      rig.r0 = Math.max(0.1, rig.d0.length()); rig.d0.divideScalar(rig.r0);
      rig.t1.copy(target);
      rig.d1.copy(dir || rig.d0).normalize();
      rig.r1 = dist; rig.t = 0; rig.dur = Math.max(0.01, dur); rig.lift = lift; rig.fly = true;
      controls.enabled = false;
    }
    function snapTo(s) {
      const h = HOME[s];
      controls.target.copy(h.target);
      camera.position.copy(h.target).addScaledVector(h.dir, h.dist * stageMul(s));
      camera.lookAt(h.target);
      driftPrev.set(0, 0, 0);
      rig.fly = false;
    }
    function stepRig(dt) {
      camera.position.sub(driftPrev);
      if (rig.fly) {
        rig.t += dt / rig.dur;
        const e = easeInOut(Math.min(1, rig.t));
        tmpT.lerpVectors(rig.t0, rig.t1, e);
        tmpD.lerpVectors(rig.d0, rig.d1, e);
        if (tmpD.lengthSq() < 1e-6) tmpD.copy(rig.d1);
        tmpD.normalize();
        tmpD.y += Math.sin(Math.PI * e) * 0.18 * rig.lift;   // a gentle crane up and over between stations
        tmpD.normalize();
        const r = Math.exp(lerp(Math.log(rig.r0), Math.log(rig.r1), e));
        camera.position.copy(tmpT).addScaledVector(tmpD, r);
        controls.target.copy(tmpT);
        camera.lookAt(tmpT);
        if (rig.t >= 1) { rig.fly = false; controls.enabled = active && !grabbing; }
      } else {
        controls.update();
      }
      const amp = (REDUCED ? 0.15 : 1) * 0.016 * camera.position.distanceTo(controls.target);
      driftPrev.set(Math.sin(T * 0.21) * amp, Math.sin(T * 0.17 + 1.3) * amp * 0.6, Math.cos(T * 0.13) * amp * 0.5);
      camera.position.add(driftPrev);
    }

    // ------------------------------------------------------------ actions
    function hideHint() { hintUp = 0; ui.hideHint(); }
    function setStage(s, instant = false) {
      s = clamp(Math.round(s), 1, 3);
      if (s === stage && !instant) return;
      const from = stage;
      stage = s;
      if (s !== 3) { s3.focus = -1; s3.swapTo = -1; s3.hover = -1; }
      controls.enableZoom = s !== 1;
      controls.minDistance = s === 2 ? 2.6 : 1.6;
      controls.maxDistance = s === 2 ? 13 : 32;
      if (instant) snapTo(s);
      else flyTo(HOME[s].target, HOME[s].dir, HOME[s].dist * stageMul(s), from === 1 || s === 1 ? 2.8 : 2.4, 0.6);
      ctx.setPost(POST[s]);
      if (active) { buildPanels(); showHint(); }
      if (!instant) {
        audio.noise({ freq: 240, sweepTo: 1400, dur: 1.5, gain: 0.03, q: 1.1, attack: 0.35 });
        audio.swell(audio.freq(s === 1 ? 12 : s === 2 ? 15 : 19), { gain: 0.055, attack: 0.9, dur: 1.3 });
      }
    }

    // Detents at every whole power of two: at R = 2^k every (n, w) up to 4 sounds a D or an A over the drone.
    function setRlog(v, fromSlider = false) {
      let x = clamp(v, -2, 2);
      const r = Math.round(x);
      if (Math.abs(x - r) < 0.025) x = r;
      s1.Rlog = x;
      if (!fromSlider && panel) panel.set('R', x);
      s1.touched = true;
      if (hintUp === 1) hideHint();
    }
    // on release (tube drag or slider), ease to a nearby whole power of two
    function settleR() {
      const r = Math.round(s1.Rlog);
      if (s1.Rlog !== r && Math.abs(s1.Rlog - r) < 0.12) { s1.Rlog = r; if (panel) panel.set('R', r); }
    }
    function addQuantum(isMomentum) {
      // buttons follow the description on screen: in the dual view the cyan state is a winding
      const cyanIsMomentum = s1.dualT < 0.5;
      const bumpCyan = isMomentum === cyanIsMomentum;
      const wrapped = bumpCyan ? s1.n === 4 : s1.w === 4;
      if (bumpCyan) s1.n = (s1.n % 4) + 1; else s1.w = (s1.w % 4) + 1;
      const R = Math.pow(2, s1.Rlog);
      const f = clamp(D4 * (bumpCyan ? s1.n / R : s1.w * R), 60, 1760);
      audio.pluck(audio.freq(pentaSemis(f)), { gain: 0.2, decay: 2.2, bright: 0.5 * Math.min(1, 500 / f) + 0.1, pan: bumpCyan ? -0.35 : 0.35 });
      if (wrapped) {
        const id = isMomentum ? 'mom' : 'win';
        if (panel) { panel.set(id, true); later(() => { if (panel) panel.set(id, false); }, 700); }
        const isN = (bumpCyan && cyanIsMomentum) || (!bumpCyan && !cyanIsMomentum);
        ui.toast(isN ? `Back to ${sym('n')} = 1, the lightest momentum state. This picture stops at 4.` : `Back to ${sym('w')} = 1, a single wrap. This picture stops at 4.`, 2600);
      }
      buildReadout();
    }
    function toggleDual() {
      s1.dualT = s1.dualT > 0.5 ? 0 : 1;
      if (panel) panel.set('dual', s1.dualT ? 'View on R' : 'View on 1/R');
      buildReadout();
      ui.toast(s1.dualT
        ? 'The same strings described on a circle of radius 1/R. Every energy is unchanged, so the sound does not change.'
        : 'Back to radius R. Momentum and winding swap names again.', 3800);
      audio.noise({ freq: 600, sweepTo: 2400, dur: 0.9, gain: 0.028, q: 2 });
    }
    function toggleZoom() {
      const far = s1.zoomLog < -2;
      s1.intro = -1;
      s1.zoomLog = far ? 0 : Math.log(0.012);
      updateChip();
    }
    function startIntroZoom() {
      s1.intro = 0;
      s1.zoomD = s1.zoomLog = Math.log(0.01);
    }

    // The zoom control lives in the readout's "From here" row as a chip (stage 1 keeps to five panel controls).
    const zoomChip = document.createElement('button');
    zoomChip.type = 'button';
    zoomChip.style.cssText = 'all:unset;box-sizing:border-box;cursor:pointer;display:inline-flex;align-items:center;gap:7px;padding:2px 9px 3px;'
      + 'border:1px solid rgba(236,231,246,.3);border-radius:999px;color:#ece7f6;font:inherit;letter-spacing:.02em;white-space:nowrap;transition:border-color .25s,background .25s;';
    const chipHi = (on) => { zoomChip.style.borderColor = on ? '#ffc46b' : 'rgba(236,231,246,.3)'; zoomChip.style.background = on ? 'rgba(255,196,107,.08)' : 'transparent'; };
    zoomChip.addEventListener('click', () => toggleZoom());
    zoomChip.addEventListener('pointerenter', () => chipHi(true));
    zoomChip.addEventListener('pointerleave', () => chipHi(false));
    zoomChip.addEventListener('focus', () => chipHi(true));
    zoomChip.addEventListener('blur', () => chipHi(false));
    let chipKey = '';
    function updateChip() {
      const far = (s1.far ?? 1) > 0.5, zin = s1.zoomLog < -2, nar = narrow();
      const key = `${far}${zin}${nar}`;
      if (key === chipKey) return;
      chipKey = key;
      const seen = nar ? (far ? '1D' : '2D') : (far ? 'a line · 1D' : 'a tube · 2D');
      zoomChip.innerHTML = `<span>${seen}</span><span style="opacity:.62;border-left:1px solid rgba(236,231,246,.25);padding-left:7px">${zin ? 'zoom in' : 'zoom out'}</span>`;
      const lab = `${zin ? 'Zoom in' : 'Zoom out'} (key Z)`;
      zoomChip.title = lab; zoomChip.setAttribute('aria-label', lab);
    }

    // Degree changes: the readout, the copies and the hum follow at once; the visible shape crossfades as soon as the
    // previous crossfade has finished (see step), so rapid changes never pop a half-dissolved slot.
    function setDegree(n, quiet = false) {
      n = clamp(Math.round(n), 3, 6);
      if (n === s2.n) return;
      s2.n = n;
      setInstDegree(n);
      if (panel && stage === 2) panel.set('n', n);
      if (stage === 2) buildReadout();
      if (hum) setHumTimbre();
      if (!quiet) {
        const g = genus(n);
        ui.toast(n === 3
          ? 'Degree 3: a torus, one hole. This is the one-dimensional Calabi–Yau.'
          : `Degree ${n}: a surface with ${g} holes.`, 3000);
        audio.bell(audio.freq(CHORD[Math.min(4, n - 2)] + 12), { gain: 0.1 });
      }
    }
    function swapSlots(n) {
      const old = slotCur; slotCur = slotOld; slotOld = old;
      slotCur.setDegree(n); slotCur.rev = 0; slotCur.target = 1; slotCur.show(true);
      slotOld.target = 0;
    }
    function alphaDeg() {
      const d = ((((s2.alpha + CYU.uAlphaOff.value) * 180) / Math.PI) % 360 + 360) % 360;
      return Math.round(clamp(d > 180 ? 360 - d : d, 0, 180));
    }
    function setAlphaUser(deg) {
      s2.base = (deg * Math.PI) / 180 - CYU.uAlphaOff.value; s2.auto = 0; s2.hold = 6;
    }
    function pluckLoop(at) {
      if (T - s2.lastPluck < 0.15) return;
      s2.lastPluck = T;
      s2.pluckA = 1; s2.pluckS = at; s2.pluckT = 0;
      const n = s2.n;
      for (let i = 0; i < n; i++) {
        audio.pluck(audio.freq(ARP[i]), { gain: 0.15 - i * 0.015, decay: 2.8, bright: 0.45, when: i * 0.07, pan: -0.4 + (0.8 * i) / (n - 1) });
      }
      if (!s2.plucked) { s2.plucked = true; if (hintUp === 2) hideHint(); }
    }

    // Looking into a point: the hero is always "the copy at heroIdx" when it is small (same place, size, orientation
    // and hidden angle), so swapping it with a copy is invisible. It then grows out of that very copy.
    function swapHeroTo(i) {
      const old = s3.heroIdx;
      if (old === i) return;
      const took = iScale[i];
      iBase[old] = iScale[old] = heroScale / INST_SCALE;   // the copy takes over exactly where the hero stood
      iBase[i] = iScale[i] = 0;
      iScaleDirty = true;
      s3.heroIdx = i;
      heroPos.set(lat[i * 3], lat[i * 3 + 1], lat[i * 3 + 2]);
      heroScale = INST_SCALE * Math.max(0.02, took);   // and the hero takes over copy i at the size it had
    }
    function focusPoint(i) {
      if (i < 0 || i === s3.focus) { unfocus(); return; }
      if (heroScale > INST_SCALE * 1.3) s3.swapTo = i;   // fold the big shape back into its point first
      else { swapHeroTo(i); s3.swapTo = -1; }
      s3.focus = i;
      V.set(lat[i * 3], lat[i * 3 + 1], lat[i * 3 + 2]);
      tmpD.copy(camera.position).sub(driftPrev).sub(controls.target).normalize();
      flyTo(V, tmpD, 7.4 * stageMul(3, fitMul, true), 2.0, 0.15);
      audio.bell(audio.freq(24), { gain: 0.12 });
      audio.swell(audio.freq(12), { gain: 0.05, attack: 0.6, dur: 1.4 });
      if (!s3.toasted) { s3.toasted = true; ui.toast('Every point in your room would hide one of these.', 3600); }
      hideHint();
      if (panel) panel.set('look', 'Back out');
    }
    function unfocus() {
      if (s3.focus < 0) return;
      s3.focus = -1; s3.swapTo = -1;
      tmpD.copy(camera.position).sub(driftPrev).sub(controls.target).normalize();
      flyTo(HOME[3].target, tmpD, HOME[3].dist * stageMul(3), 2.0, 0.15);
      audio.swell(audio.freq(7), { gain: 0.04, attack: 0.5, dur: 1 });
      if (panel) panel.set('look', 'Look inside a point');
    }
    function lookRandom() {
      if (s3.focus >= 0) { unfocus(); return; }
      // pick a point near the middle of the frame, not too far away
      let best = -1, bs = -1e9;
      for (let i = 0; i < LN; i++) {
        V.set(lat[i * 3], lat[i * 3 + 1], lat[i * 3 + 2]);
        const d = V.distanceTo(camera.position);
        V.project(camera);
        if (V.z > 1) continue;
        const score = -Math.hypot(V.x * 1.2, V.y) * 3 - Math.abs(d - 11) * 0.25 + Math.random() * 1.2;
        if (score > bs) { bs = score; best = i; }
      }
      focusPoint(best);
    }
    // Hovering a copy: it swells and plucks, and the copies within two lattice steps swell a little in turn and answer
    // with quieter notes, so the lattice rings sympathetically outward from the pointer.
    function ringFrom(h, pan) {
      const deg = ((latI[h * 3] * 2 + latI[h * 3 + 1] * 3 + latI[h * 3 + 2]) % 7 + 7) % 7;
      audio.pluck(audio.degree(5 + deg, 0), { gain: 0.07, decay: 1.6, bright: 0.3, pan });
      if (T - s3.lastRing < 0.4) return;
      s3.lastRing = T;
      for (let i = 0; i < LN; i++) {
        if (i === h) continue;
        const dx = latI[i * 3] - latI[h * 3], dy = latI[i * 3 + 1] - latI[h * 3 + 1], dz = latI[i * 3 + 2] - latI[h * 3 + 2];
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 <= 4) iRing[i] = T + 0.08 * Math.sqrt(d2);
      }
      for (let sh = 1; sh <= 4; sh++) {   // one quiet note per shell (distance 1, √2, √3, 2)
        audio.pluck(audio.degree(6 + deg + sh, 0), { gain: 0.034 - sh * 0.005, decay: 1.4, bright: 0.2, pan: pan * 0.6, when: 0.08 * Math.sqrt(sh) });
      }
    }

    // ------------------------------------------------------------ picking
    function hitTube(ndc) {
      raycaster.setFromCamera(ndc, camera);
      const o = raycaster.ray.origin, d = raycaster.ray.direction;
      // either tube can be grabbed: the visible one may be the 1/R ghost
      const r = Math.max(tubeA.material.uniforms.uRadius.value, tubeB.material.uniforms.uRadius.value);
      // a thin tube is still easy to grab: at least ~14 px of tolerance
      const dist = Math.hypot(o.y, o.z);
      const tol = Math.max(r * 1.08, (dist * tanHalf() * 2 * 14) / ctx.height + r);
      const a = d.y * d.y + d.z * d.z, b = 2 * (o.y * d.y + o.z * d.z), c = o.y * o.y + o.z * o.z - tol * tol;
      if (a < 1e-9) return c <= 0;
      const disc = b * b - 4 * a * c;
      if (disc < 0) return false;
      const t = (-b - Math.sqrt(disc)) / (2 * a);
      return t > 0 && Math.abs(o.x + t * d.x) < 30;
    }
    function hitLoop(ev, tolPx = ctx.isTouch ? 28 : 16) {
      if (loopRib.shared.uIntensity.value < 0.2) return -1;
      hero.updateMatrixWorld();
      let best = -1, bd = tolPx * tolPx;
      for (let i = 0; i < NLOOP; i += 2) {
        V.set(loopOut[i * 3], loopOut[i * 3 + 1], loopOut[i * 3 + 2]).applyMatrix4(hero.matrixWorld).project(camera);
        const sx = ((V.x + 1) / 2) * ctx.width, sy = ((1 - V.y) / 2) * ctx.height;
        const dd = (sx - ev.x) ** 2 + (sy - ev.y) ** 2;
        if (dd < bd) { bd = dd; best = i; }
      }
      return best < 0 ? -1 : best / NLOOP;
    }
    function pickLattice(ndc) {
      raycaster.setFromCamera(ndc, camera);
      const o = raycaster.ray.origin, d = raycaster.ray.direction, th = tanHalf();
      let best = -1, bt = Infinity;
      for (let i = 0; i < LN; i++) {
        const vx = lat[i * 3] - o.x, vy = lat[i * 3 + 1] - o.y, vz = lat[i * 3 + 2] - o.z;
        const t = vx * d.x + vy * d.y + vz * d.z;
        if (t < 0.4) continue;
        const d2 = vx * vx + vy * vy + vz * vz - t * t;
        const tol = 0.2 + (t * th * 2 * (ctx.isTouch ? 22 : 12)) / ctx.height;
        if (d2 < tol * tol && t < bt) { bt = t; best = i; }
      }
      return best;
    }
    /** Is the screen point inside the hero's silhouette disc (roughly)? */
    function onHero(ev) {
      V.copy(hero.position); toScreen(V); const cx = SP.x, cy = SP.y;
      V.copy(camera.position).sub(hero.position).normalize().cross(camera.up).normalize().multiplyScalar(CY_SIZE * 1.1 * hero.scale.x).add(hero.position);
      toScreen(V);
      return Math.hypot(ev.x - cx, ev.y - cy) < Math.hypot(SP.x - cx, SP.y - cy);
    }

    // ------------------------------------------------------------ UI
    const TALLY = (() => {
      let bars = '';
      for (let i = 0; i < 10; i++) {
        const c = i === 0 ? '#ffc46b' : i < 4 ? '#ece7f6' : '#8b6cff';
        bars += `<i style="display:block;flex:1;height:${i < 4 ? 14 : 7}px;border-radius:1px;background:${c};opacity:${i < 4 ? 0.85 : 1};${i >= 4 ? `box-shadow:0 0 8px ${c}` : ''}"></i>`;
      }
      return `<div style="display:flex;gap:3px;align-items:flex-end;margin:2px 0 5px">${bars}</div>`
        + '<div style="display:flex;justify-content:space-between;font-size:9.5px;letter-spacing:.1em;color:rgba(236,231,246,.62);margin-bottom:8px">'
        + '<span style="color:#ffc46b">1 TIME</span><span>3 SPACE</span><span style="color:#a48cff">6 CURLED</span></div>';
    })();
    // The energy ladder: the lowest (n, w) energies at R (top row) and at 1/R (bottom row) on a log axis. A faint
    // vertical connector joins every tick to its equal in the other row: the two lists are identical, colours swapped.
    const LX = (e) => 44 + (194 * (Math.log(e) - LN_EMIN)) / (LN_EMAX - LN_EMIN);
    const CN_DIM = 'rgba(236,231,246,.18)', CN_GOLD = 'rgba(255,196,107,.9)';
    function ladderHTML() {
      const font = 'font-family="IBM Plex Mono, monospace"';
      const lab = `${font} font-size="8.5" letter-spacing=".08em" fill="rgba(236,231,246,.62)"`;
      const sc = `${font} font-size="7.5" fill="rgba(236,231,246,.45)" text-anchor="middle"`;
      let s = '<svg viewBox="0 0 240 60" width="100%" height="60" style="display:block;margin:0 0 6px;overflow:visible" aria-label="Lowest string energies at radius R and at 1/R">';
      s += `<text x="0" y="20" ${lab}>AT R</text><text x="0" y="44" ${lab}>AT 1/R</text><text x="0" y="57" ${font} font-size="7.5" letter-spacing=".08em" fill="rgba(236,231,246,.45)">ENERGY</text>`;
      s += '<line x1="42" x2="240" y1="22.5" y2="22.5" stroke="rgba(236,231,246,.14)"/><line x1="42" x2="240" y1="46.5" y2="46.5" stroke="rgba(236,231,246,.14)"/>';
      for (const e of [0.25, 1, 4]) {
        const x = LX(e).toFixed(1);
        s += `<line x1="${x}" x2="${x}" y1="46.5" y2="49.5" stroke="rgba(236,231,246,.3)"/><text x="${x}" y="57.5" ${sc}>${e}</text>`;
      }
      for (let i = 0; i < LAD_N.length; i++) s += `<line class="cn" x1="-10" x2="-10" y1="22.5" y2="34" stroke="${CN_DIM}" stroke-width="0.85" style="transition:stroke .35s"/>`;
      for (let row = 0; row < 2; row++) {
        for (let i = 0; i < LAD_N.length; i++) {
          const n = LAD_N[i], w = LAD_W[i];
          const col = w === 0 ? '#6ff3ff' : n === 0 ? '#ff4fbf' : '#ffc46b';
          const op = n + w === 1 ? 1 : n + w === 2 ? 0.7 : 0.42;
          s += `<rect class="t${row}" x="-10" y="${row ? 34 : 10}" width="2" height="12" rx="1" fill="${col}" opacity="${op}"/>`;
        }
      }
      s += `<text class="gn" x="-20" y="7" ${font} font-size="8" fill="#6ff3ff" text-anchor="middle">n</text>`;
      s += `<text class="gw" x="-20" y="7" ${font} font-size="8" fill="#ff4fbf" text-anchor="middle">w</text>`;
      return s + '</svg>';
    }
    function updateLadder(R) {
      if (!ladder) return;
      for (let i = 0; i < LAD_N.length; i++) {
        const n = LAD_N[i], w = LAD_W[i];
        const xa = tick(ladder.a[i], Math.hypot(n / R, w * R));
        tick(ladder.b[i], Math.hypot(n * R, w / R));
        const c = ladder.c[i], v = xa < 0 ? '-10' : (xa + 1).toFixed(1);
        c.setAttribute('x1', v); c.setAttribute('x2', v);
      }
      // the n and w glyphs over the (1, 0) and (0, 1) ticks, nudged apart where they meet at R = 1
      let xn = LX(1 / R) + 1, xw = LX(R) + 1;
      if (Math.abs(xn - xw) < 8) { const m = (xn + xw) / 2, sg = xn <= xw ? 1 : -1; xn = m - 4 * sg; xw = m + 4 * sg; }
      ladder.gn.setAttribute('x', xn.toFixed(1)); ladder.gw.setAttribute('x', xw.toFixed(1));
      // crossing the self-dual radius: the connectors flash gold
      const sg = Math.sign(Math.log2(R));
      if (sg !== 0 && s1.lastRS !== 0 && sg !== s1.lastRS) flashLadder();
      if (sg !== 0) s1.lastRS = sg;
    }
    function flashLadder() {
      if (!ladder) return;
      for (let i = 0; i < ladder.c.length; i++) ladder.c[i].setAttribute('stroke', CN_GOLD);
      later(() => { if (ladder) for (let i = 0; i < ladder.c.length; i++) ladder.c[i].setAttribute('stroke', CN_DIM); }, 600);
    }
    function tick(el, e) {
      const x = LX(e);
      if (x < 42 || x > 239) { el.setAttribute('x', '-10'); return -1; }
      el.setAttribute('x', x.toFixed(1));
      return x;
    }

    function buildReadout() {
      if (!active) return;
      ladder = null; ladderR = -1;
      const n_ = sym('n'), w_ = sym('w');
      if (stage === 1) {
        const dual = s1.dualT > 0.5, R = Math.pow(2, s1.Rlog);
        ro = narrow() ? ui.readout({
          title: 'A curled dimension',
          html: ladderHTML(),
          rows: [
            { id: 'R', k: 'R · 1/R', v: '', cls: 'gold' },
            { id: 'Ec', k: dual ? `${w_} = ${s1.n} on 1/R` : `${n_} = ${s1.n} · ${n_}/R`, v: '', cls: 'cyan' },
            { id: 'Em', k: dual ? `${n_} = ${s1.w} on 1/R` : `${w_} = ${s1.w} · ${w_}R`, v: '', cls: 'mag' },
            { id: 'seen', k: 'View', v: '' },
          ],
        }) : ui.readout({
          title: 'A curled dimension',
          html: ladderHTML(),
          rows: [
            { id: 'R', k: 'Radius R', v: fmt2(R), cls: 'gold' },
            { id: 'Rd', k: 'Dual radius 1/R', v: fmt2(1 / R) },
            { id: 'Ec', k: dual ? `Winding ${w_} = ${s1.n} on 1/R` : `Momentum ${n_} = ${s1.n} · E = ${n_}/R`, v: '', cls: 'cyan' },
            { id: 'Em', k: dual ? `Momentum ${n_} = ${s1.w} on 1/R` : `Winding ${w_} = ${s1.w} · E = ${w_}R`, v: '', cls: 'mag' },
            { id: 'seen', k: 'From here', v: '' },
          ],
          note: '',
        });
        const svg = ro.el.querySelector('svg');
        if (svg) ladder = { a: svg.querySelectorAll('rect.t0'), b: svg.querySelectorAll('rect.t1'), c: svg.querySelectorAll('line.cn'), gn: svg.querySelector('text.gn'), gw: svg.querySelector('text.gw') };
        // the zoom chip owns the last row's value cell
        const cells = ro.el.querySelectorAll('.row .v');
        const cell = cells[cells.length - 1];
        if (cell) { cell.textContent = ''; cell.appendChild(zoomChip); }
        chipKey = '';
        updateChip();
      } else if (stage === 2) {
        const n = s2.n, g = genus(n);
        ro = narrow() ? ui.readout({
          title: 'Six dimensions, folded',
          html: TALLY,
          rows: [
            { id: 'n', k: `Degree ${n_}`, v: String(n) },
            { id: 'g', k: 'Holes', v: String(g), cls: 'gold' },
            { id: 'q', k: 'Quintic', v: '100 families', cls: 'mag' },
            { id: 'us', k: 'Ours', v: '3 families', cls: 'cyan' },
          ],
        }) : ui.readout({
          title: 'Six dimensions, folded',
          html: TALLY,
          rows: [
            { id: 'n', k: `Degree ${n_}`, v: String(n) },
            { id: 'g', k: 'Holes (genus)', v: String(g), cls: 'gold' },
            { id: 'chi', k: `Euler ${sym('χ')} (closed curve)`, v: signed(2 - 2 * g) },
            { id: 'q', k: 'Quintic, 6D', v: 'χ = −200 → 100 families', cls: 'mag' },
            { id: 'us', k: 'Our universe', v: '3 families → |χ| = 6', cls: 'cyan' },
          ],
          note: n === 5
            ? 'A 2D slice of a 6D shape, drawn in 3D. Degree 5 is the usual stand-in for a slice of the quintic. On this slice the gold string wraps a handle and cannot shrink away.'
            : 'A 2D slice of a 6D shape, drawn in 3D. On this slice the gold string wraps a handle and cannot shrink away.',
        });
      } else {
        ro = narrow() ? ui.readout({
          title: 'At every point of space',
          rows: [
            { id: 'pts', k: 'Drawn', v: String(LN) },
            { id: 'real', k: 'In reality', v: 'every point', cls: 'gold' },
            { id: 'lim', k: 'Collider limit', v: '< 10⁻¹⁹ m' },
          ],
        }) : ui.readout({
          title: 'At every point of space',
          html: TALLY,
          rows: [
            { id: 'pts', k: 'Points drawn', v: String(LN) },
            { id: 'real', k: 'Points in reality', v: 'every one', cls: 'gold' },
            { id: 'lim', k: 'Collider limit', v: '< 10⁻¹⁹ m' },
            { id: 'mod', k: 'Often assumed', v: '10⁻³⁵–10⁻³² m', cls: 'cyan' },
          ],
          note: 'The spacing is hugely exaggerated. Space is a continuum, so the copies would sit at every point.',
        });
      }
      readoutT = 1;
    }

    function buildPanels() {
      if (!active) return;
      const seg = {
        type: 'segmented', id: 'stage', label: 'Stage', value: stage,
        options: [{ value: 1, label: '1 Circle', kbd: '1' }, { value: 2, label: '2 Folded', kbd: '2' }, { value: 3, label: '3 Everywhere', kbd: '3' }],
        onChange: (v) => setStage(+v),
      };
      if (stage === 1) {
        panel = ui.controls([
          seg,
          { type: 'slider', id: 'R', label: 'Radius R', min: -2, max: 2, step: 0.01, value: s1.Rlog, format: (v) => `${fmt2(Math.pow(2, v))} √α′`, onInput: (v) => setRlog(v, true) },
          { type: 'button', id: 'mom', label: '+ Momentum', kbd: 'm', onClick: () => addQuantum(true) },
          { type: 'button', id: 'win', label: '+ Winding', kbd: 'w', onClick: () => addQuantum(false) },
          { type: 'button', id: 'dual', label: s1.dualT > 0.5 ? 'View on R' : 'View on 1/R', kbd: 't', onClick: toggleDual },
        ]);
        const rIn = panel.get('R');
        if (rIn) {
          rIn.addEventListener('change', settleR);   // our own element, rebuilt with the panel
          const o = rIn.closest('.ctl')?.querySelector('output');
          if (o) o.style.textTransform = 'none';      // keeps α′ from turning into A′
        }
      } else if (stage === 2) {
        panel = ui.controls([
          seg,
          { type: 'slider', id: 'n', label: `Degree ${sym('n')}`, min: 3, max: 6, step: 1, value: s2.n, format: (v) => `${v} · ${genus(v)} hole${genus(v) === 1 ? '' : 's'}`, onInput: (v) => setDegree(v) },
          { type: 'slider', id: 'alpha', label: `Hidden angle ${sym('α')}`, min: 0, max: 180, step: 1, value: alphaDeg(), format: (v) => `${Math.round(v)}°`, onInput: (v) => setAlphaUser(v) },
          { type: 'button', id: 'pluck', label: 'Pluck the loop', kbd: ' ', onClick: () => pluckLoop(Math.random()) },
        ]);
      } else {
        panel = ui.controls([
          seg,
          { type: 'button', id: 'look', label: s3.focus >= 0 ? 'Back out' : 'Look inside a point', kbd: ' ', onClick: lookRandom },
        ]);
      }
      buildReadout();
    }
    function showHint() {
      if (!active) return;
      const touch = ctx.isTouch, nar = narrow();
      // on phones the hint sits mid-screen, over the scene, so it always leaves on its own
      if (stage === 1) ui.hint(touch ? 'Drag the tube to resize its circle' : 'Drag the tube or press ↑ ↓ to resize its circle', nar ? 5500 : s1.touched ? 4000 : 0);
      else if (stage === 2) ui.hint(touch ? 'Drag to orbit · tap the gold loop' : 'Drag to orbit · click the gold loop or press space', nar ? 4000 : 9000);
      else ui.hint(touch ? 'Tap a point of space to look inside it' : 'Click any point of space to look inside it', nar ? 4000 : 9000);
      hintUp = stage;
    }

    // ------------------------------------------------------------ audio
    const HUM_AMPS = [1, 0.2, 0.1, 0.05];
    function setHumTimbre() {
      const g = genus(s2.n) / 10;   // more holes, richer overtones
      HUM_AMPS[1] = 0.18 + 0.2 * g; HUM_AMPS[2] = 0.08 + 0.16 * g; HUM_AMPS[3] = 0.03 + 0.12 * g;
      if (hum) for (let i = 0; i < hum.length; i++) hum[i].setAmps(HUM_AMPS, 0.4);
      humN = s2.n;
    }
    function stopVoices() {
      if (vMom) { vMom.stop(0.6); vMom = null; }
      if (vWin) { vWin.stop(0.6); vWin = null; }
      if (hum) { for (let i = 0; i < hum.length; i++) hum[i].stop(0.8); hum = null; }
    }
    const comp = (f) => Math.min(1.4, Math.pow(300 / f, 0.3));
    const aLast = new Float32Array(4).fill(-1);   // last written freq/gain of the two stage-1 tones
    function updateAudio(fM, fW) {
      if (!audio.on) return;
      const w1 = W[1];
      if (w1 > 0.01) {
        if (!vMom || !vWin) aLast.fill(-1);
        if (!vMom) { vMom = audio.additive(fM, 3, { gain: 0 }); vMom.setAmps([1, 0.2, 0.06], 0.02); vMom.setPan(-0.35); }
        if (!vWin) { vWin = audio.additive(fW, 4, { gain: 0 }); vWin.setAmps([1, 0.38, 0.2, 0.09], 0.02); vWin.setPan(0.35); }
        const gM = 0.07 * w1 * comp(fM), gW = 0.065 * w1 * comp(fW);
        // high notes get darker, so a small circle never turns piercing
        if (Math.abs(fM - aLast[0]) > 0.05) { aLast[0] = fM; vMom.setFreq(fM, 0.05); const d = Math.min(1, 500 / fM); vMom.setAmps([1, 0.2 * d, 0.06 * d * d], 0.08); }
        if (Math.abs(fW - aLast[1]) > 0.05) { aLast[1] = fW; vWin.setFreq(fW, 0.05); const d = Math.min(1, 500 / fW); vWin.setAmps([1, 0.38 * d, 0.2 * d * d, 0.09 * d * d], 0.08); }
        if (Math.abs(gM - aLast[2]) > 1e-4) { aLast[2] = gM; vMom.setGain(gM, 0.1); }
        if (Math.abs(gW - aLast[3]) > 1e-4) { aLast[3] = gW; vWin.setGain(gW, 0.1); }
      } else if (w1 < 0.004 && (vMom || vWin)) {
        if (vMom) { vMom.stop(0.5); vMom = null; }
        if (vWin) { vWin.stop(0.5); vWin = null; }
      }
      const wh = W[2] + W[3] * 0.45;
      if (wh > 0.01) {
        if (!hum) {
          hum = [];
          for (let i = 0; i < CHORD.length; i++) { const v = audio.additive(audio.freq(CHORD[i]), 4, { gain: 0 }); v.setPan(CHORD_PAN[i]); hum.push(v); }
          setHumTimbre();
        }
        for (let i = 0; i < hum.length; i++) {
          const on = i < s2.n - 1 ? 1 : 0;
          hum[i].setGain(on * 0.03 * wh * (0.82 + 0.18 * Math.sin(T * 0.7 + i * 1.7)), 0.25);
        }
      } else if (hum && wh < 0.004) {
        for (let i = 0; i < hum.length; i++) hum[i].stop(0.8);
        hum = null;
      }
    }

    // ------------------------------------------------------------ initial state + debug params
    slotCur.setDegree(5); slotCur.target = 1;
    setInstDegree(5);
    {
      const qs = parseInt(Q.get('stage'), 10);
      const qn = parseInt(Q.get('deg'), 10);
      const qr = parseFloat(Q.get('rad'));
      if (qn >= 3 && qn <= 6) { s2.n = qn; slotCur.setDegree(qn); setInstDegree(qn); }
      if (qr > 0) s1.Rlog = s1.RlogD = clamp(Math.log2(qr), -2, 2);
      if (Q.has('dual')) s1.dualT = s1.dualD = 1;
      if (Q.has('alpha')) s2.base = s2.alpha = ((parseFloat(Q.get('alpha')) || 45) * Math.PI) / 180;
      stage = qs >= 1 && qs <= 3 ? qs : 1;
      for (let i = 1; i <= 3; i++) W[i] = i === stage ? 1 : 0;
      if (stage !== 1) { slotCur.rev = 1; }
      controls.enableZoom = stage !== 1;
      snapTo(stage);
    }
    const qFocus = Q.get('focus');

    // ------------------------------------------------------------ per-frame: state integration
    // step(h) only advances cheap state (weights, camera, phases, damped parameters). It may run several
    // times per frame when the wall clock jumped further than the engine's clamped dt (a long frame, or a
    // headless renderer that paints rarely), so animations stay on real time. draw() then builds geometry once.
    let iScaleDirty = false;
    function step(h) {
      T += h;
      const kw = damp(2.3, h);
      for (let i = 1; i <= 3; i++) W[i] += ((i === stage ? 1 : 0) - W[i]) * kw;
      stepRig(h);

      // stage 1
      s1.RlogD += (s1.Rlog - s1.RlogD) * damp(9, h);
      if (s1.intro >= 0) {
        s1.intro = Math.min(1, s1.intro + h / 4.6);
        s1.zoomD = s1.zoomLog = lerp(Math.log(0.01), 0, easeInOut(s1.intro));
        if (s1.intro >= 1) s1.intro = -1;
      } else {
        s1.zoomD += (s1.zoomLog - s1.zoomD) * damp(2.2, h);
      }
      s1.dualD += (s1.dualT - s1.dualD) * damp(2.6, h);
      const R = Math.pow(2, s1.RlogD);
      const EM = s1.n / R, EW = s1.w * R;                 // cyan: (n, 0) at R; magenta: (0, w) at R
      s1.thM += h * (0.3 + 0.42 * Math.min(EM, 8));       // more momentum, faster orbit
      s1.thW += h * (0.3 + 0.42 * Math.min(EW, 8));
      s1.phC += h * 0.11; s1.phM += h * 0.09;
      s1.fM = clamp(D4 * EM, 55, 1760); s1.fW = clamp(D4 * EW, 55, 1760);
      // tone crossing (at R = 1 when n = w): a bell and a toast
      const sign = s1.fM >= s1.fW ? 1 : -1;
      if (s1.lastSign === 0) s1.lastSign = sign;
      else if (sign !== s1.lastSign) {
        s1.lastSign = sign;
        if (active && W[1] > 0.5 && T - s1.lastBell > 1.2) {
          s1.lastBell = T;
          audio.bell(audio.freq(pentaSemis(clamp(s1.fM, 110, 1200) * 2)), { gain: 0.09 });
          if (T - s1.lastToast > 5) {
            s1.lastToast = T;
            // the tones cross where n/R = wR, i.e. R = sqrt(n/w): the self-dual radius when n = w
            ui.toast(s1.n === s1.w
              ? 'Self-dual radius R = √α′. The momentum and winding tones meet.'
              : `The tones cross where n/R = wR, at R = ${fmt2(Math.sqrt(s1.n / s1.w))}.`, 3200);
          }
        }
      }

      // the key point of T-duality, said once: shrinking past the string scale buys nothing new
      if (!s1.smallTold && s1.touched && active && W[1] > 0.5 && s1.RlogD < -0.75 && T - s1.lastToast > 3.4) {
        s1.smallTold = true; s1.lastToast = T;
        ui.toast('Smaller than √α′ gives nothing new. This circle has exactly the physics of one with radius 1/R.', 4200);
      }

      // stage 2: hidden angle drifts slowly unless the user is holding it
      if (s2.hold > 0) s2.hold -= h; else s2.auto += h * (REDUCED ? 0.3 : 1);
      s2.alpha = s2.base + 0.55 * Math.sin(0.085 * s2.auto);
      slotCur.rev += (slotCur.target - slotCur.rev) * damp(1.5, h);
      if (slotCur.rev > 0.996) slotCur.rev = 1;   // lets the shader skip the dissolve noise
      slotOld.rev += (slotOld.target - slotOld.rev) * damp(2.2, h);
      if (slotCur.n !== s2.n && slotOld.rev < 0.05) swapSlots(s2.n);
      // hero placement: the origin in stages 1-2, a lattice point in stage 3
      const hi = s3.heroIdx;
      if (stage === 3) heroPosT.set(lat[hi * 3], lat[hi * 3 + 1], lat[hi * 3 + 2]); else heroPosT.set(0, 0, 0);
      heroPos.lerp(heroPosT, damp(stage === 3 ? 2.2 : 3, h));
      if (s3.swapTo >= 0 && heroScale < INST_SCALE * 1.3) { swapHeroTo(s3.swapTo); s3.swapTo = -1; }
      const sT = stage === 3 && (s3.focus < 0 || s3.swapTo >= 0) ? INST_SCALE : 1;
      heroScale += (sT - heroScale) * damp(2.4, h);
      heroSpin += h * SPIN;
      s3.focusK += ((s3.focus >= 0 ? 1 : 0) - s3.focusK) * damp(2.5, h);
      // the string on the handle swaps its curve while faded out
      const want = s2.loopN === slotCur.n ? 1 : 0;
      s2.loopVis += (want - s2.loopVis) * damp(want ? 1.4 : 7, h);
      if (!want && s2.loopVis < 0.03) s2.loopN = slotCur.n;
      s2.pluckA *= Math.exp(-h * 0.6);
      s2.pluckT += h;

      // stage 3: lattice copies grow, shrink and step aside
      if (W[3] > 0.002 || iScaleDirty) {
        const f = s3.focus;
        const fx = f >= 0 ? lat[f * 3] : 0, fy = f >= 0 ? lat[f * 3 + 1] : 0, fz = f >= 0 ? lat[f * 3 + 2] : 0;
        const k = damp(6, h), kh = damp(9, h);
        for (let i = 0; i < LN; i++) {
          let tgt = i === s3.heroIdx ? 0 : i === s3.hover ? 3.2 : 1;
          if (f >= 0 && i !== s3.heroIdx) {
            const d = Math.hypot(lat[i * 3] - fx, lat[i * 3 + 1] - fy, lat[i * 3 + 2] - fz);
            tgt *= 0.12 + 0.88 * smoothstep(1.4, 4.6, d);
          }
          iBase[i] += (tgt - iBase[i]) * (i === s3.hover ? kh : k);
          const rt = T - iRing[i];
          const nv = iBase[i] * (rt > 0 && rt < 0.7 ? 1 + 0.15 * Math.sin((Math.PI * rt) / 0.7) : 1);
          if (Math.abs(nv - iScale[i]) > 1e-4) { iScale[i] = nv; iScaleDirty = true; }
        }
      }
    }

    // ------------------------------------------------------------ per-frame: visuals
    function draw1() {
      const w1 = W[1];
      const R = Math.pow(2, s1.RlogD), Z = Math.exp(s1.zoomD);
      const du = smoothstep(0, 1, s1.dualD);
      const coll = Math.pow(w1, 0.7);   // leaving stage 1, the hose collapses into a thread of space
      const rA = RW * Z * R * coll, rB = (RW * Z * coll) / R;

      // how big is the circle on screen? below a few pixels the hose reads as a line
      const axisDist = Math.max(0.3, Math.hypot(camera.position.y, camera.position.z));
      const pxPerUnit = (ctx.height / 2) / (axisDist * tanHalf());
      const rpx = Math.max(rA, rB * 0.35) * pxPerUnit;
      const far = 1 - smoothstep(1.5, 7, rpx);
      s1.far = far;
      axis.setIntensity(w1 * (0.1 + 0.95 * far) + W[2] * 0.36 + W[3] * 0.12);   // in stage 3 it is one grid line among many
      // away from stage 1 the thread is a backdrop: a soft, wide core stays under the bloom threshold, so the bloom's
      // low-resolution mips never see a one-pixel line (which they alias into a row of beads along the diagonal)
      const w23 = Math.min(1, W[2] + W[3]);
      axis.shared.uCore.value = Math.max(0, 0.8 - 0.66 * w23 - 0.14 * W[3]);
      axis.shared.uWidth.value = axis.baseWidth * ctx.pixelRatio * (1 + 1.4 * w23);

      const ua = tubeA.material.uniforms, ub = tubeB.material.uniforms;
      ua.uRadius.value = Math.max(rA, 1e-5); ub.uRadius.value = Math.max(rB, 1e-5);
      ua.uRing.value = ub.uRing.value = RW * Z;
      ua.uGhost.value = du; ub.uGhost.value = 1 - du;
      const merge = 1 - smoothstep(0.0, 0.07, Math.abs(s1.RlogD));
      ua.uMerge.value = ub.uMerge.value = merge;
      // sub-pixel tubes alias into dashes; the axis thread takes over instead
      const fadeA = smoothstep(0.7, 2.5, rA * pxPerUnit), fadeB = smoothstep(0.7, 2.5, rB * pxPerUnit);
      ua.uIntensity.value = w1 * lerp(1, 0.55, du) * fadeA;
      ub.uIntensity.value = w1 * lerp(0.55, 1, du) * (1 - merge * 0.8) * fadeB;
      tubeA.visible = ua.uIntensity.value > 0.002 && !Q.has('m3notube');
      tubeB.visible = ub.uIntensity.value > 0.002 && !Q.has('m3notube');

      const near = 1 - far;
      const strI = w1 * w1 * near * near * near;   // squeezed to a few pixels the strings would only be a glare on the line
      if (strI > 0.002) {
        const n = s1.n, w = s1.w, EM = n / R, EW = w * R;
        const xs = 1.15 * Z + 0.35 * lerp(rA, rB, du);
        const beat = Math.abs(s1.fM - s1.fW);
        const pulse = beat < 12 ? 0.84 + 0.16 * Math.cos(TAU * beat * T) : 1;   // the visual beats with the audio
        // cyan: momentum loop on R  <->  winding coil on 1/R
        loopShape(bufA, glA, NS, rA, Z, -xs, s1.thM, T, 0);
        coilShape(bufB, glB, NS, rB, Z, -xs, n, s1.phC, T);
        for (let i = 0; i < NS * 3; i++) bufOut[i] = bufA[i] + (bufB[i] - bufA[i]) * du;
        for (let i = 0; i < NS; i++) glOut[i] = glA[i] + (glB[i] - glA[i]) * du;
        cyanStr.set(bufOut, glOut);
        // magenta: winding coil on R  <->  momentum loop on 1/R
        coilShape(bufA, glA, NS, rA, Z, xs, w, s1.phM, T);
        loopShape(bufB, glB, NS, rB, Z, xs, s1.thW, T, 1.7);
        for (let i = 0; i < NS * 3; i++) bufOut[i] = bufA[i] + (bufB[i] - bufA[i]) * du;
        for (let i = 0; i < NS; i++) glOut[i] = glA[i] + (glB[i] - glA[i]) * du;
        magStr.set(bufOut, glOut);
        cyanStr.setIntensity(strI * pulse);
        magStr.setIntensity(strI * pulse);
        // the wave ring and tail belong to whichever string is the momentum state in this description
        const ic = strI * (1 - du), im = strI * du;
        cyanRing.setIntensity(ic); cyanTail.setIntensity(ic);
        magRing.setIntensity(im); magTail.setIntensity(im);
        if (ic > 0.002) {
          ringShape(rA, -xs, n, s1.thM); cyanRing.set(ringBuf, ringGlow);
          tailShape(rA, -xs, s1.thM, Math.min(3.6, 1.2 + 0.3 * EM)); cyanTail.set(tailBuf, tailGlow);
        }
        if (im > 0.002) {
          ringShape(rB, xs, w, s1.thW); magRing.set(ringBuf, ringGlow);
          tailShape(rB, xs, s1.thW, Math.min(3.6, 1.2 + 0.3 * EW)); magTail.set(tailBuf, tailGlow);
        }
        const lop = active && !(ctx.engine && ctx.engine.transition) ? clamp((strI - 0.3) * 2.2) * smoothstep(0.55, 0.9, Z) : 0;
        const rc = lerp(rA, rB, du);
        const isNarrow = narrow();
        // rebuild the label text only when what it says changes (no string churn per frame)
        const key = (n * 8 + w) * 4 + (du > 0.5 ? 2 : 0) + (isNarrow ? 1 : 0) + Math.round(EM * 100) * 1e3 + Math.round(EW * 100) * 1e7;
        if (key !== labelKey) {
          labelKey = key;
          labelHTML(labels.cyan, du > 0.5
            ? `${sym('w')} = ${n}${isNarrow ? '' : ' · winding'}<br><span style="opacity:.6">E = ${fmt2(EM)}</span>`
            : `${sym('n')} = ${n}${isNarrow ? '' : ' · momentum'}<br><span style="opacity:.6">E = ${fmt2(EM)}</span>`);
          labelHTML(labels.mag, du > 0.5
            ? `${sym('n')} = ${w}${isNarrow ? '' : ' · momentum'}<br><span style="opacity:.6">E = ${fmt2(EW)}</span>`
            : `${sym('w')} = ${w}${isNarrow ? '' : ' · winding'}<br><span style="opacity:.6">E = ${fmt2(EW)}</span>`);
        }
        if (isNarrow) {   // phones: the readout owns the top of the frame, so label from below
          placeLabel(labels.cyan, -xs, -rc * 1.3 - 0.12, 0, 10, lop, true);
          placeLabel(labels.mag, xs, -rc * 1.3 - 0.12, 0, 10, lop, true);
        } else {
          placeLabel(labels.cyan, -xs, rc * 1.3 + 0.12, 0, -10, lop);
          placeLabel(labels.mag, xs, rc * 1.3 + 0.12, 0, -10, lop);
        }
      } else {
        cyanStr.setIntensity(0); magStr.setIntensity(0);
        cyanRing.setIntensity(0); magRing.setIntensity(0); cyanTail.setIntensity(0); magTail.setIntensity(0);
        placeLabel(labels.cyan, 0, 0, 0, 0, 0); placeLabel(labels.mag, 0, 0, 0, 0, 0);
      }
      return R;
    }

    function draw2() {
      const wH = Math.min(1, W[2] + W[3]);   // only the hose (stage 1) dissolves the shape; 2 <-> 3 just rescales it
      CYU.uAlpha.value = s2.alpha;
      CYU.uBreath.value = 0.9 + 0.1 * Math.sin(T * 0.7);
      // 2 -> 3: the shape shrinks and the camera pulls out first, then the copies ripple in, never brighter than at rest
      CYU.uGrow.value = smoothstep(0.45, 1, W[3]) * 1.6;
      instMat.uniforms.uIntensity.value = 0.45 + 0.55 * W[3] * W[3];
      if (slotOld.rev < 0.004) { slotOld.rev = 0; slotOld.show(false); }
      const rw = wH > 0.87 ? 1 : wH * 1.15;
      slotCur.su.uReveal.value = slotCur.rev * rw;
      slotOld.su.uReveal.value = slotOld.rev * rw;
      slotCur.show(wH > 0.002);
      hero.visible = wH > 0.002;
      const breath = 1 + (REDUCED ? 0 : 0.012) * Math.sin(T * 0.6);
      hero.position.copy(heroPos);
      hero.scale.setScalar(heroScale * breath);
      // shrunk into the lattice, the hero is exactly the copy at its point: same orientation and hidden angle
      const cpk = W[3] * (1 - smoothstep(INST_SCALE * 1.05, 0.55, heroScale)), i4 = s3.heroIdx * 4;
      eul.set(heroRot.x, heroRot.y + heroSpin, heroRot.z); qB.setFromEuler(eul);
      if (cpk > 1e-3) {
        const pz = iData[i4 + 2], az = iData[i4 + 1] * TAU;
        axV.set(Math.sin(pz) * Math.cos(az), Math.cos(pz), Math.sin(pz) * Math.sin(az)).normalize();
        qA.setFromAxisAngle(axV, iData[i4] * TAU + T * (0.1 + 0.18 * iData[i4 + 1]));
        hero.quaternion.copy(qB).slerp(qA, cpk);
      } else hero.quaternion.copy(qB);
      CYU.uAlphaOff.value = cpk * iData[i4 + 3];
      // lattice-copy size: the mid mesh and no x-ray (a few thousand triangles instead of 130k)
      const lod = heroScale < 0.45 ? 1 : 0;
      slotCur.setLod(lod); if (slotOld.n) slotOld.setLod(lod);
      slotCur.xray.visible = hero.visible && !LOW && !lod;
      slotOld.xray.visible = false;
      aura.position.copy(heroPos);
      aura.material.opacity = Math.min(1, W[2] + W[3] * s3.focusK) * (0.85 + 0.15 * Math.sin(T * 0.5));
      aura.visible = aura.material.opacity > 0.002;

      // shrunk to a lattice point, the screen-space string would stay fat and smear into a glare: thin it and dim it
      const big = smoothstep(INST_SCALE, 1, heroScale);
      loopRib.shared.uWidth.value = loopRib.baseWidth * ctx.pixelRatio * (0.3 + 0.7 * big);
      const li = s2.loopVis * Math.max(W[2], W[3] * big) * smoothstep(0.55, 0.95, slotCur.rev);
      loopRib.setIntensity(li);
      if (li > 0.002) updateLoop();
      const lo = active && W[2] > 0.6 && !narrow() && !(ctx.engine && ctx.engine.transition) ? li * 0.75 : 0;
      if (lo > 0.01) labelHTML(labels.loop, 'a string wrapped<br>on a handle');
      placeLoopLabel(lo);
    }

    function updateLoop() {
      const L = getLoop(s2.loopN);
      const N = L.N, p4 = L.p4, u4 = L.u4, v4 = L.v4;
      const al = s2.alpha + CYU.uAlphaOff.value, c = Math.cos(al), s = Math.sin(al), norm = cyNorm(s2.loopN);
      hero.updateMatrixWorld();
      invM.copy(hero.matrixWorld).invert();
      camLocal.copy(camera.position).applyMatrix4(invM);
      const pa = s2.pluckA, ps = s2.pluckS, pv = s2.pluckT * 0.16;
      for (let i = 0; i < N; i++) {
        const i4 = i * 4;
        const px = p4[i4] * norm, py = p4[i4 + 2] * norm, pz = (c * p4[i4 + 1] + s * p4[i4 + 3]) * norm;
        const ux = u4[i4], uy = u4[i4 + 2], uz = c * u4[i4 + 1] + s * u4[i4 + 3];
        const vx = v4[i4], vy = v4[i4 + 2], vz = c * v4[i4 + 1] + s * v4[i4 + 3];
        let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
        const nl = Math.hypot(nx, ny, nz) || 1;
        nx /= nl; ny /= nl; nz /= nl;
        if (nx * (camLocal.x - px) + ny * (camLocal.y - py) + nz * (camLocal.z - pz) < 0) { nx = -nx; ny = -ny; nz = -nz; }
        const f = i / N;
        // a plucked closed string: two pulses running in opposite directions (left- and right-movers)
        const g1 = wrap01(f - ps - pv), g2 = wrap01(f - ps + pv);
        const pk = pa * (Math.exp(-(g1 * g1) / 0.0016) + Math.exp(-(g2 * g2) / 0.0016));
        const a1 = wrap01(f - T * 0.07), a2 = wrap01(f - T * 0.07 - 0.5);
        const idle = Math.exp(-(a1 * a1) / 0.004) + Math.exp(-(a2 * a2) / 0.004);
        const disp = 0.02 + 0.011 * Math.sin(TAU * 3 * f - 2.2 * T) + 0.11 * pk;
        loopOut[i * 3] = px + nx * disp; loopOut[i * 3 + 1] = py + ny * disp; loopOut[i * 3 + 2] = pz + nz * disp;
        loopGlow[i] = 0.62 + 0.38 * idle + 1.4 * pk;
      }
      loopRib.set(loopOut, loopGlow);
    }

    function draw3() {
      const w3 = W[3];
      gridMat.uniforms.uOpacity.value = w3 * (1 - 0.25 * s3.focusK);
      const fi = s3.heroIdx;
      instMat.uniforms.uFocus.value.set(lat[fi * 3], lat[fi * 3 + 1], lat[fi * 3 + 2], s3.focusK);
      grid.visible = w3 > 0.002;
      inst.visible = w3 > 0.002;
      if (iScaleDirty) { iScaleAttr.needsUpdate = true; iScaleDirty = false; }
      const h = s3.hover;
      const lop = active && h >= 0 && s3.focus < 0 && !rig.fly && w3 > 0.6 ? 0.9 : 0;
      if (h >= 0) { labelHTML(labels.pt, 'one point of space'); placeLabel(labels.pt, lat[h * 3], lat[h * 3 + 1] + 0.35, lat[h * 3 + 2], -6, lop); }
      else placeLabel(labels.pt, 0, 0, 0, 0, 0);
    }

    function updateReadout(R) {
      if (!ro) return;
      if (stage === 1) {
        if (narrow()) ro.set('R', `${fmt2(R)} · ${fmt2(1 / R)}`);
        else { ro.set('R', fmt2(R)); ro.set('Rd', fmt2(1 / R)); }
        ro.set('Ec', fmt2(s1.n / R)); ro.set('Em', fmt2(s1.w * R));
        updateChip();   // the chip owns the 'From here' cell; writing text there would remove it
        if (Math.abs(R - ladderR) > 0.002) { ladderR = R; updateLadder(R); }
        let note;
        if (s1.far > 0.5) note = 'From far away the circle is invisible and the tube looks like a line.';
        else if (Math.abs(s1.RlogD) < 0.02) note = 'Self-dual radius. Momentum and winding cost the same, and the tube meets its dual.';
        else if (R > 1) note = 'Large circle: momentum states are light, winding states are heavy. The two rows above hold the same energies.';
        else note = 'Small circle: winding states are now the light ones, exactly as on a circle of radius 1/R.';
        if (!narrow()) ro.note(note);
      }
    }
    let lastPerf = 0;
    let wasNarrow = ctx.width <= 760;
    const PERF = Q.has('perf') ? { n: 0, sum: 0, done: false } : null;

    // ------------------------------------------------------------ the movement instance
    const self = {
      scene, camera,
      post: POST[stage],

      enter() {
        active = true;
        lastPerf = 0;
        labelLayer.style.display = '';
        controls.connect();
        controls.enabled = !rig.fly;
        ctx.setPost(POST[stage]);
        if (stage === 1 && !Q.has('nozoom')) startIntroZoom();
        buildPanels();
        showHint();
        if (qFocus !== null && stage === 3) later(() => focusPoint(qFocus === 'c' ? centerIdx + 1 : clamp(parseInt(qFocus, 10) || 0, 0, LN - 1)), 400);
      },

      exit() {
        active = false;
        controls.enabled = false;
        { const ta = ctx.canvas.style.touchAction; controls.disconnect(); ctx.canvas.style.touchAction = ta; }   // drops its document keydown listener
        grabbing = null;
        ctx.setCursor('');
        stopVoices();
        timers.forEach((id) => clearTimeout(id));
        timers.clear();
        labelLayer.style.display = 'none';
        s3.hover = -1;
        panel = null; ro = null; ladder = null;
      },

      update(dt) {
        // wall-clock catch-up: if the real gap is much longer than the clamped dt, integrate it in small steps
        const now = performance.now();
        let span = dt;
        if (lastPerf > 0) { const gap = (now - lastPerf) / 1000; if (gap > dt + 0.03) span = Math.min(gap, 8); }
        lastPerf = now;
        const frameSpan = span;
        while (span > 1e-6) { const h = Math.min(span, 1 / 30); step(h); span -= h; }
        camera.updateMatrixWorld();   // DOM labels project with this frame's camera, not last frame's

        if (PERF) {   // debug: ?perf logs the steady-state frame time between 3 s and 7 s of movement time
          if (T > 3 && T < 7) { PERF.n++; PERF.sum += dt; }
          else if (T >= 7 && !PERF.done) {
            PERF.done = true;
            const n = s2.n, gh = slotCur.lod ? 12 : (LOW ? Math.round(hiG[n] * 0.6) : hiG[n]);
            const tris = (hero.visible ? n * n * gh * gh * 2 * (slotCur.xray.visible ? 3 : 2) : 0) + (inst.visible ? LN * n * n * loG[n] * loG[n] * 2 : 0);
            console.log(`m3 perf stage ${stage}: ${(PERF.sum / Math.max(1, PERF.n) * 1000).toFixed(1)} ms/frame, pr ${ctx.pixelRatio.toFixed(2)}, ~${Math.round(tris / 1000)}k CY tris`);
          }
        }
        SH.uTime.value = T;
        stars.userData.update(T);
        dust.userData.update(T);
        const R = draw1();
        draw2();
        draw3();

        // keep the alpha slider honest while it drifts (but never fight the user's thumb)
        sliderT += frameSpan;
        if (panel && stage === 2 && s2.hold <= 0 && sliderT > 0.2) { sliderT = 0; panel.set('alpha', Math.round((((s2.alpha * 180) / Math.PI) % 360 + 360) % 360)); }
        readoutT += frameSpan;
        if (readoutT > 0.12) { readoutT = 0; updateReadout(R); }
        audioT += frameSpan;
        if (audioT > 0.05) { audioT = 0; if (active) updateAudio(s1.fM, s1.fW); }
      },

      resize(w, h) {
        const pr = ctx.pixelRatio;
        SH.uRes.value.set(w * pr, h * pr);
        for (let i = 0; i < ribbons.length; i++) ribbons[i].shared.uWidth.value = ribbons[i].baseWidth * pr;
        const aspect = w / Math.max(1, h);
        const mul = aspect < 1 ? Math.min(2.3, Math.pow(1 / aspect, 0.8)) : 1;
        if (Math.abs(mul - fitMul) > 0.01) {
          const k = stageMul(stage, mul, s3.focus >= 0) / stageMul(stage, fitMul, s3.focus >= 0);
          if (!rig.fly) {
            camera.position.sub(driftPrev);
            V.copy(camera.position).sub(controls.target).multiplyScalar(k);
            camera.position.copy(controls.target).add(V);
            driftPrev.set(0, 0, 0);
          } else rig.r1 *= k;
          fitMul = mul;
        }
        // compose: hero slightly right of centre on wide screens. On phones the free band sits between the
        // readout (top) and the controls sheet (bottom), which is close to the vertical centre.
        camera.setViewOffset(w, h, aspect < 1 ? 0 : -w * 0.035, aspect < 1 ? -h * 0.02 : 0, w, h);
        const nw = w <= 760;
        if (active && nw !== wasNarrow) buildReadout();
        wasNarrow = nw;
      },

      onPointerDown(ev) {
        if (stage === 1 && W[1] > 0.5 && hitTube(ev.ndc)) {
          grabbing = 'tube'; grabY = ev.y; grabLog = s1.Rlog;
          controls.enabled = false;
          ctx.setCursor('grabbing');
          return;
        }
        if ((stage === 2 && W[2] > 0.5) || (stage === 3 && s3.focusK > 0.8 && s3.swapTo < 0)) {
          const at = hitLoop(ev);
          if (at >= 0) { pluckLoop(at); grabbing = 'loop'; controls.enabled = false; return; }
        }
        s3.downX = ev.x; s3.downY = ev.y; s3.downT = T;
      },
      onPointerMove(ev) {
        if (grabbing === 'tube') { setRlog(grabLog - (ev.y - grabY) * 0.008); return; }
        if (hintUp > 1 && narrow()) hideHint();   // phones: the mid-screen hint leaves at the first touch
        if (grabbing || ctx.pointer.down) return;
        if (stage === 1) ctx.setCursor(W[1] > 0.5 && hitTube(ev.ndc) ? 'grab' : '');
        else if (stage === 2) ctx.setCursor(hitLoop(ev) >= 0 ? 'point' : '');
        else if (stage === 3 && W[3] > 0.6) {
          const onLoop = s3.focusK > 0.8 && hitLoop(ev) >= 0;
          const h = onLoop ? -1 : pickLattice(ev.ndc);
          const hv = h === s3.heroIdx && s3.focus >= 0 ? -1 : h;
          if (hv !== s3.hover) {
            s3.hover = hv;
            if (hv >= 0 && T - s3.lastHover > 0.09) { s3.lastHover = T; ringFrom(hv, clamp(ev.ndc.x * 0.7, -0.8, 0.8)); }
          }
          ctx.setCursor(hv >= 0 || onLoop ? 'point' : '');
        }
      },
      onPointerUp(ev) {
        if (grabbing) {
          if (grabbing === 'tube') settleR();
          grabbing = null;
          controls.enabled = active && !rig.fly;
          ctx.setCursor('');
          return;
        }
        if (stage === 3 && W[3] > 0.6) {
          const moved = Math.hypot(ev.x - s3.downX, ev.y - s3.downY);
          if (moved < 7 && T - s3.downT < 0.6) {
            const h = pickLattice(ev.ndc);
            if (h >= 0 && h !== s3.focus) focusPoint(h);
            else if (s3.focus >= 0) unfocus();
          }
        }
      },
      onWheel(ev) {
        if (stage !== 1) return;
        s1.intro = -1;
        s1.zoomLog = clamp(s1.zoomLog - ev.deltaY * 0.0016, Math.log(0.006), Math.log(2.4));
        updateChip();
      },
      onKey(ev) {
        const k = ev.key;
        if (stage === 1 && (k === 'z' || k === 'Z') && !ev.repeat) { toggleZoom(); return true; }
        if (stage === 1 && (k === 'ArrowUp' || k === 'ArrowDown' || k === '=' || k === '+' || k === '-')) {
          setRlog(s1.Rlog + (k === 'ArrowUp' || k === '=' || k === '+' ? 0.05 : -0.05));
          return true;
        }
        if (stage === 2) {
          if (!ev.repeat && (k === 'ArrowUp' || k === '=' || k === '+')) { setDegree(s2.n + 1); return true; }
          if (!ev.repeat && (k === 'ArrowDown' || k === '-')) { setDegree(s2.n - 1); return true; }
          if (k === ',' || k === '.') { setAlphaUser(clamp((s2.alpha * 180) / Math.PI + (k === '.' ? 4 : -4), 0, 180)); if (panel) panel.set('alpha', Math.round((s2.base * 180) / Math.PI)); return true; }
        }
        if (stage === 3 && k === 'Escape' && s3.focus >= 0) { unfocus(); return true; }
        return false;
      },

      demo() {
        setDegree(5, true);
        setStage(2);
        later(() => pluckLoop(0.1), 1800);
        later(() => pluckLoop(0.62), 5000);
      },
    };
    // Debug: ?m3test runs a scripted tour through every handler (keys, synthetic pointer events, wheel, stage,
    // degree, pluck, focus) so a headless run surfaces any exception. Harmless when absent.
    if (Q.has('m3test')) {
      const P = (nx, ny) => ({ x: (nx + 1) * 0.5 * ctx.width, y: (1 - ny) * 0.5 * ctx.height, ndc: new THREE.Vector2(nx, ny), dx: 0, dy: 0, button: 0, buttons: 1, pointerType: 'mouse' });
      const K = (key) => self.onKey({ key, repeat: false, type: 'down' });
      const steps = [
        [400, () => { for (let i = 0; i < 30; i++) K('ArrowDown'); }],
        [900, () => { toggleDual(); addQuantum(true); addQuantum(false); addQuantum(false); }],
        [1500, () => { self.onPointerDown(P(0, 0.05)); self.onPointerMove(P(0, -0.2)); self.onPointerUp(P(0, -0.2)); }],
        [1900, () => { self.onWheel({ deltaY: 400 }); self.onWheel({ deltaY: -900 }); toggleZoom(); toggleZoom(); console.log(`m3test zoom ${s1.zoomLog.toFixed(2)} intro ${s1.intro.toFixed(2)} T ${T.toFixed(2)}`); }],
        [2300, () => { toggleDual(); for (let i = 0; i < 50; i++) K('ArrowUp'); }],
        [2800, () => setStage(2)],
        [4300, () => { K('ArrowDown'); K('ArrowDown'); }],
        [5000, () => { K('ArrowUp'); K('ArrowUp'); K('ArrowUp'); pluckLoop(0.3); }],
        [5600, () => { setAlphaUser(120); K(','); K('.'); self.onPointerMove(P(0.05, 0.05)); self.onPointerDown(P(0.05, 0.05)); self.onPointerUp(P(0.05, 0.05)); }],
        [6200, () => setStage(3)],
        [8200, () => { self.onPointerMove(P(0.02, 0.02)); self.onPointerMove(P(-0.3, 0.2)); }],
        [8600, () => { const e = P(-0.3, 0.2); self.onPointerDown(e); self.onPointerUp(e); }],
        [10600, () => K('Escape')],
        [11400, () => lookRandom()],
        [12600, () => { self.resize(ctx.width, ctx.height); setDegree(5, true); setStage(1); }],
        [14200, () => console.log(`m3test done: stage ${stage}, n ${s2.n}, R ${Math.pow(2, s1.Rlog).toFixed(2)}, nan ${!Number.isFinite(camera.position.x + heroPos.x + s1.RlogD)}, zoom ${s1.zoomLog.toFixed(2)}/${s1.zoomD.toFixed(2)} intro ${s1.intro.toFixed(2)} T ${T.toFixed(2)}`)],
      ];
      for (const [ms, fn] of steps) later(fn, ms);
    }
    return self;
  },
};
