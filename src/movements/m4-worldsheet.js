// IV. Worldsheets — a point draws a line through spacetime, a string draws a surface.
//
// Scene graph (time runs up the +y axis):
//   background sphere (faint nebula, noise per vertex)  ·  starfield  ·  upward-drifting "time dust"  ·  bokeh
//     ·  one large backlight behind the crotch
//   worldsheet: a box-bounded raymarched SDF drawn as a soap film (the Nambu–Goto action is the sheet's area):
//     clear face-on, thin-film interference swirls at grazing angles, a gold key rim and a cyan back rim,
//     contours of lab time that fill in behind the slice, and the gold "now" contour (white-hot core, gold halo)
//     with an afterglow of recent slices. A gold wave spreads from the join whenever the topology changes.
//   the observer's "now": a tilted disc (plane t = vx·x + vz·z + T) with an instrument rim; a "now" tick rides
//     the time axis where the slice crosses it (relativity of simultaneity at a glance)
//   markers (magenta): where THIS observer sees the loops first touch (CPU saddle search), plus the locus of that
//     point for observers from −0.8c to +0.8c, drawn as a smooth magenta curve with −0.8c / 0 / +0.8c ticks.
//     Dragging a marker along the locus picks the observer.
//   particles mode: two cyan worldlines meeting a warm outgoing one at an ember vertex, with an "∞", sparks and
//     an anamorphic flare that grows as the slice nears the vertex
//
// Input: drag the sheet up/down = time, sideways = observer (direction locks after 6 px), shift-drag = tilt in
//   both axes, drag the marker = observer (on touch: sideways only; vertical starts a time drag), drag elsewhere =
//   orbit, two fingers = pinch zoom.
// Keys: ↑/↓ time (shift: ×4), , and . tilt, 1/2/3 mode, space sweep, 0 rest frame.
// Debug params (harmless when absent): slice, tilt, mode, nosheet, m4log, m4sweep, m4dragtest.
//
// The CPU and GPU share one shape definition (shapeAt / SHAPE_GLSL). Keep them identical.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { palette, hdr, glowTexture, makeStarfield, GLSL, clamp, lerp, damp } from '../fx.js';

const V_MAX = 0.8;                 // fastest observer offered, in units of c (tilt = atan(v))
const T_MAX = 2.6;                 // slice range (lab time at x = 0), in string units ℓs/c
const BOX = { x: 2.05, y: 3.7, z: 1.25 };   // the sheet fades out by |y| = 3.6, so nothing above that is marched
const DISK_R = 2.45;
const AXIS = { x: -3.3, z: -1.5 };   // where the quiet time axis stands
const VERTEX = new THREE.Vector3(0.0, -0.3, 0.0);   // particles mode: where the worldlines meet
const LOC_N = 33;                  // observers sampled for the locus of interaction points

// ---- the worldsheet as a signed distance function ------------------------------------
// At height y the cross-section is an interpolation (by m) between two leg discs (two closed strings) and one
// waist disc (one string): d = mix(dLegs, dWaist, m(y)). Interpolating distance fields gives the crotch a round
// arch, so the saddle has real curvature and the "first touch" visibly slides as the observer changes.
// W morphs between the pair of pants (0) and the handle (1). The legs move slower than light (|dx/dy| < 0.2).
// This JS version must stay identical to SHAPE_GLSL below.
const K_LEG = 0.45, R_WAIST = 0.85, W_SOFT = 0.15;   // waist field is sqrt(r² + s²): no cone tip, so the arch is one dome
const R_WAIST_S = Math.sqrt(R_WAIST * R_WAIST + W_SOFT);
const sstep = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
const SH = new Float64Array(7); // cAx, cAz, cBx, cBz, rA, rB, m
function shapeAt(y, W) {
  const lx = 0.16 * Math.sin(0.5 * y + 0.7), lz = 0.10 * Math.sin(0.37 * y - 0.4);
  const flare = 0.07 * Math.max(0, -y - 1.2) * (1 - W);
  SH[0] = lerp(-1.18, -0.95, W) - flare + lx; SH[1] = lerp(0.2, 0.12, W) + lz;
  SH[2] = lerp(1.08, 0.95, W) + flare + lx; SH[3] = lerp(-0.16, -0.12, W) + lz;
  SH[4] = lerp(0.5, 0.42, W); SH[5] = lerp(0.4, 0.36, W);
  const bump = sstep(-2.4, -0.8, y) * (1 - sstep(0.8, 2.4, y));
  SH[6] = lerp(sstep(-1.4, 1.0, y), 1 - bump, W);
}
function smin(a, b, k) { const h = Math.min(1, Math.max(0, 0.5 + 0.5 * (b - a) / k)); return b + (a - b) * h - k * h * (1 - h); }
function sdf(x, y, z, W) {
  shapeAt(y, W);
  const ax = x - SH[0], az = z - SH[1], bx = x - SH[2], bz = z - SH[3];
  const dL = smin(Math.sqrt(ax * ax + az * az) - SH[4], Math.sqrt(bx * bx + bz * bz) - SH[5], K_LEG);
  const wx = x - 0.5 * (SH[0] + SH[2]), wz = z - 0.5 * (SH[1] + SH[3]);
  return dL + (Math.sqrt(wx * wx + wz * wz + W_SOFT) - R_WAIST_S - dL) * SH[6];
}

// ---- slice analysis on the CPU ---------------------------------------------------------
// A grid over (x, z); each cell is lifted onto the observer's plane y = T + vx·x + vz·z and the SDF sampled.
// Inside cells (sdf < 0) are flood-filled into connected components: one component = one string.
const GX0 = -2.4, GX1 = 2.4, GZ0 = -1.6, GZ1 = 1.6;
class SliceGrid {
  constructor(nx, nz) {
    this.nx = nx; this.nz = nz; const n = nx * nz;
    this.val = new Float32Array(n); this.lab = new Int32Array(n); this.stack = new Int32Array(n);
    this.area = new Int32Array(256); this.cx = new Float64Array(256);
    this.count = 0; this.big0 = -1; this.big1 = -1;
    this.window(GX0, GX1, GZ0, GZ1);
  }
  window(x0, x1, z0, z1) { this.x0 = x0; this.z0 = z0; this.dx = (x1 - x0) / (this.nx - 1); this.dz = (z1 - z0) / (this.nz - 1); }
  fill(vx, vz, T, W) {
    const nx = this.nx, nz = this.nz, val = this.val, dx = this.dx, dz = this.dz, x0 = this.x0, z0 = this.z0;
    for (let j = 0, k = 0; j < nz; j++) {
      const z = z0 + j * dz;
      for (let i = 0; i < nx; i++, k++) { const x = x0 + i * dx; val[k] = sdf(x, T + vx * x + vz * z, z, W); }
    }
  }
  label() {
    const nx = this.nx, nz = this.nz, n = nx * nz, val = this.val, lab = this.lab, st = this.stack;
    lab.fill(0);
    let c = 0;
    for (let s = 0; s < n; s++) {
      if (val[s] >= 0 || lab[s] !== 0 || c >= 255) continue;
      c++; let sp = 0, a = 0, ax = 0;
      st[sp++] = s; lab[s] = c;
      while (sp > 0) {
        const q = st[--sp], qi = q % nx, qj = (q - qi) / nx;
        a++; ax += qi;
        if (qi > 0 && val[q - 1] < 0 && lab[q - 1] === 0) { lab[q - 1] = c; st[sp++] = q - 1; }
        if (qi < nx - 1 && val[q + 1] < 0 && lab[q + 1] === 0) { lab[q + 1] = c; st[sp++] = q + 1; }
        if (qj > 0 && val[q - nx] < 0 && lab[q - nx] === 0) { lab[q - nx] = c; st[sp++] = q - nx; }
        if (qj < nz - 1 && val[q + nx] < 0 && lab[q + nx] === 0) { lab[q + nx] = c; st[sp++] = q + nx; }
      }
      this.area[c] = a; this.cx[c] = this.x0 + (ax / a) * this.dx;
    }
    let count = 0, b0 = -1, b1 = -1;
    for (let l = 1; l <= c; l++) {
      if (this.area[l] < 2) continue; // specks at the grid edge are not strings
      count++;
      if (b0 < 0 || this.area[l] > this.area[b0]) { b1 = b0; b0 = l; } else if (b1 < 0 || this.area[l] > this.area[b1]) b1 = l;
    }
    this.count = count; this.big0 = b0; this.big1 = b1;
    return count;
  }
}
const G = new SliceGrid(64, 48);       // fine grid: counting, bisection, lens drawing
const GC = new SliceGrid(32, 24);      // coarse grid: scanning for topology changes
const GL = new SliceGrid(40, 40);      // local window around a bridge, for sub-cell precision
const BA = new Int32Array(64 * 48), BB = new Int32Array(64 * 48);
const LENS_VAL = new Float32Array(64 * 48);   // the current slice, kept apart from G (which the saddle search reuses)

/** Closest pair of boundary cells between components la and lb of grid g; midpoint (x, z) into out[0..1]. */
function closestBridge(g, la, lb, out) {
  const nx = g.nx, nz = g.nz, lab = g.lab;
  let na = 0, nb = 0;
  for (let q = 0; q < nx * nz; q++) {
    const l = lab[q]; if (l !== la && l !== lb) continue;
    const i = q % nx, j = (q - i) / nx;
    const edge = i === 0 || i === nx - 1 || j === 0 || j === nz - 1 || lab[q - 1] !== l || lab[q + 1] !== l || lab[q - nx] !== l || lab[q + nx] !== l;
    if (!edge) continue;
    if (l === la) BA[na++] = q; else BB[nb++] = q;
  }
  if (!na || !nb) return false;
  let best = Infinity, qa = 0, qb = 0;
  for (let a = 0; a < na; a++) {
    const ia = BA[a] % nx, ja = (BA[a] - ia) / nx;
    for (let b = 0; b < nb; b++) {
      const ib = BB[b] % nx, jb = (BB[b] - ib) / nx;
      const ex = (ia - ib) * g.dx, ez = (ja - jb) * g.dz, d2 = ex * ex + ez * ez;
      if (d2 < best) { best = d2; qa = BA[a]; qb = BB[b]; }
    }
  }
  const ia = qa % nx, ja = (qa - ia) / nx, ib = qb % nx, jb = (qb - ib) / nx;
  out[0] = g.x0 + 0.5 * (ia + ib) * g.dx; out[1] = g.z0 + 0.5 * (ja + jb) * g.dz;
  return true;
}

/**
 * The places where, for the observer (vx, vz), the cross-section changes topology: the saddle points of the
 * observer's time function on the surface. Scan the plane offset T on a coarse grid, bisect each change of
 * the component count on the fine grid, take the bridge between the two loops at the critical T and
 * project it onto the surface. Writes up to two points (x, y, z) into out; returns how many.
 */
const PT = new Float64Array(2);
function solveSaddles(vx, vz, W, out) {
  let n = 0, prevC = -1, prevT = 0;
  const NS = 35, TA = -3.4, TB = 3.4;
  for (let k = 0; k < NS; k++) {
    const T = TA + (TB - TA) * k / (NS - 1);
    GC.fill(vx, vz, T, W);
    const c = GC.label();
    if (k > 0 && c !== prevC && n < 2) {
      let lo = prevT - 0.15, hi = T + 0.15;   // widened: the fine grid's transition can sit just outside the coarse step
      G.fill(vx, vz, lo, W); const cLo = G.label();
      G.fill(vx, vz, hi, W); const cHi = G.label();
      if (cLo !== cHi && Math.max(cLo, cHi) >= 2) {
        for (let it = 0; it < 13; it++) {
          const mid = 0.5 * (lo + hi);
          G.fill(vx, vz, mid, W);
          if (G.label() === cLo) lo = mid; else hi = mid;
        }
        const Tm = cLo > cHi ? lo : hi;
        G.fill(vx, vz, Tm, W); G.label();
        if (G.big1 > 0 && closestBridge(G, G.big0, G.big1, PT)) {
          // refine on a fine local window: bisect again there, then take the bridge
          const two = cLo > cHi ? lo : hi, one = cLo > cHi ? hi : lo;
          GL.window(PT[0] - 0.36, PT[0] + 0.36, PT[1] - 0.36, PT[1] + 0.36);
          const sg = two > one ? 1 : -1;
          let a2 = two + 0.03 * sg, a1 = one - 0.03 * sg;   // widen: the fine grid's critical time differs slightly
          GL.fill(vx, vz, a2, W); const c2 = GL.label();
          GL.fill(vx, vz, a1, W); const c1 = GL.label();
          if (c2 >= 2 && c1 === 1) {
            for (let it = 0; it < 14; it++) {
              const mid = 0.5 * (a2 + a1);
              GL.fill(vx, vz, mid, W);
              if (GL.label() >= 2) a2 = mid; else a1 = mid;
            }
            GL.fill(vx, vz, a2, W); GL.label();
            if (GL.big1 > 0) closestBridge(GL, GL.big0, GL.big1, PT);
          }
          toSurface(PT[0], Tm + vx * PT[0] + vz * PT[1], PT[1], W, out, n * 3); n++;
        }
      }
    }
    prevC = c; prevT = T;
  }
  return n;
}
/** Newton steps along the SDF gradient: the nearest point on the surface to (px, py, pz), into out[o..o+2]. */
function toSurface(px, py, pz, W, out, o) {
  for (let r = 0; r < 4; r++) {
    const d = sdf(px, py, pz, W), e = 1e-3;
    const gx = (sdf(px + e, py, pz, W) - sdf(px - e, py, pz, W)) / (2 * e);
    const gy = (sdf(px, py + e, pz, W) - sdf(px, py - e, pz, W)) / (2 * e);
    const gz = (sdf(px, py, pz + e, W) - sdf(px, py, pz - e, W)) / (2 * e);
    const g2 = gx * gx + gy * gy + gz * gz || 1;
    px -= d * gx / g2; py -= d * gy / g2; pz -= d * gz / g2;
  }
  out[o] = px; out[o + 1] = py; out[o + 2] = pz;
}

// The grid answer is quantised to a few hundredths. The locus of interaction points is a smooth function of the
// observer's speed, so once all observers along x are sampled a least-squares cubic per coordinate (with one
// outlier-rejection pass) gives a smooth curve; checked offline: every fitted point lies on the surface
// (|sdf| < 0.005) and the slice through it changes the string count. Runs once per mode, so it may allocate.
function fitCubic(vs, ys, w, n, c) {
  const M = new Float64Array(20), P = [0, 0, 0, 0];
  for (let i = 0; i < n; i++) {
    if (!w[i]) continue;
    const v = vs[i]; P[0] = 1; P[1] = v; P[2] = v * v; P[3] = v * v * v;
    for (let r = 0; r < 4; r++) { for (let k = 0; k < 4; k++) M[r * 5 + k] += P[r] * P[k]; M[r * 5 + 4] += P[r] * ys[i]; }
  }
  for (let col = 0; col < 4; col++) {
    let piv = col;
    for (let r = col + 1; r < 4; r++) if (Math.abs(M[r * 5 + col]) > Math.abs(M[piv * 5 + col])) piv = r;
    if (piv !== col) for (let k = 0; k < 5; k++) { const t = M[col * 5 + k]; M[col * 5 + k] = M[piv * 5 + k]; M[piv * 5 + k] = t; }
    const d = M[col * 5 + col] || 1e-12;
    for (let r = 0; r < 4; r++) if (r !== col) { const f = M[r * 5 + col] / d; for (let k = col; k < 5; k++) M[r * 5 + k] -= f * M[col * 5 + k]; }
  }
  for (let r = 0; r < 4; r++) c[r] = M[r * 5 + 4] / (M[r * 5 + r] || 1e-12);
}
const poly3 = (c, v) => c[0] + v * (c[1] + v * (c[2] + v * c[3]));
/** Fit one branch of a locus: samples (vs, pts) → { cx, cy, cz } (cubic coefficients in v). */
function fitBranch(vs, pts) {
  const n = vs.length, w = new Uint8Array(n).fill(1), xs = new Float64Array(n), ys = new Float64Array(n), zs = new Float64Array(n);
  for (let i = 0; i < n; i++) { xs[i] = pts[i * 3]; ys[i] = pts[i * 3 + 1]; zs[i] = pts[i * 3 + 2]; }
  const cx = new Float64Array(4), cy = new Float64Array(4), cz = new Float64Array(4), res = new Float64Array(n);
  for (let pass = 0; pass < 2; pass++) {
    fitCubic(vs, xs, w, n, cx); fitCubic(vs, ys, w, n, cy); fitCubic(vs, zs, w, n, cz);
    if (pass) break;
    for (let i = 0; i < n; i++) res[i] = Math.hypot(poly3(cx, vs[i]) - xs[i], poly3(cy, vs[i]) - ys[i], poly3(cz, vs[i]) - zs[i]);
    const med = Float64Array.from(res).sort()[n >> 1];
    for (let i = 0; i < n; i++) w[i] = res[i] < Math.max(0.06, 3 * med) ? 1 : 0;
  }
  return { cx, cy, cz };
}
/** A point of a fitted locus branch for observer speed v, projected onto the surface. */
function evalBranch(f, v, W, out, o) { toSurface(poly3(f.cx, v), poly3(f.cy, v), poly3(f.cz, v), W, out, o); }

// Pitch: a closed string's fundamental falls as its loop grows (f ∝ 1/L). Snapped into D minor pentatonic.
const PENTA_STEPS = [0, 3, 5, 7, 10, 12];
function semisForLength(L) {
  const raw = clamp(24 + 12 * Math.log2(3.0 / Math.max(0.3, L)), 8, 38);
  const o = Math.floor(raw / 12), r = raw - 12 * o;
  let best = 0, bd = 99;
  for (let i = 0; i < PENTA_STEPS.length; i++) { const d = Math.abs(r - PENTA_STEPS[i]); if (d < bd) { bd = d; best = PENTA_STEPS[i]; } }
  return 12 * o + best;
}
const STRING_AMPS = [1, 0.42, 0.2, 0.1, 0.05, 0.025];
const BUZZ_AMPS = [0.6, 0.42, 0.3, 0.26, 0.2, 0.17, 0.14, 0.12, 0.1, 0.09];

const fmtS = (v, d = 2) => (v < 0 ? '−' : '+') + Math.abs(v).toFixed(d);

// ---- shaders -----------------------------------------------------------------------------
const SHAPE_GLSL = /* glsl */`
  uniform float uW;
  #define K_LEG ${K_LEG.toFixed(3)}
  #define R_WAIST ${R_WAIST.toFixed(3)}
  #define R_WAIST_S ${R_WAIST_S.toFixed(5)}
  #define W_SOFT ${W_SOFT.toFixed(3)}
  void shape(float y, out vec2 cA, out vec2 cB, out vec2 rr, out float m) {
    vec2 lean = vec2(0.16 * sin(0.5 * y + 0.7), 0.10 * sin(0.37 * y - 0.4));
    float flare = 0.07 * max(0.0, -y - 1.2) * (1.0 - uW);
    cA = vec2(mix(-1.18, -0.95, uW) - flare, mix(0.2, 0.12, uW)) + lean;
    cB = vec2(mix(1.08, 0.95, uW) + flare, mix(-0.16, -0.12, uW)) + lean;
    rr = vec2(mix(0.5, 0.42, uW), mix(0.4, 0.36, uW));
    float bump = smoothstep(-2.4, -0.8, y) * (1.0 - smoothstep(0.8, 2.4, y));
    m = mix(smoothstep(-1.4, 1.0, y), 1.0 - bump, uW);
  }
  float smin(float a, float b, float k) { float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0); return mix(b, a, h) - k * h * (1.0 - h); }
  float sdf(vec3 p) {
    vec2 cA, cB, rr; float m; shape(p.y, cA, cB, rr, m);
    float dL = smin(length(p.xz - cA) - rr.x, length(p.xz - cB) - rr.y, K_LEG);
    vec2 q = p.xz - 0.5 * (cA + cB);
    return mix(dL, sqrt(dot(q, q) + W_SOFT) - R_WAIST_S, m);
  }
`;

const SHEET_VERT = /* glsl */`
  varying vec3 vWorld;
  void main() { vec4 w = modelMatrix * vec4(position, 1.0); vWorld = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }
`;

const TRAILS = 7;   // afterglow: the gold contour at the last few slice positions, each with the tilt it had then
const SHEET_FRAG = /* glsl */`
  uniform float uTime, uSheet, uPix, uSliceOn, uFlash, uTrOn, uJoinT;
  uniform vec3 uSlice;              // (vx, vz, T): the observer's now is y = vx x + vz z + T
  uniform vec3 uBoxMin, uBoxMax;
  uniform vec3 uJoinP;              // where the topology last changed: the join wave spreads from here
  uniform vec3 uGold, uKey, uBack, uInner, uLine, uGlow;
  uniform vec4 uTrP[${TRAILS}];     // afterglow planes: unit normal (xyz) and offset (w)
  uniform float uTrA[${TRAILS}];
  varying vec3 vWorld;
  ${GLSL.noise}
  ${GLSL.iridescence}
  ${SHAPE_GLSL}
  #define DT 0.5
  vec3 calcNormal(vec3 p) {
    const vec2 e = vec2(1.0, -1.0) * 0.0025;
    return normalize(e.xyy * sdf(p + e.xyy) + e.yyx * sdf(p + e.yyx) + e.yxy * sdf(p + e.yxy) + e.xxx * sdf(p + e.xxx));
  }
  vec2 boxHit(vec3 ro, vec3 rd) {
    vec3 inv = 1.0 / rd;
    vec3 t0 = (uBoxMin - ro) * inv, t1 = (uBoxMax - ro) * inv;
    vec3 a = min(t0, t1), b = max(t0, t1);
    return vec2(max(max(a.x, a.y), a.z), min(min(b.x, b.y), b.z));
  }
  // both ends dissolve into mist inside the frame
  float endFade(float y) { return 1.0 - smoothstep(2.2, 3.6, abs(y)); }
  void main() {
    if (uSheet < 0.004) discard;
    vec3 ro = cameraPosition, rd = normalize(vWorld - ro);
    vec2 tb = boxHit(ro, rd);
    float t = max(tb.x, 0.0), tEnd = tb.y;
    if (tEnd <= t) discard;
    float sl = inversesqrt(1.0 + dot(uSlice.xy, uSlice.xy));
    vec3 sN = vec3(-uSlice.x, 1.0, -uSlice.y) * sl;
    vec3 L = normalize(vec3(-0.72, 0.62, 0.3));          // key light from the upper left
    vec3 H = normalize(L - rd);
    float age = uTime - uJoinT;
    float joinR = age * 1.8, joinA = age >= 0.0 && age < 3.0 ? 1.8 * exp(-age * 1.6) : 0.0;
    vec3 acc = vec3(0.0);
    float trans = 1.0, glow = 0.0;
    int layers = 0;
    bool armed = true;
    for (int i = 0; i < MAX_STEPS; i++) {
      if (t > tEnd) break;
      vec3 p = ro + rd * t;
      float d = sdf(p), ad = abs(d), pix = t * uPix;
      if (armed && ad < 0.0012 + pix * 0.4) {
        vec3 n = calcNormal(p);
        float vd = dot(n, -rd), f = 1.0 - abs(vd);
        float f2 = f * f, f4 = f2 * f2, fres = 0.04 + 0.96 * f4 * f;
        float endF = endFade(p.y);
        float g = dot(sN, p) - uSlice.z * sl;              // signed distance to the observer's now
        float past = 1.0 - smoothstep(-0.05, 0.05, g);
        float front = layers == 0 ? 1.0 : 0.3;             // back walls stay quiet, so the lines never form a cage
        // soap film: thin-film interference whose thickness drifts slowly upward. Face-on it is nearly clear.
        float film = 0.15 + 1.6 * f + 0.08 * p.y
          + 0.6 * snoise(vec3(p.xz * 1.1, p.y * 0.5 - uTime * 0.06))
          + 0.25 * snoise(p * 2.7 + vec3(0.0, -uTime * 0.1, 0.0));
        vec3 irid = iridescence(film);
        irid = max(mix(vec3(dot(irid, vec3(0.333))), irid, 1.9), 0.0);
        vec3 c = irid * (0.04 + 2.2 * fres);
        float nl = dot(n, L);
        c += uKey * f4 * max(nl, 0.0) + uBack * f4 * max(-nl, 0.0);     // gold key rim, cyan back rim
        if (vd < 0.0) c += uInner;                           // looking at the inside wall
        // the observer's past is solid glass; their future is a cool, faint ghost
        c = mix(c * vec3(0.42, 0.55, 1.0) * 0.55, c, past);
        float a = (0.02 + 0.6 * fres) * mix(0.42, 1.0, past) * endF;
        acc += trans * a * c;
        // a small specular glint of the key light (reflection: not scaled by the film's opacity)
        vec3 nf = vd < 0.0 ? -n : n;
        vec3 e = vec3(1.0, 0.93, 0.85) * pow(max(dot(nf, H), 0.0), 90.0) * 0.32 * front * mix(0.35, 1.0, past);
        // contours of constant lab time, only in this observer's past: the slice leaves a map of time behind it
        float k = floor(p.y / DT + 0.5), yd = abs(p.y - k * DT);
        float lw = pix / max(length(n.xz), 0.25);
        float tl = 1.0 - smoothstep(lw * 0.5, lw * 1.5, yd);
        float maj = 1.0 - step(0.5, mod(k, 2.0));
        e += uLine * tl * mix(0.02, 0.10, maj) * front * past;
        // the gold contour: the strings as they exist at this observer's instant.
        // sinA = |slope of g along the surface|; a band |g| < w*sinA has the same width on the surface everywhere,
        // so the line stays one crisp thread even where the slice grazes the saddle.
        float sinA = length(cross(n, sN));
        float sa = max(sinA, 0.22);
        float gw = pix * 1.35 * sa;
        float core = exp(-g * g / (gw * gw));
        float halo = exp(-abs(g) / (0.008 + 0.022 * sa)) * smoothstep(0.05, 0.45, sinA);
        float wave = 0.72 + 0.28 * sin(3.1 * p.x - 2.3 * p.z + 1.5 * p.y - uTime * 2.2);
        // seen edge-on, the whole loop collapses onto one screen line and its layers stack: dim it there
        float edgeOn = mix(0.4, 1.0, smoothstep(0.04, 0.32, abs(dot(sN, rd))));
        // white only in the one-pixel centre; the bloom then glows gold around a white-hot thread
        e += ((uGold * 2.2 * wave + vec3(0.14)) * core + uGold * halo * (0.5 + 0.9 * uFlash)) * uSliceOn * edgeOn;
        // afterglow: where the slice was a moment ago (skipped entirely once it has faded)
        if (uTrOn > 0.5) {
          float tr = 0.0;
          for (int j = 0; j < ${TRAILS}; j++) {
            float gj = dot(uTrP[j].xyz, p) - uTrP[j].w;
            tr += uTrA[j] * exp(-gj * gj / (gw * gw * 1.8));
          }
          e += uGold * tr * 0.45 * uSliceOn;
        }
        // the join wave: one ring of gold light spreading over the membrane from where two strings became one
        if (joinA > 0.002) { float q = (length(p - uJoinP) - joinR) / 0.05; e += uGold * joinA * exp(-q * q); }
        acc += trans * e * endF;
        trans *= 1.0 - a;
        layers++;
        if (layers >= MAX_LAYERS || trans < 0.04) break;
        armed = false; t += 0.03;
        continue;
      }
      if (!armed && ad > 0.015) armed = true;
      // far from the membrane the field is close to a true distance, so the march takes longer strides there
      float st = armed ? (ad > 0.25 ? ad * 0.85 : max(ad * 0.7, 0.002)) : max(ad, 0.012);
      // soft light scattered just outside the membrane: rays that graze it collect the most
      glow += trans * exp(-ad * 9.0) * st * endFade(p.y);
      t += st;
    }
    acc += uGlow * glow;
    gl_FragColor = vec4(acc * uSheet, (1.0 - trans) * uSheet);
  }
`;

const DISK_VERT = /* glsl */`
  varying vec2 vL;
  void main() { vL = position.xz; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
`;
const DISK_FRAG = /* glsl */`
  uniform float uR, uA, uTime;
  uniform vec3 uGold, uTint;
  varying vec2 vL;
  void main() {
    float r = length(vL), fw = max(fwidth(r), 1e-4);
    float inside = 1.0 - smoothstep(uR - fw, uR, r);
    float fill = (0.006 + 0.07 * pow(r / uR, 4.0)) * inside;
    vec2 gq = vL / 0.5;
    vec2 gd = abs(fract(gq + 0.5) - 0.5) / max(fwidth(gq), vec2(1e-4));
    float grid = (1.0 - min(min(gd.x, gd.y), 1.0)) * 0.05 * inside * (1.0 - smoothstep(0.35 * uR, uR, r));
    float ring = exp(-pow((r - uR * 0.99) / (fw * 1.3), 2.0));
    float ring2 = exp(-pow((r - uR * 0.93) / (fw * 1.0), 2.0)) * 0.35;
    float ang = atan(vL.y, vL.x);
    float tk = abs(fract(ang / 6.2831853 * 96.0 + 0.5) - 0.5) * 6.2831853 / 96.0 * r;
    float tkMaj = abs(fract(ang / 6.2831853 * 12.0 + 0.5) - 0.5) * 6.2831853 / 12.0 * r;
    float ticks = (1.0 - smoothstep(fw * 0.5, fw * 1.5, tk)) * step(uR * 0.955, r) * step(r, uR * 0.985);
    ticks += (1.0 - smoothstep(fw * 0.6, fw * 1.8, tkMaj)) * step(uR * 0.93, r) * step(r, uR * 0.985);
    float shimmer = 0.85 + 0.15 * sin(ang * 3.0 - uTime * 0.7);
    vec3 col = uTint * fill + uGold * (grid + (ring * 0.28 + ring2 * 0.15 + ticks * 0.18) * shimmer);
    gl_FragColor = vec4(col * uA, 1.0);
  }
`;

const LINE_VERT = /* glsl */`
  varying vec3 vN; varying vec3 vV; varying float vY;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vN = normalize(normalMatrix * normal); vV = normalize(-mv.xyz); vY = position.y;
    gl_Position = projectionMatrix * mv;
  }
`;
const LINE_FRAG = /* glsl */`
  uniform vec3 uColor; uniform float uA, uPow;
  varying vec3 vN; varying vec3 vV; varying float vY;
  void main() {
    float k = abs(dot(normalize(vN), normalize(vV)));
    float fade = 1.0 - smoothstep(2.9, 4.1, abs(vY));
    gl_FragColor = vec4(uColor * pow(k, uPow) * uA * fade, 1.0);
  }
`;

const DUST_VERT = /* glsl */`
  attribute float seed;
  uniform float uTime, uScale, uSize, uSpeed;
  varying float vA; varying float vT;
  void main() {
    vec3 p = position;
    vT = step(0.62, fract(seed * 13.7));
    p.y = mod(p.y + uTime * uSpeed * (0.6 + 0.8 * seed) + 7.0, 14.0) - 7.0;   // time flows upward
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    vA = (1.0 - smoothstep(4.5, 7.0, abs(p.y))) * (0.3 + 0.7 * fract(seed * 7.7));
    gl_PointSize = clamp(uScale * uSize * (0.6 + 1.4 * fract(seed * 3.3)) / max(0.5, -mv.z), 1.0, uSize * 0.9);
    gl_Position = projectionMatrix * mv;
  }
`;
const DUST_FRAG = /* glsl */`
  uniform vec3 uColor, uTint; uniform float uSoft;
  varying float vA; varying float vT;
  void main() {
    float d = length(gl_PointCoord - 0.5) * 2.0;
    float a = mix(1.0 - smoothstep(0.0, 1.0, d), (1.0 - smoothstep(0.75, 1.0, d)) * (0.45 + 0.55 * smoothstep(0.55, 0.95, d)), uSoft);
    gl_FragColor = vec4(mix(uColor, uTint, vT) * a * vA, 1.0);
  }
`;

const SPARK_VERT = /* glsl */`
  attribute vec3 dir; attribute float seed;
  uniform float uTime, uScale, uHeat; uniform vec3 uV;
  varying float vA;
  void main() {
    float life = fract(uTime * (0.3 + 0.55 * seed) / 1.4 + seed * 17.0);
    vec3 p = uV + dir * (0.03 + pow(life, 0.55) * (0.35 + 0.9 * fract(seed * 5.1)) * (0.8 + 0.6 * uHeat));
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    vA = (1.0 - life) * (1.0 - life);
    gl_PointSize = clamp(uScale * 46.0 / max(0.5, -mv.z), 1.0, 9.0);
    gl_Position = projectionMatrix * mv;
  }
`;
const SPARK_FRAG = /* glsl */`
  uniform vec3 uColor; uniform float uA;
  varying float vA;
  void main() { float d = length(gl_PointCoord - 0.5) * 2.0; gl_FragColor = vec4(uColor * (1.0 - smoothstep(0.0, 1.0, d)) * vA * uA, 1.0); }
`;

const LOCUS_VERT = /* glsl */`
  uniform float uScale;
  void main() { vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_PointSize = clamp(uScale * 70.0 / max(0.5, -mv.z), 3.5, 9.0); gl_Position = projectionMatrix * mv; }
`;
const LOCUS_FRAG = /* glsl */`
  uniform vec3 uColor; uniform float uA;
  void main() { float d = length(gl_PointCoord - 0.5) * 2.0; float a = exp(-d * d * 7.0) + 0.6 * (1.0 - smoothstep(0.18, 0.32, d)); gl_FragColor = vec4(uColor * a * uA, 1.0); }
`;

// The nebula is low-frequency, so its noise is evaluated per vertex of a dense sphere, not per pixel.
const BG_VERT = /* glsl */`
  uniform float uTime;
  varying vec3 vDir; varying float vN;
  ${GLSL.noise}
  void main() {
    vDir = position;
    vec3 d = normalize(position);
    vN = snoise(d * 1.8 + vec3(0.0, uTime * 0.006, 0.0)) * 0.6 + snoise(d * 4.3 - vec3(uTime * 0.004)) * 0.4;
    vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0); gl_Position = p.xyww;
  }
`;
const BG_FRAG = /* glsl */`
  uniform vec3 uDeep, uViolet, uCyan;
  varying vec3 vDir; varying float vN;
  void main() {
    vec3 d = normalize(vDir);
    float hz = 1.0 - abs(d.y);
    vec3 col = uDeep * (0.35 + 0.65 * hz * hz);
    col += uViolet * smoothstep(0.05, 0.9, vN) * 0.07;
    col += uCyan * pow(max(0.0, dot(d, normalize(vec3(0.35, 0.25, -1.0)))), 5.0) * 0.05;
    gl_FragColor = vec4(col, 1.0);
  }
`;

// ---- small procedural textures and sprites ---------------------------------------------------
function ringTexture() {
  const s = 128, c = document.createElement('canvas'); c.width = c.height = s;
  const g = c.getContext('2d');
  g.strokeStyle = '#fff'; g.shadowColor = '#fff';
  g.shadowBlur = 10; g.lineWidth = 3; g.beginPath(); g.arc(s / 2, s / 2, 46, 0, Math.PI * 2); g.stroke();
  g.shadowBlur = 0; g.lineWidth = 1.5; g.stroke();
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}
function textSprite(text, { font = 'italic 400 104px "Instrument Serif", Georgia, serif', color, height = 0.3, w = 512, h = 160, glow = 0 } = {}) {
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const g = c.getContext('2d');
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
  const draw = () => {
    g.clearRect(0, 0, w, h); g.font = font; g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillStyle = '#fff';
    if (glow) { g.shadowColor = 'rgba(255,255,255,0.7)'; g.shadowBlur = glow; g.fillText(text, w / 2, h / 2); g.shadowBlur = 0; }
    g.fillText(text, w / 2, h / 2); tex.needsUpdate = true;
  };
  draw();
  if (document.fonts && document.fonts.load) document.fonts.load(font).then(draw).catch(() => {});
  const mat = new THREE.SpriteMaterial({ map: tex, color, transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending });
  const sp = new THREE.Sprite(mat); sp.scale.set(height * w / h, height, 1);
  return sp;
}
/** The "∞" at the particle vertex: a solid glowing glyph, ember on the left warming to gold on the right. */
function infinitySprite(color, height) {
  const w = 256, h = 220, font = '400 200px "Instrument Serif", Georgia, serif';
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const g = c.getContext('2d');
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
  const draw = () => {
    g.clearRect(0, 0, w, h); g.font = font; g.textAlign = 'center'; g.textBaseline = 'middle';
    const grd = g.createLinearGradient(w * 0.2, 0, w * 0.8, 0);
    grd.addColorStop(0, '#ff6a3d'); grd.addColorStop(1, '#ffc46b');
    g.fillStyle = grd; g.strokeStyle = grd; g.lineWidth = 3; g.lineJoin = 'round';
    g.shadowColor = 'rgba(255,140,80,0.85)'; g.shadowBlur = 6;
    for (let i = 0; i < 3; i++) g.fillText('∞', w / 2, h / 2 + 6);   // stacked fills: a solid glyph, not an outline
    g.shadowBlur = 0; g.strokeText('∞', w / 2, h / 2 + 6); g.fillText('∞', w / 2, h / 2 + 6);
    tex.needsUpdate = true;
  };
  draw();
  if (document.fonts && document.fonts.load) document.fonts.load(font).then(draw).catch(() => {});
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, color, transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending }));
  sp.scale.set(height * w / h, height, 1);
  return sp;
}
function glowSprite(color, scale) {
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture(), color, transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending }));
  sp.scale.setScalar(scale);
  return sp;
}

export default {
  id: 'worldsheet',
  numeral: 'IV',
  title: 'Worldsheets',
  tempo: 'Scherzo, legato',
  caption: 'A moving point draws a line through spacetime. A moving string draws a surface, its <em>worldsheet</em>. When particles collide, their lines meet at a sharp point. When strings join, the surface stays smooth. Drag the sheet up and down to slice through time, and sideways to see another observer’s <em>now</em>.',
  fine: '<p>Time runs upward. Each cut across the picture is one instant, and the two directions across stand in for all of space. A point particle’s history is a <em>worldline</em>. A closed string’s history is a tube, its <em>worldsheet</em>. The gold contour is the string at the instant you chose. A string’s action is the area of its worldsheet (the Nambu–Goto action), so the sheet is drawn as a soap film, the everyday surface whose shape is set by its area.</p>'
    + '<p>When point particles interact, their worldlines meet at a vertex, and every observer agrees on where and when. A quantum calculation must also add up loops of virtual particles, including ones where two of these sharp meeting points come arbitrarily close. For gravity these sums diverge more strongly at each order, and absorbing the infinities would take infinitely many new constants: gravity is <em>non-renormalizable</em>. Goroff and Sagnotti showed in 1985 that pure gravity diverges at two loops.</p>'
    + '<p>A worldsheet has no vertex. A moving observer slices spacetime along a tilted plane (the relativity of simultaneity, t′ = γ(t − vx/c²)), and the place where the two loops first touch slides around the crotch along the magenta curve. In the observer’s own frame, lengths along their motion are shorter by 1/γ, which the lens in the readout shows. No point is special, so the interaction is spread over the string length √α′ and the short-distance infinities do not appear. Drawn in real time, the crotch has to be flatter than light’s slope near the join, and every observer’s join point lies there. This is one reason the exact calculation is done in imaginary time, where the worldsheet becomes a smooth Riemann surface with no distinguished point. A surface with h handles is an h-loop correction, suppressed by g<sub>s</sub><sup>2h</sup> relative to the simplest surface. String amplitudes have been shown finite in every case worked out so far; a general proof to all orders is still an active topic.</p>'
    + '<p>Simplified: the shapes are hand-drawn illustrations and the film’s colours are decoration. Real worldsheets move through ten dimensions, and the string length is unknown (often taken near the Planck length, 10⁻³⁵ m). String theory has no experimental confirmation.</p>',
  drone: { notes: [0, 7, 15], opts: { gain: 0.085, cutoff: 760 } },
  post: { bloomStrength: 0.95, bloomRadius: 0.55, bloomThreshold: 0.2, exposure: 1.0, aberration: 0.0016, grain: 0.032, vignette: 0.9 },

  create(ctx) {
    const low = ctx.quality === 'low';
    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const params = new URLSearchParams(location.search);
    const audio = ctx.audio;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(34, ctx.aspect, 0.05, 400);
    camera.position.set(-4.9, 4.7, 13.0);
    const controls = new OrbitControls(camera, ctx.canvas);
    controls.enabled = false;
    controls.enableDamping = true; controls.dampingFactor = 0.07;
    controls.enablePan = false; controls.rotateSpeed = 0.55; controls.zoomSpeed = 0.6;
    controls.minDistance = 5.5; controls.maxDistance = 22;
    controls.minPolarAngle = Math.PI * 0.16; controls.maxPolarAngle = Math.PI * 0.84;
    const TARGET_Y = -0.25;
    controls.target.set(0, TARGET_Y, 0);
    controls.update();

    const pxScale = () => ctx.height / 1080 * ctx.pixelRatio;   // the engine's pixel ratio, which drops under load

    // ---- background: nebula sphere, stars, upward time dust, foreground bokeh ----
    const bgMat = new THREE.ShaderMaterial({
      side: THREE.BackSide, depthWrite: false,
      uniforms: { uTime: { value: 0 }, uDeep: { value: palette.deep.clone().multiplyScalar(0.9) }, uViolet: { value: palette.violet.clone() }, uCyan: { value: palette.harmonic.clone() } },
      vertexShader: BG_VERT, fragmentShader: BG_FRAG,
    });
    const bg = new THREE.Mesh(new THREE.SphereGeometry(150, 128, 64), bgMat);
    bg.renderOrder = -3; bg.frustumCulled = false; scene.add(bg);
    const stars = makeStarfield({ count: low ? 1300 : 2600, radius: 110, inner: 45, size: 1.5, intensity: 1.0 });
    stars.renderOrder = -2; scene.add(stars);

    function makeDust(count, rMin, rMax, size, speed, color, tint, soft, seed0) {
      const pos = new Float32Array(count * 3), seed = new Float32Array(count);
      let a = seed0 >>> 0;
      const rnd = () => { a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
      for (let i = 0; i < count; i++) {
        const r = rMin + (rMax - rMin) * Math.sqrt(rnd()), th = rnd() * Math.PI * 2;
        pos[i * 3] = r * Math.cos(th); pos[i * 3 + 1] = rnd() * 14 - 7; pos[i * 3 + 2] = r * Math.sin(th);
        seed[i] = rnd();
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      geo.setAttribute('seed', new THREE.BufferAttribute(seed, 1));
      const mat = new THREE.ShaderMaterial({
        transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
        uniforms: { uTime: { value: 0 }, uScale: { value: pxScale() }, uSize: { value: size }, uSpeed: { value: speed }, uColor: { value: color }, uTint: { value: tint }, uSoft: { value: soft } },
        vertexShader: DUST_VERT, fragmentShader: DUST_FRAG,
      });
      const pts = new THREE.Points(geo, mat); pts.frustumCulled = false;
      return pts;
    }
    const dust = makeDust(low ? 700 : 1500, 2.2, 10, 30, 0.12, palette.ink.clone().multiplyScalar(0.32), palette.violet.clone().multiplyScalar(0.5), 0, 7);
    dust.renderOrder = -1; scene.add(dust);
    const bokeh = makeDust(low ? 14 : 28, 5.5, 9.5, 150, 0.05, palette.violet.clone().multiplyScalar(0.07), palette.planck.clone().multiplyScalar(0.05), 1, 31);
    bokeh.renderOrder = -1; scene.add(bokeh);
    // one large, faint backlight behind the crotch: a rim of light that separates the glass from the void
    const BACK_SHEET = palette.violet.clone().multiplyScalar(0.12).lerp(palette.planck.clone().multiplyScalar(0.12), 0.3);
    const BACK_LINES = palette.violet.clone().multiplyScalar(0.14).lerp(palette.ember.clone().multiplyScalar(0.14), 0.35);   // warmer behind the vertex
    const backlight = glowSprite(BACK_SHEET, 9);
    backlight.position.set(0, -0.35, -0.8); backlight.renderOrder = -1; scene.add(backlight);

    // ---- the observer's now: a tilted disc ----
    const diskGroup = new THREE.Group(); scene.add(diskGroup);
    const diskMat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
      uniforms: { uR: { value: DISK_R }, uA: { value: 1 }, uTime: { value: 0 }, uGold: { value: palette.planck.clone().multiplyScalar(0.7) }, uTint: { value: palette.violet.clone().lerp(palette.planck, 0.25).multiplyScalar(0.55) } },
      vertexShader: DISK_VERT, fragmentShader: DISK_FRAG,
    });
    const disk = new THREE.Mesh(new THREE.CircleGeometry(DISK_R, 192).rotateX(-Math.PI / 2), diskMat);
    disk.renderOrder = 1; diskGroup.add(disk);
    // "now" rides the time axis at the height where this observer's slice crosses it
    const nowLabel = textSprite('now', { color: hdr(palette.planck, 1.25), height: 0.34 });
    nowLabel.renderOrder = 6; scene.add(nowLabel);
    const nowTick = glowSprite(hdr(palette.planck, 2.2), 0.22); nowTick.renderOrder = 6; scene.add(nowTick);

    // ---- the worldsheet (raymarched inside its bounding box) ----
    const sheetMat = new THREE.ShaderMaterial({
      defines: { MAX_STEPS: low ? 48 : 96, MAX_LAYERS: 3 },
      uniforms: {
        uTime: { value: 0 }, uW: { value: 0 }, uSheet: { value: 1 }, uPix: { value: 0.001 }, uSliceOn: { value: 1 }, uFlash: { value: 0 },
        uSlice: { value: new THREE.Vector3() },
        uBoxMin: { value: new THREE.Vector3(-BOX.x, -BOX.y, -BOX.z) }, uBoxMax: { value: new THREE.Vector3(BOX.x, BOX.y, BOX.z) },
        uJoinP: { value: new THREE.Vector3(0, 99, 0) }, uJoinT: { value: -99 },
        uGold: { value: palette.planck.clone() },
        uKey: { value: palette.planck.clone().multiplyScalar(0.5) }, uBack: { value: palette.harmonic.clone().multiplyScalar(0.6) },
        uInner: { value: palette.violet.clone().multiplyScalar(0.07) }, uLine: { value: palette.ink.clone().lerp(palette.harmonic, 0.4).multiplyScalar(0.85) },
        uGlow: { value: palette.violet.clone().lerp(palette.harmonic, 0.3).multiplyScalar(0.17) },
        uTrP: { value: new Float32Array(TRAILS * 4) }, uTrA: { value: new Float32Array(TRAILS) }, uTrOn: { value: 0 },
      },
      vertexShader: SHEET_VERT, fragmentShader: SHEET_FRAG,
      transparent: true, depthWrite: false, depthTest: false, side: THREE.BackSide,
      blending: THREE.CustomBlending, blendEquation: THREE.AddEquation, blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
    });
    const sheet = new THREE.Mesh(new THREE.BoxGeometry(BOX.x * 2, BOX.y * 2, BOX.z * 2), sheetMat);
    sheet.renderOrder = 2; sheet.frustumCulled = false; scene.add(sheet);
    const SU = sheetMat.uniforms;

    // ---- interaction-point markers (two: the handle has a split and a rejoin) ----
    const ringTex = ringTexture();
    const markers = [0, 1].map(() => {
      const g = new THREE.Group();
      const glow = glowSprite(hdr(palette.resonance.clone().lerp(palette.ink, 0.15), 3.2), 0.42);
      const core = glowSprite(hdr(palette.ink, 5), 0.1);
      const ring = new THREE.Sprite(new THREE.SpriteMaterial({ map: ringTex, color: hdr(palette.resonance.clone().lerp(palette.ink, 0.15), 1.7), transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending }));
      g.add(glow, core, ring); g.renderOrder = 5; glow.renderOrder = core.renderOrder = ring.renderOrder = 5;
      g.visible = false; scene.add(g);
      return { g, glow, core, ring, pos: new THREE.Vector3(), tgt: new THREE.Vector3(), a: 0, on: false };
    });

    // locus: the interaction point as seen by observers moving at -0.8c ... +0.8c along x. It draws itself in as
    // dots (one observer per frame); once complete it is fitted and becomes a continuous magenta curve with ticks.
    const locus = [0, 1].map(() => {
      const arr = new Float32Array(LOC_N * 2 * 3);
      const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(arr, 3)); geo.setDrawRange(0, 0);
      const mat = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
        uniforms: { uScale: { value: pxScale() }, uColor: { value: hdr(palette.resonance.clone().lerp(palette.ink, 0.3), 1.5) }, uA: { value: 0 } }, vertexShader: LOCUS_VERT, fragmentShader: LOCUS_FRAG });
      const pts = new THREE.Points(geo, mat); pts.frustumCulled = false; pts.renderOrder = 4; scene.add(pts);
      const group = new THREE.Group(); group.visible = false; scene.add(group);
      return { arr, geo, mat, pts, group, k: 0, n: 0, v: new Float32Array(LOC_N * 2), which: new Uint8Array(LOC_N * 2), fit: null, a: 0, mats: [] };
    });
    const LOCUS_LINE = hdr(palette.resonance, 1.4);
    const TICK_V = [-V_MAX, 0, V_MAX], TICK_TXT = ['−0.8c', '0', '+0.8c'];
    /** The locus is complete: fit each branch, snap the dots onto the fit, build the curve, its ticks and labels. */
    function finishLocus(li) {
      const L = locus[li], W = li, nb = li === 1 ? 2 : 1;
      L.fit = [];
      for (let b = 0; b < nb; b++) {
        const vs = [], pts = [];
        for (let j = 0; j < L.n; j++) if (L.which[j] === b) { vs.push(L.v[j]); pts.push(L.arr[j * 3], L.arr[j * 3 + 1], L.arr[j * 3 + 2]); }
        if (vs.length < 8) { L.fit = null; return; }   // too few samples to trust a fit: keep the raw dots
        L.fit.push(fitBranch(vs, pts));
      }
      // the dots become evenly spaced samples of the fitted curve (dragMarker reads them too)
      const P = new Float64Array(3);
      L.n = 0;
      for (let b = 0; b < nb; b++) {
        for (let k = 0; k < LOC_N; k++) {
          const v = -V_MAX + 2 * V_MAX * k / (LOC_N - 1);
          evalBranch(L.fit[b], v, W, P, 0);
          L.arr[L.n * 3] = P[0]; L.arr[L.n * 3 + 1] = P[1]; L.arr[L.n * 3 + 2] = P[2];
          L.v[L.n] = v; L.which[L.n] = b; L.n++;
        }
        const cpts = [];
        for (let k = 0; k <= 48; k++) { evalBranch(L.fit[b], -V_MAX + 2 * V_MAX * k / 48, W, P, 0); cpts.push(new THREE.Vector3(P[0], P[1], P[2])); }
        const curve = new THREE.CatmullRomCurve3(cpts, false, 'centripetal');
        const lineMat = new THREE.MeshBasicMaterial({ color: LOCUS_LINE, transparent: true, opacity: 0, depthTest: false, depthWrite: false, blending: THREE.AdditiveBlending });
        const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, 128, 0.011, 6, false), lineMat);
        tube.renderOrder = 4; L.group.add(tube); L.mats.push({ m: lineMat, k: 0.75 });
        for (let i = 0; i < 3; i++) {
          evalBranch(L.fit[b], TICK_V[i], W, P, 0);
          const bead = glowSprite(hdr(palette.resonance.clone().lerp(palette.ink, 0.35), 2.4), 0.11);
          bead.position.set(P[0], P[1], P[2]); bead.renderOrder = 4; L.group.add(bead); L.mats.push({ m: bead.material, k: 1 });
          const lab = textSprite(TICK_TXT[i], { font: '500 64px "IBM Plex Mono", ui-monospace, monospace', color: palette.resonance.clone().multiplyScalar(0.9), height: 0.13, w: 256, h: 96 });
          // labels sit above the lower branch and below the upper one, clear of the surface's own contour
          lab.position.set(P[0], P[1] + (b === 1 ? -0.15 : 0.15), P[2]); lab.renderOrder = 4; L.group.add(lab); L.mats.push({ m: lab.material, k: 0.95 });
        }
      }
      L.geo.attributes.position.needsUpdate = true; L.geo.setDrawRange(0, L.n);
      L.group.visible = true;
      saddleDirty = true;   // the markers can now ride the smooth curve
    }

    // ---- particles mode: worldlines meeting at a vertex ----
    const lineGroup = new THREE.Group(); scene.add(lineGroup);
    const V = VERTEX;
    const curves = [
      new THREE.CatmullRomCurve3([new THREE.Vector3(-1.5, -4.3, 0.35), new THREE.Vector3(-1.0, -2.6, 0.24), new THREE.Vector3(-0.45, -1.25, 0.1), V.clone()]),
      new THREE.CatmullRomCurve3([new THREE.Vector3(1.35, -4.3, -0.3), new THREE.Vector3(0.9, -2.5, -0.2), new THREE.Vector3(0.38, -1.2, -0.07), V.clone()]),
      new THREE.CatmullRomCurve3([V.clone(), new THREE.Vector3(0.1, 1.0, 0.0), new THREE.Vector3(0.05, 2.6, 0.06), new THREE.Vector3(0.22, 4.3, 0.0)]),
    ];
    // the two incoming particles are cool cyan, the one that leaves the vertex warm white: the kink reads at a glance
    const lineMat = (color, pow) => new THREE.ShaderMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { uColor: { value: color }, uA: { value: 0 }, uPow: { value: pow } }, vertexShader: LINE_VERT, fragmentShader: LINE_FRAG });
    const lineMats = [
      lineMat(hdr(palette.ink.clone().lerp(palette.harmonic, 0.55), 2.0), 0.5), lineMat(hdr(palette.harmonic, 0.34), 3.0),
      lineMat(hdr(palette.ink.clone().lerp(palette.planck, 0.3), 2.2), 0.5), lineMat(hdr(palette.planck.clone().lerp(palette.ember, 0.3), 0.3), 3.0),
    ];
    const WL = []; // sampled worldlines for slice intersections and hit tests
    for (let ci = 0; ci < curves.length; ci++) {
      const c = curves[ci], out = ci === 2 ? 2 : 0;
      const a = new THREE.Mesh(new THREE.TubeGeometry(c, 256, 0.013, 6, false), lineMats[out]);
      const b = new THREE.Mesh(new THREE.TubeGeometry(c, 256, 0.085, 12, false), lineMats[out + 1]);
      a.renderOrder = b.renderOrder = 3; lineGroup.add(b, a);
      const pts = c.getSpacedPoints(160), arr = new Float32Array(pts.length * 3);
      pts.forEach((p, i) => p.toArray(arr, i * 3));
      WL.push(arr);
    }
    lineGroup.visible = false;
    const vertexGroup = new THREE.Group(); vertexGroup.position.copy(V); scene.add(vertexGroup);
    const vGlow = glowSprite(hdr(palette.ember, 3.8), 1.3);
    const vCore = glowSprite(hdr(palette.ink.clone().lerp(palette.ember, 0.3), 5), 0.16);
    const infLabel = infinitySprite(hdr(palette.ink, 2.2), 0.95);
    infLabel.position.set(0.72, 0.5, 0);
    // an anamorphic flare: two thin streaks through the vertex that grow as the slice approaches it
    const streakH = glowSprite(hdr(palette.ember, 2.5), 1), streakV = glowSprite(hdr(palette.ember, 2.5), 1);
    streakH.scale.set(3.0, 0.06, 1); streakV.scale.set(0.06, 1.4, 1);
    vGlow.renderOrder = vCore.renderOrder = infLabel.renderOrder = streakH.renderOrder = streakV.renderOrder = 6;
    vertexGroup.add(vGlow, streakH, streakV, vCore, infLabel);
    const SPARKS = low ? 260 : 520;
    const sparkGeo = new THREE.BufferGeometry();
    {
      const dir = new Float32Array(SPARKS * 3), sd = new Float32Array(SPARKS), pos = new Float32Array(SPARKS * 3);
      for (let i = 0; i < SPARKS; i++) {
        const u = Math.random() * 2 - 1, th = Math.random() * Math.PI * 2, s = Math.sqrt(1 - u * u);
        dir[i * 3] = s * Math.cos(th); dir[i * 3 + 1] = u; dir[i * 3 + 2] = s * Math.sin(th); sd[i] = Math.random();
      }
      sparkGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      sparkGeo.setAttribute('dir', new THREE.BufferAttribute(dir, 3));
      sparkGeo.setAttribute('seed', new THREE.BufferAttribute(sd, 1));
    }
    const sparkMat = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { uTime: { value: 0 }, uScale: { value: pxScale() }, uHeat: { value: 0 }, uV: { value: V.clone() }, uColor: { value: hdr(palette.ember, 3.2) }, uA: { value: 0 } },
      vertexShader: SPARK_VERT, fragmentShader: SPARK_FRAG });
    const sparks = new THREE.Points(sparkGeo, sparkMat); sparks.frustumCulled = false; sparks.renderOrder = 5; scene.add(sparks);
    vertexGroup.visible = false; sparks.visible = false;
    const beads = [0, 1, 2, 3].map(() => {
      const g = new THREE.Group();
      const a = glowSprite(hdr(palette.planck, 2.6), 0.5), b = glowSprite(hdr(palette.ink, 4), 0.09);
      a.renderOrder = b.renderOrder = 5; g.add(a, b); g.visible = false; scene.add(g);
      return g;
    });

    // ---- a quiet time axis ----
    const axisGroup = new THREE.Group(); axisGroup.position.set(AXIS.x, 0, AXIS.z); scene.add(axisGroup);
    {
      const pts = [0, -3.4, 0, 0, 3.3, 0, -0.07, 3.12, 0, 0, 3.3, 0, 0.07, 3.12, 0];
      const ticks = [];
      for (let k = -13; k <= 12; k++) { const y = k * 0.25; const w = k % 4 === 0 ? 0.09 : 0.04; ticks.push(-w, y, 0, w, y, 0); }
      const g1 = new THREE.BufferGeometry(); g1.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
      const g2 = new THREE.BufferGeometry(); g2.setAttribute('position', new THREE.Float32BufferAttribute(ticks, 3));
      const m = new THREE.LineBasicMaterial({ color: palette.ink.clone().multiplyScalar(0.2), transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
      axisGroup.add(new THREE.Line(g1, m), new THREE.LineSegments(g2, m));
      const tl = textSprite('time', { color: palette.ink.clone().multiplyScalar(0.42), height: 0.3 });
      tl.position.set(0, 3.62, 0); axisGroup.add(tl);
    }

    // ---- state ------------------------------------------------------------------------------
    // mode: 0 strings (pair of pants), 1 point particles, 2 string loop (handle)
    const S = { mode: 0, W: 0, Wt: 0, sheetA: 1, lineA: 0, T: -1.05, Tt: -1.05, vx: 0.32, vxt: 0.32, vz: 0, vzt: 0, count: 2, act: 0 };
    {
      const pSlice = parseFloat(params.get('slice')), pTilt = parseFloat(params.get('tilt')), pMode = parseInt(params.get('mode'), 10);
      if (isFinite(pSlice)) S.T = S.Tt = clamp(pSlice, -1, 1) * T_MAX;
      if (isFinite(pTilt)) S.vx = S.vxt = clamp(pTilt, -V_MAX, V_MAX);
      if (pMode >= 1 && pMode <= 3) {
        S.mode = pMode - 1; S.W = S.Wt = S.mode === 2 ? 1 : 0;
        S.sheetA = S.mode === 1 ? 0 : 1; S.lineA = S.mode === 1 ? 1 : 0;
      }
    }
    let computedT = NaN, computedVx = NaN, computedVz = NaN, computedW = NaN;
    let saddleDirty = true, lastSolve = -1, nSad = 0;
    const SAD = new Float64Array(6), TMP6 = new Float64Array(6);
    let drag = null, idle = 99, entered = false, quietFrames = 0, time = 0;
    const sweep = { on: false, t: 0, from: 0, to: 0, dur: 1 };
    const told = { merge: false, tilt: false, particles: false, loop: false, markerTip: false };
    let hintShown = false, hintUntil = 0, flash = 0, vFlash = 0, lastEvent = -1, lastTick = -1, prevVertexSide = 0;
    let joinIdx = -1, joinTime = -99;   // the marker where the topology last changed, and when (for its pulse)
    // afterglow ring buffer: a faint gold contour is left wherever the slice was a moment ago. Each ghost keeps the
    // plane it had then (normal + offset), so tilting afterwards does not swing the old rings around.
    const trail = { head: 0, lastT: S.T, lastVx: S.vx, lastVz: S.vz, clock: 0 };
    function clearTrail() {
      const P = SU.uTrP.value;
      SU.uTrA.value.fill(0); SU.uTrOn.value = 0;
      for (let i = 0; i < TRAILS; i++) { P[i * 4] = 0; P[i * 4 + 1] = 1; P[i * 4 + 2] = 0; P[i * 4 + 3] = 99; }
      trail.lastT = S.T; trail.lastVx = S.vx; trail.lastVz = S.vz; trail.clock = 0;
    }
    function stepTrail(dt) {
      const A = SU.uTrA.value, P = SU.uTrP.value;
      let on = 0;
      for (let i = 0; i < TRAILS; i++) { A[i] *= Math.exp(-dt * 2.6); if (A[i] > 0.01) on = 1; }
      trail.clock += dt;
      // one ghost ring per 0.13 of travel, so they read as a strobe of separate instants
      if (Math.abs(S.T - trail.lastT) > 0.13) {
        trail.clock = 0;
        const sl = 1 / Math.sqrt(1 + trail.lastVx * trail.lastVx + trail.lastVz * trail.lastVz), h = trail.head * 4;
        P[h] = -trail.lastVx * sl; P[h + 1] = sl; P[h + 2] = -trail.lastVz * sl; P[h + 3] = trail.lastT * sl;
        A[trail.head] = 0.5; on = 1;
        trail.head = (trail.head + 1) % TRAILS; trail.lastT = S.T; trail.lastVx = S.vx; trail.lastVz = S.vz;
      } else if (trail.clock > 0.6) { trail.lastT = S.T; trail.lastVx = S.vx; trail.lastVz = S.vz; trail.clock = 0; }
      SU.uTrOn.value = on;
    }
    const lastTickPos = new THREE.Vector3(0, 99, 0);
    let semA = 24, semB = 27, merged = false;
    let vA = null, vB = null, vZ1 = null, vZ2 = null;
    let sentA = NaN, sentB = NaN, sentZ = NaN, sentMerged = null, gainA = -1, gainB = -1, gainZ = -1, panMerged = null;
    let ro = null, uiC = null, narrowUI = false, noteUI = false, lensCanvas = null, lensCtx = null, lensDirty = true, roTimer = 0;
    let lensFont = '', lensGamma = NaN, lensGammaTxt = '';
    let sliderSyncT = NaN, sliderSyncV = NaN, lensCheck = 0, framed = null;
    // pointer: where a press started (for the direction lock), the primary finger, and a two-finger pinch
    const press = { x: 0, y: 0 }, prim = { id: -1, x: 0, y: 0 };
    const pinch = { id: -1, x: 0, y: 0, d0: 1, len0: 15 };
    const hits = new Float32Array(12); let nHits = 0;
    const UP = new THREE.Vector3(0, 1, 0), nrm = new THREE.Vector3(), off = new THREE.Vector3(), SCR = new THREE.Vector3();
    let dragMarkerIdx = 0;
    const raycaster = new THREE.Raycaster();
    const postScratch = { aberration: 0.0016 };
    const infBase = infLabel.scale.clone();
    const DEBUG_LOG = params.has('m4log'); let dbgFrames = 0;
    const DEBUG_SWEEP = params.has('m4sweep'); let enterTime = -1, dbgSwept = false;   // m4sweep: start a time sweep 1 s after entering
    // m4dragtest: a synthetic drag of the interaction point to the far end of the locus (tests markerHit + dragMarker)
    const DEBUG_DRAG = params.has('m4dragtest'); let dbgDrag = 0;
    const dbgEv = { x: 0, y: 0, ndc: new THREE.Vector2(), dx: 0, dy: 0, shift: false, pointerType: 'mouse' };
    function debugDrag(et) {
      if (dbgDrag === 0) {
        const m = markers[0]; toScreen(m.pos.x, m.pos.y, m.pos.z); dbgEv.x = SX; dbgEv.y = SY;
        api.onPointerDown(dbgEv); console.log('m4dragtest down', drag, SX.toFixed(0), SY.toFixed(0)); dbgDrag = 1;
      } else if (dbgDrag === 1 && et > 2.0) {
        const L = locus[0]; toScreen(L.arr[0], L.arr[1], L.arr[2]); dbgEv.x = SX + 6; dbgEv.y = SY + 4;
        api.onPointerMove(dbgEv); dbgDrag = 2;
      } else if (dbgDrag === 2 && et > 2.4) { api.onPointerUp(dbgEv); console.log('m4dragtest up, vxt', S.vxt.toFixed(2)); dbgDrag = 3; }
    }

    const gamma = () => 1 / Math.sqrt(Math.max(1e-6, 1 - S.vx * S.vx - S.vz * S.vz));
    const vertexT = () => V.y - S.vx * V.x - S.vz * V.z;   // when this observer's slice crosses the vertex

    function hintText() {
      if (S.mode === 1) return ctx.isTouch ? 'Drag a worldline to move through time' : 'Drag a worldline up/down for time · sideways to tilt';
      return ctx.isTouch ? 'Up/down: time · sideways: observer' : 'Drag up/down for time · sideways to change observer';
    }
    function showHint() {
      hintShown = true; hintUntil = time + 7;
      ctx.ui.hint(hintText(), ctx.isTouch ? 7000 : 0);
    }
    function hideHint() { if (hintShown) { ctx.ui.hideHint(); hintShown = false; } }
    function setMode(m) {
      if (m === S.mode) return;
      S.mode = m;
      if (m !== 1) S.Wt = m === 2 ? 1 : 0;
      saddleDirty = true; lensDirty = true; computedT = NaN; prevVertexSide = 0; quietFrames = 4;
      if (uiC) uiC.set('mode', String(m));
      if (m === 1 && !told.particles) { told.particles = true; ctx.ui.toast('Every observer agrees where the worldlines meet. In quantum gravity, loops built from such sharp meeting points blow up.', 5600); }
      if (m === 2 && !told.loop) { told.loop = true; ctx.ui.toast('A handle: the string splits and rejoins. Each handle adds one quantum loop.', 5200); }
      audio.pluck(audio.freq(m === 1 ? 12 : m === 2 ? 22 : 19), { gain: 0.14, decay: 1.8, bright: 0.45, pan: 0.2 });
      // a hint that is still showing follows the mode (there is no sheet to drag in particles mode)
      if (hintShown && entered && (!ctx.isTouch || time < hintUntil)) { ctx.ui.hint(hintText(), ctx.isTouch ? Math.max(500, (hintUntil - time) * 1000) : 0); }
      paintReadoutStatic();
    }
    function setTilt(vx, vz) {
      const m = Math.sqrt(vx * vx + vz * vz);
      if (m > V_MAX) { vx *= V_MAX / m; vz *= V_MAX / m; }
      S.vxt = vx; S.vzt = vz;
    }
    function userTilt() {
      idle = 0; hideHint();
      if (!told.tilt && S.mode !== 1) { told.tilt = true; ctx.ui.toast('A moving observer slices spacetime at an angle. Watch the join slide along the crotch.', 4600); }
    }
    function userTime() { idle = 0; hideHint(); }
    function startSweep(from, to, dur) { sweep.on = true; sweep.t = 0; sweep.from = from; sweep.to = to; sweep.dur = dur; }
    function sweepButton() {
      if (sweep.on) { sweep.on = false; return; }
      startSweep(Math.min(S.T, -T_MAX * 0.92), T_MAX * 0.92, 7.5); userTime();
    }

    // ---- slice: connectivity, loop lengths, particle crossings -------------------------------------
    function recomputeSlice() {
      let n;
      if (S.mode === 1) {
        nHits = 0;
        for (let c = 0; c < 3; c++) {
          const a = WL[c], m = a.length / 3;
          let gp = a[1] - S.vx * a[0] - S.vz * a[2] - S.T;
          for (let i = 1; i < m; i++) {
            const g = a[i * 3 + 1] - S.vx * a[i * 3] - S.vz * a[i * 3 + 2] - S.T;
            if ((gp <= 0) !== (g <= 0) && nHits < 4) {
              const u = gp / (gp - g), x = lerp(a[i * 3 - 3], a[i * 3], u), y = lerp(a[i * 3 - 2], a[i * 3 + 1], u), z = lerp(a[i * 3 - 1], a[i * 3 + 2], u);
              let dup = false; // the vertex belongs to all three lines
              for (let h = 0; h < nHits; h++) { const ex = hits[h * 3] - x, ey = hits[h * 3 + 1] - y, ez = hits[h * 3 + 2] - z; if (ex * ex + ey * ey + ez * ez < 0.004) dup = true; }
              if (!dup) { hits[nHits * 3] = x; hits[nHits * 3 + 1] = y; hits[nHits * 3 + 2] = z; nHits++; }
            }
            gp = g;
          }
        }
        n = nHits;
      } else {
        G.fill(S.vx, S.vz, S.T, S.W);
        n = G.label();
        LENS_VAL.set(G.val);
        // loop lengths in the observer's own frame: lengths along their motion are shorter by 1/γ
        const cellA = G.dx * G.dz / gamma();
        if (n >= 2) {
          const l0 = G.big0, l1 = G.big1, left = G.cx[l0] <= G.cx[l1] ? l0 : l1, right = left === l0 ? l1 : l0;
          semA = semisForLength(2 * Math.sqrt(Math.PI * G.area[left] * cellA));
          semB = semisForLength(2 * Math.sqrt(Math.PI * G.area[right] * cellA));
          merged = false;
        } else if (n === 1) {
          semA = semB = semisForLength(2 * Math.sqrt(Math.PI * G.area[G.big0] * cellA));
          merged = true;
        }
      }
      if (n !== S.count) {
        const settled = Math.abs(S.W - S.Wt) < 0.02 && quietFrames <= 0 && entered;
        if (settled && S.mode !== 1 && time - lastEvent > 0.12) {
          lastEvent = time; flash = 1;
          // light and sound land on the same point: a gold wave spreads from the interaction point just crossed
          let bi = -1, bd = Infinity;
          for (let i = 0; i < 2; i++) {
            const m = markers[i]; if (!m.on) continue;
            const d = Math.abs(m.tgt.y - S.vx * m.tgt.x - S.vz * m.tgt.z - S.T);
            if (d < bd) { bd = d; bi = i; }
          }
          if (bi >= 0) { SU.uJoinP.value.copy(markers[bi].tgt); SU.uJoinT.value = time; joinIdx = bi; joinTime = time; }
          if (n < S.count) {
            audio.bell(audio.freq(36), { gain: 0.13, pan: -0.1 });
            if (!told.merge && S.mode === 0) { told.merge = true; ctx.ui.toast('Two strings became one. Look for the corner where it happened: there is none.', 4600); }
          } else audio.bell(audio.freq(31), { gain: 0.12, pan: 0.1 });
        }
        S.count = n;
      }
      lensDirty = true;
    }

    // the locus: where observers moving at -0.8c ... +0.8c (along x) place the interaction
    function stepLocus(li) {
      const L = locus[li];
      if (L.k >= LOC_N) return;
      const v = -V_MAX + 2 * V_MAX * L.k / (LOC_N - 1);
      const n = solveSaddles(v, 0, li, TMP6);
      for (let i = 0; i < n && L.n < LOC_N * 2; i++) {
        L.arr[L.n * 3] = TMP6[i * 3]; L.arr[L.n * 3 + 1] = TMP6[i * 3 + 1]; L.arr[L.n * 3 + 2] = TMP6[i * 3 + 2];
        // the handle's split lies below its middle (y = 0) and the rejoin above, whatever the scan found first
        L.v[L.n] = v; L.which[L.n] = li === 1 && TMP6[i * 3 + 1] > 0 ? 1 : 0; L.n++;
      }
      L.k++; L.geo.attributes.position.needsUpdate = true; L.geo.setDrawRange(0, L.n);
      if (L.k === LOC_N) finishLocus(li);
    }
    /** True when the current observer moves along x and the locus fit for this mode is ready. */
    function onFit() { const L = locus[S.mode === 2 ? 1 : 0]; return S.mode !== 1 && !!L.fit && Math.abs(S.vz) < 1e-4; }

    function solveNow() {
      lastSolve = time; saddleDirty = false;
      let n = 0;
      if (S.mode !== 1) {
        const li = S.mode === 2 ? 1 : 0;
        if (onFit()) {
          // an observer moving along x: ride the smooth fitted locus (exact to the grid's precision, and continuous)
          const F = locus[li].fit;
          for (let b = 0; b < F.length; b++) evalBranch(F[b], S.vx, li, SAD, b * 3);
          n = F.length;
        } else {
          n = solveSaddles(S.vx, S.vz, li, SAD);
          if (li === 1 && n === 2 && SAD[1] > SAD[4]) for (let k = 0; k < 3; k++) { const tmp = SAD[k]; SAD[k] = SAD[k + 3]; SAD[k + 3] = tmp; }
        }
      }
      if (S.mode !== 1 && n < (S.mode === 2 ? 2 : 1) && nSad > 0) return;   // rare grid miss: keep the last answer
      nSad = n;
      for (let i = 0; i < 2; i++) {
        const m = markers[i];
        m.on = i < nSad;
        if (m.on) { m.tgt.set(SAD[i * 3], SAD[i * 3 + 1], SAD[i * 3 + 2]); if (m.a < 0.02) m.pos.copy(m.tgt); }
      }
      // a soft tick each time the interaction point slides a little further
      if (nSad && markers[0].tgt.distanceTo(lastTickPos) > 0.12 && time - lastTick > 0.11) {
        if (lastTickPos.y < 90 && entered) {
          const deg = clamp(Math.round((markers[0].tgt.x + 1.2) * 3), 0, 9);
          audio.pluck(audio.degree(deg, 2), { gain: 0.06, decay: 0.9, bright: 0.3, pan: clamp(markers[0].tgt.x * 0.4, -0.6, 0.6) });
        }
        lastTick = time; lastTickPos.copy(markers[0].tgt);
      }
    }

    // ---- the lens: this observer's slice in their own frame (marching squares on the slice grid) --------
    // On the slice t = v·r + T the induced metric is ds² = |dr|² − (v·dr)², so in the observer's frame lengths along
    // their motion are shorter by 1/γ. The lens draws the loops contracted that way, with the lab outline as a ghost.
    let EX = 0, EZ = 0, LK = 0, LUX = 1, LUZ = 0;
    function edgePt(e, a, b, c, d, x0, z0) {
      const dx = G.dx, dz = G.dz;
      if (e === 0) { EX = x0 + dx * a / (a - b); EZ = z0; }
      else if (e === 1) { EX = x0 + dx; EZ = z0 + dz * b / (b - c); }
      else if (e === 2) { EX = x0 + dx * d / (d - c); EZ = z0 + dz; }
      else { EX = x0; EZ = z0 + dz * a / (a - d); }
      const s = (EX * LUX + EZ * LUZ) * LK;   // contract along the observer's velocity, about the lens centre
      EX -= s * LUX; EZ -= s * LUZ;
    }
    function seg(g, e1, e2, a, b, c, d, x0, z0, ox, oy, sc) {
      edgePt(e1, a, b, c, d, x0, z0); g.moveTo(ox + EX * sc, oy + EZ * sc);
      edgePt(e2, a, b, c, d, x0, z0); g.lineTo(ox + EX * sc, oy + EZ * sc);
    }
    function contour(g, ox, oy, sc) {
      const nx = G.nx, nz = G.nz, val = LENS_VAL;
      g.beginPath();
      for (let j = 0; j < nz - 1; j++) {
        const z0 = GZ0 + j * G.dz;
        for (let i = 0; i < nx - 1; i++) {
          const k = j * nx + i, a = val[k], b = val[k + 1], c = val[k + nx + 1], d = val[k + nx];
          const cs = (a < 0 ? 1 : 0) | (b < 0 ? 2 : 0) | (c < 0 ? 4 : 0) | (d < 0 ? 8 : 0);
          if (cs === 0 || cs === 15) continue;
          const x0 = GX0 + i * G.dx;
          if (cs === 5) { seg(g, 0, 3, a, b, c, d, x0, z0, ox, oy, sc); seg(g, 1, 2, a, b, c, d, x0, z0, ox, oy, sc); continue; }
          if (cs === 10) { seg(g, 0, 1, a, b, c, d, x0, z0, ox, oy, sc); seg(g, 2, 3, a, b, c, d, x0, z0, ox, oy, sc); continue; }
          let e1 = -1, e2 = -1;
          const cr0 = (a < 0) !== (b < 0), cr1 = (b < 0) !== (c < 0), cr2 = (d < 0) !== (c < 0), cr3 = (a < 0) !== (d < 0);
          if (cr0) e1 = 0;
          if (cr1) { if (e1 < 0) e1 = 1; else e2 = 1; }
          if (cr2) { if (e1 < 0) e1 = 2; else e2 = 2; }
          if (cr3) e2 = 3;
          if (e1 >= 0 && e2 >= 0) seg(g, e1, e2, a, b, c, d, x0, z0, ox, oy, sc);
        }
      }
      g.stroke();
    }
    const COUNT_S = ['NO STRING', '1 STRING', '2 STRINGS', '3 STRINGS', '4 STRINGS', 'STRINGS'];
    const COUNT_P = ['NO PARTICLE', '1 PARTICLE', '2 PARTICLES', '3 PARTICLES', '4 PARTICLES', 'PARTICLES'];
    function drawLens() {
      lensDirty = false;
      if (!lensCtx) return;
      const g = lensCtx, cw = lensCanvas.width, ch = lensCanvas.height;
      if (!cw || !ch) return;
      const dpr = cw / Math.max(1, lensCanvas.clientWidth);
      g.clearRect(0, 0, cw, ch);
      const sc = ch / 2.5, ox = cw * 0.5, oy = ch * 0.56;
      g.lineWidth = 1; g.strokeStyle = 'rgba(236,231,246,0.07)';
      g.beginPath(); g.moveTo(0, oy); g.lineTo(cw, oy); g.moveTo(ox, 0); g.lineTo(ox, ch); g.stroke();
      const v = Math.sqrt(S.vx * S.vx + S.vz * S.vz), gm = gamma();
      if (S.mode !== 1 && gm > 1.01) {   // the lab frame's outline, uncontracted, as a faint ghost
        LK = 0; g.strokeStyle = 'rgba(236,231,246,0.13)'; g.lineWidth = 1 * dpr; contour(g, ox, oy, sc);
      }
      LK = 1 - 1 / gm;
      if (v > 1e-6) { LUX = S.vx / v; LUZ = S.vz / v; } else { LUX = 1; LUZ = 0; }
      g.shadowColor = '#ffc46b'; g.shadowBlur = 9 * dpr;
      if (S.mode === 1) {
        g.fillStyle = '#ffd59a';
        for (let i = 0; i < nHits; i++) {
          const x = hits[i * 3], z = hits[i * 3 + 2], s = (x * LUX + z * LUZ) * LK;
          g.beginPath(); g.arc(ox + (x - s * LUX) * sc, oy + (z - s * LUZ) * sc, 3 * dpr, 0, Math.PI * 2); g.fill();
        }
      } else {
        g.strokeStyle = '#ffd59a'; g.lineWidth = 1.6 * dpr; g.lineCap = 'round';
        contour(g, ox, oy, sc);
      }
      g.shadowBlur = 0;
      g.font = lensFont; g.textBaseline = 'bottom';
      g.fillStyle = 'rgba(255,196,107,0.9)'; g.textAlign = 'right';
      g.fillText((S.mode === 1 ? COUNT_P : COUNT_S)[Math.min(S.count, 5)], cw - 8 * dpr, ch - 6 * dpr);
      if (gm > 1.005) {
        if (Math.abs(gm - lensGamma) > 0.005) { lensGamma = gm; lensGammaTxt = `LENGTHS ÷ ${gm.toFixed(2)}`; }
        g.fillStyle = 'rgba(236,231,246,0.45)'; g.textAlign = 'left';
        g.fillText(lensGammaTxt, 8 * dpr, ch - 6 * dpr);
      }
    }
    function sizeLens() {
      if (!lensCanvas) return;
      const r = lensCanvas.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
      if (r.width < 4) { lensCtx = null; return; }
      lensCanvas.width = Math.round(r.width * dpr); lensCanvas.height = Math.round(r.height * dpr);
      lensCtx = lensCanvas.getContext('2d'); lensDirty = true;
      lensFont = `500 ${Math.round(9.5 * dpr)}px "IBM Plex Mono", ui-monospace, monospace`;
    }

    // ---- programme panels ----------------------------------------------------------------------
    const LENS_HTML = '<div class="m4-lens" style="position:relative;height:104px;margin:0 0 8px;border:1px solid rgba(236,231,246,.08);background:radial-gradient(ellipse at 50% 56%,rgba(255,196,107,.06),rgba(5,4,10,.25) 70%)">'
      + '<canvas style="position:absolute;inset:0;width:100%;height:100%;display:block"></canvas>'
      + '<span style="position:absolute;left:8px;top:7px;font:400 9.5px/1 \'IBM Plex Mono\',monospace;letter-spacing:.14em;text-transform:uppercase;color:rgba(236,231,246,.5)">This observer’s now, in their frame</span></div>';
    const NOTES = [
      'The magenta curve: where observers moving from −0.8c to +0.8c place the join. Drag the bright point along it.',
      'Tilt the slice: the vertex stays put. Every observer agrees on the event.',
      'A handle is a quantum loop: the string splits, then rejoins. Both points move with the observer.',
    ];
    const RQ = { obs: NaN, t: NaN, n: -1, ip: NaN };   // last values written, so the readout only formats on change
    function buildUI() {
      narrowUI = ctx.width <= 760;
      noteUI = !narrowUI && ctx.height >= 820;   // the note only where there is height to spare
      // phones: two short rows and no title, so the panel stays clear of the caption title
      const rows = narrowUI
        ? [{ id: 'n', k: 'Now', v: '', cls: 'gold' }, { id: 'ip', k: 'Join', v: '' }]
        : [{ id: 'obs', k: 'Observer', v: '' }, { id: 't', k: 'Slice time <span style="text-transform:none">t′</span>', v: '' }, { id: 'n', k: 'In this slice', v: '', cls: 'gold' },
          { id: 'ip', k: 'Interaction point', v: '' }, { id: 'g', k: 'Gravity there', v: '' }];
      ro = ctx.ui.readout({ title: narrowUI ? undefined : 'Spacetime slice', html: narrowUI ? '' : LENS_HTML, rows, note: noteUI ? '' : undefined });
      lensCanvas = narrowUI ? null : ro.el.querySelector('.m4-lens canvas');
      lensCtx = null; sizeLens();
      uiC = ctx.ui.controls([
        { type: 'segmented', id: 'mode', label: 'History of', value: String(S.mode), onChange: (v) => setMode(+v),
          options: [{ value: '0', label: 'Strings', kbd: '1' }, { value: '1', label: 'Particles', kbd: '2' }, { value: '2', label: 'Loop', kbd: '3' }] },
        { type: 'slider', id: 'time', label: 'Time of slice &nbsp;↑↓', min: -T_MAX, max: T_MAX, step: 0.005, value: S.Tt, format: (v) => fmtS(v),
          onInput: (v) => { sweep.on = false; S.Tt = v; sliderSyncT = v; userTime(); } },
        { type: 'slider', id: 'tilt', label: 'Observer speed &nbsp;, .', min: -V_MAX, max: V_MAX, step: 0.01, value: S.vxt, format: (v) => `${fmtS(v)} c`,
          onInput: (v) => { setTilt(v, S.vzt); sliderSyncV = v; userTilt(); } },
        { type: 'button', id: 'sweep', label: 'Sweep time', kbd: ' ', onClick: sweepButton },
        { type: 'button', id: 'rest', label: 'Rest frame', kbd: '0', onClick: () => { setTilt(0, 0); userTilt(); } },
      ]);
      sliderSyncT = S.Tt; sliderSyncV = S.vxt;
      roTimer = 0;
      paintReadoutStatic();
    }
    function paintReadoutStatic() {
      if (!ro) return;
      RQ.obs = RQ.t = RQ.ip = NaN; RQ.n = -1;
      if (narrowUI) ro.set('ip', S.mode === 1 ? 'fixed' : 'relative', S.mode === 1 ? 'mag' : 'gold');
      if (S.mode === 1) ro.set('g', 'loops diverge', 'mag');
      else ro.set('g', S.mode === 0 ? 'smooth, finite' : 'one loop, finite', 'cyan');
      if (noteUI) ro.note(NOTES[S.mode]);
      updateReadout();
    }
    function updateReadout() {
      if (!ro) return;
      const q = Math.round(S.vx * 100) * 1000 + Math.round(S.vz * 100);
      if (q !== RQ.obs) {
        RQ.obs = q;
        ro.set('obs', Math.abs(S.vz) > 0.005 ? `|v| ${Math.sqrt(S.vx * S.vx + S.vz * S.vz).toFixed(2)} c` : `v ${fmtS(S.vx)} c`);
      }
      const tq = Math.round(gamma() * S.T * 100);
      if (tq !== RQ.t) { RQ.t = tq; ro.set('t', `${fmtS(tq / 100)} ℓₛ/c`); }
      const nq = S.count * 10 + S.mode;
      if (nq !== RQ.n) {
        RQ.n = nq;
        ro.set('n', S.mode === 1 ? (S.count === 1 ? '1 particle' : `${S.count} particles`) : (S.count === 1 ? '1 string' : `${S.count} strings`), 'gold');
      }
      if (!narrowUI) {
        // the interaction point, with its lab coordinates (x, t) on a second, dimmer line
        const has = S.mode === 1 || nSad > 0, px = S.mode === 1 ? V.x : markers[0].tgt.x, py = S.mode === 1 ? V.y : markers[0].tgt.y;
        const iq = has ? (Math.round(px * 100) * 10000 + Math.round(py * 100)) * 4 + S.mode : -1 - S.mode;
        if (iq !== RQ.ip) {
          RQ.ip = iq;
          const head = S.mode === 1 ? 'fixed for everyone' : 'depends on observer';
          ro.set('ip', has ? `${head}<br><span style="color:var(--ink-dim);font-size:10px">at x ${fmtS(px)}, t ${fmtS(py)}</span>` : head, S.mode === 1 ? 'mag' : 'gold');
        }
      }
    }

    // ---- sound: the slice is a chord ------------------------------------------------------------
    function stopVoices(rel = 0.6) {
      if (vA) vA.stop(rel); if (vB) vB.stop(rel); if (vZ1) vZ1.stop(rel); if (vZ2) vZ2.stop(rel);
      vA = vB = vZ1 = vZ2 = null;
      sentA = sentB = sentZ = NaN; sentMerged = null; gainA = gainB = gainZ = -1; panMerged = null;
    }
    function updateAudio() {
      if (!audio.on) return;
      if (S.mode !== 1) {
        if (vZ1) { vZ1.stop(0.5); vZ2.stop(0.5); vZ1 = vZ2 = null; gainZ = -1; sentZ = NaN; }
        if (!vA) {
          // two strings, two tones; their pitch follows each loop's length (snapped into the key)
          vA = audio.additive(audio.freq(semA), 6, { gain: 0 }); vA.setAmps(STRING_AMPS);
          vB = audio.additive(audio.freq(semB), 6, { gain: 0 }); vB.setAmps(STRING_AMPS);
          sentA = semA; sentB = semB; sentMerged = false; panMerged = null; gainA = gainB = -1;
        }
        if (semA !== sentA) { vA.setFreq(audio.freq(semA), merged ? 0.22 : 0.07); sentA = semA; }
        // the merged voice is detuned a hair against its twin, so the one string beats softly; resend when that changes
        if (semB !== sentB || merged !== sentMerged) { vB.setFreq(audio.freq(semB) * (merged ? 1.003 : 1), merged ? 0.35 : 0.07); sentB = semB; sentMerged = merged; }
        if (panMerged !== merged) { vA.setPan(merged ? -0.06 : -0.42); vB.setPan(merged ? 0.06 : 0.42); panMerged = merged; }
        const base = (0.045 + 0.06 * S.act) * S.sheetA, gB = merged ? base * 0.6 : base;
        if (Math.abs(base - gainA) > 0.003) { vA.setGain(base, 0.15); gainA = base; }
        if (Math.abs(gB - gainB) > 0.003) { vB.setGain(gB, 0.15); gainB = gB; }
      } else {
        if (vA) { vA.stop(0.6); vB.stop(0.6); vA = vB = null; sentA = sentB = NaN; gainA = gainB = -1; }
        if (!vZ1) {
          // a buzzing D3 + A3 whose beating quickens as the slice nears the vertex
          vZ1 = audio.additive(audio.freq(12), 10, { gain: 0 }); vZ1.setAmps(BUZZ_AMPS); vZ1.setPan(-0.2);
          vZ2 = audio.additive(audio.freq(19), 8, { gain: 0 }); vZ2.setAmps(BUZZ_AMPS); vZ2.setPan(0.2);
          gainZ = -1; sentZ = NaN;
        }
        const prox = Math.exp(-Math.abs(S.T - vertexT()) / 0.3);
        const g = (0.022 + 0.085 * prox) * S.lineA;
        if (Math.abs(g - gainZ) > 0.003) { vZ1.setGain(g, 0.1); vZ2.setGain(g * 0.8, 0.1); gainZ = g; }
        const det = 1.004 + 0.03 * prox;
        if (Math.abs(det - sentZ) > 0.0008) { vZ2.setFreq(audio.freq(19) * det, 0.08); sentZ = det; }
      }
    }

    // ---- picking ---------------------------------------------------------------------------
    // Screen position of a world point, in canvas CSS pixels, into SX/SY.
    let SX = 0, SY = 0;
    function toScreen(x, y, z) {
      SCR.set(x, y, z).project(camera);
      SX = (SCR.x * 0.5 + 0.5) * ctx.width; SY = (-SCR.y * 0.5 + 0.5) * ctx.height;
    }
    /** Index of the interaction-point marker under the pointer, or -1. */
    function markerHit(ev) {
      if (S.mode === 1) return -1;
      const r = ctx.isTouch ? 42 : 26;
      for (let i = 0; i < 2; i++) {
        const m = markers[i];
        if (!m.on || m.a < 0.3) continue;
        toScreen(m.pos.x, m.pos.y, m.pos.z);
        if ((SX - ev.x) * (SX - ev.x) + (SY - ev.y) * (SY - ev.y) < r * r) return i;
      }
      return -1;
    }
    /**
     * Dragging a marker: pick the observer whose interaction point (on the locus) is nearest the pointer. The
     * pointer is projected onto each screen-space segment between neighbouring samples, so the speed is continuous.
     */
    function dragMarker(ev) {
      const L = locus[S.mode === 2 ? 1 : 0];
      let bd = Infinity, bv = NaN, px = 0, py = 0, pv = 0, have = false;
      for (let j = 0; j < L.n; j++) {
        if (L.which[j] !== dragMarkerIdx) continue;
        toScreen(L.arr[j * 3], L.arr[j * 3 + 1], L.arr[j * 3 + 2]);
        if (have) {
          const ex = SX - px, ey = SY - py, l2 = ex * ex + ey * ey;
          const u = l2 > 1e-6 ? clamp(((ev.x - px) * ex + (ev.y - py) * ey) / l2, 0, 1) : 0;
          const qx = px + ex * u - ev.x, qy = py + ey * u - ev.y, d = qx * qx + qy * qy;
          if (d < bd) { bd = d; bv = lerp(pv, L.v[j], u); }
        } else {
          const d = (SX - ev.x) * (SX - ev.x) + (SY - ev.y) * (SY - ev.y);
          if (d < bd) { bd = d; bv = L.v[j]; }
        }
        px = SX; py = SY; pv = L.v[j]; have = true;
      }
      if (isFinite(bv)) { setTilt(bv, 0); userTilt(); }
    }
    function worldPerPx() {
      const d = camera.position.distanceTo(controls.target);
      return 2 * d * Math.tan(THREE.MathUtils.degToRad(camera.fov) * 0.5) / Math.max(1, ctx.height);
    }
    function hitTest(ndc) {
      raycaster.setFromCamera(ndc, camera);
      const o = raycaster.ray.origin, d = raycaster.ray.direction;
      if (S.mode === 1) {
        for (let c = 0; c < 3; c++) {
          const a = WL[c];
          for (let i = 0; i < a.length; i += 6) {
            const px = a[i] - o.x, py = a[i + 1] - o.y, pz = a[i + 2] - o.z, tp = px * d.x + py * d.y + pz * d.z;
            if (tp > 0 && px * px + py * py + pz * pz - tp * tp < 0.1) return true;
          }
        }
        return false;
      }
      let t0 = 0, t1 = 1e9;
      for (let ax = 0; ax < 3; ax++) {
        const oo = ax === 0 ? o.x : ax === 1 ? o.y : o.z, dd = ax === 0 ? d.x : ax === 1 ? d.y : d.z, B = ax === 0 ? BOX.x : ax === 1 ? BOX.y : BOX.z;
        if (Math.abs(dd) < 1e-9) { if (oo < -B || oo > B) return false; continue; }
        let ta = (-B - oo) / dd, tb = (B - oo) / dd;
        if (ta > tb) { const tmp = ta; ta = tb; tb = tmp; }
        t0 = Math.max(t0, ta); t1 = Math.min(t1, tb);
        if (t0 > t1) return false;
      }
      for (let i = 0, t = t0; i < 90 && t < t1; i++) {
        const ds = sdf(o.x + d.x * t, o.y + d.y * t, o.z + d.z * t, S.W) - 0.16;
        if (ds < 0.01) return true;
        t += Math.max(ds * 0.7, 0.015);
      }
      return false;
    }

    const api = {
      scene, camera,

      enter() {
        entered = true; controls.enabled = true; drag = null; idle = 99; sweep.on = false;
        buildUI();
        showHint();
        computedT = NaN; saddleDirty = true; lensDirty = true; quietFrames = 4; prevVertexSide = 0; clearTrail();
        solveNow();
        for (let i = 0; i < 2; i++) markers[i].pos.copy(markers[i].tgt);
        enterTime = -1;   // the locus of interaction points draws itself in, one observer per frame (see update)
      },

      exit() {
        entered = false; controls.enabled = false; drag = null; sweep.on = false;
        stopVoices();
        ctx.setCursor('');
        ro = null; uiC = null; lensCanvas = null; lensCtx = null;
      },

      resize(w, h) {
        const portrait = w / Math.max(1, h) < 1, landPhone = !portrait && ctx.isTouch && h < 520;
        const frame = portrait ? 'portrait' : landPhone ? 'land' : 'desk';
        camera.fov = portrait ? 50 : landPhone ? 40 : 34;
        // reframe only when the framing changes (the engine also calls resize when it lowers the pixel
        // ratio, and that must not undo the viewer's zoom)
        if (frame !== framed) {
          framed = frame;
          off.copy(camera.position).sub(controls.target);
          if (off.lengthSq() < 1e-6) off.set(0, 0, 1);
          off.setLength(portrait ? 15.5 : landPhone ? 17 : 15.0);
          camera.position.copy(controls.target).add(off);
        }
        axisGroup.visible = nowLabel.visible = nowTick.visible = frame === 'desk';   // no room for the axis on a phone
        if (portrait) camera.setViewOffset(w, h, 0, h * 0.07, w, h);
        // landscape phone: the bottom sheet covers the lower half, so lift the hero into the top half, right of the title
        else if (landPhone) camera.setViewOffset(w, h, -w * 0.04, h * 0.24, w, h);
        else camera.setViewOffset(w, h, -w * 0.045, 0, w, h);
        camera.updateProjectionMatrix();
        const sc = pxScale();
        dust.material.uniforms.uScale.value = sc; bokeh.material.uniforms.uScale.value = sc;
        sparkMat.uniforms.uScale.value = sc; locus[0].mat.uniforms.uScale.value = sc; locus[1].mat.uniforms.uScale.value = sc;
        // crossing 760 px wide (or 820 px tall, for the note) needs the other readout layout
        const wantNarrow = ctx.width <= 760, wantNote = !wantNarrow && ctx.height >= 820;
        if (entered && (wantNarrow !== narrowUI || wantNote !== noteUI)) buildUI();
        else if (lensCanvas) sizeLens();
      },

      update(dt, t) {
        time = t; idle += dt;
        if (DEBUG_LOG) { dbgFrames++; if (dbgFrames % 60 === 0) console.log('m4 frames', dbgFrames, 't', t.toFixed(2), 'perf', (performance.now() / 1000).toFixed(2), 'pr', ctx.pixelRatio.toFixed(2)); }
        if (quietFrames > 0) quietFrames--;
        if (enterTime < 0) enterTime = t;
        if (DEBUG_SWEEP && entered && !dbgSwept && t - enterTime > 1) { dbgSwept = true; sweepButton(); }
        if (DEBUG_DRAG && entered && t - enterTime > 1.6 && dbgDrag < 3) debugDrag(t - enterTime);

        // ---- state easing ----
        S.W += (S.Wt - S.W) * damp(2.4, dt);
        if (Math.abs(S.Wt - S.W) < 0.004) S.W = S.Wt;
        S.sheetA += ((S.mode === 1 ? 0 : 1) - S.sheetA) * damp(3.2, dt);
        S.lineA += ((S.mode === 1 ? 1 : 0) - S.lineA) * damp(3.2, dt);
        if (sweep.on) {
          sweep.t += dt;
          const u = Math.min(1, sweep.t / sweep.dur), e = u * u * (3 - 2 * u);
          S.Tt = lerp(sweep.from, sweep.to, e);
          if (u >= 1) sweep.on = false;
        }
        const T0 = S.T;
        S.T += (S.Tt - S.T) * damp(drag === 'time' ? 22 : 9, dt);
        S.vx += (S.vxt - S.vx) * damp(7, dt);
        S.vz += (S.vzt - S.vz) * damp(7, dt);
        if (Math.abs(S.Tt - S.T) < 1e-4) S.T = S.Tt;
        if (Math.abs(S.vxt - S.vx) < 1e-4) S.vx = S.vxt;
        if (Math.abs(S.vzt - S.vz) < 1e-4) S.vz = S.vzt;
        S.act += (Math.min(1, Math.abs(S.T - T0) / Math.max(dt, 1e-3) * 0.6) - S.act) * damp(4, dt);

        // ---- analysis (only when something moved) ----
        const vMoved = S.vx !== computedVx || S.vz !== computedVz, wMoved = S.W !== computedW;
        if (vMoved || wMoved || S.T !== computedT) {
          recomputeSlice();
          computedT = S.T; computedVx = S.vx; computedVz = S.vz; computedW = S.W;
          if (vMoved || wMoved) saddleDirty = true;
        }
        const settled = S.W === S.Wt;
        if (saddleDirty && settled && (time - lastSolve > 0.07 || onFit())) solveNow();   // the fitted locus is cheap
        // fill in the locus of interaction points, one observer per frame
        if (S.mode !== 1 && settled && !drag) stepLocus(S.mode === 2 ? 1 : 0);

        // ---- worldsheet + slice ----
        flash = Math.max(0, flash - dt * 1.3);
        stepTrail(dt);
        SU.uTime.value = t; SU.uW.value = S.W; SU.uSheet.value = S.sheetA; SU.uFlash.value = flash * 1.4;
        SU.uPix.value = 2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) * 0.5) / Math.max(1, ctx.height * ctx.pixelRatio);
        SU.uSlice.value.set(S.vx, S.vz, S.T);
        sheet.visible = S.sheetA > 0.004 && !params.has('nosheet');
        nrm.set(-S.vx, 1, -S.vz).normalize();
        diskGroup.quaternion.setFromUnitVectors(UP, nrm);
        diskGroup.position.set(0, S.T, 0);
        diskMat.uniforms.uTime.value = t;
        diskMat.uniforms.uA.value = 0.9 + 0.35 * flash;
        {
          const ny = clamp(S.T + S.vx * AXIS.x + S.vz * AXIS.z, -3.35, 3.25);
          nowLabel.position.set(AXIS.x - 0.42, ny + 0.02, AXIS.z); nowTick.position.set(AXIS.x, ny, AXIS.z);
          nowLabel.material.opacity = 0.9; nowTick.material.opacity = 0.8 + 0.2 * Math.sin(t * 2.4);
        }

        // ---- interaction-point markers ----
        for (let i = 0; i < 2; i++) {
          const m = markers[i];
          const want = m.on && S.mode !== 1 && settled ? S.sheetA : 0;
          m.a += (want - m.a) * damp(5, dt);
          m.pos.lerp(m.tgt, damp(10, dt));
          m.g.visible = m.a > 0.01;
          if (m.g.visible) {
            m.g.position.copy(m.pos);
            // the marker where the strings just joined (or split) swells to 1.8x and settles back over 0.4 s
            const jp = i === joinIdx ? Math.max(0, 1 - (t - joinTime) / 0.4) : 0;
            m.glow.scale.setScalar(0.5 * (1 + 0.18 * Math.sin(t * 4.2 + i * 1.7) + flash * 0.5) * (1 + 0.8 * jp));
            m.core.scale.setScalar(0.1 * (1 + 0.8 * jp));
            m.glow.material.opacity = m.a; m.core.material.opacity = m.a;
            const R = (t * 0.42) % 1;
            m.ring.scale.setScalar(0.24 + R * 1.0); m.ring.material.opacity = m.a * (1 - R);
          }
        }
        for (let li = 0; li < 2; li++) {
          const L = locus[li], w = S.sheetA * (li === 1 ? S.W : 1 - S.W);
          // dots while the locus draws itself in; then a continuous curve with the dots as faint beads on it
          L.mat.uniforms.uA.value = w * (L.fit ? 0.45 : 0.9);
          L.pts.visible = L.mat.uniforms.uA.value > 0.01;
          L.a += ((L.fit ? w : 0) - L.a) * damp(3, dt);
          L.group.visible = L.a > 0.01;
          if (L.group.visible) for (let k = 0; k < L.mats.length; k++) L.mats[k].m.opacity = L.a * L.mats[k].k;
        }

        // ---- particles: worldlines, the vertex, the beads where the slice crosses them ----
        const la = S.lineA, show = la > 0.01;
        lineGroup.visible = vertexGroup.visible = sparks.visible = show;
        vFlash = Math.max(0, vFlash - dt * 1.5);
        if (show) {
          const TV = vertexT(), prox = Math.exp(-Math.abs(S.T - TV) / 0.3);
          for (let k = 0; k < 4; k++) lineMats[k].uniforms.uA.value = la;
          vGlow.scale.setScalar(1.3 + 0.25 * Math.sin(t * 5.3) + 0.7 * prox + vFlash);
          vGlow.material.opacity = la * (0.7 + 0.3 * prox); vCore.material.opacity = la;
          // the anamorphic flare grows as the slice approaches the vertex, and flashes when it crosses
          const fl = la * Math.min(1, 0.16 + 0.84 * prox + vFlash);
          streakH.material.opacity = fl; streakV.material.opacity = fl * 0.75;
          streakH.scale.set((2.0 + 1.0 * prox + 1.4 * vFlash) * (1 + 0.04 * Math.sin(t * 7.1)), 0.06, 1);
          streakV.scale.set(0.06, 0.8 + 0.6 * prox + 0.6 * vFlash, 1);
          const ip = 1 + 0.07 * Math.sin(t * 3.1) + 0.15 * vFlash;
          infLabel.scale.set(infBase.x * ip, infBase.y * ip, 1);
          infLabel.material.opacity = la * (0.75 + 0.25 * Math.sin(t * 3.1));
          sparkMat.uniforms.uTime.value = t; sparkMat.uniforms.uA.value = la; sparkMat.uniforms.uHeat.value = prox + vFlash;
          const side = S.T > TV ? 1 : -1;
          if (S.mode === 1 && prevVertexSide !== 0 && side !== prevVertexSide && time - lastEvent > 0.12) {
            lastEvent = time; vFlash = 1;
            audio.noise({ freq: 2200, q: 0.9, dur: 0.6, gain: 0.07, sweepTo: 280 });
            audio.pluck(audio.freq(12), { gain: 0.2, decay: 1.6, bright: 1, partials: 14, inharm: 0.0012 });
          }
          prevVertexSide = S.mode === 1 ? side : 0;
          postScratch.aberration = 0.0016 + 0.0018 * prox * la;
          ctx.setPost(postScratch);
        } else if (postScratch.aberration !== 0.0016) {
          postScratch.aberration = 0.0016; ctx.setPost(postScratch);
        }
        for (let i = 0; i < 4; i++) {
          const b = beads[i];
          b.visible = show && S.mode === 1 && i < nHits;
          if (b.visible) {
            b.position.set(hits[i * 3], hits[i * 3 + 1], hits[i * 3 + 2]);
            b.children[0].material.opacity = la; b.children[1].material.opacity = la;
          }
        }

        backlight.material.color.lerpColors(BACK_SHEET, BACK_LINES, la);

        // ---- camera: slow idle sway + breathing ----
        if (!drag && idle > 3 && !reduced) {
          const a = 0.05 * Math.cos(t * 0.11) * dt, c = Math.cos(a), s = Math.sin(a);
          off.copy(camera.position).sub(controls.target);
          const x = off.x * c + off.z * s, z = -off.x * s + off.z * c;
          camera.position.set(controls.target.x + x, camera.position.y, controls.target.z + z);
        }
        controls.target.y = TARGET_Y + (reduced ? 0 : 0.06 * Math.sin(t * 0.33));
        controls.update(dt);

        // ---- atmosphere ----
        bgMat.uniforms.uTime.value = t;
        stars.userData.update(t);
        dust.material.uniforms.uTime.value = t;
        bokeh.material.uniforms.uTime.value = t;

        // ---- panels and sound ----
        if (lensDirty) drawLens();
        roTimer -= dt;
        if (ro && roTimer <= 0) {
          roTimer = 0.1; updateReadout();
          // the readout can change width after the webfonts arrive: keep the lens canvas at its real size
          if (lensCanvas && ++lensCheck % 5 === 0) {
            const want = Math.round(lensCanvas.clientWidth * Math.min(window.devicePixelRatio || 1, 2));
            if (want > 4 && want !== lensCanvas.width) sizeLens();
          }
        }
        if (uiC) {
          if (Math.abs(S.Tt - sliderSyncT) > 0.004) { uiC.set('time', S.Tt); sliderSyncT = S.Tt; }
          if (Math.abs(S.vxt - sliderSyncV) > 0.004) { uiC.set('tilt', S.vxt); sliderSyncV = S.vxt; }
        }
        updateAudio();
      },

      // drag states: 'pend' (on the sheet, direction not yet known), 'pendM' (on a marker, touch), 'time', 'tilt1'
      // (sideways on the sheet), 'tilt2' (shift-drag, both axes), 'marker', 'pinch', 'orbit'
      onPointerDown(ev) {
        idle = 0;
        const o = ev.original, id = o && o.pointerId !== undefined ? o.pointerId : -1;
        if (o && o.isPrimary === false) {
          // a second finger: pinch to zoom. On empty space OrbitControls already tracks both fingers; when the first
          // finger is on the sheet or a marker OrbitControls never saw it, so the pinch is done here.
          if (drag && drag !== 'orbit' && drag !== 'pinch') {
            pinch.id = id; pinch.x = ev.x; pinch.y = ev.y;
            pinch.d0 = Math.max(20, Math.hypot(ev.x - prim.x, ev.y - prim.y));
            pinch.len0 = camera.position.distanceTo(controls.target);
            drag = 'pinch'; controls.enabled = false; ctx.setCursor('');
          }
          return;
        }
        prim.id = id; prim.x = press.x = ev.x; prim.y = press.y = ev.y;
        if (ev.shift) { drag = 'tilt2'; controls.enabled = false; ctx.setCursor('grabbing'); return; }
        const mh = markerHit(ev);
        if (mh >= 0) { drag = ctx.isTouch ? 'pendM' : 'marker'; dragMarkerIdx = mh; sweep.on = false; controls.enabled = false; ctx.setCursor('grabbing'); return; }
        if (hitTest(ev.ndc)) { drag = 'pend'; sweep.on = false; controls.enabled = false; ctx.setCursor('grabbing'); return; }
        drag = 'orbit';
      },
      onPointerMove(ev) {
        const o = ev.original, id = o && o.pointerId !== undefined ? o.pointerId : -1;
        if (drag === 'pinch') {
          if (id === pinch.id) { pinch.x = ev.x; pinch.y = ev.y; } else if (id === prim.id) { prim.x = ev.x; prim.y = ev.y; }
          const d = Math.max(20, Math.hypot(pinch.x - prim.x, pinch.y - prim.y));
          off.copy(camera.position).sub(controls.target).setLength(clamp(pinch.len0 * pinch.d0 / d, controls.minDistance, controls.maxDistance));
          camera.position.copy(controls.target).add(off); idle = 0;
          return;
        }
        if (o && o.isPrimary === false) return;   // only the first finger drives time, tilt and the marker
        if (id === prim.id) { prim.x = ev.x; prim.y = ev.y; }
        if (drag === 'pend' || drag === 'pendM') {
          // the direction locks after 6 px: up/down moves through time, sideways changes the observer
          const ddx = ev.x - press.x, ddy = ev.y - press.y;
          if (ddx * ddx + ddy * ddy < 36) return;
          if (Math.abs(ddy) > Math.abs(ddx)) { drag = 'time'; S.Tt = clamp(S.Tt - ddy * worldPerPx(), -T_MAX, T_MAX); userTime(); }
          else if (drag === 'pendM') { drag = 'marker'; dragMarker(ev); }
          else { drag = 'tilt1'; setTilt(S.vxt + ddx * 0.003, S.vzt); userTilt(); }
          return;
        }
        if (drag === 'time') { S.Tt = clamp(S.Tt - ev.dy * worldPerPx(), -T_MAX, T_MAX); userTime(); }
        else if (drag === 'tilt1') { setTilt(S.vxt + ev.dx * 0.003, S.vzt); userTilt(); }
        else if (drag === 'tilt2') { setTilt(S.vxt + ev.dx * 0.003, S.vzt - ev.dy * 0.003); userTilt(); }
        else if (drag === 'marker') dragMarker(ev);
        else if (drag === 'orbit') idle = 0;
        else if (ev.pointerType === 'mouse') ctx.setCursor(markerHit(ev) >= 0 ? 'point' : hitTest(ev.ndc) ? 'grab' : '');
      },
      onPointerUp() {
        // phones have no note: after the first drag through time, point at the next thing to try
        if (drag === 'time' && ctx.isTouch && !told.markerTip && S.mode !== 1 && nSad > 0) {
          told.markerTip = true; ctx.ui.toast('Drag the magenta point to change the observer.', 3800);
        }
        if (drag && drag !== 'orbit') controls.enabled = entered;
        drag = null; ctx.setCursor('');
      },
      onWheel() { idle = 0; },
      onKey(ev) {
        const k = ev.key;
        if (k === 'ArrowUp' || k === 'ArrowDown') {
          sweep.on = false;
          S.Tt = clamp(S.Tt + (k === 'ArrowUp' ? 1 : -1) * (ev.shift ? 0.4 : 0.1), -T_MAX, T_MAX);
          userTime(); return true;
        }
        if (k === ',' || k === '<' || k === '.' || k === '>') {
          setTilt(S.vxt + (k === ',' || k === '<' ? -0.05 : 0.05), S.vzt); userTilt(); return true;
        }
        return false;
      },

      demo() {
        // strings, a moving observer, the slice just after this observer's join: gold contour + marker both visible
        setMode(0);
        S.W = S.Wt;
        setTilt(0.5, 0); S.vx = S.vxt; S.vz = S.vzt;
        const n = solveSaddles(0.5, 0, 0, TMP6);
        const Tc = n ? TMP6[1] - 0.5 * TMP6[0] : -0.3;   // this observer's time of the join
        sweep.on = false; S.T = Tc - 0.06; S.Tt = Tc + 0.26;
        saddleDirty = true; solveNow();
        idle = 0; ctx.ui.hideHint(); hintShown = false;
      },
    };
    return api;
  },
};
