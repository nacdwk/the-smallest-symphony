// VI. The Landscape (Finale, maestoso)
//
// Night flight over the string landscape. Height is vacuum energy. One periodic heightfield
// (256 x 256 Float32Array, built once on the CPU) drives everything: the GPU samples it with a
// cubic B-spline (four bilinear taps) to displace the terrain and draw anti-aliased contour lines,
// and the marble physics evaluates the very same B-spline (sixteen taps) on the CPU, so the marbles
// roll on exactly the surface you see. Because the field tiles seamlessly, the map has no edges.
//
// Λ = 0 is drawn as a glassy level. Valleys below it (negative Λ) are dark lakes; a thin mint
// shore just above it holds the valleys whose Λ is tiny and positive, like ours. A steepest-descent
// watershed labels every texel with the valley it drains into; a valley's floor height sets its Λ
// and its id seeds the rest of its laws. Universes land, roll, settle and inflate into bubbles whose
// behaviour follows their fate: they crunch, dilute, flatten, or fill with galaxies.
import * as THREE from 'three';
import { palette, glowTexture, makeStarfield, GLSL, clamp, lerp, smoothstep, damp, sup, rng } from '../fx.js';

// ---- constants -------------------------------------------------------------------------------
const TAU = Math.PI * 2;
const N = 256;                  // heightfield texels per side
const MASK = N - 1;
const NN = N * N;
const P = 128;                  // world units per period: the map tiles seamlessly every P units
const CELL = P / N;
const BAND = 0.5;               // floors in [0, BAND] have a tiny positive Λ (drawn absurdly thick)
const SPACING = 0.62;           // contour interval in world units (major line every 5th)
const MAXU = 36;                // universes alive at once
const NAMB = 7;                 // faint far bubbles (eternal inflation elsewhere)
const NGHOST = 3;               // collapsing bubbles left behind by tunnelling
const NBUB = MAXU + NAMB + NGHOST;
const TRAIL = 44;               // trail samples per marble
const NL = 16;                  // light slots that tint the contour lines
const NPULSE = 6;               // expanding rings of light across the map
const NRIP = 8;                 // ripples on the Λ = 0 lakes
const NPATH = 320;              // dots in the hover descent path
const MR = 0.34;                // marble radius
const G_FALL = 38, G_ROLL = 34, FRICTION = 1.75;
const PH_FALL = 0, PH_ROLL = 1, PH_GLIDE = 2, PH_LIVE = 3, PH_TUNNEL = 4;
const T_PLANCK = 5.39e-44;      // s
const YEAR = 3.156e7;           // s
const MINT = '#7dffc4', EMBER = '#ff6a3d', GOLD = '#ffc46b';

const DRONE = { notes: [0, 7, 12, 15, 19], opts: { gain: 0.1, cutoff: 1050, wobble: 0.35 } };
const POST = { bloomStrength: 0.8, bloomRadius: 0.55, bloomThreshold: 0.62, exposure: 1.0, aberration: 0.0015, grain: 0.034, vignette: 0.92 };
const POST_CODA = { bloomStrength: 1.0, bloomRadius: 0.7, bloomThreshold: 0.55, exposure: 1.04, vignette: 1.0 };

const CLOSING = 'String theory has not been tested by experiment. In most versions its strings are far too small for any collider we can imagine, and the landscape makes sharp predictions hard. It is still our most developed idea for joining quantum mechanics and gravity, and it has already changed mathematics and our understanding of black holes. If it is right, somewhere below every particle in your body, a string is playing.';

// ---- the landscape: built once on the CPU ------------------------------------------------------
/** Periodic gradient noise + fbm on the unit torus, so the map tiles with no seams. */
function makeNoise(seed) {
  const R = rng(seed);
  const perm = new Uint8Array(256);
  for (let i = 0; i < 256; i++) perm[i] = i;
  for (let i = 255; i > 0; i--) { const j = Math.floor(R() * (i + 1)); const t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
  const gx = new Float32Array(256), gy = new Float32Array(256);
  for (let i = 0; i < 256; i++) { const a = R() * TAU; gx[i] = Math.cos(a); gy[i] = Math.sin(a); }
  const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10);
  const pn = (x, y, per, o) => {
    const ix = Math.floor(x), iy = Math.floor(y), fx = x - ix, fy = y - iy;
    const x0 = ((ix % per) + per) % per, y0 = ((iy % per) + per) % per;
    const x1 = (x0 + 1) % per, y1 = (y0 + 1) % per;
    const h00 = perm[(perm[(x0 + o) & 255] + y0) & 255], h10 = perm[(perm[(x1 + o) & 255] + y0) & 255];
    const h01 = perm[(perm[(x0 + o) & 255] + y1) & 255], h11 = perm[(perm[(x1 + o) & 255] + y1) & 255];
    const n00 = gx[h00] * fx + gy[h00] * fy, n10 = gx[h10] * (fx - 1) + gy[h10] * fy;
    const n01 = gx[h01] * fx + gy[h01] * (fy - 1), n11 = gx[h11] * (fx - 1) + gy[h11] * (fy - 1);
    const u = fade(fx), v = fade(fy);
    const a = n00 + (n10 - n00) * u, b = n01 + (n11 - n01) * u;
    return 1.41 * (a + (b - a) * v);
  };
  const fbm = (u, v, f, oct, gain, o) => {
    let s = 0, a = 1, n = 0;
    for (let k = 0; k < oct; k++) { s += a * pn(u * f, v * f, f, o + k * 31); n += a; a *= gain; f *= 2; }
    return s / n;
  };
  return { pn, fbm };
}

function quantile(sorted, q) { return sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(q * (sorted.length - 1))))]; }

/** Cubic B-spline weights and their derivatives (shared by CPU sampling). */
function bsw(f, w, d) {
  const f2 = f * f, f3 = f2 * f, m = 1 - f;
  w[0] = m * m * m / 6; w[1] = (3 * f3 - 6 * f2 + 4) / 6; w[3] = f3 / 6; w[2] = 1 - w[0] - w[1] - w[3];
  d[0] = -0.5 * m * m; d[1] = 1.5 * f2 - 2 * f; d[3] = 0.5 * f2; d[2] = -d[0] - d[1] - d[3];
}

function buildLandscape(seed) {
  const R = rng(seed * 7 + 3);
  const { pn, fbm } = makeNoise(seed);
  const H = new Float32Array(NN);
  const base = new Float32Array(NN), ridge = new Float32Array(NN);
  // pass 1: warped fbm continents and ridged ranges
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const u = (i + 0.5) / N, v = (j + 0.5) / N;
    const qx = fbm(u + 0.13, v + 0.71, 2, 3, 0.5, 11), qy = fbm(u + 0.57, v + 0.29, 2, 3, 0.5, 23);
    const wu = u + 0.12 * qx, wv = v + 0.12 * qy;
    base[j * N + i] = fbm(wu, wv, 2, 5, 0.5, 51);
    let r = 0, a = 1;
    for (let k = 0; k < 3; k++) { const f = 4 << k; const q = pn(wu * f + 0.3 * qy, wv * f - 0.3 * qx, f, 97 + k * 13); const n = 1 - Math.sqrt(q * q + 0.02); r += a * n * n; a *= 0.48; }
    ridge[j * N + i] = r / 1.71;
  }
  const bs = Float32Array.from(base).sort();
  const b0 = quantile(bs, 0.02), b1 = quantile(bs, 0.98);
  // explicit basins: a hero crater near the default view, big valleys, many small ones
  const basins = [{ x: 0.5 + 9 / P, z: 0.5 - 7 / P, r: 13 / P, d: 7.5, flat: 1 }];
  for (let i = 0; i < 7; i++) basins.push({ x: R(), z: R(), r: (6 + R() * 7) / P, d: 2.5 + R() * 4.5, flat: R() < 0.5 ? 1 : 0 });
  for (let i = 0; i < 170; i++) basins.push({ x: R(), z: R(), r: (1.8 + R() * 3.6) / P, d: 0.7 + R() * 2.4, flat: 0 });
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const u = (i + 0.5) / N, v = (j + 0.5) / N, p = j * N + i;
    const b = clamp((base[p] - b0) / (b1 - b0), 0, 1);
    let h = 13 * Math.pow(b, 1.5) + 4.5 * ridge[p] * smoothstep(0.42, 0.85, b);
    h += 1.3 * fbm(u, v, 20, 2, 0.5, 151) + 0.5 * pn(u * 36, v * 36, 36, 163);
    for (let k = 0; k < basins.length; k++) {
      const B = basins[k];
      let dx = Math.abs(u - B.x); dx = Math.min(dx, 1 - dx);
      let dz = Math.abs(v - B.z); dz = Math.min(dz, 1 - dz);
      const r = Math.sqrt(dx * dx + dz * dz) / B.r;
      if (r > 2.4) continue;
      const bowl = B.flat ? 1 / (1 + r * r * r * r) : Math.exp(-r * r);
      const rim = (r - 1.25) / 0.42;
      h += -B.d * bowl + 0.3 * B.d * Math.exp(-rim * rim);
    }
    H[p] = h;
  }

  // The B-spline surface passes through the [1 4 1]/6 smoothed grid at the texel centres.
  const tmp = new Float32Array(NN), Hs = new Float32Array(NN);
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const r = j * N;
    tmp[r + i] = (H[r + ((i - 1) & MASK)] + 4 * H[r + i] + H[r + ((i + 1) & MASK)]) / 6;
  }
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    Hs[j * N + i] = (tmp[((j - 1) & MASK) * N + i] + 4 * tmp[j * N + i] + tmp[((j + 1) & MASK) * N + i]) / 6;
  }

  // Catchments: D8 steepest descent, then merge basins that are too shallow or tiny to hold a marble.
  const DI = [1, -1, 0, 0, 1, 1, -1, -1], DJ = [0, 0, 1, -1, 1, -1, 1, -1];
  const down = new Int32Array(NN);
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const p = j * N + i; let best = p, bs2 = 0;
    for (let k = 0; k < 8; k++) {
      const q = ((j + DJ[k]) & MASK) * N + ((i + DI[k]) & MASK);
      const s = (Hs[p] - Hs[q]) * (k < 4 ? 1 : 0.70710678);
      if (s > bs2) { bs2 = s; best = q; }
    }
    down[p] = best;
  }
  const root = new Int32Array(NN).fill(-1), stack = new Int32Array(NN);
  for (let p = 0; p < NN; p++) {
    let q = p, n = 0;
    while (root[q] < 0 && down[q] !== q) { stack[n++] = q; q = down[q]; }
    if (root[q] < 0) root[q] = q;
    const r = root[q];
    while (n > 0) root[stack[--n]] = r;
  }
  const uf = new Int32Array(NN); for (let p = 0; p < NN; p++) uf[p] = p;
  const find = (a) => { while (uf[a] !== a) { uf[a] = uf[uf[a]]; a = uf[a]; } return a; };
  const minH = new Float32Array(NN), area = new Int32Array(NN), spillH = new Float32Array(NN), spillTo = new Int32Array(NN);
  for (let round = 0; round < 10; round++) {
    minH.fill(Infinity); area.fill(0); spillH.fill(Infinity); spillTo.fill(-1);
    for (let p = 0; p < NN; p++) { const r = find(root[p]); area[r]++; if (Hs[p] < minH[r]) minH[r] = Hs[p]; }
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const p = j * N + i, rp = find(root[p]);
      for (let k = 0; k < 2; k++) {
        const q = k === 0 ? j * N + ((i + 1) & MASK) : ((j + 1) & MASK) * N + i;
        const rq = find(root[q]);
        if (rp === rq) continue;
        const s = Hs[p] > Hs[q] ? Hs[p] : Hs[q];
        if (s < spillH[rp]) { spillH[rp] = s; spillTo[rp] = rq; }
        if (s < spillH[rq]) { spillH[rq] = s; spillTo[rq] = rp; }
      }
    }
    let merged = 0;
    for (let p = 0; p < NN; p++) {
      if (uf[p] !== p || area[p] === 0 || spillTo[p] < 0) continue;
      if (spillH[p] - minH[p] < 0.06 || area[p] < 8) {
        const t = find(spillTo[p]);
        if (t !== p) { uf[p] = t; merged++; }
      }
    }
    if (!merged) break;
  }
  const lab = new Float32Array(NN), idOf = new Int32Array(NN).fill(-1), valleys = [];
  for (let p = 0; p < NN; p++) {
    const r = find(root[p]);
    if (idOf[r] < 0) { idOf[r] = valleys.length; valleys.push({ id: valleys.length, p, h: Hs[p], area: 0 }); }
    const v = valleys[idOf[r]];
    v.area++; if (Hs[p] < v.h) { v.h = Hs[p]; v.p = p; }
    lab[p] = idOf[r];
  }

  // Refine each valley floor on the continuous B-spline surface (what the marbles feel).
  const W = { wx: new Float32Array(4), dx: new Float32Array(4), wz: new Float32Array(4), dz: new Float32Array(4) };
  const out = new Float32Array(3);
  const sample = (x, z) => sampleBS(H, x, z, out, W);
  for (const v of valleys) {
    let x = ((v.p % N) + 0.5) * CELL - P / 2, z = (Math.floor(v.p / N) + 0.5) * CELL - P / 2;
    sample(x, z); let h = out[0], step = 0.25;
    for (let it = 0; it < 60 && step > 0.002; it++) {
      const gx = out[1], gz = out[2], g = Math.hypot(gx, gz);
      if (g < 1e-5) break;
      const nx = x - (gx / g) * step, nz = z - (gz / g) * step;
      sample(nx, nz);
      if (out[0] < h) { x = nx; z = nz; h = out[0]; step *= 1.2; } else { step *= 0.5; sample(x, z); }
    }
    v.x = x; v.z = z; v.h = h;
  }

  // Sea level (Λ = 0): put a cluster of floors just above it, keep tiny-negative floors off the map,
  // and flood a modest fraction of the area.
  const hsSorted = Float32Array.from(Hs).sort();
  const fracBelow = (s) => { let lo = 0, hi = NN; while (lo < hi) { const m = (lo + hi) >> 1; if (hsSorted[m] < s) lo = m + 1; else hi = m; } return lo / NN; };
  const mins = valleys.map((v) => v.h).sort((a, b) => a - b);
  let sea = quantile(hsSorted, 0.12), best = -1e9;
  for (let k = 0; k < mins.length; k++) {
    const s = mins[k] - 0.06;
    const fb = fracBelow(s);
    if (fb < 0.05 || fb > 0.24) continue;
    let inBand = 0, near = 0;
    for (let q = 0; q < mins.length; q++) { const h = mins[q] - s; if (h > 0 && h <= BAND) inBand++; else if (h <= 0 && h > -0.7) near++; }
    const score = inBand * 2 - near * 6 - Math.abs(fb - 0.13) * 25;
    if (score > best) { best = score; sea = s; }
  }
  for (let p = 0; p < NN; p++) { H[p] -= sea; Hs[p] -= sea; }
  for (const v of valleys) v.h -= sea;

  // Pitch, Λ class and colour per valley. Deeper valleys sing lower.
  const byH = valleys.slice().sort((a, b) => a.h - b.h);
  byH.forEach((v, r) => { v.deg = Math.min(9, Math.floor((r / byH.length) * 10)); });
  let nBand = 0, nLake = 0;
  for (const v of valleys) {
    v.kind = v.h < 0 ? 2 : v.h <= BAND ? 1 : 0;
    if (v.kind === 1) nBand++; else if (v.kind === 2) nLake++;
  }
  const hTop = byH[byH.length - 1].h, hBot = byH[0].h;

  // Texture payload: h, dh/dx, dh/dz, and how far below its surroundings each point lies (mist).
  const blurA = new Float32Array(NN), blurB = new Float32Array(NN);
  const RAD = 22, sig = 9, ker = new Float32Array(2 * RAD + 1);
  let ks = 0; for (let k = -RAD; k <= RAD; k++) { ker[k + RAD] = Math.exp(-(k * k) / (2 * sig * sig)); ks += ker[k + RAD]; }
  for (let k = 0; k < ker.length; k++) ker[k] /= ks;
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) { let s = 0; for (let k = -RAD; k <= RAD; k++) s += ker[k + RAD] * H[j * N + ((i + k) & MASK)]; blurA[j * N + i] = s; }
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) { let s = 0; for (let k = -RAD; k <= RAD; k++) s += ker[k + RAD] * blurA[((j + k) & MASK) * N + i]; blurB[j * N + i] = s; }
  const rgba = new Float32Array(NN * 4);
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const p = j * N + i, o = p * 4;
    rgba[o] = H[p];
    rgba[o + 1] = (H[j * N + ((i + 1) & MASK)] - H[j * N + ((i - 1) & MASK)]) / (2 * CELL);
    rgba[o + 2] = (H[((j + 1) & MASK) * N + i] - H[((j - 1) & MASK) * N + i]) / (2 * CELL);
    rgba[o + 3] = blurB[p] - H[p];
  }
  const sortedH = Float32Array.from(H).sort();
  return {
    H, rgba, lab, valleys, nBand, nLake, hTop, hBot,
    hLo: quantile(sortedH, 0.03), hHi: quantile(sortedH, 0.985), lakeFrac: fracBelow(sea) ,
  };
}

/** Cubic B-spline height and gradient of the periodic field at world (x, z). Same maths as the shader. */
function sampleBS(H, x, z, out, W) {
  const sx = (x / P + 0.5) * N - 0.5, sz = (z / P + 0.5) * N - 0.5;
  const ix = Math.floor(sx), iz = Math.floor(sz);
  bsw(sx - ix, W.wx, W.dx); bsw(sz - iz, W.wz, W.dz);
  let h = 0, hx = 0, hz = 0;
  for (let b = 0; b < 4; b++) {
    const row = ((iz - 1 + b) & MASK) * N;
    let r = 0, rd = 0;
    for (let a = 0; a < 4; a++) { const v = H[row + ((ix - 1 + a) & MASK)]; r += W.wx[a] * v; rd += W.dx[a] * v; }
    h += W.wz[b] * r; hx += W.wz[b] * rd; hz += W.dz[b] * r;
  }
  out[0] = h; out[1] = hx / CELL; out[2] = hz / CELL;
  return out;
}

/** Small periodic value-noise texture for mist and shimmer. */
function makeNoiseTexture() {
  const S = 128, R = rng(4242), data = new Uint8Array(S * S);
  const acc = new Float32Array(S * S);
  let amp = 1, tot = 0;
  for (let per = 4; per <= 64; per *= 2) {
    const lat = new Float32Array(per * per); for (let k = 0; k < lat.length; k++) lat[k] = R();
    for (let j = 0; j < S; j++) for (let i = 0; i < S; i++) {
      const x = (i / S) * per, y = (j / S) * per, x0 = Math.floor(x), y0 = Math.floor(y);
      let fx = x - x0, fy = y - y0; fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy);
      const a = lat[(y0 % per) * per + (x0 % per)], b = lat[(y0 % per) * per + ((x0 + 1) % per)];
      const c = lat[((y0 + 1) % per) * per + (x0 % per)], d = lat[((y0 + 1) % per) * per + ((x0 + 1) % per)];
      acc[j * S + i] += amp * ((a + (b - a) * fx) * (1 - fy) + (c + (d - c) * fx) * fy);
    }
    tot += amp; amp *= 0.55;
  }
  let lo = 1e9, hi = -1e9; for (let k = 0; k < acc.length; k++) { lo = Math.min(lo, acc[k]); hi = Math.max(hi, acc[k]); }
  for (let k = 0; k < acc.length; k++) data[k] = Math.round(255 * (acc[k] - lo) / (hi - lo));
  const t = new THREE.DataTexture(data, S, S, THREE.RedFormat, THREE.UnsignedByteType);
  t.wrapS = t.wrapT = THREE.RepeatWrapping; t.magFilter = THREE.LinearFilter; t.minFilter = THREE.LinearMipmapLinearFilter;
  t.generateMipmaps = true; t.needsUpdate = true;
  return t;
}

/** Non-uniform grid axis: fine near the centre, geometric growth outward. */
function axisCoords(fine, step, outer, q) {
  const n = Math.round(fine / step), mid = [];
  for (let k = -n; k <= n; k++) mid.push(k * step);
  const ext = []; let x = n * step, s = step;
  while (x < outer) { s *= q; x += s; ext.push(x); }
  return ext.map((v) => -v).reverse().concat(mid, ext);
}
function gridGeometry(xs, zs) {
  const nx = xs.length, nz = zs.length;
  const pos = new Float32Array(nx * nz * 3);
  for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) { const o = (j * nx + i) * 3; pos[o] = xs[i]; pos[o + 2] = zs[j]; }
  const idx = new Uint32Array((nx - 1) * (nz - 1) * 6);
  let t = 0;
  for (let j = 0; j < nz - 1; j++) for (let i = 0; i < nx - 1; i++) {
    const a = j * nx + i, b = a + 1, c = a + nx, d = c + 1;
    idx[t++] = a; idx[t++] = c; idx[t++] = b; idx[t++] = b; idx[t++] = c; idx[t++] = d;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  return g;
}

// ---- physics prose: what each valley's Λ means ------------------------------------------------------
function fmtTime(s) {
  if (s >= 1e9 * YEAR) { const g = s / (1e9 * YEAR); return `${g < 10 ? g.toFixed(1) : Math.round(g)} billion years`; }
  if (s >= 1e6 * YEAR) return `${Math.round(s / (1e6 * YEAR))} million years`;
  if (s >= 1e4 * YEAR) return `${(Math.round(s / (1e3 * YEAR)) * 1000).toLocaleString('en-US')} years`;
  if (s >= YEAR) return `${Math.round(s / YEAR).toLocaleString('en-US')} years`;
  if (s >= 86400) return `${Math.round(s / 86400)} days`;
  if (s >= 3600) return `${Math.round(s / 3600)} hours`;
  if (s >= 60) return `${Math.round(s / 60)} minutes`;
  if (s >= 1) return `${Math.round(s)} seconds`;
  return `10${sup(Math.floor(Math.log10(s)))} s`;
}

// ---- GLSL ------------------------------------------------------------------------------------------
const GLSL_HF = /* glsl */`
  uniform highp sampler2D uHF;
  // Cubic B-spline of the periodic heightfield via four bilinear taps (GPU Gems 2, ch. 20).
  vec4 hfS(vec2 xz) {
    vec2 st = (xz * ${(1 / P).toFixed(9)} + 0.5) * ${N}.0 - 0.5;
    vec2 i = floor(st), f = st - i;
    vec2 f2 = f * f, f3 = f2 * f, m = 1.0 - f;
    vec2 w0 = m * m * m / 6.0, w1 = (3.0 * f3 - 6.0 * f2 + 4.0) / 6.0, w3 = f3 / 6.0;
    vec2 w2 = 1.0 - w0 - w1 - w3;
    vec2 g0 = w0 + w1, g1 = w2 + w3;
    vec2 p0 = (i - 0.5 + w1 / g0) / ${N}.0, p1 = (i + 1.5 + w3 / g1) / ${N}.0;
    vec4 a = textureLod(uHF, vec2(p0.x, p0.y), 0.0), b = textureLod(uHF, vec2(p1.x, p0.y), 0.0);
    vec4 c = textureLod(uHF, vec2(p0.x, p1.y), 0.0), d = textureLod(uHF, vec2(p1.x, p1.y), 0.0);
    return g0.y * (g0.x * a + g1.x * b) + g1.y * (g0.x * c + g1.x * d);
  }
`;
const GLSL_SKY = /* glsl */`
  uniform float uFogD, uFogStart;
  // The night sky and the haze the far terrain dissolves into (they must match at the horizon).
  vec3 skyBase(vec3 d) {
    float e = d.y;
    vec3 zen = vec3(0.0012, 0.0010, 0.0042);
    vec3 hor = vec3(0.016, 0.0085, 0.030);
    vec3 c = mix(hor, zen, smoothstep(-0.03, 0.45, e));
    float g = max(e, -0.25) / 0.06;
    c += vec3(0.050, 0.017, 0.030) * exp(-g * g) + vec3(0.012, 0.005, 0.016) * exp(-abs(g) * 0.35);
    return c;
  }
  float fogF(float dist) { float x = max(dist - uFogStart, 0.0) * uFogD; return 1.0 - exp(-x * x); }
`;

const TERRAIN_VS = /* glsl */`
  ${GLSL_HF}
  uniform float uDrop;
  varying vec3 vW;
  void main() {
    vec4 wp = modelMatrix * vec4(position, 1.0);
    wp.y = hfS(wp.xz).x - uDrop;
    vW = wp.xyz;
    gl_Position = projectionMatrix * viewMatrix * wp;
  }
`;
const TERRAIN_FS = /* glsl */`
  ${GLSL_HF}
  ${GLSL_SKY}
  uniform highp sampler2D uLab;
  uniform sampler2D uNoise;
  uniform float uTime, uHlo, uHhi, uBasins, uCoda, uLines;
  uniform vec3 uDbg;
  uniform vec4 uHover;               // x, z, strength, label
  uniform vec4 uLights[${NL}];       // xyz, intensity
  uniform vec3 uLightC[${NL}];
  uniform vec4 uPulse[${NPULSE}];    // x, z, start time, strength
  uniform vec3 uPulseC[${NPULSE}];
  varying vec3 vW;

  float contour(float v, float w) {
    float fw = max(fwidth(v), 1e-5);
    float d = abs(fract(v + 0.5) - 0.5) / fw;
    return 1.0 - smoothstep(w - 0.65, w + 0.65, d);
  }
  vec3 basinHue(float l) {
    float t = fract(l * 0.6180339);
    vec3 a = vec3(1.0, 0.77, 0.42), b = vec3(1.0, 0.31, 0.75), c = vec3(0.55, 0.42, 1.0), d = vec3(0.44, 0.95, 1.0), e = vec3(0.49, 1.0, 0.77);
    float x = t * 5.0;
    return x < 1.0 ? mix(a, b, x) : x < 2.0 ? mix(b, c, x - 1.0) : x < 3.0 ? mix(c, d, x - 2.0) : x < 4.0 ? mix(d, e, x - 3.0) : mix(e, a, x - 4.0);
  }

  void main() {
    vec4 s = hfS(vW.xz);
    float h = s.x;
    vec3 n = normalize(vec3(-s.y, 1.0, -s.z));
    vec3 V = cameraPosition - vW; float dist = length(V); V /= dist;
    float t = clamp((h - uHlo) / (uHhi - uHlo), 0.0, 1.0);

    // height ramp: gold valley floors, rose slopes, violet ridges
    vec3 cLo = vec3(1.0, 0.52, 0.13), cMid = vec3(1.0, 0.27, 0.50), cHi = vec3(0.50, 0.36, 1.0);
    vec3 lc = t < 0.32 ? cLo : t < 0.6 ? mix(cLo, cMid, (t - 0.32) / 0.28) : mix(cMid, cHi, smoothstep(0.6, 0.9, t));
    float lum = mix(1.0, 0.24, smoothstep(0.0, 0.75, t));
    // below Λ = 0 the vacuum energy is negative: the drowned contours turn cold
    float under = smoothstep(0.0, -1.2, h);
    lc = mix(lc, vec3(0.38, 0.72, 1.0), under);
    lum *= 1.0 - 0.35 * under;

    // contour lines (screen-space anti-aliased); minor lines fade where they crowd closer than ~3 px
    float v = h / ${SPACING.toFixed(3)};
    float fw = fwidth(v);
    float minor = contour(v, 0.5) * (1.0 - smoothstep(0.16, 0.45, fw));
    float major = contour(v / 5.0, 0.8) * (1.0 - smoothstep(0.12, 0.34, fw / 5.0));
    float lines = (minor * 0.26 + major) * uLines + smoothstep(0.15, 0.8, fw) * 0.07;

    // slow shimmer so the map breathes
    vec2 q = vW.xz * 0.011 + vec2(uTime * 0.0035, uTime * 0.0022);
    float nz = texture(uNoise, q).r * 0.6 + texture(uNoise, q * 2.7 - vec2(uTime * 0.003)).r * 0.4;

    // surface: dark, with a whisper of relief so the forms read
    vec3 Lm = normalize(vec3(-0.45, 0.55, -0.7));
    float dif = max(dot(n, Lm), 0.0);
    vec3 col = mix(vec3(0.010, 0.007, 0.018), vec3(0.014, 0.010, 0.032), t) + dif * vec3(0.014, 0.010, 0.028);
    col += pow(1.0 - max(dot(n, V), 0.0), 3.0) * vec3(0.022, 0.014, 0.05);
    // mist pooled in valleys
    float mist = smoothstep(0.2, 3.4, s.w) * (0.35 + 0.65 * nz) * (1.0 - 0.85 * smoothstep(0.0, -0.6, h));
    col += mist * vec3(0.075, 0.032, 0.014) * (1.0 - 0.5 * t);

    // light from marbles, bubbles and the cursor, caught mostly by the lines
    vec3 lit = vec3(0.0);
    for (int i = 0; i < ${NL}; i++) {
      vec4 Lp = uLights[i];
      if (Lp.w <= 0.0) break;            // the list is packed: the first empty slot ends it
      vec3 d = Lp.xyz - vW;
      float d2 = dot(d, d);
      if (d2 > 900.0) continue;
      // a local pool of light (about 5 units across) with a faint tail
      lit += uLightC[i] * Lp.w * (exp(-d2 * 0.035) + 0.6 / (1.0 + d2 * 0.6));
    }
    // rings of light that sweep the map when a universe settles
    vec3 pulse = vec3(0.0);
    for (int i = 0; i < ${NPULSE}; i++) {
      vec4 p = uPulse[i];
      float age = uTime - p.z;
      if (age < 0.0 || age > 5.0 || p.w <= 0.0) continue;
      if (abs(length(vW.xz - p.xy) - age * 14.0) > 8.0) continue;
      float r = length(vW.xz - p.xy), fr = age * 14.0, wdt = 0.9 + age * 0.6;
      float x = (r - fr) / wdt;
      float life = 1.0 - age / 5.0;
      pulse += uPulseC[i] * exp(-x * x) * life * life * p.w * 0.6;
    }

    // catchment under the cursor, and the basins-of-attraction map
    float hov = 0.0, hovEdge = 0.0; vec3 bcol = vec3(0.0); float border = 0.0;
    if (uHover.z > 0.001 || uBasins > 0.001) {
      vec2 st = (vW.xz * ${(1 / P).toFixed(9)} + 0.5) * ${N}.0 - 0.5;
      vec2 fi = floor(st), f = st - fi;
      ivec2 ii = ivec2(fi);
      vec4 Lb = vec4(
        texelFetch(uLab, ivec2(ii.x & ${MASK}, ii.y & ${MASK}), 0).r,
        texelFetch(uLab, ivec2((ii.x + 1) & ${MASK}, ii.y & ${MASK}), 0).r,
        texelFetch(uLab, ivec2(ii.x & ${MASK}, (ii.y + 1) & ${MASK}), 0).r,
        texelFetch(uLab, ivec2((ii.x + 1) & ${MASK}, (ii.y + 1) & ${MASK}), 0).r);
      vec4 wts = vec4((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
      if (uHover.z > 0.001) {
        float m = dot(vec4(equal(Lb, vec4(uHover.w))), wts);
        hov = smoothstep(0.3, 0.7, m) * uHover.z;
        // the catchment's outline: an anti-aliased line where the mask crosses one half
        hovEdge = (1.0 - smoothstep(0.0, max(fwidth(m) * 1.4, 0.02), abs(m - 0.5))) * uHover.z * (1.0 - smoothstep(50.0, 140.0, dist));
      }
      if (uBasins > 0.001) {
        bcol = basinHue(Lb.x) * wts.x + basinHue(Lb.y) * wts.y + basinHue(Lb.z) * wts.z + basinHue(Lb.w) * wts.w;
        float mx = max(max(dot(vec4(equal(Lb, vec4(Lb.x))), wts), dot(vec4(equal(Lb, vec4(Lb.y))), wts)),
                       max(dot(vec4(equal(Lb, vec4(Lb.z))), wts), dot(vec4(equal(Lb, vec4(Lb.w))), wts)));
        border = smoothstep(0.2, 0.45, 1.0 - mx) * (1.0 - smoothstep(60.0, 160.0, dist));
      }
    }

    vec3 lineCol = lc * lum * (0.85 + 0.3 * nz) * (1.0 - 0.25 * uCoda);
    col += lineCol * lines * (1.0 + 1.2 * hov);
    col += lit * (0.012 + 0.85 * lines) + pulse * (0.02 + 1.0 * lines);
    col += hovEdge * vec3(1.0, 0.8, 0.5) * 0.75;

    // Λ = 0: the shoreline, and the thin band of tiny positive Λ just above it
    float sh = abs(h) / max(fwidth(h), 1e-5);
    float shore = 1.0 - smoothstep(0.35, 1.3, sh);
    float band = step(0.0, h) * (1.0 - smoothstep(${(BAND * 0.7).toFixed(3)}, ${BAND.toFixed(3)}, h));
    col += vec3(0.42, 1.0, 0.72) * (shore * 0.62 * (1.0 + 0.6 * uCoda) + band * 0.03);

    // cursor ring
    float cr = length(vW.xz - uHover.xy);
    float ringW = max(fwidth(cr) * 1.2, 0.03);
    col += vec3(1.0, 0.86, 0.6) * (1.0 - smoothstep(0.0, ringW, abs(cr - 1.2))) * uHover.z * 0.9;

    if (uBasins > 0.001) col = mix(col, col * 0.6 + bcol * (0.035 + 0.16 * lines) + bcol * border * 0.42, uBasins);

    col = mix(col, skyBase(-V), fogF(dist));
    gl_FragColor = vec4(col + uDbg, 1.0);
  }
`;

const WATER_VS = /* glsl */`
  varying vec3 vW;
  void main() { vec4 wp = modelMatrix * vec4(position, 1.0); vW = wp.xyz; gl_Position = projectionMatrix * viewMatrix * wp; }
`;
const WATER_FS = /* glsl */`
  ${GLSL_HF}
  ${GLSL_SKY}
  uniform sampler2D uNoise;
  uniform float uTime;
  uniform vec4 uRip[${NRIP}];
  uniform vec4 uLights[${NL}];
  uniform vec3 uLightC[${NL}];
  varying vec3 vW;
  float hash(vec3 p) { p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
  void main() {
    float depth = -hfS(vW.xz).x;
    if (depth < 0.0) discard;
    vec3 V = cameraPosition - vW; float dist = length(V); V /= dist;
    vec2 g = vec2(0.0);
    vec2 q = vW.xz * 0.07 + uTime * vec2(0.010, 0.007);
    float e = 0.01;
    float n0 = texture(uNoise, q).r;
    g += vec2(texture(uNoise, q + vec2(e, 0.0)).r - n0, texture(uNoise, q + vec2(0.0, e)).r - n0) * 0.9;
    for (int i = 0; i < ${NRIP}; i++) {
      vec4 r = uRip[i];
      float age = uTime - r.z;
      if (age < 0.0 || age > 4.5 || r.w <= 0.0) continue;
      vec2 d = vW.xz - r.xy; float rr = length(d) + 1e-3, fr = age * 4.2;
      float x = (rr - fr) / (0.9 + age * 0.5);
      g += (d / rr) * cos((rr - fr) * 4.0) * exp(-x * x) * (1.0 - age / 4.5) * r.w * 0.3;
    }
    vec3 N = normalize(vec3(-g.x, 1.0, -g.y));
    float nv = max(dot(N, V), 0.0);
    float F = 0.03 + 0.97 * pow(1.0 - nv, 5.0);
    vec3 R = reflect(-V, N);
    vec3 refl = skyBase(R) * 1.1;
    // stars in the water
    vec3 sp = floor(R * 260.0);
    refl += vec3(0.9, 0.85, 1.0) * step(0.9965, hash(sp)) * smoothstep(0.05, 0.3, R.y) * 0.8;
    // bubbles and marbles mirrored in the lake
    for (int i = 0; i < 10; i++) {
      vec4 Lp = uLights[i];
      if (Lp.w <= 0.0) break;
      vec3 Ld = Lp.xyz - vW; float dl = length(Ld); Ld /= dl;
      float c = max(dot(R, Ld), 0.0);
      refl += uLightC[i] * Lp.w * (pow(c, 700.0) * 10.0 + pow(c, 60.0) * 0.07);
    }
    // a faint sheen so the level reads as a surface even when it reflects little
    float sheen = 0.5 + 0.5 * sin(dot(vW.xz, vec2(0.9, 0.4)) * 0.6 + n0 * 9.0 + uTime * 0.3);
    vec3 col = refl * F + vec3(0.004, 0.008, 0.016) + vec3(0.02, 0.05, 0.06) * pow(sheen, 6.0) * (0.25 + F);
    float foam = 1.0 - smoothstep(0.0, max(0.035, fwidth(depth) * 1.6), depth);
    col += vec3(0.49, 1.0, 0.77) * foam * 0.35;
    float a = mix(0.42, 0.8, smoothstep(0.0, 5.0, depth));
    float fg = fogF(dist);
    col = mix(col, skyBase(-V) * a, fg);
    gl_FragColor = vec4(col, a);
  }
`;

const MIST_VS = /* glsl */`
  ${GLSL_HF}
  uniform float uLift;
  varying vec3 vW; varying float vDep;
  void main() {
    vec4 wp = modelMatrix * vec4(position, 1.0);
    vec4 s = hfS(wp.xz);
    wp.y = max(s.x, 0.0) + uLift;
    vW = wp.xyz; vDep = s.w;
    gl_Position = projectionMatrix * viewMatrix * wp;
  }
`;
const MIST_FS = /* glsl */`
  ${GLSL_SKY}
  uniform sampler2D uNoise;
  uniform float uTime, uAmt, uLayer;
  uniform vec2 uCenter;
  varying vec3 vW; varying float vDep;
  void main() {
    vec2 q = vW.xz * 0.018 + vec2(uTime * 0.007 * (uLayer - 0.5), uTime * 0.004);
    float n = texture(uNoise, q).r * texture(uNoise, q * 2.3 + 0.37 + uLayer).r * 1.8;
    float d = smoothstep(0.25, 3.0, vDep) * smoothstep(0.18, 0.8, n);
    float dist = length(cameraPosition - vW);
    float edge = 1.0 - smoothstep(70.0, 96.0, length(vW.xz - uCenter));
    float fade = (1.0 - fogF(dist)) * smoothstep(4.0, 22.0, dist) * edge;
    vec3 c = mix(vec3(1.0, 0.55, 0.26), vec3(0.62, 0.36, 0.95), uLayer) * d * uAmt * fade;
    gl_FragColor = vec4(c, 1.0);
  }
`;

const SKY_VS = /* glsl */`
  varying vec3 vD;
  void main() { vD = position; vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0); gl_Position = p.xyww; }
`;
const SKY_FS = /* glsl */`
  ${GLSL_SKY}
  uniform sampler2D uNoise;
  uniform float uCoda;
  varying vec3 vD;
  void main() {
    vec3 d = normalize(vD);
    vec3 c = skyBase(d);
    // a faint galactic band across the sky
    vec3 gN = normalize(vec3(0.25, 0.5, 0.83));
    float gb = dot(d, gN) / 0.16;
    float band = exp(-gb * gb) * smoothstep(0.0, 0.25, d.y);
    vec2 sq = vec2(atan(d.z, d.x) * 0.9549, d.y * 3.0);
    float n = texture(uNoise, sq).r * 0.65 + texture(uNoise, sq * 3.1 + 0.4).r * 0.35;
    c += vec3(0.07, 0.045, 0.095) * band * n * n * (1.0 + 0.6 * uCoda);
    gl_FragColor = vec4(c, 1.0);
  }
`;

const BUBBLE_VS = /* glsl */`
  attribute vec3 aCol;
  attribute vec4 aP;           // alpha, seed, glow, galaxy/heat
  varying vec3 vW, vN, vL, vCol; varying vec4 vP;
  void main() {
    vec4 wp = modelMatrix * instanceMatrix * vec4(position, 1.0);
    vec3 sc = vec3(length(instanceMatrix[0].xyz), length(instanceMatrix[1].xyz), length(instanceMatrix[2].xyz));
    vN = normalize(mat3(modelMatrix) * (normal / max(sc, vec3(1e-4))));
    vW = wp.xyz; vL = position; vCol = aCol; vP = aP;
    gl_Position = projectionMatrix * viewMatrix * wp;
  }
`;
const BUBBLE_FS = /* glsl */`
  ${GLSL_SKY}
  ${GLSL.iridescence}
  uniform sampler2D uNoise;
  uniform float uTime;
  varying vec3 vW, vN, vL, vCol; varying vec4 vP;
  void main() {
    if (vP.x < 0.003) discard;
    vec3 N = normalize(vN);
    vec3 V = normalize(cameraPosition - vW);
    float a = abs(dot(N, V));
    float fres = pow(1.0 - a, 2.4);
    // a thin soap film: thickness swirls and drains toward the bottom
    vec2 fuv = vec2(atan(vL.z, vL.x) * 0.3183 + vP.y * 3.7, vL.y * 0.6 - uTime * 0.025);
    float sw = (texture(uNoise, fuv).r + texture(uNoise, fuv * 2.3 + vec2(uTime * 0.01, 0.0)).r) - 1.0;
    float thick = 0.42 + 0.22 * sw - 0.2 * vL.y + 0.22 * a;
    // the film keeps its rainbow, but leans toward the colour of this universe's fate
    vec3 film = mix(iridescence(thick * 1.9 + vP.y), vCol, 0.32);
    float side = gl_FrontFacing ? 1.0 : 0.5;
    vec3 col = film * (0.03 + 0.95 * fres) * (0.8 + 0.5 * vP.z) + vCol * pow(fres, 1.6) * (0.42 + 0.9 * vP.z);
    col += vCol * pow(a, 5.0) * (0.02 + 0.06 * vP.z);
    // window glints
    vec3 R = reflect(-V, N);
    float sp = pow(max(dot(R, normalize(vec3(-0.4, 0.75, 0.35))), 0.0), 150.0) * 3.2
             + pow(max(dot(R, normalize(vec3(0.62, 0.32, -0.55))), 0.0), 260.0) * 1.6;
    col += vec3(1.0, 0.95, 0.9) * sp * (gl_FrontFacing ? 1.0 : 0.0);
    // heat for a universe in its final crunch
    col += vec3(1.0, 0.35, 0.12) * vP.w * (0.4 + fres * 2.0);
    col *= side * vP.x * (1.0 - fogF(length(cameraPosition - vW)));
    gl_FragColor = vec4(col, 1.0);
  }
`;

const GLOW_VS = /* glsl */`
  attribute vec3 aCol; attribute float aSize;
  uniform float uScale;
  varying vec3 vC;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    vC = aCol;
    gl_PointSize = aSize > 0.0 ? min(aSize * uScale * 520.0 / max(1.0, -mv.z), 34.0 * uScale) : 0.0;
  }
`;
const GLOW_FS = /* glsl */`
  varying vec3 vC;
  void main() {
    vec2 p = gl_PointCoord - 0.5; float d2 = dot(p, p) * 4.0;
    float a = exp(-d2 * 5.0) * 0.8 + exp(-d2 * 40.0) * 0.6;
    gl_FragColor = vec4(vC * a, 1.0);
  }
`;

const TRAIL_VS = /* glsl */`
  attribute float aA; attribute vec3 aC;
  varying float vA; varying vec3 vC;
  void main() { vA = aA; vC = aC; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
`;
const TRAIL_FS = /* glsl */`
  varying float vA; varying vec3 vC;
  void main() { if (vA < 0.004) discard; gl_FragColor = vec4(vC * vA, 1.0); }
`;

const DOTS_VS = /* glsl */`
  ${GLSL_SKY}
  attribute float aKind, aLabel, aSeed;
  uniform float uTime, uScale, uHoverL, uAmt, uWaveT;
  uniform vec2 uWaveC;
  varying vec3 vC; varying float vA;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    float d = -mv.z;
    float hov = abs(aLabel - uHoverL) < 0.5 ? 1.0 : 0.0;
    vC = aKind < 0.5 ? vec3(1.0, 0.86, 0.68) : aKind < 1.5 ? vec3(0.49, 1.0, 0.77) : vec3(0.6, 0.82, 1.0);
    float tw = 0.7 + 0.3 * sin(uTime * (0.7 + aSeed * 1.9) + aSeed * 40.0);
    vec3 wpos = (modelMatrix * vec4(position, 1.0)).xyz;
    float wave = uWaveT > 0.0 ? smoothstep(0.0, 14.0, uWaveT * 22.0 - length(wpos.xz - uWaveC)) : 0.0;
    vA = (aKind > 0.5 && aKind < 1.5 ? 2.6 : 1.7) * tw * (1.0 - fogF(d) * (1.0 - 0.6 * wave)) * (1.0 + 2.2 * hov) * uAmt * (1.0 + 2.2 * wave);
    gl_PointSize = (5.0 + 4.0 * hov) * uScale * clamp(70.0 / d, 0.6, 2.0);
  }
`;
const DOTS_FS = /* glsl */`
  varying vec3 vC; varying float vA;
  void main() { vec2 p = gl_PointCoord - 0.5; float r = length(p) * 2.0; float a = exp(-r * r * 14.0) * 1.2 + exp(-r * r * 3.5) * 0.18; gl_FragColor = vec4(vC * vA * a, 1.0); }
`;

const GAL_VS = /* glsl */`
  attribute vec4 aOff; attribute float aSlot;
  uniform vec4 uBub[${MAXU}];
  uniform float uBubG[${MAXU}];
  uniform float uTime, uScale;
  varying float vA; varying vec3 vC;
  void main() {
    int k = int(aSlot + 0.5);
    vec4 b = uBub[k]; float g = uBubG[k];
    float ang = uTime * (0.12 + 0.2 * aOff.w);
    float c = cos(ang), s = sin(ang);
    vec3 o = aOff.xyz; o.xz = mat2(c, -s, s, c) * o.xz;
    vec4 mv = modelViewMatrix * vec4(b.xyz + o * b.w * 0.8, 1.0);
    gl_Position = projectionMatrix * mv;
    vA = g * (0.55 + 0.45 * sin(uTime * (1.0 + 3.0 * aOff.w) + aOff.w * 60.0));
    vC = mix(vec3(1.0, 0.86, 0.62), vec3(0.62, 0.85, 1.0), step(0.6, fract(aOff.w * 7.0)));
    gl_PointSize = g > 0.001 ? (1.2 + 2.0 * aOff.w) * uScale * clamp(70.0 / -mv.z, 0.4, 2.5) : 0.0;
  }
`;
const GAL_FS = /* glsl */`
  varying float vA; varying vec3 vC;
  void main() { vec2 p = gl_PointCoord - 0.5; float a = smoothstep(0.5, 0.0, length(p)); gl_FragColor = vec4(vC * vA * a * 1.6, 1.0); }
`;

const ARC_VS = /* glsl */`
  attribute float aT;
  uniform float uProg, uTime, uScale;
  varying float vA, vT;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    vT = aT;
    float dash = step(0.42, fract(aT * 34.0 - uTime * 2.2));
    float head = exp(-pow((aT - uProg) * 14.0, 2.0));
    vA = step(aT, uProg) * (0.35 + 0.65 * dash) + head * 2.5;
    gl_PointSize = (2.4 + 3.0 * head) * uScale * clamp(90.0 / -mv.z, 0.6, 2.2);
  }
`;
const ARC_FS = /* glsl */`
  uniform float uAlpha; uniform vec3 uColA, uColB;
  varying float vA, vT;
  void main() { vec2 p = gl_PointCoord - 0.5; float a = smoothstep(0.5, 0.05, length(p)); gl_FragColor = vec4(mix(uColA, uColB, vT) * vA * a * uAlpha, 1.0); }
`;

const PATH_VS = /* glsl */`
  attribute float aI;
  uniform float uCount, uTime, uScale, uAmt;
  varying float vA;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    float on = step(aI, uCount - 1.0);
    float flow = pow(fract(aI * 0.06 - uTime * 0.9), 4.0);
    vA = on * (0.28 + 0.9 * flow) * uAmt * smoothstep(0.0, 3.0, aI);
    gl_PointSize = on * (2.2 + 2.0 * flow) * uScale * clamp(80.0 / -mv.z, 0.6, 2.0);
  }
`;
const PATH_FS = /* glsl */`
  varying float vA;
  void main() { vec2 p = gl_PointCoord - 0.5; float a = smoothstep(0.5, 0.0, length(p)); gl_FragColor = vec4(vec3(1.0, 0.86, 0.6) * vA * a * 2.2, 1.0); }
`;

// ---- the movement --------------------------------------------------------------------------------------
export default {
  id: 'landscape',
  numeral: 'VI',
  title: 'The Landscape',
  tempo: 'Finale, maestoso',
  caption: 'String theory seems to allow a vast number of ways to fold its hidden dimensions, perhaps 10<sup>500</sup>. Each is a <em>vacuum</em>: a possible universe with its own laws. Height on this map is vacuum energy. Click to drop a universe and see where it settles.',
  fine: '<p>The curled-up dimensions can be threaded by <em>fluxes</em>, generalised magnetic fields that come in whole-number units. Each choice of flux numbers helps fix the shape and size of the hidden space and gives a different vacuum energy. Bousso and Polchinski (2000) argued that the many choices produce an almost continuous spread of values, fine enough to include one as small as ours. Kachru, Kallosh, Linde and Trivedi (KKLT, 2003) proposed how such vacua could be long-lived (metastable) with a small positive energy. Counting flux choices on typical Calabi–Yau shapes gives estimates like 10<sup>500</sup>; later counts on other geometries are far larger. All of these are rough estimates.</p>' +
    '<p>The real field space has hundreds of directions; this map shows two. Heights are not to scale. Our dark energy is about 10<sup>−122</sup> in Planck units, so the mint band of valleys like ours is drawn enormously too thick. Each valley\'s other laws are illustrative draws, not computed from a real compactification. The doubling and crunch times follow from each Λ by ordinary cosmology.</p>' +
    '<p>Why is our Λ so small? One contested answer is anthropic: if eternal inflation fills the landscape with bubble universes, observers can only arise where Λ is tiny (Weinberg 1987). Many physicists find this unsatisfying. The <em>swampland</em> programme goes further and asks whether long-lived vacua with positive Λ exist in string theory at all.</p>' +
    '<p>Tunnelling between valleys is vacuum decay: a bubble of lower-energy vacuum nucleates and its wall expands at nearly the speed of light (Coleman and De Luccia, 1980). Real vacua are reached through cosmic history, and the rolling marbles only picture it. None of this has been tested by experiment.</p>',
  drone: DRONE,
  post: POST,

  create(ctx) {
    const A = ctx.audio, ui = ctx.ui;
    const low = ctx.quality === 'low';
    const params = new URLSearchParams(location.search);
    const reduced = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
    const motion = reduced ? 0.15 : 1;

    // ---- the landscape -----------------------------------------------------------------------
    const LS = buildLandscape(11);
    const HF = LS.H, LAB = LS.lab, valleys = LS.valleys, K = valleys.length;
    const W = { wx: new Float32Array(4), dx: new Float32Array(4), wz: new Float32Array(4), dz: new Float32Array(4) };
    const S = new Float32Array(3);
    const sample = (x, z) => sampleBS(HF, x, z, S, W);
    const heightAt = (x, z) => sampleBS(HF, x, z, S, W)[0];
    const labelAt = (x, z) => {
      const i = Math.round((x / P + 0.5) * N - 0.5) & MASK, j = Math.round((z / P + 0.5) * N - 0.5) & MASK;
      return LAB[j * N + i] | 0;
    };
    const wrapD = (d) => d - Math.round(d / P) * P;

    // Half floats filter at full rate everywhere (RGBA32F filtering is slow or missing on many GPUs). The
    // quantisation (under 0.01 units) disappears in the B-spline, which blends sixteen texels per sample.
    const h16 = new Uint16Array(LS.rgba.length);
    for (let i = 0; i < h16.length; i++) h16[i] = THREE.DataUtils.toHalfFloat(LS.rgba[i]);
    const hfTex = new THREE.DataTexture(h16, N, N, THREE.RGBAFormat, THREE.HalfFloatType);
    hfTex.wrapS = hfTex.wrapT = THREE.RepeatWrapping;
    hfTex.magFilter = hfTex.minFilter = THREE.LinearFilter; hfTex.generateMipmaps = false; hfTex.needsUpdate = true;
    const labTex = new THREE.DataTexture(LAB, N, N, THREE.RedFormat, THREE.FloatType);
    labTex.wrapS = labTex.wrapT = THREE.RepeatWrapping;
    labTex.magFilter = labTex.minFilter = THREE.NearestFilter; labTex.generateMipmaps = false; labTex.needsUpdate = true;
    const noiseTex = makeNoiseTexture();

    // ---- scene ---------------------------------------------------------------------------------
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, ctx.aspect, 0.5, 2400);
    let clock = 0;
    const fogU = { uFogD: { value: 1 / 150 }, uFogStart: { value: 55 } };
    const timeU = { value: 0 };
    const scaleU = { value: 1 };
    const lightsU = { value: Array.from({ length: NL }, () => new THREE.Vector4(0, 0, 0, 0)) };
    const lightsCU = { value: Array.from({ length: NL }, () => new THREE.Color(0, 0, 0)) };
    const pulseU = { value: Array.from({ length: NPULSE }, () => new THREE.Vector4(0, 0, -99, 0)) };
    const pulseCU = { value: Array.from({ length: NPULSE }, () => new THREE.Color(1, 1, 1)) };
    const ripU = { value: Array.from({ length: NRIP }, () => new THREE.Vector4(0, 0, -99, 0)) };
    const hoverU = { value: new THREE.Vector4(0, 0, 0, -1) };
    const codaU = { value: 0 };

    // sky dome and stars (both follow the camera, so they sit at infinity)
    const sky = new THREE.Mesh(new THREE.SphereGeometry(1000, 48, 24), new THREE.ShaderMaterial({
      vertexShader: SKY_VS, fragmentShader: SKY_FS, side: THREE.BackSide, depthWrite: false,
      uniforms: { ...fogU, uCoda: codaU, uNoise: { value: noiseTex } },
    }));
    // drawn after the opaque terrain: its depth sits on the far plane, so only visible sky pixels are shaded
    sky.renderOrder = 10; sky.frustumCulled = false;
    scene.add(sky);
    const stars = makeStarfield({ count: low ? 1600 : 3200, radius: 860, inner: 720, size: 40, intensity: 1.6 });
    {
      const pa = stars.geometry.attributes.position.array;
      for (let i = 0; i < pa.length; i += 3) { const r = Math.hypot(pa[i], pa[i + 1], pa[i + 2]); pa[i + 1] = Math.abs(pa[i + 1]) + r * 0.05; }
    }
    stars.renderOrder = -9;
    scene.add(stars);

    // terrain: nested square grids centred ahead of the camera (fine to coarse). Every vertex sits on its
    // level's fixed world lattice, so nothing swims. Overlaps are harmless because shading depends only on
    // world xz; coarser levels sit a hair lower so the finer one wins wherever both exist.
    const LODS = low
      ? [{ s: 1.0, n: 120, drop: 0 }, { s: 2.4, n: 100, drop: 0.2 }, { s: 7.5, n: 70, drop: 1.0 }, { s: 24, n: 44, drop: 4 }]
      : [{ s: 0.6, n: 192, drop: 0 }, { s: 1.25, n: 184, drop: 0.14 }, { s: 3.6, n: 128, drop: 0.8 }, { s: 13, n: 76, drop: 3.2 }];
    const terrainMat = new THREE.ShaderMaterial({
      vertexShader: TERRAIN_VS, fragmentShader: TERRAIN_FS,
      uniforms: {
        uHF: { value: hfTex }, uLab: { value: labTex }, uNoise: { value: noiseTex }, uTime: timeU, ...fogU,
        uHlo: { value: LS.hLo }, uHhi: { value: LS.hHi }, uBasins: { value: 0 }, uDrop: { value: 0 }, uDbg: { value: new THREE.Color(0) }, uCoda: codaU, uLines: { value: 1 },
        uHover: hoverU, uLights: lightsU, uLightC: lightsCU, uPulse: pulseU, uPulseC: pulseCU,
      },
    });
    const terrains = LODS.map((l, i) => {
      const ax = []; for (let k = -l.n / 2; k <= l.n / 2; k++) ax.push(k * l.s);
      const mat = i === 0 ? terrainMat : terrainMat.clone();
      // share every uniform object (so per-frame writes reach all levels) except the level's own drop
      mat.uniforms = { ...terrainMat.uniforms, uDrop: { value: l.drop }, uDbg: { value: new THREE.Color(params.has('m6lod') ? ['#300', '#030', '#003', '#330'][i] : '#000') } };
      const m = new THREE.Mesh(gridGeometry(ax, ax), mat);
      m.userData.s = l.s;
      m.frustumCulled = false; m.renderOrder = i;
      scene.add(m);
      return m;
    });

    // Λ = 0: the glassy level
    const water = new THREE.Mesh(new THREE.PlaneGeometry(960, 960, 1, 1).rotateX(-Math.PI / 2), new THREE.ShaderMaterial({
      vertexShader: WATER_VS, fragmentShader: WATER_FS, transparent: true, depthWrite: false,
      blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
      blendSrcAlpha: THREE.ZeroFactor, blendDstAlpha: THREE.OneFactor,
      uniforms: { uHF: { value: hfTex }, uNoise: { value: noiseTex }, uTime: timeU, uRip: ripU, uLights: lightsU, uLightC: lightsCU, ...fogU },
    }));
    water.renderOrder = 4; water.frustumCulled = false;
    scene.add(water);

    // mist shells hugging the valleys (they follow the view, snapped to their own grid)
    const MIST_STEP = low ? 3 : 2;
    const mistGeo = gridGeometry(...[0, 0].map(() => axisCoords(96, MIST_STEP, 96, 1.5)));
    const mists = [];
    const mistCenter = new THREE.Vector2();
    for (let k = 0; k < (low ? 1 : 2); k++) {
      const m = new THREE.Mesh(mistGeo, new THREE.ShaderMaterial({
        vertexShader: MIST_VS, fragmentShader: MIST_FS, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
        uniforms: { uHF: { value: hfTex }, uNoise: { value: noiseTex }, uTime: timeU, uAmt: { value: k ? 0.03 : 0.045 }, uLayer: { value: k }, uLift: { value: 0.7 + k * 1.3 }, uCenter: { value: mistCenter }, ...fogU },
      }));
      m.renderOrder = 5; m.frustumCulled = false;
      scene.add(m); mists.push(m);
    }

    // hundreds of faint dots: every local minimum on the map, repeated over the neighbouring tiles
    const dots = (() => {
      const n = K * 9, pos = new Float32Array(n * 3), kind = new Float32Array(n), labA = new Float32Array(n), seed = new Float32Array(n);
      const R = rng(77);
      let o = 0;
      for (let ty = -1; ty <= 1; ty++) for (let tx = -1; tx <= 1; tx++) for (let k = 0; k < K; k++) {
        const v = valleys[k];
        pos[o * 3] = v.x + tx * P; pos[o * 3 + 1] = v.h + 0.5; pos[o * 3 + 2] = v.z + ty * P;
        kind[o] = v.kind; labA[o] = k; seed[o] = R(); o++;
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      g.setAttribute('aKind', new THREE.BufferAttribute(kind, 1));
      g.setAttribute('aLabel', new THREE.BufferAttribute(labA, 1));
      g.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1));
      const m = new THREE.Points(g, new THREE.ShaderMaterial({
        vertexShader: DOTS_VS, fragmentShader: DOTS_FS, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
        uniforms: { uTime: timeU, uScale: scaleU, uHoverL: { value: -1 }, uAmt: { value: 1 }, uWaveT: { value: 0 }, uWaveC: { value: new THREE.Vector2() }, ...fogU },
      }));
      m.frustumCulled = false; m.renderOrder = 1;
      return m;
    })();
    scene.add(dots);

    // ---- universes: marbles, trails, bubbles, galaxies --------------------------------------------
    const U = [];
    for (let i = 0; i < MAXU; i++) {
      U.push({
        slot: i, active: false, id: 0, phase: 0, x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, still: 0, rollT: 0, wet: false,
        tx: 0, tz: 0, gx0: 0, gz0: 0, gt: 0, k: -1, laws: null, born: 0, R: 2.6, seed: 0, bx: 0, by: 0, bz: 0, floorY: 0,
        flash: 0, dying: 0, decays: 0, crunched: false, alpha: 0, scale: 0, heat: 0, galaxy: 0,
        trail: new Float32Array(TRAIL * 3), tHead: 0, tCount: 0, tAlpha: 0, tLast: 0,
        col: new THREE.Color(), arc: null, arcT: 0, everLife: false, hx0: 0, hy0: 0, hz0: 0,
      });
    }
    const coreMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(MR, 18, 12), new THREE.MeshBasicMaterial({ color: 0xffffff }), MAXU);
    coreMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    coreMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(MAXU * 3), 3);
    coreMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    coreMesh.frustumCulled = false; coreMesh.renderOrder = 2;
    scene.add(coreMesh);

    const glowPos = new Float32Array(MAXU * 3), glowCol = new Float32Array(MAXU * 3), glowSize = new Float32Array(MAXU);
    const glowGeo = new THREE.BufferGeometry();
    glowGeo.setAttribute('position', new THREE.BufferAttribute(glowPos, 3).setUsage(THREE.DynamicDrawUsage));
    glowGeo.setAttribute('aCol', new THREE.BufferAttribute(glowCol, 3).setUsage(THREE.DynamicDrawUsage));
    glowGeo.setAttribute('aSize', new THREE.BufferAttribute(glowSize, 1).setUsage(THREE.DynamicDrawUsage));
    const glows = new THREE.Points(glowGeo, new THREE.ShaderMaterial({
      vertexShader: GLOW_VS, fragmentShader: GLOW_FS, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { uScale: scaleU },
    }));
    glows.frustumCulled = false; glows.renderOrder = 7;
    scene.add(glows);

    const SEGS = MAXU * (TRAIL - 1);
    const trailPos = new Float32Array(SEGS * 6), trailA = new Float32Array(SEGS * 2), trailC = new Float32Array(SEGS * 6);
    const trailGeo = new THREE.BufferGeometry();
    trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3).setUsage(THREE.DynamicDrawUsage));
    trailGeo.setAttribute('aA', new THREE.BufferAttribute(trailA, 1).setUsage(THREE.DynamicDrawUsage));
    trailGeo.setAttribute('aC', new THREE.BufferAttribute(trailC, 3).setUsage(THREE.DynamicDrawUsage));
    const trails = new THREE.LineSegments(trailGeo, new THREE.ShaderMaterial({
      vertexShader: TRAIL_VS, fragmentShader: TRAIL_FS, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    }));
    trails.frustumCulled = false; trails.renderOrder = 3;
    scene.add(trails);

    const bubbleGeo = new THREE.SphereGeometry(1, low ? 32 : 44, low ? 22 : 30);
    const bCol = new Float32Array(NBUB * 3), bP = new Float32Array(NBUB * 4);
    bubbleGeo.setAttribute('aCol', new THREE.InstancedBufferAttribute(bCol, 3).setUsage(THREE.DynamicDrawUsage));
    bubbleGeo.setAttribute('aP', new THREE.InstancedBufferAttribute(bP, 4).setUsage(THREE.DynamicDrawUsage));
    const bubbles = new THREE.InstancedMesh(bubbleGeo, new THREE.ShaderMaterial({
      vertexShader: BUBBLE_VS, fragmentShader: BUBBLE_FS, transparent: true, depthWrite: false,
      blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
      uniforms: { uTime: timeU, uNoise: { value: noiseTex }, ...fogU },
    }), NBUB);
    bubbles.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    bubbles.frustumCulled = false; bubbles.renderOrder = 6;
    scene.add(bubbles);
    const bM = bubbles.instanceMatrix.array;
    const setInst = (arr, i, x, y, z, sx, sy, sz) => {
      const o = i * 16;
      arr[o] = sx; arr[o + 1] = 0; arr[o + 2] = 0; arr[o + 3] = 0;
      arr[o + 4] = 0; arr[o + 5] = sy; arr[o + 6] = 0; arr[o + 7] = 0;
      arr[o + 8] = 0; arr[o + 9] = 0; arr[o + 10] = sz; arr[o + 11] = 0;
      arr[o + 12] = x; arr[o + 13] = y; arr[o + 14] = z; arr[o + 15] = 1;
    };

    // galaxies inside the universes that could hold them: a little cosmic web in each bubble
    const GPS = low ? 48 : 96;
    const galBub = Array.from({ length: MAXU }, () => new THREE.Vector4(0, -999, 0, 0));
    const galG = new Float32Array(MAXU);
    const galaxies = (() => {
      const n = MAXU * GPS, off = new Float32Array(n * 4), slot = new Float32Array(n), R = rng(31);
      for (let s = 0; s < MAXU; s++) {
        const cl = [];
        for (let c = 0; c < 4; c++) { let x, y, z; do { x = R() * 2 - 1; y = R() * 2 - 1; z = R() * 2 - 1; } while (x * x + y * y + z * z > 0.5); cl.push([x, y, z]); }
        for (let k = 0; k < GPS; k++) {
          const o = s * GPS + k; let x, y, z;
          if (R() < 0.7) { const c = cl[k % 4], sp = 0.22; x = c[0] + (R() - 0.5) * sp * 2; y = c[1] + (R() - 0.5) * sp; z = c[2] + (R() - 0.5) * sp * 2; }
          else { do { x = R() * 2 - 1; y = R() * 2 - 1; z = R() * 2 - 1; } while (x * x + y * y + z * z > 0.9); }
          off[o * 4] = x; off[o * 4 + 1] = y; off[o * 4 + 2] = z; off[o * 4 + 3] = R(); slot[o] = s;
        }
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
      g.setAttribute('aOff', new THREE.BufferAttribute(off, 4));
      g.setAttribute('aSlot', new THREE.BufferAttribute(slot, 1));
      const m = new THREE.Points(g, new THREE.ShaderMaterial({
        vertexShader: GAL_VS, fragmentShader: GAL_FS, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
        uniforms: { uBub: { value: galBub }, uBubG: { value: galG }, uTime: timeU, uScale: scaleU },
      }));
      m.frustumCulled = false; m.renderOrder = 8;
      return m;
    })();
    scene.add(galaxies);

    // tunnelling arcs: dashed glowing dots along a quadratic arc
    const ARCN = 180;
    const arcs = [];
    for (let k = 0; k < 3; k++) {
      const g = new THREE.BufferGeometry();
      const t = new Float32Array(ARCN); for (let i = 0; i < ARCN; i++) t[i] = i / (ARCN - 1);
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(ARCN * 3), 3));
      g.setAttribute('aT', new THREE.BufferAttribute(t, 1));
      const mat = new THREE.ShaderMaterial({
        vertexShader: ARC_VS, fragmentShader: ARC_FS, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
        uniforms: { uProg: { value: 0 }, uTime: timeU, uScale: scaleU, uAlpha: { value: 0 }, uColA: { value: new THREE.Color() }, uColB: { value: new THREE.Color() } },
      });
      const pts = new THREE.Points(g, mat);
      pts.frustumCulled = false; pts.renderOrder = 9; pts.visible = false;
      scene.add(pts);
      arcs.push({ pts, mat, t0: -99, busy: false, ax: 0, ay: 0, az: 0, bx: 0, by: 0, bz: 0, cx: 0, cy: 0, cz: 0 });
    }

    // hover: the predicted path of steepest descent from the cursor to its valley floor
    const pathPos = new Float32Array(NPATH * 3);
    const path = (() => {
      const g = new THREE.BufferGeometry();
      const ia = new Float32Array(NPATH); for (let i = 0; i < NPATH; i++) ia[i] = i;
      g.setAttribute('position', new THREE.BufferAttribute(pathPos, 3).setUsage(THREE.DynamicDrawUsage));
      g.setAttribute('aI', new THREE.BufferAttribute(ia, 1));
      const m = new THREE.Points(g, new THREE.ShaderMaterial({
        vertexShader: PATH_VS, fragmentShader: PATH_FS, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
        uniforms: { uCount: { value: 0 }, uTime: timeU, uScale: scaleU, uAmt: { value: 0 } },
      }));
      m.frustumCulled = false; m.renderOrder = 9;
      return m;
    })();
    scene.add(path);

    // ambient bubbles far away and ghosts of tunnelled universes
    const amb = Array.from({ length: NAMB }, () => ({ on: false, t0: 0, life: 1, x: 0, y: 0, z: 0, R: 1, col: new THREE.Color() }));
    const ghosts = Array.from({ length: NGHOST }, () => ({ on: false, t0: 0, x: 0, y: 0, z: 0, R: 1, col: new THREE.Color(), flat: false }));
    let ambNext = 1.5;
    const ambR = rng(1234);

    // ---- colours -------------------------------------------------------------------------------
    const C = {
      life: new THREE.Color(palette.mint), large: new THREE.Color(palette.planck), neg: new THREE.Color(palette.ember),
      flat: new THREE.Color(palette.harmonic), hyper: new THREE.Color(palette.violet), empty: new THREE.Color(palette.resonance),
      marble: new THREE.Color(1.0, 0.82, 0.58), cursor: new THREE.Color(1.0, 0.86, 0.62),
    };
    const ambCols = [palette.violet, palette.resonance, palette.planck, palette.harmonic].map((c) => new THREE.Color(c));

    // ---- laws of each valley ---------------------------------------------------------------------
    const lawCache = new Array(K).fill(null);
    function lawsFor(k) {
      if (lawCache[k]) return lawCache[k];
      const v = valleys[k], r = rng(1000003 + k * 7919);
      let e, sign, cls;
      if (v.h > BAND) { sign = 1; cls = 'large'; e = -118 + 116 * Math.pow(clamp((v.h - BAND) / Math.max(0.1, LS.hTop - BAND)), 0.55); }
      else if (v.h >= 0) { sign = 1; cls = 'tiny'; e = -122.6 + 2.8 * (v.h / BAND); }
      else { sign = -1; cls = 'neg'; e = -110 + 108 * Math.pow(clamp(-v.h / Math.max(0.1, -LS.hBot)), 0.55); }
      e = Math.round(e);
      const xd = r(); const dims = xd < 0.8 ? 3 : xd < 0.88 ? 2 : xd < 0.95 ? 4 : 5;
      const xf = r(); const fam = xf < 0.1 ? 0 : xf < 0.26 ? 1 : xf < 0.46 ? 2 : xf < 0.72 ? 3 : xf < 0.88 ? 4 : 5;
      const em = r() < 0.85, strong = r() < 0.87, weak = r() < 0.76;
      const lam = Math.pow(10, e);
      let fate;
      if (sign > 0) fate = `doubles every ${fmtTime(Math.LN2 * Math.sqrt(3 / lam) * T_PLANCK)}`;
      else fate = `crunches after ${fmtTime((TAU / Math.sqrt(3 * lam)) * T_PLANCK)}`;
      let reason = '', kind = 'life';
      if (cls === 'large') { reason = 'Λ is too large. Space expands so fast that galaxies never form.'; kind = 'large'; }
      else if (cls === 'neg') { reason = 'Λ is negative. Space recollapses in a big crunch before any star can form.'; kind = 'neg'; }
      else if (dims === 2) { reason = 'Only two large space dimensions. Gravity cannot pull matter together there, so no stars.'; kind = 'flat'; }
      else if (dims > 3) { reason = `${dims} large space dimensions. Orbits and atoms are unstable, so no chemistry.`; kind = 'hyper'; }
      else if (fam === 0) { reason = 'No families of matter particles. There is nothing to build atoms from.'; kind = 'empty'; }
      else if (!em) { reason = 'No electromagnetism. Atoms cannot hold together, so no chemistry.'; kind = 'empty'; }
      else if (!strong) { reason = 'No strong force. Quarks never bind into protons or nuclei, so no ordinary atoms and no fusion in stars.'; kind = 'empty'; }
      const forces = ['gravity'];
      if (em) forces.push('EM'); if (strong) forces.push('strong'); if (weak) forces.push('weak');
      const tag = cls === 'tiny' ? (e <= -121 ? 'like ours' : 'tiny') : cls === 'large' ? 'huge' : 'negative';
      const L = {
        e, sign, cls, dims, fam, em, strong, weak, fate, reason, kind, life: kind === 'life',
        lamText: `${sign > 0 ? '+' : '−'}10${sup(e)}`, tag, forces: forces.join(' · '), forcesShort: forces.length + ' of 4',
      };
      lawCache[k] = L;
      return L;
    }
    // does any valley hold a universe that could support chemistry? (used by demo)
    const lifeValleys = [];
    for (let k = 0; k < K; k++) if (valleys[k].kind === 1 && lawsFor(k).life) lifeValleys.push(k);
    if (params.has('m6dbg')) console.log('m6 stats', JSON.stringify({ K, band: LS.nBand, lake: LS.nLake, life: lifeValleys.length, lakeFrac: +LS.lakeFrac.toFixed(3), hLo: +LS.hLo.toFixed(2), hHi: +LS.hHi.toFixed(2), hTop: +LS.hTop.toFixed(2), hBot: +LS.hBot.toFixed(2) }));

    // ---- state -----------------------------------------------------------------------------------
    let active = false;
    let made = 0, lifeFound = 0;
    let selected = null;
    let basinsOn = params.has('m6basins');
    let basinsAmt = basinsOn ? 1 : 0;
    let readout = null, controls = null, panelMode = '', panelKey = '';
    const seen = { first: false, life: false, neg: false, tunnel: false, offer: false, same: false };
    let lastInput = 0;
    let rollVoice = null;
    const snd = { land: -9, settle: -9, hover: -9, drop: -9 };
    const queue = [];
    const later = (sec, fn) => queue.push({ at: clock + sec, fn });
    let pulseIdx = 0, ripIdx = 0;
    const coda = { on: false, t: 0, idx: 0, order: [], interval: 0.3, chord: false, silenced: false, auto: false, glow: 0, waveT0: 0 };
    let codaOfferT = -1;
    let pinUntil = -1, demoPinLife = false, settledN = 0;

    // camera rig: orbit around a target with damping, idle drift and a gentle breath
    const DEF = { yaw: 2.4, pitch: 0.26, dist: 72, tx: 9, tz: -7 };
    {
      const c = params.get('m6cam');
      if (c) { const a = c.split(',').map(Number); ['yaw', 'pitch', 'dist', 'tx', 'tz'].forEach((k, i) => { if (Number.isFinite(a[i])) DEF[k] = a[i]; }); }
    }
    const cam = { ...DEF };
    const goal = { ...DEF };
    let portrait = false;

    // input
    const drag = { on: false, moved: false, pan: false, pinch: false, x0: 0, y0: 0, t0: 0, touch: false };
    // two-finger pinch: the engine routes every finger through the same handlers, so track them by id
    const touches = new Map();
    let pinchD = 0;
    const touchSpread = () => { let ax = 0, ay = 0, bx = 0, by = 0, i = 0; touches.forEach((p) => { if (i === 0) { ax = p.x; ay = p.y; } else if (i === 1) { bx = p.x; by = p.y; } i++; }); return Math.hypot(ax - bx, ay - by); };
    const hover = { dirty: false, ndc: new THREE.Vector2(), has: false, label: -1, x: 0, z: 0, amt: 0, last: -9, n: 0, bubble: null };
    let rollT = 0;
    const ray = new THREE.Raycaster();
    const hit = new THREE.Vector3();
    const tmpV = new THREE.Vector3(), tmpV2 = new THREE.Vector3();
    const ndcTmp = new THREE.Vector2();

    // ---- helpers ----------------------------------------------------------------------------------
    function pulse(x, z, col, strength) {
      const i = pulseIdx++ % NPULSE;
      pulseU.value[i].set(x, z, clock, strength); pulseCU.value[i].copy(col);
    }
    function ripple(x, z, amp) {
      const i = ripIdx++ % NRIP;
      ripU.value[i].set(x, z, clock, amp);
    }
    function panOf(x, y, z) {
      tmpV2.set(x, y, z).project(camera);
      return clamp(tmpV2.x * 0.7, -0.8, 0.8);
    }

    /** Ray-march the B-spline terrain. Returns true and fills `hit` if the ray meets the ground. */
    function pickTerrain(ndc) {
      ray.setFromCamera(ndc, camera);
      const o = ray.ray.origin, d = ray.ray.direction;
      let tPrev = 0, t = 0.5;
      if (o.y - heightAt(o.x, o.z) < 0) return false;
      for (let i = 0; i < 500 && t < 900; i++) {
        const x = o.x + d.x * t, y = o.y + d.y * t, z = o.z + d.z * t;
        const above = y - heightAt(x, z);
        if (above < 0) {
          let a = tPrev, b = t;
          for (let k = 0; k < 14; k++) { const m = (a + b) * 0.5; if (o.y + d.y * m - heightAt(o.x + d.x * m, o.z + d.z * m) < 0) b = m; else a = m; }
          hit.set(o.x + d.x * b, 0, o.z + d.z * b); hit.y = heightAt(hit.x, hit.z);
          return true;
        }
        tPrev = t;
        t += Math.max(0.2, above * 0.4, t * 0.003);
        if (d.y > 0 && y > 60) return false;
      }
      return false;
    }
    /** Nearest bubble along the pointer ray (distance), or null. Reuses one result object. */
    const pickRes = { u: null, t: 0 };
    function pickBubble(ndc) {
      ray.setFromCamera(ndc, camera);
      const o = ray.ray.origin, d = ray.ray.direction;
      let best = null, bt = Infinity;
      for (let i = 0; i < MAXU; i++) {
        const u = U[i];
        if (!u.active || u.phase !== PH_LIVE || u.dying || u.crunched || u.scale < 0.3) continue;
        const r = u.R * u.scale, ox = o.x - u.bx, oy = o.y - u.by, oz = o.z - u.bz;
        const b = ox * d.x + oy * d.y + oz * d.z, c = ox * ox + oy * oy + oz * oz - r * r, disc = b * b - c;
        if (disc < 0) continue;
        const t = -b - Math.sqrt(disc);
        if (t > 0 && t < bt) { bt = t; best = u; }
      }
      if (!best) return null;
      pickRes.u = best; pickRes.t = bt;
      return pickRes;
    }

    // ---- sound -------------------------------------------------------------------------------------
    function chord(v, laws, pan) {
      if (!A.on) return;
      const d = v.deg, oct = laws.cls === 'neg' ? 0 : 1;
      A.pluck(A.degree(d, oct), { gain: 0.2, decay: 3.4, bright: 0.45, pan });
      A.pluck(A.degree(d + 2, oct), { gain: 0.13, decay: 3.0, bright: 0.4, pan: pan * 0.8, when: 0.07 });
      A.pluck(A.degree(d + 4, oct), { gain: 0.1, decay: 2.8, bright: 0.4, pan: pan * 0.6, when: 0.14 });
      A.swell(A.degree(d, oct + 1), { gain: 0.04, attack: 0.6, dur: 1.4, pan, when: 0.12 });
      if (laws.life) { A.bell(A.degree(d + 7, 2), { gain: 0.08, when: 0.3, pan }); A.bell(A.degree(d + 9, 2), { gain: 0.05, when: 0.55, pan }); }
    }
    function makeRollVoice() {
      const c = A.ctx;
      const len = Math.floor(c.sampleRate * 2), buf = c.createBuffer(1, len, c.sampleRate), d = buf.getChannelData(0);
      let last = 0;
      for (let i = 0; i < len; i++) { last = (last + 0.02 * (Math.random() * 2 - 1)) / 1.02; d[i] = last * 3.2; }
      const src = c.createBufferSource(); src.buffer = buf; src.loop = true;
      const bp = c.createBiquadFilter(); bp.type = 'bandpass'; bp.frequency.value = 320; bp.Q.value = 0.8;
      const g = c.createGain(); g.gain.value = 0;
      src.connect(bp).connect(g).connect(A.out);
      src.start();
      let stopped = false;
      return {
        set(level, freq) { if (stopped) return; const t = c.currentTime; g.gain.setTargetAtTime(level, t, 0.09); bp.frequency.setTargetAtTime(freq, t, 0.12); },
        stop() {
          if (stopped) return; stopped = true;
          const t = c.currentTime; g.gain.setTargetAtTime(0, t, 0.08);
          try { src.stop(t + 0.5); } catch (e) { /* already stopped */ }
          setTimeout(() => { try { g.disconnect(); bp.disconnect(); src.disconnect(); } catch (e) { /* gone */ } }, 900);
        },
      };
    }

    // ---- universes -----------------------------------------------------------------------------------
    function allocSlot() {
      for (let i = 0; i < MAXU; i++) if (!U[i].active) return U[i];
      let old = null;
      for (let i = 0; i < MAXU; i++) { const u = U[i]; if (u !== selected && u.phase === PH_LIVE && (!old || u.id < old.id)) old = u; }
      if (!old) old = U[0];
      deactivate(old);
      return old;
    }
    function deactivate(u) {
      u.active = false; u.alpha = 0; u.scale = 0; u.galaxy = 0; u.tCount = 0; u.tAlpha = 0;
      if (u.arc) { u.arc.busy = false; u.arc = null; }
      if (selected === u) selected = null;
      if (hover.bubble === u) hover.bubble = null;
    }

    function drop(x, z, opts = {}) {
      const u = allocSlot();
      made++;
      u.active = true; u.id = made; u.phase = PH_FALL;
      u.x = x; u.z = z;
      const g = heightAt(x, z);
      u.y = g + 7.5 + (opts.high || 0); u.vx = (Math.random() - 0.5) * 0.4; u.vz = (Math.random() - 0.5) * 0.4; u.vy = -3;
      u.still = 0; u.rollT = 0; u.wet = u.y < 0; u.k = -1; u.laws = null; u.dying = 0; u.flash = 0; u.decays = 0;
      u.crunched = false; u.alpha = 0; u.scale = 0; u.heat = 0; u.galaxy = 0; u.seed = Math.random(); u.everLife = false;
      u.tHead = 0; u.tCount = 0; u.tAlpha = 1; u.tLast = -1;
      u.col.copy(C.marble);
      pushTrail(u);
      if (A.on && clock - snd.drop > 0.07) { snd.drop = clock; A.noise({ freq: 2600, sweepTo: 650, dur: 0.6, gain: 0.03, q: 2.4, pan: panOf(x, g, z) }); }
      lastInput = clock;
      updateCounter();
      return u;
    }
    function pushTrail(u) {
      const o = u.tHead * 3;
      u.trail[o] = u.x; u.trail[o + 1] = u.y; u.trail[o + 2] = u.z;
      u.tHead = (u.tHead + 1) % TRAIL; if (u.tCount < TRAIL) u.tCount++;
    }

    function stepMarble(u, dt) {
      if (u.phase === PH_FALL) {
        u.vy -= G_FALL * dt; u.y += u.vy * dt;
        const g = heightAt(u.x, u.z) + MR;
        if (!u.wet && u.y < 0 && g < 0) { u.wet = true; ripple(u.x, u.z, 1.0); if (A.on) A.noise({ freq: 1500, q: 1.2, dur: 0.4, gain: 0.04, type: 'lowpass', pan: panOf(u.x, 0, u.z) }); }
        if (u.y <= g) {
          u.y = g; u.phase = PH_ROLL;
          const impact = clamp(-u.vy / 26);
          sample(u.x, u.z);
          // the drop's vertical momentum turns partly into a push down the local slope
          u.vx += -S[1] * 2.2; u.vz += -S[2] * 2.2;
          pulse(u.x, u.z, C.marble, 0.35);
          if (A.on && clock - snd.land > 0.08) {
            snd.land = clock;
            const v = valleys[labelAt(u.x, u.z)];
            A.pluck(A.degree((v ? v.deg : 4) + 3, 2), { gain: 0.07 + 0.06 * impact, decay: 1.3, bright: 0.7, pan: panOf(u.x, u.y, u.z) });
          }
        }
        return;
      }
      if (u.phase === PH_ROLL) {
        // a solid ball rolling without slipping: a = (5/7) g sinθ along the slope, with rolling friction
        const sub = 4, h = dt / sub;
        for (let s = 0; s < sub; s++) {
          sample(u.x, u.z);
          const gx = S[1], gz = S[2], inv = 1 / (1 + gx * gx + gz * gz);
          const sp2 = u.vx * u.vx + u.vz * u.vz;
          const fr = FRICTION + (sp2 < 0.16 ? 2.4 : 0);
          u.vx += (-(5 / 7) * G_ROLL * gx * inv - fr * u.vx) * h;
          u.vz += (-(5 / 7) * G_ROLL * gz * inv - fr * u.vz) * h;
          u.x += u.vx * h; u.z += u.vz * h;
        }
        sample(u.x, u.z);
        u.y = S[0] + MR;
        if (!u.wet && S[0] < 0) { u.wet = true; ripple(u.x, u.z, 0.6); }
        u.rollT += dt;
        const sp2 = u.vx * u.vx + u.vz * u.vz, g2 = S[1] * S[1] + S[2] * S[2];
        if (sp2 < 0.2 && g2 < 0.01) u.still += dt; else u.still = 0;
        if (u.still > 0.25 || u.rollT > 11) beginGlide(u);
        return;
      }
      if (u.phase === PH_GLIDE) {
        u.gt += dt / 0.6;
        const k = smoothstep(0, 1, u.gt);
        u.x = lerp(u.gx0, u.tx, k); u.z = lerp(u.gz0, u.tz, k);
        u.y = heightAt(u.x, u.z) + MR;
        if (u.gt >= 1) settle(u);
      }
    }
    function beginGlide(u) {
      const k = labelAt(u.x, u.z), v = valleys[k];
      u.k = k; u.phase = PH_GLIDE; u.gt = 0; u.gx0 = u.x; u.gz0 = u.z;
      u.tx = v.x + Math.round((u.x - v.x) / P) * P; u.tz = v.z + Math.round((u.z - v.z) / P) * P;
      u.vx = u.vz = 0;
    }

    function bubbleSpot(u) {
      // several universes in one valley share its vacuum: they cluster like foam
      const v = valleys[u.k];
      let n = 0;
      for (let i = 0; i < MAXU; i++) {
        const w = U[i];
        if (w !== u && w.active && w.phase === PH_LIVE && w.k === u.k && !w.dying && Math.abs(wrapD(w.tx - u.tx)) < 1 && Math.abs(wrapD(w.tz - u.tz)) < 1) n++;
      }
      const flat = u.laws.dims === 2;
      u.floorY = v.h;
      if (n) u.R *= 0.62 + 0.18 * u.seed;   // later arrivals form smaller foam around the first bubble
      const ang = n * 2.399 + u.seed, rr = n ? u.R * 2.1 * Math.sqrt(n) : 0;
      u.bx = u.tx + Math.cos(ang) * rr; u.bz = u.tz + Math.sin(ang) * rr;
      u.by = Math.max(v.h, 0) + (flat ? 0.9 : u.R * 1.1) + (n ? 0.5 * n : 0);
      return n;
    }

    function settle(u) {
      const v = valleys[u.k];
      u.phase = PH_LIVE; u.born = clock; u.tLast = clock;
      u.laws = lawsFor(u.k);
      u.R = 2.4 + 0.9 * u.seed;
      const same = bubbleSpot(u);
      u.col.copy(C[u.laws.kind]);
      u.alpha = 0; u.scale = 0; u.heat = 0; u.galaxy = 0; u.crunched = false;
      u.hx0 = u.tx; u.hy0 = v.h + MR; u.hz0 = u.tz;
      if (u.laws.life && !u.everLife) { u.everLife = true; lifeFound++; }
      pulse(u.tx, u.tz, u.col, u.laws.life ? 1.5 : 0.9);
      if (v.h < 0) ripple(u.bx, u.bz, 0.8);
      const pan = panOf(u.bx, u.by, u.bz);
      if (A.on && clock - snd.settle > 0.12) { snd.settle = clock; chord(v, u.laws, pan); }
      // the card follows the newest universe unless the user (or the demo) pinned another one
      if (clock > pinUntil || !selected || !selected.active || (demoPinLife && u.laws.life)) select(u, false);
      if (demoPinLife && u.laws.life) { pinUntil = clock + 8; demoPinLife = false; }
      if (!seen.first && !u.laws.life) { seen.first = true; ui.hideHint(); ui.toast(`Universe ${u.id} settled. Its laws are on the card, top right.`, 3600); }
      else if (u.laws.life && !seen.life) {
        seen.first = true; ui.hideHint(); seen.life = true; ui.toast(`Universe ${u.id} could support chemistry and stars. Only some valleys on the mint shore can.`, 5200); }
      else if (u.laws.kind === 'neg' && !seen.neg) { seen.neg = true; ui.toast(`Universe ${u.id} has negative Λ. Watch it recollapse.`, 3800); }
      else if (same && !seen.same) { seen.same = true; ui.toast(`Universe ${u.id} fell into an occupied valley. Same vacuum, same laws.`, 3800); }
      updateCounter();
      settledN++;
      if (settledN >= 8 && !seen.offer && !coda.on) { seen.offer = true; codaOfferT = clock; later(2.6, () => { if (!coda.on && active) ui.toast(`Eight universes made. ${ctx.isTouch ? 'Tap Coda' : 'Press C'} for the coda, or keep exploring.`, 4600); }); }
    }

    function animateBubble(u, dt) {
      const age = clock - u.born, L = u.laws;
      let s, a;
      if (L.kind === 'large') {
        // empty and diluting: it keeps inflating and thins out
        s = (1 - Math.exp(-2.6 * age)) * (1 + 0.55 * (1 - Math.exp(-0.5 * age)));
        a = 0.3 + 1.1 * Math.exp(-age * 0.9);
      } else if (L.kind === 'neg') {
        if (age < 2.6) { s = 1 - Math.exp(-3.2 * age) * Math.cos(5.5 * age); a = 0.55 + 0.9 * Math.exp(-age * 1.3); u.heat = smoothstep(1.4, 2.6, age) * 0.4; }
        else {
          const c = clamp((age - 2.6) / 2.2);
          s = Math.max(0, 1 - c * c * c); a = 0.7 + 0.8 * c; u.heat = 0.4 + 0.9 * c;
          if (c >= 1 && !u.crunched) {
            u.crunched = true;
            pulse(u.bx, u.bz, C.neg, 0.8);
            if (A.on) { A.noise({ freq: 900, sweepTo: 110, dur: 1.4, gain: 0.05, q: 1.1, type: 'lowpass', pan: panOf(u.bx, u.by, u.bz) }); A.pluck(A.degree(u.laws ? valleys[u.k].deg : 0, 0), { gain: 0.12, decay: 2.2, bright: 0.15 }); }
          }
        }
      } else {
        s = 1 - Math.exp(-3.2 * age) * Math.cos(5.5 * age);
        a = 0.52 + 0.95 * Math.exp(-age * 1.2);
      }
      u.scale = s; u.alpha = a * (1 - u.dying);
      u.galaxy = L.life ? smoothstep(0.8, 2.6, age) * (1 - u.dying) : 0;
      u.flash = Math.max(0, u.flash - dt * 0.8);
    }

    function select(u, sound = true) {
      selected = u;
      if (sound && u && A.on) A.bell(A.degree(valleys[u.k].deg + 2, 2), { gain: 0.08, pan: panOf(u.bx, u.by, u.bz) });
      if (u) u.flash = Math.max(u.flash, 0.6);
      buildReadout();
    }
    function latestLive() {
      let best = null;
      for (let i = 0; i < MAXU; i++) { const u = U[i]; if (u.active && u.phase === PH_LIVE && !u.dying && !u.crunched && (!best || u.born > best.born)) best = u; }
      return best;
    }

    /** Is a world point in the clear part of the frame (away from the HUD panels)? */
    function inClear(x, y, z, loose = false) {
      tmpV.set(x, y, z).project(camera);
      if (tmpV.z >= 1) return false;
      if (loose) return Math.abs(tmpV.x) < 0.7 && tmpV.y > -0.55 && tmpV.y < 0.45;
      return portrait ? Math.abs(tmpV.x) < 0.6 && tmpV.y > -0.3 && tmpV.y < 0.24
        : tmpV.x > -0.55 && tmpV.x < 0.42 && tmpV.y > -0.4 && tmpV.y < 0.34;
    }
    /** Does the valley that drains (x, z) sit in the clear part of the frame? */
    function settlesInClear(x, z) {
      const v = valleys[labelAt(x, z)];
      return inClear(v.x + Math.round((x - v.x) / P) * P, Math.max(v.h, 0) + 2.4, v.z + Math.round((z - v.z) / P) * P);
    }
    function randomDrop() {
      // aim where the universe will settle, so its bubble is not hidden behind a panel
      let fx = 0, fz = 0, any = false;
      for (let tries = 0; tries < 14; tries++) {
        ndcTmp.set(-0.5 + Math.random() * 0.95, -0.3 + Math.random() * 0.58);
        if (portrait) ndcTmp.set((Math.random() * 2 - 1) * 0.62, -0.32 + Math.random() * 0.54);
        if (!pickTerrain(ndcTmp)) continue;
        if (settlesInClear(hit.x, hit.z)) { drop(hit.x, hit.z); return true; }
        if (!any) { any = true; fx = hit.x; fz = hit.z; }
      }
      if (any) { drop(fx, fz); return true; }
      return false;
    }
    function rain(n = 12) {
      if (coda.on) return;
      for (let i = 0; i < n; i++) later(i * 0.11, () => randomDrop());
      if (A.on) A.noise({ freq: 5000, sweepTo: 1200, dur: 1.8, gain: 0.025, q: 0.8, attack: 0.4 });
      lastInput = clock;
    }
    function clearAll() {
      if (coda.on) return;
      for (let i = 0; i < MAXU; i++) { const u = U[i]; if (u.active) u.dying = Math.max(u.dying, 0.001); }
      selected = null; buildReadout();
      if (A.on) A.noise({ freq: 700, sweepTo: 3000, dur: 0.7, gain: 0.03, q: 1.5 });
      lastInput = clock;
    }

    // ---- tunnelling: vacuum decay into a neighbouring lower valley ----------------------------------------
    function tunnel() {
      if (coda.on) return;
      lastInput = clock;
      let u = selected && selected.active && selected.phase === PH_LIVE && !selected.dying && !selected.crunched ? selected : latestLive();
      if (!u) { ui.toast('Drop a universe first. Tunnelling needs a vacuum to decay.', 3000); return; }
      const v = valleys[u.k];
      let best = -1, bd = Infinity;
      for (let k = 0; k < K; k++) {
        const w = valleys[k];
        if (k === u.k || w.h > v.h - 0.15) continue;
        const dx = wrapD(w.x - u.tx), dz = wrapD(w.z - u.tz), d = Math.hypot(dx, dz);
        if (d < 4 || d > 36) continue;
        // prefer near neighbours, with a nudge toward bigger drops in energy
        const score = d - (v.h - w.h) * 0.8;
        if (score < bd) { bd = score; best = k; }
      }
      if (best < 0) { ui.toast(`Universe ${u.id} has no lower valley nearby on this map, so it stays put.`, 3400); if (A.on) A.pluck(A.degree(v.deg, 1), { gain: 0.1, decay: 1.2, bright: 0.2 }); return; }
      const arc = arcs.find((a) => !a.busy);
      if (!arc) return;
      const w = valleys[best];
      const tx = u.tx + wrapD(w.x - u.tx), tz = u.tz + wrapD(w.z - u.tz);
      // a ghost of the old bubble collapses where it was
      const gh = ghosts.find((g) => !g.on) || ghosts[0];
      gh.on = true; gh.t0 = clock; gh.x = u.bx; gh.y = u.by; gh.z = u.bz; gh.R = u.R * u.scale; gh.col.copy(u.col); gh.flat = u.laws.dims === 2;
      // the arc
      arc.busy = true; arc.t0 = clock; arc.pts.visible = true;
      arc.ax = u.bx; arc.ay = u.by; arc.az = u.bz;
      arc.bx = tx; arc.bz = tz; arc.by = Math.max(w.h, 0) + u.R * 1.1;
      const dd = Math.hypot(tx - u.bx, tz - u.bz);
      arc.cx = (arc.ax + arc.bx) / 2; arc.cz = (arc.az + arc.bz) / 2; arc.cy = Math.max(arc.ay, arc.by) + dd * 0.45 + 3;
      const pa = arc.pts.geometry.attributes.position.array;
      for (let i = 0; i < ARCN; i++) {
        const t = i / (ARCN - 1), m = 1 - t;
        pa[i * 3] = m * m * arc.ax + 2 * m * t * arc.cx + t * t * arc.bx;
        pa[i * 3 + 1] = m * m * arc.ay + 2 * m * t * arc.cy + t * t * arc.by;
        pa[i * 3 + 2] = m * m * arc.az + 2 * m * t * arc.cz + t * t * arc.bz;
      }
      arc.pts.geometry.attributes.position.needsUpdate = true;
      arc.mat.uniforms.uColA.value.copy(u.col).multiplyScalar(1.6);
      const nl = lawsFor(best);
      arc.mat.uniforms.uColB.value.copy(C[nl.kind]).multiplyScalar(2.2);
      u.tCount = 0; u.tHead = 0;
      u.arc = arc; u.arcT = 0; u.phase = PH_TUNNEL; u.alpha = 0; u.scale = 0; u.galaxy = 0;
      u.k = best; u.tx = tx; u.tz = tz; u.decays++;
      if (A.on) {
        const hi = v.deg + 7;
        for (let i = 0; i < 6; i++) A.bell(A.degree(hi - i, 2), { gain: 0.06, decay: 2.4, when: i * 0.075, pan: panOf(arc.ax, arc.ay, arc.az) * (1 - i / 6) + panOf(arc.bx, arc.by, arc.bz) * (i / 6) });
        A.noise({ freq: 3200, sweepTo: 260, dur: 1.0, gain: 0.035, q: 2 });
      }
      if (!seen.tunnel) { seen.tunnel = true; ui.toast(`Vacuum decay. A bubble of lower vacuum energy nucleated and replaced Universe ${u.id}. Its laws changed.`, 4600); }
    }
    function stepTunnel(u, dt) {
      const arc = u.arc;
      u.arcT += dt / 0.95;
      const t = smoothstep(0, 1, Math.min(1, u.arcT)), m = 1 - t;
      u.x = m * m * arc.ax + 2 * m * t * arc.cx + t * t * arc.bx;
      u.y = m * m * arc.ay + 2 * m * t * arc.cy + t * t * arc.by;
      u.z = m * m * arc.az + 2 * m * t * arc.cz + t * t * arc.bz;
      arc.mat.uniforms.uProg.value = t;
      if (u.arcT >= 1) {
        // a new bubble nucleates in the lower valley; its wall races outward
        u.laws = lawsFor(u.k);
        u.phase = PH_LIVE; u.born = clock; u.crunched = false; u.heat = 0;
        u.R = 2.4 + 0.9 * u.seed;
        bubbleSpot(u);
        u.col.copy(C[u.laws.kind]); u.flash = 1;
        u.hx0 = u.x; u.hy0 = u.y; u.hz0 = u.z;   // the heart continues from where the arc ended
        if (u.laws.life && !u.everLife) { u.everLife = true; lifeFound++; }
        pulse(u.tx, u.tz, u.col, 1.3);
        if (valleys[u.k].h < 0) ripple(u.bx, u.bz, 1.0);
        if (A.on) chord(valleys[u.k], u.laws, panOf(u.bx, u.by, u.bz));
        u.arc = null;
        arc.t0 = clock;     // the arc now fades
        arc.done = true;
        select(u, false);
      }
    }

    // ---- coda ------------------------------------------------------------------------------------------
    function startCoda() {
      if (coda.on || !active) return;
      coda.on = true; coda.t = 0; coda.idx = 0; coda.chord = false; coda.silenced = false;
      coda.order = U.filter((u) => u.active && u.phase === PH_LIVE && !u.dying).sort((a, b) => a.id - b.id);
      coda.interval = Math.min(0.42, 5.2 / Math.max(1, coda.order.length));
      goal.pitch = 0.62; goal.dist = 118 * (portrait ? 1.2 : 1); goal.yaw = cam.yaw - 0.35;
      ctx.setPost({ ...POST, ...POST_CODA });
      ui.hideHint();
      lastInput = clock;
      buildPanels();
    }
    function endCoda(restore) {
      if (!coda.on) return;
      coda.on = false; coda.glow = 0; dots.material.uniforms.uWaveT.value = 0;
      goal.pitch = DEF.pitch; goal.dist = DEF.dist * (portrait ? 1.22 : 1);
      ctx.setPost({ ...POST });
      if (coda.silenced && restore) A.setDrone(DRONE.notes, DRONE.opts);
      coda.silenced = false;
      if (restore) buildPanels();
    }
    function stepCoda(dt) {
      coda.t += dt;
      const n = coda.order.length;
      while (coda.idx < n && coda.t > 1.4 + coda.idx * coda.interval) {
        const u = coda.order[coda.idx++];
        if (!u.active) continue;
        u.flash = 1.2;
        pulse(u.tx, u.tz, u.col, 0.7);
        if (A.on) {
          const d = valleys[u.k].deg;
          A.pluck(A.degree(d, 2), { gain: 0.13, decay: 2.6, bright: 0.55, pan: panOf(u.bx, u.by, u.bz) });
          if (u.laws.life) A.bell(A.degree(d + 5, 2), { gain: 0.06, when: 0.1 });
        }
      }
      const tArp = 1.4 + n * coda.interval;
      if (!coda.chord && coda.t > tArp + 0.5) {
        coda.chord = true; coda.waveT0 = coda.t;
        dots.material.uniforms.uWaveC.value.set(cam.tx, cam.tz);
        pulse(cam.tx, cam.tz, C.life, 1.4);
        if (A.on) [0, 7, 12, 15, 19, 24, 27].forEach((s, i) => A.swell(A.freq(s), { gain: 0.042, attack: 2.4, dur: 5.5, when: i * 0.42, pan: (i / 6 - 0.5) * 0.9 }));
      }
      if (!coda.silenced && coda.t > tArp + 9) { coda.silenced = true; A.setDrone([], {}); }
      coda.glow = smoothstep(tArp, tArp + 3.5, coda.t);
      dots.material.uniforms.uWaveT.value = coda.chord ? coda.t - coda.waveT0 : 0;
    }

    // ---- panels --------------------------------------------------------------------------------------
    const narrow = () => ctx.width <= 760;
    const keyTag = (k) => `<span style="opacity:.42;margin-left:.55em">${k}</span>`;
    function counterHTML(extra = '') {
      return `<div style="display:flex;align-items:center;gap:12px;margin:0 0 6px"><span class="big" data-m6="n">${made}</span>` +
        `<span style="font-family:var(--f-mono);font-size:var(--t-xs);letter-spacing:.12em;text-transform:uppercase;color:var(--ink-dim);line-height:1.5">Universes<br>you have made${extra}</span></div>`;
    }
    function updateCounter() {
      if (!readout) return;
      const el = readout.el.querySelector('[data-m6="n"]');
      if (el && el.textContent !== String(made)) el.textContent = String(made);
      if (panelMode === 'map') readout.set('hab', `${lifeFound} of ${made}`);
    }
    function buildReadout() {
      if (!active) return;
      const nw = narrow();
      if (coda.on) {
        panelMode = 'coda'; panelKey = '';
        readout = ui.readout({
          title: 'Coda',
          html: counterHTML() + `<div class="row" style="border-top:0"><span class="k">Could hold stars</span><span class="v" style="color:${MINT}">${lifeFound}</span></div>` +
            `<div class="row"><span class="k">Vacua in the landscape</span><span class="v">~10${sup(500)}</span></div>`,
          note: `${nw ? CLOSING.split('. ')[0] + '. ' + CLOSING.split('. ').slice(-1)[0] : CLOSING}<div style="margin-top:12px"><button type="button" class="btn" data-m6="again" style="display:block;width:100%;box-sizing:border-box">Play again${keyTag('P')}</button></div>`,
        });
        const b = readout.el.querySelector('[data-m6="again"]');
        if (b) b.addEventListener('click', playAgain);
        return;
      }
      const u = selected && selected.active && selected.laws ? selected : null;
      if (!u) {
        panelMode = 'map'; panelKey = '';
        readout = ui.readout({
          title: 'The landscape',
          html: counterHTML(),
          rows: nw ? [{ id: 'vac', k: 'Vacua', v: `~10${sup(500)}` }, { id: 'hab', k: 'Hold stars', v: `${lifeFound} of ${made}`, cls: 'gold' }]
            : [{ id: 'vac', k: 'Vacua in the landscape', v: `~10${sup(500)}` }, { id: 'val', k: 'Valleys on this map', v: String(K) }, { id: 'hab', k: 'Could hold stars', v: `${lifeFound} of ${made}`, cls: 'gold' }],
          note: nw ? undefined : `Height is vacuum energy, felt as dark energy Λ. The glassy level is Λ = 0. Valleys on the <span style="color:${MINT}">mint shore</span> just above it have a tiny positive Λ, like ours.`,
        });
        return;
      }
      const L = u.laws;
      panelMode = 'uni'; panelKey = `${u.id}:${u.k}`;
      const lamCls = L.cls === 'tiny' ? 'cyan' : L.cls === 'neg' ? '' : 'gold';
      const lamV = (txt) => (L.cls === 'neg' ? `<span style="color:${EMBER}">${txt}</span>` : txt);
      const verdict = L.life ? `<span style="color:${MINT}">● Could support chemistry and stars.</span>` : `<span style="color:${EMBER}">${L.reason}</span>`;
      const decayed = u.decays ? `, decayed ${u.decays === 1 ? 'once' : u.decays + ' times'}` : '';
      readout = ui.readout({
        title: `Universe ${u.id}${decayed}`,
        html: counterHTML(),
        rows: nw ? [
          { id: 'lam', k: 'Λ', v: lamV(L.lamText), cls: lamCls },
          { id: 'dim', k: 'Space + time', v: `${L.dims} + 1` },
          { id: 'fam', k: 'Families', v: String(L.fam) },
        ] : [
          { id: 'lam', k: 'Dark energy Λ', v: lamV(`${L.lamText} · ${L.tag}`), cls: lamCls },
          { id: 'fate', k: 'Space', v: L.fate },
          { id: 'dim', k: 'Space + time', v: `${L.dims} + 1` },
          { id: 'fam', k: 'Particle families', v: String(L.fam) },
          { id: 'for', k: 'Forces', v: L.forces },
          { id: 'vac', k: 'Vacua in the landscape', v: `~10${sup(500)}` },
        ],
        note: nw ? (L.life ? `<span style="color:${MINT}">Could support chemistry and stars.</span>` : `<span style="color:${EMBER}">${L.reason.split('. ')[0]}.</span>`) : verdict,
      });
    }
    function buildControls() {
      if (!active) return;
      if (coda.on) {
        // Play again lives on the coda card (key P, see onKey); the controls only offer the way back
        controls = ui.controls([
          { type: 'button', id: 'back', label: `Back to the map${keyTag('C')}`, kbd: 'c', onClick: () => endCoda(true) },
        ]);
        return;
      }
      controls = ui.controls([
        { type: 'button', id: 'rain', label: `Rain${keyTag('R')}`, kbd: 'r', onClick: () => rain(12) },
        { type: 'button', id: 'tunnel', label: `Tunnel${keyTag('T')}`, kbd: 't', onClick: tunnel },
        { type: 'button', id: 'clear', label: `Clear${keyTag('X')}`, kbd: 'x', onClick: clearAll },
        { type: 'toggle', id: 'basins', label: 'Basins of attraction', kbd: 'b', value: basinsOn, onChange: (v) => { basinsOn = v; lastInput = clock; if (A.on) A.swell(A.degree(v ? 7 : 5, 1), { gain: 0.035, attack: 0.3, dur: 0.6 }); } },
        { type: 'button', id: 'coda', label: `Coda${keyTag('C')}`, kbd: 'c', onClick: startCoda },
      ]);
    }
    function buildPanels() { buildReadout(); buildControls(); }
    function playAgain() {
      resetOnExit = true;
      ctx.engine.goTo(0);
    }
    let resetOnExit = false;

    // ---- per-frame writers -------------------------------------------------------------------------------
    const lightOrder = new Int32Array(MAXU);
    function writeLights() {
      let n = 0;
      const Lp = lightsU.value, Lc = lightsCU.value;
      // moving marbles first (dimmer when a whole shower is in flight), then bubbles by recency
      let moving = 0;
      for (let i = 0; i < MAXU; i++) if (U[i].active && U[i].phase !== PH_LIVE) moving++;
      const mk = Math.min(1, 3.5 / Math.max(1, moving));
      for (let i = 0; i < MAXU && n < NL - 1; i++) {
        const u = U[i];
        if (!u.active || u.phase === PH_LIVE) continue;
        Lp[n].set(u.x, u.y + 0.6, u.z, (u.phase === PH_TUNNEL ? 1.6 : 1.1) * (u.phase === PH_TUNNEL ? 1 : mk));
        Lc[n].copy(u.phase === PH_TUNNEL ? u.col : C.marble); n++;
      }
      let m = 0;
      for (let i = 0; i < MAXU; i++) if (U[i].active && U[i].phase === PH_LIVE && U[i].alpha > 0.01) lightOrder[m++] = i;
      // simple insertion sort by birth time (newest first); m ≤ 36
      for (let a = 1; a < m; a++) { const x = lightOrder[a]; let b = a - 1; while (b >= 0 && U[lightOrder[b]].born < U[x].born) { lightOrder[b + 1] = lightOrder[b]; b--; } lightOrder[b + 1] = x; }
      if (selected && selected.active && selected.phase === PH_LIVE && n < NL - 1) {
        const u = selected; Lp[n].set(u.crunched ? u.tx : u.bx, u.crunched ? u.floorY + 0.8 : u.by, u.crunched ? u.tz : u.bz, (0.7 + 1.2 * u.flash) * Math.min(1, u.alpha + 0.3) * (u.crunched ? 0.3 : 1) + coda.glow * 0.6); Lc[n].copy(u.col); n++;
      }
      for (let a = 0; a < m && n < NL - 1; a++) {
        const u = U[lightOrder[a]];
        if (u === selected) continue;
        Lp[n].set(u.crunched ? u.tx : u.bx, u.crunched ? u.floorY + 0.8 : u.by, u.crunched ? u.tz : u.bz, (0.42 + 1.2 * u.flash) * Math.min(1, u.alpha + 0.3) * (u.crunched ? 0.3 : 1) + coda.glow * 0.6); Lc[n].copy(u.col); n++;
      }
      if (hideSet.has('lights')) n = 0;
      if (hover.amt > 0.01) { Lp[n].set(hover.x, heightAt(hover.x, hover.z) + 2.2, hover.z, 0.55 * hover.amt); Lc[n].copy(C.cursor); n++; }
      for (; n < NL; n++) Lp[n].w = 0;
    }

    function writeBubbles() {
      const cA = bCol, pA = bP;
      for (let i = 0; i < MAXU; i++) {
        const u = U[i];
        const o3 = i * 3, o4 = i * 4;
        if (!u.active || u.phase !== PH_LIVE || u.scale <= 0.001 || u.alpha <= 0.003) {
          setInst(bM, i, 0, -999, 0, 0.0001, 0.0001, 0.0001); pA[o4] = 0;
          galG[i] = 0; continue;
        }
        const L = u.laws;
        const bob = 0.22 * Math.sin(clock * 0.7 + u.seed * 6) * motion;
        const r = u.R * u.scale;
        let sx = r, sy = r, sz = r;
        if (L.dims === 2) { sy = r * 0.12; }
        else if (L.dims > 3) { const w = 0.12 * Math.sin(clock * 1.7 + u.seed * 9); sx = r * (1 + w); sy = r * (1 - w); sz = r * (1 + 0.7 * Math.sin(clock * 2.3 + u.seed * 4)); }
        setInst(bM, i, u.bx, u.by + bob, u.bz, sx, sy, sz);
        const sel = (u === selected ? 0.35 + 0.15 * Math.sin(clock * 2.2) : 0) + (u === hover.bubble ? 0.35 : 0);
        cA[o3] = u.col.r; cA[o3 + 1] = u.col.g; cA[o3 + 2] = u.col.b;
        pA[o4] = u.alpha * (1 + 0.6 * coda.glow); pA[o4 + 1] = u.seed; pA[o4 + 2] = sel + u.flash * 0.8 + coda.glow * 0.5; pA[o4 + 3] = u.heat;
        galBub[i].set(u.bx, u.by + bob, u.bz, r);
        galG[i] = u.galaxy * (1 + coda.glow);
      }
      // far ambient bubbles (eternal inflation elsewhere)
      for (let k = 0; k < NAMB; k++) {
        const b = amb[k], i = MAXU + k, o3 = i * 3, o4 = i * 4;
        if (!b.on) { setInst(bM, i, 0, -999, 0, 0.0001, 0.0001, 0.0001); pA[o4] = 0; continue; }
        const age = (clock - b.t0) / b.life;
        if (age >= 1) { b.on = false; pA[o4] = 0; continue; }
        const s = b.R * (1 - Math.exp(-age * 5));
        setInst(bM, i, b.x, b.y, b.z, s, s, s);
        cA[o3] = b.col.r; cA[o3 + 1] = b.col.g; cA[o3 + 2] = b.col.b;
        pA[o4] = 0.5 * smoothstep(0, 0.15, age) * (1 - smoothstep(0.55, 1, age)) * (1 + coda.glow); pA[o4 + 1] = k * 0.13; pA[o4 + 2] = 0.2; pA[o4 + 3] = 0;
      }
      // ghosts of bubbles left by tunnelling: they collapse
      for (let k = 0; k < NGHOST; k++) {
        const g = ghosts[k], i = MAXU + NAMB + k, o3 = i * 3, o4 = i * 4;
        if (!g.on) { setInst(bM, i, 0, -999, 0, 0.0001, 0.0001, 0.0001); pA[o4] = 0; continue; }
        const c = (clock - g.t0) / 1.3;
        if (c >= 1) { g.on = false; pA[o4] = 0; continue; }
        const s = g.R * (1 - c * c);
        setInst(bM, i, g.x, g.y, g.z, s, g.flat ? s * 0.12 : s, s);
        cA[o3] = g.col.r; cA[o3 + 1] = g.col.g; cA[o3 + 2] = g.col.b;
        pA[o4] = 0.8 + c; pA[o4 + 1] = 0.5; pA[o4 + 2] = 0.8 * c; pA[o4 + 3] = 0.6 * c;
      }
      bubbles.instanceMatrix.needsUpdate = true;
      bubbleGeo.attributes.aCol.needsUpdate = true;
      bubbleGeo.attributes.aP.needsUpdate = true;
    }

    function writeMarbles() {
      const cm = coreMesh.instanceMatrix.array, cc = coreMesh.instanceColor.array;
      for (let i = 0; i < MAXU; i++) {
        const u = U[i];
        const o3 = i * 3;
        if (!u.active) { setInst(cm, i, 0, -999, 0, 0, 0, 0); glowSize[i] = 0; continue; }
        let x = u.x, y = u.y, z = u.z, s = 1, gs = 1.9, ci = 4.5;
        let cr = C.marble.r, cg = C.marble.g, cb = C.marble.b;
        if (u.phase === PH_LIVE) {
          // the marble rises into its bubble and becomes its heart
          const age = clock - u.born, k = smoothstep(0.1, 1.5, age);
          const bob = 0.22 * Math.sin(clock * 0.7 + u.seed * 6) * motion;
          x = lerp(u.hx0, u.bx, k); z = lerp(u.hz0, u.bz, k); y = lerp(u.hy0, u.by + bob, k);
          s = lerp(1, 0.5, k) * (1 - u.dying); gs = lerp(1.9, 2.6, k) * (1 - u.dying); ci = lerp(4.5, 2.2, k);
          cr = lerp(C.marble.r, u.col.r, k); cg = lerp(C.marble.g, u.col.g, k); cb = lerp(C.marble.b, u.col.b, k);
          if (u.crunched) { const ember = 0.75 + 0.25 * Math.sin(clock * 1.3 + u.seed * 7); s = 0.4; gs = 1.7 * ember; ci = 3.2 * ember; cr = C.neg.r; cg = C.neg.g * 0.55; cb = C.neg.b * 0.45; y = u.floorY + 0.6; x = u.tx; z = u.tz; }
          ci *= 1 + u.flash + coda.glow * 0.8;
          gs *= 1 + coda.glow * 1.4;
        }
        if (u.phase === PH_TUNNEL) { cr = u.col.r; cg = u.col.g; cb = u.col.b; ci = 6; gs = 2.6; }
        setInst(cm, i, x, y, z, s, s, s);
        cc[o3] = cr * ci; cc[o3 + 1] = cg * ci; cc[o3 + 2] = cb * ci;
        glowPos[o3] = x; glowPos[o3 + 1] = y; glowPos[o3 + 2] = z;
        const ga = u.phase === PH_LIVE ? 0.4 : 0.55;
        glowCol[o3] = cr * ga; glowCol[o3 + 1] = cg * ga; glowCol[o3 + 2] = cb * ga;
        glowSize[i] = gs;
      }
      coreMesh.instanceMatrix.needsUpdate = true; coreMesh.instanceColor.needsUpdate = true;
      glowGeo.attributes.position.needsUpdate = true; glowGeo.attributes.aCol.needsUpdate = true; glowGeo.attributes.aSize.needsUpdate = true;
    }

    function writeTrails() {
      let o = 0;
      for (let i = 0; i < MAXU; i++) {
        const u = U[i];
        const on = u.active && u.tCount > 1 && u.tAlpha > 0.003;
        for (let k = 0; k < TRAIL - 1; k++, o++) {
          if (!on || k + 1 >= u.tCount) { trailA[o * 2] = 0; trailA[o * 2 + 1] = 0; continue; }
          const a = (u.tHead - 1 - k + TRAIL * 2) % TRAIL, b = (a - 1 + TRAIL) % TRAIL;
          trailPos[o * 6] = u.trail[a * 3]; trailPos[o * 6 + 1] = u.trail[a * 3 + 1]; trailPos[o * 6 + 2] = u.trail[a * 3 + 2];
          trailPos[o * 6 + 3] = u.trail[b * 3]; trailPos[o * 6 + 4] = u.trail[b * 3 + 1]; trailPos[o * 6 + 5] = u.trail[b * 3 + 2];
          const f0 = 1 - k / (TRAIL - 1), f1 = 1 - (k + 1) / (TRAIL - 1);
          trailA[o * 2] = f0 * f0 * u.tAlpha; trailA[o * 2 + 1] = f1 * f1 * u.tAlpha;
          trailC[o * 6] = trailC[o * 6 + 3] = 2.2; trailC[o * 6 + 1] = trailC[o * 6 + 4] = 1.55; trailC[o * 6 + 2] = trailC[o * 6 + 5] = 0.9;
        }
      }
      trailGeo.attributes.position.needsUpdate = true; trailGeo.attributes.aA.needsUpdate = true; trailGeo.attributes.aC.needsUpdate = true;
    }

    // ---- hover --------------------------------------------------------------------------------------------
    function computeHover() {
      hover.dirty = false;
      // a bubble under the cursor takes precedence: it can be clicked to read its laws
      const b = drag.on || coda.on ? null : pickBubble(hover.ndc);
      const terrainHit = !drag.on && !coda.on && pickTerrain(hover.ndc);
      const hb = b && (!terrainHit || b.t < camera.position.distanceTo(hit)) ? b.u : null;
      if (hb !== hover.bubble) { hover.bubble = hb; ctx.setCursor(hb ? 'point' : ''); }
      if (!terrainHit || hb) { hover.has = false; return; }
      hover.has = true; hover.x = hit.x; hover.z = hit.z;
      const lb = labelAt(hit.x, hit.z);
      if (lb !== hover.label) {
        hover.label = lb;
        if (A.on && clock - snd.hover > 0.11) { snd.hover = clock; A.pluck(A.degree(valleys[lb].deg + 2, 2), { gain: 0.035, decay: 0.8, bright: 0.25, partials: 4, pan: panOf(hit.x, hit.y, hit.z) }); }
      }
      // steepest-descent path from the cursor to the valley floor (gradient flow on the B-spline)
      let x = hit.x, z = hit.z, n = 0, acc = 0;
      const v = valleys[lb], tx = v.x + Math.round((x - v.x) / P) * P, tz = v.z + Math.round((z - v.z) / P) * P;
      pathPos[0] = x; pathPos[1] = heightAt(x, z) + 0.3; pathPos[2] = z; n = 1;
      for (let it = 0; it < 900 && n < NPATH; it++) {
        sample(x, z);
        const g = Math.hypot(S[1], S[2]);
        const dx = tx - x, dz = tz - z, dist = Math.hypot(dx, dz);
        if (dist < 0.3) break;
        let sx, sz;
        if (g < 0.02 || dist < 1.2) { sx = dx / dist; sz = dz / dist; } else { sx = -S[1] / g; sz = -S[2] / g; }
        const st = 0.12;
        x += sx * st; z += sz * st; acc += st;
        if (acc >= 0.42) { acc = 0; pathPos[n * 3] = x; pathPos[n * 3 + 1] = heightAt(x, z) + 0.3; pathPos[n * 3 + 2] = z; n++; }
      }
      hover.n = n;
      path.geometry.attributes.position.needsUpdate = true;
    }

    // ---- ambient nucleations far away -------------------------------------------------------------------------
    function spawnAmbient() {
      let b = null;
      for (let i = 0; i < NAMB && !b; i++) if (!amb[i].on) b = amb[i];
      if (!b) return;
      const fx = cam.tx - camera.position.x, fz = cam.tz - camera.position.z, fl = Math.hypot(fx, fz) || 1;
      const th = Math.atan2(fx / fl, fz / fl) + (ambR() - 0.5) * 1.3;
      const d = 120 + ambR() * 140;
      b.x = camera.position.x + Math.sin(th) * d; b.z = camera.position.z + Math.cos(th) * d;
      b.R = 4 + ambR() * 6; b.y = Math.max(heightAt(b.x, b.z), 0) + b.R * 0.9;
      b.t0 = clock; b.life = 6 + ambR() * 5; b.on = true;
      b.col.copy(ambCols[Math.floor(ambR() * ambCols.length)]);
    }

    // ---- camera ----------------------------------------------------------------------------------------------
    function updateCamera(dt) {
      const idle = clock - lastInput;
      if (!drag.on && idle > 4) goal.yaw += dt * (coda.on ? 0.035 : 0.011) * motion;
      const k = damp(drag.on ? 9 : 2.6, dt);
      cam.yaw += (goal.yaw - cam.yaw) * k; cam.pitch += (goal.pitch - cam.pitch) * damp(coda.on ? 0.55 : 2.6, dt);
      cam.dist += (goal.dist - cam.dist) * damp(coda.on ? 0.45 : 3, dt);
      cam.tx += (goal.tx - cam.tx) * damp(4, dt); cam.tz += (goal.tz - cam.tz) * damp(4, dt);
      const yaw = cam.yaw + 0.012 * Math.sin(clock * 0.11) * motion;
      const pitch = cam.pitch + 0.008 * Math.sin(clock * 0.07 + 1) * motion;
      const dist = cam.dist + 1.2 * Math.sin(clock * 0.05) * motion;
      const ty = 0;
      let px = cam.tx + dist * Math.cos(pitch) * Math.sin(yaw);
      let pz = cam.tz + dist * Math.cos(pitch) * Math.cos(yaw);
      let py = ty + dist * Math.sin(pitch);
      const ground = Math.max(heightAt(px, pz), 0) + 6;
      if (py < ground) { py = ground; goal.pitch = Math.min(1.3, goal.pitch + dt * 0.4); }
      camera.position.set(px, py, pz);
      camera.lookAt(cam.tx, ty, cam.tz);
      sky.position.copy(camera.position);
      stars.position.copy(camera.position);
      // the terrain and dots snap by whole periods: invisible, since the field repeats exactly
      const sx = Math.round(cam.tx / P) * P, sz = Math.round(cam.tz / P) * P;
      dots.position.set(sx, 0, sz);
      const fx = cam.tx - px, fz = cam.tz - pz, fl = Math.hypot(fx, fz) || 1;
      const lead = Math.min(62, fl * 0.75);
      const cx = px + (fx / fl) * lead, cz = pz + (fz / fl) * lead;
      for (let i = 0; i < terrains.length; i++) { const m = terrains[i], st = m.userData.s; m.position.set(Math.round(cx / st) * st, 0, Math.round(cz / st) * st); }
      water.position.set(camera.position.x, 0, camera.position.z);
      const mx = Math.round(cam.tx / MIST_STEP) * MIST_STEP, mz = Math.round(cam.tz / MIST_STEP) * MIST_STEP;
      for (let i = 0; i < mists.length; i++) mists[i].position.set(mx, 0, mz);
      mistCenter.set(mx, mz);
    }

    // ---- debug staging (harmless when absent) -------------------------------------------------------------------
    function stageDebug() {
      const hv = params.get('m6hover');
      if (hv) { const [fx, fy] = hv.split(',').map(Number); later(0.4, () => { hover.ndc.set(fx * 2 - 1, -(fy * 2 - 1)); hover.dirty = true; lastInput = clock; }); }
    }

    // debug: hide systems to profile them (?m6hide=bubbles,water,mist,glows,trails,galaxies,lights,dots,sky)
    const hideSet = new Set((params.get('m6hide') || '').split(',').filter(Boolean));
    const hideMap = { bubbles, water, glows, trails, galaxies, dots, sky, cores: coreMesh };
    hideSet.forEach((k) => { if (hideMap[k]) hideMap[k].visible = false; if (k === 'mist') mists.forEach((m) => { m.visible = false; }); });
    // debug: GPU time per frame via a timer query spanning one update to the next (?m6gpu)
    let gpuProbe = null;
    if (params.has('m6gpu')) {
      const gl = ctx.renderer.getContext(), ext = gl.getExtension('EXT_disjoint_timer_query_webgl2');
      if (!ext) console.log('m6gpu: timer query unavailable');
      else {
        let q = null; const pend = []; const res = []; let reported = false;
        gpuProbe = () => {
          if (q) { gl.endQuery(ext.TIME_ELAPSED_EXT); pend.push(q); q = null; }
          for (let i = pend.length - 1; i >= 0; i--) {
            if (gl.getQueryParameter(pend[i], gl.QUERY_RESULT_AVAILABLE)) { if (!gl.getParameter(ext.GPU_DISJOINT_EXT)) res.push(gl.getQueryParameter(pend[i], gl.QUERY_RESULT) / 1e6); gl.deleteQuery(pend[i]); pend.splice(i, 1); }
          }
          if (clock > 3 && !reported) { q = gl.createQuery(); gl.beginQuery(ext.TIME_ELAPSED_EXT, q); }
          if (res.length >= 120 && !reported) {
            reported = true; res.sort((x, y) => x - y);
            console.log(`m6gpu ms: median ${res[60].toFixed(2)} p90 ${res[108].toFixed(2)} max ${res[119].toFixed(2)}`);
          }
        };
      }
    }

    // ---- the instance --------------------------------------------------------------------------------------------
    return {
      scene, camera,
      post: POST,

      enter() {
        active = true;
        if (coda.on) endCoda(false);
        coda.silenced = false;
        goal.pitch = DEF.pitch; goal.dist = DEF.dist * (portrait ? 1.22 : 1);
        lastInput = clock;
        basinsAmt = basinsOn ? 1 : 0;
        buildPanels();
        if (!seen.first) ui.hint(ctx.isTouch ? 'Tap the map to drop a universe' : 'Click the map to drop a universe · drag to fly · scroll to zoom');
        stageDebug();
      },

      exit() {
        active = false;
        if (rollVoice) { rollVoice.stop(); rollVoice = null; }
        queue.length = 0;
        drag.on = false; hover.has = false; hover.amt = 0; hover.label = -1; hover.bubble = null; touches.clear(); drag.pinch = false;
        ctx.setCursor('');
        if (coda.on) endCoda(false);
        if (resetOnExit) {
          resetOnExit = false;
          for (let i = 0; i < MAXU; i++) deactivate(U[i]);
          made = 0; lifeFound = 0; settledN = 0; selected = null;
          for (const k in seen) seen[k] = false;
          Object.assign(goal, DEF); Object.assign(cam, DEF);
        }
        readout = null; controls = null;
      },

      resize(w, h) {
        const asp = w / Math.max(1, h);
        const wasPortrait = portrait;
        portrait = asp < 1;
        camera.fov = portrait ? 64 : asp < 1.35 ? 54 : 50;
        // composition: hero a touch right of centre on desktop, in the upper-middle on phones
        camera.setViewOffset(w, h, portrait ? 0 : -0.045 * w, portrait ? 0.1 * h : 0.015 * h, w, h);
        camera.updateProjectionMatrix();
        scaleU.value = (h / 900) * ctx.pixelRatio;
        if (portrait !== wasPortrait && !coda.on) goal.dist = DEF.dist * (portrait ? 1.22 : 1);
        if (active && readout) buildReadout();
      },

      update(dt, t) {
        if (gpuProbe) gpuProbe();
        clock += dt;
        timeU.value = clock;
        for (let i = queue.length - 1; i >= 0; i--) if (queue[i].at <= clock) { const f = queue[i].fn; queue.splice(i, 1); f(); }

        updateCamera(dt);
        stars.userData.update(clock);
        if (hover.dirty) computeHover();
        hover.amt += ((hover.has && !drag.on && !coda.on && clock - lastInput < 6 ? 1 : 0) - hover.amt) * damp(6, dt);
        hoverU.value.set(hover.x, hover.z, hover.amt, hover.has ? hover.label : -1);
        dots.material.uniforms.uHoverL.value = hover.amt > 0.05 ? hover.label : -1;
        path.material.uniforms.uCount.value = hover.has ? hover.n : 0;
        path.material.uniforms.uAmt.value = hover.amt;
        basinsAmt += ((basinsOn && !coda.on ? 1 : 0) - basinsAmt) * damp(5, dt);
        terrainMat.uniforms.uBasins.value = basinsAmt;

        // universes
        let rollE = 0, rollN = 0;
        for (let i = 0; i < MAXU; i++) {
          const u = U[i];
          if (!u.active) continue;
          if (u.phase === PH_FALL || u.phase === PH_ROLL || u.phase === PH_GLIDE) {
            stepMarble(u, dt);
            if (u.phase === PH_ROLL) { rollE += Math.hypot(u.vx, u.vz); rollN++; }
          } else if (u.phase === PH_TUNNEL) stepTunnel(u, dt);
          if (u.phase === PH_LIVE) animateBubble(u, dt);
          // trail sampling while moving; it fades once the universe has settled
          if (u.phase !== PH_LIVE) { if (clock - u.tLast > 0.025) { u.tLast = clock; pushTrail(u); } u.tAlpha = 1; }
          else u.tAlpha = Math.max(0, u.tAlpha - dt * 0.7);
          if (u.dying) { u.dying += dt * 1.6; if (u.dying >= 1) deactivate(u); }
        }
        // arcs fade after their universe has arrived
        for (let k = 0; k < arcs.length; k++) {
          const a = arcs[k];
          if (!a.busy) { a.pts.visible = false; continue; }
          if (a.done) {
            const f = 1 - (clock - a.t0) / 1.6;
            a.mat.uniforms.uAlpha.value = Math.max(0, f);
            if (f <= 0) { a.busy = false; a.done = false; a.pts.visible = false; }
          } else a.mat.uniforms.uAlpha.value = 1;
        }

        // eternal inflation, far off
        if (clock > ambNext) { spawnAmbient(); ambNext = clock + 1.6 + ambR() * 2.4; }

        // coda
        if (coda.on) stepCoda(dt);
        else if (codaOfferT > 0 && clock - codaOfferT > 3 && clock - lastInput > 16 && !coda.auto && active) { coda.auto = true; startCoda(); }
        codaU.value += ((coda.on ? coda.glow : 0) - codaU.value) * damp(2, dt);

        writeLights(); writeBubbles(); writeMarbles(); writeTrails();

        // the rolling bed: one soft noise voice that follows the total motion
        if (!rollVoice && A.on && A.ctx && rollN > 0) rollVoice = makeRollVoice();
        if (rollVoice && clock - rollT > 0.05) { rollT = clock; rollVoice.set(A.on ? clamp(rollE * 0.006, 0, 0.07) : 0, 240 + (rollN ? rollE / rollN : 0) * 70); }

        if (readout && panelMode === 'uni' && selected && !selected.active) { selected = null; buildReadout(); }
      },

      onPointerDown(ev) {
        const o = ev.original;
        if (ev.pointerType === 'touch' && o) {
          if (o.isPrimary) { touches.clear(); drag.pinch = false; }
          touches.set(o.pointerId, { x: ev.x, y: ev.y });
          if (touches.size >= 2) { drag.pinch = true; drag.moved = true; pinchD = touchSpread(); lastInput = clock; return; }
        }
        drag.on = true; drag.moved = false; drag.x0 = ev.x; drag.y0 = ev.y; drag.t0 = clock;
        drag.pan = ev.button === 2 || ev.shift; drag.touch = ev.pointerType === 'touch';
        lastInput = clock;
      },
      onPointerMove(ev) {
        const o = ev.original;
        if (ev.pointerType === 'touch' && o && touches.has(o.pointerId)) {
          touches.get(o.pointerId).x = ev.x; touches.get(o.pointerId).y = ev.y;
          if (drag.pinch) {
            if (touches.size >= 2 && !coda.on) {
              const d = touchSpread();
              if (pinchD > 1 && d > 1) goal.dist = clamp(goal.dist * pinchD / d, 30, 175);
              pinchD = d; lastInput = clock;
            }
            return;
          }
        }
        if (drag.on && ctx.pointer.down) {
          if (!drag.moved && Math.hypot(ev.x - drag.x0, ev.y - drag.y0) > (drag.touch ? 10 : 5)) { drag.moved = true; ctx.setCursor('grabbing'); hover.has = false; }
          if (drag.moved) {
            lastInput = clock;
            if (drag.pan) {
              const s = cam.dist * 0.0016, sy = Math.sin(cam.yaw), cy = Math.cos(cam.yaw);
              goal.tx += (-ev.dx * cy - ev.dy * sy) * s; goal.tz += (ev.dx * sy - ev.dy * cy) * s;
            } else {
              goal.yaw -= ev.dx * 0.0052;
              goal.pitch = clamp(goal.pitch + ev.dy * 0.0035, 0.1, 1.3);
            }
          }
          return;
        }
        if (ev.pointerType === 'touch') return;
        hover.ndc.copy(ev.ndc); hover.dirty = true; lastInput = clock;
      },
      onPointerUp(ev) {
        if (ev.original && ev.pointerType === 'touch') touches.delete(ev.original.pointerId);
        const wasDrag = drag.moved || drag.pinch;
        drag.on = false; drag.moved = false;
        ctx.setCursor(''); hover.bubble = null; hover.dirty = ev.pointerType !== 'touch';
        if (wasDrag || clock - drag.t0 > 0.8 || coda.on) return;
        const b = pickBubble(ev.ndc);
        const tHit = pickTerrain(ev.ndc) ? camera.position.distanceTo(hit) : Infinity;
        if (b && b.t < tHit) { select(b.u, true); pinUntil = clock + 6; return; }
        if (tHit < Infinity) drop(hit.x, hit.z);
      },
      onWheel(ev) {
        if (coda.on) return;
        goal.dist = clamp(goal.dist * Math.exp(ev.deltaY * 0.0011), 30, 175);
        lastInput = clock;
      },
      onKey(ev) {
        if (ev.type !== 'down') return false;
        const k = ev.key;
        if (k === ' ') { if (!ev.repeat && !coda.on) randomDrop(); return true; }
        if (k === '+' || k === '=') { goal.dist = clamp(goal.dist * 0.88, 30, 175); lastInput = clock; return true; }
        if (k === '-' || k === '_') { goal.dist = clamp(goal.dist * 1.13, 30, 175); lastInput = clock; return true; }
        if (k === 'Escape' && coda.on) { endCoda(true); return true; }
        if ((k === 'p' || k === 'P') && coda.on) { if (!ev.repeat) playAgain(); return true; }
        return false;
      },

      demo() {
        if (coda.on) return;
        // eight universes: one aimed at a valley that can hold stars, one at a lake, the rest across the view
        const picks = [];
        const nearest = (list, loose = false) => {
          let best = -1, bd = Infinity;
          for (const k of list) {
            const v = valleys[k];
            const x = v.x + Math.round((cam.tx - v.x) / P) * P, z = v.z + Math.round((cam.tz - v.z) / P) * P;
            const d = Math.hypot(x - cam.tx, z - cam.tz);
            if (d < bd && inClear(x, Math.max(v.h, 0) + 2.4, z, loose)) { bd = d; best = k; }
          }
          return best >= 0 || loose ? best : nearest(list, true);
        };
        const lifeK = nearest(lifeValleys);
        demoPinLife = lifeK >= 0;
        const lakeK = nearest(valleys.filter((v) => v.kind === 2 && v.area > 200).map((v) => v.id));
        [lifeK, lakeK].forEach((k) => { if (k >= 0) picks.push(k); });
        for (let i = 0; i < 8; i++) {
          const delay = 0.12 * i;
          later(delay, () => {
            if (i < picks.length) {
              const v = valleys[picks[i]];
              const x = v.x + Math.round((cam.tx - v.x) / P) * P, z = v.z + Math.round((cam.tz - v.z) / P) * P;
              drop(x + 0.25, z - 0.2);
            } else randomDrop();
          });
        }
        if (params.has('coda')) later(3.4, startCoda);
        if (params.has('m6rain')) later(1.6, () => rain(12));
        if (params.has('m6tunnel')) {
          const tryTunnel = () => { const u = (lifeK >= 0 && U.find((w) => w.active && w.k === lifeK && w.phase === PH_LIVE)) || latestLive(); if (u) { select(u, false); tunnel(); } else later(0.3, tryTunnel); };
          later(2.6, tryTunnel);
        }
        if (params.has('m6sel') && lifeK >= 0) later(4.2, () => { const u = U.find((w) => w.active && w.k === lifeK && w.phase === PH_LIVE); if (u) select(u, false); });
      },
    };
  },
};
