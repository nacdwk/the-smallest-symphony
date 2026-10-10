// I. Descent: one continuous powers-of-ten dive from a grain of salt to a single closed string.
//
// Scale model. `z` is the depth: the frame's smaller dimension spans 10^-z metres (z = 3 is a
// millimetre, z = 35 is the Planck scale). The camera never moves. Instead the `world` group is
// scaled by U = Href * 10^z world units per metre about a fixed pivot, and every layer sits in
// true metres relative to that pivot: the string at the origin, the up quark that holds it, the
// proton around that quark, the chlorine-35 nucleus around that proton, the chloride ion, the
// NaCl lattice and finally the crystal face of the grain. The dive is a single similarity
// transform; layers fade in while they are tiny and fade out once they grow past the camera.
// Scale-free things (dust motes, quantum foam, point-like quarks) live outside the scaled world
// and are driven by z directly. Every layer is a custom shader: an analytically ray-traced
// refracting cube, an infinite-zoom terrace texture, sphere impostors, sampled orbitals,
// GPU ribbons for flux tubes and the string, and an infinite-zoom foam whose fluctuations grow
// as the scale approaches the Planck length.
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { palette, glowTexture, makeStarfield, GLSL, clamp, smoothstep, damp, sup, rng } from '../fx.js';

// ---- physics ------------------------------------------------------------------------------
const GRAIN = 3.0e-4;                 // edge of a grain of table salt (m)
const A_NACL = 2.82e-10;              // Na-Cl nearest-neighbour distance in rock salt (m)
const SURF_K = 3;                     // the ion we dive into sits 3 layers below the (100) face
const A0 = 0.529177;                  // Bohr radius (angstrom)
const R_NUC = 1.2 * Math.cbrt(35);    // chlorine-35 nuclear radius, 1.2 fm * A^(1/3) = 3.93 fm
const R_NUCLEON = 0.84;               // proton charge radius (fm)
const L_P = 1.616e-35;                // Planck length sqrt(hbar G / c^3) (m)
const R_STR = L_P / (2 * Math.PI);    // a closed loop one Planck length around
const LIMIT_D = 1e-18;                // diameter of the "edge of experiment" ring (m)
const IOR = [1.540, 1.5443, 1.553];   // NaCl refractive index at 656 / 589 / 486 nm

// ---- staging ------------------------------------------------------------------------------
const CAM_Z = 8, FOV = 40;
const Z_MIN = 3, Z_MAX = 35;
const TAN = Math.tan(THREE.MathUtils.degToRad(FOV / 2));
const TILT = new THREE.Euler(0.5, -0.64, 0, 'YXZ'); // crystal orientation: three faces visible
const FACE_ON = new THREE.Euler(0.075, -0.095, 0.32, 'YXZ'); // later: the face turned toward us, a little off-axis

// Layers, by the depth at which they take over the readout.
const LAYERS = [
  { id: 'salt', z0: -1, name: 'A grain of salt', probe: 'Ring', by: 'your eye',
    note: 'Table salt, about 0.3 mm across. Each grain is a single crystal of sodium chloride.' },
  { id: 'face', z0: 3.85, name: 'The crystal face', probe: 'Ring', by: 'light microscope',
    note: 'A face of the crystal. It is almost perfectly flat: wide terraces separated by steps one or a few atoms high.' },
  { id: 'lattice', z0: 8.55, name: 'The crystal lattice', probe: 'Shake', by: 'X-ray diffraction',
    note: 'Sodium (gold) and chloride (cyan) ions alternate every 0.28 nm. The pattern repeats about a million times across the grain.' },
  { id: 'ion', z0: 9.2, name: 'A chloride ion', probe: 'Measure', by: 'electron scattering',
    note: 'Eighteen electrons, each spread out as a cloud of probability. Each colour is one orbital holding two of them. Almost all the mass sits in a nucleus tens of thousands of times smaller.' },
  { id: 'nucleus', z0: 12.8, name: 'The nucleus', probe: 'Kick', by: 'electron scattering',
    note: 'Chlorine-35: 17 protons (gold) and 18 neutrons (violet), about 8 fm across. It carries 99.97% of the mass of the ion.' },
  { id: 'proton', z0: 14.3, name: 'A proton', probe: 'Stir', by: 'deep inelastic scattering',
    note: 'Two up quarks and one down quark, bound by gluons, drawn here as flux tubes. Quark and antiquark pairs flicker in and out between them.' },
  { id: 'quark', z0: 16.3, name: 'An up quark', probe: 'Probe', by: 'HERA, LHC',
    note: 'As far as any experiment can tell, a quark is a point. The ring is 10⁻¹⁸ m wide, about where colliders stop seeing.' },
  { id: 'desert', z0: 18.6, name: 'The desert', probe: 'Probe', by: 'nothing yet',
    note: 'No experiment has seen anything in this range.' },
  { id: 'foam', z0: 29.6, name: 'Quantum foam', probe: 'Probe', by: 'theory only',
    note: 'Near the Planck length, quantum uncertainty is expected to make spacetime itself fluctuate, more strongly the closer you look. No experiment has tested this.' },
  { id: 'string', z0: 33.9, name: 'A string', probe: 'Pluck', by: 'theory only',
    note: 'In string theory, this is the instrument: a closed loop, drawn here 1.6 × 10⁻³⁵ m around. If an atom were blown up to the size of the observable universe, this loop would be about 140 m long, roughly the height of the Great Pyramid. Movement II lets you play it.' },
];
const PADS = [
  { z: 3.0, label: '−3', sub: 'salt' }, { z: 9.45, label: '−10', sub: 'atom' }, { z: 13.85, label: '−14', sub: 'nucleus' },
  { z: 14.45, label: '−15', sub: 'proton' }, { z: 17.55, label: '−18', sub: 'quark' }, { z: 35, label: '−35', sub: 'string' },
];
// Atmosphere tint keyed by depth (multiplied small; it is a colour grade, not a light).
const TINTS = [
  [3, '#2a1d5c'], [4.5, '#1b2650'], [8.8, '#103040'], [10, '#24184c'], [13, '#3a1630'], [15, '#40142e'],
  [17, '#1a1030'], [20, '#07060d'], [29, '#0a0816'], [32, '#24104a'], [35, '#1c1238'],
];

const SI = [['m', 0], ['mm', -3], ['µm', -6], ['nm', -9], ['pm', -12], ['fm', -15], ['am', -18], ['zm', -21], ['ym', -24], ['rm', -27], ['qm', -30]];

// ---- shared GLSL ----------------------------------------------------------------------------
const G_COMMON = /* glsl */`
  uniform vec3 uGold, uCyan, uMag, uViolet, uEmber, uMint, uInk;
  float hash11(float p){ p = fract(p * 0.1031); p *= p + 33.33; p *= p + p; return fract(p); }
  float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
  vec3 hash31(float p){ vec3 p3 = fract(vec3(p) * vec3(0.1031, 0.1030, 0.0973)); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.xxy + p3.yzz) * p3.zyx); }
  vec3 hash32(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973)); p3 += dot(p3, p3.yxz + 33.33); return fract((p3.xxy + p3.yzz) * p3.zyx); }
`;

// Screen-space ribbon: a curve(s) is evaluated three times per vertex on the GPU and the strip is
// expanded perpendicular to its projected tangent, so lines stay smooth and evenly thick.
const G_RIBBON = /* glsl */`
  uniform vec2 uRes;
  attribute float aS; attribute float aSide; attribute float aId;
  varying float vV; varying float vS; varying float vId;
`;
const ribbonMain = (width) => /* glsl */`
  vec3 p0 = curve(aS), pa = curve(aS - 0.0015), pb = curve(aS + 0.0015);
  vec4 c0 = projectionMatrix * modelViewMatrix * vec4(p0, 1.0);
  vec4 ca = projectionMatrix * modelViewMatrix * vec4(pa, 1.0);
  vec4 cb = projectionMatrix * modelViewMatrix * vec4(pb, 1.0);
  vec2 d = (cb.xy / cb.w - ca.xy / ca.w) * uRes;
  vec2 dir = d / max(length(d), 1e-6);
  float wpx = ${width};
  c0.xy += vec2(-dir.y, dir.x) * aSide * wpx / uRes * 2.0 * c0.w;
  vV = aSide; vS = aS; vId = aId;
  gl_Position = c0;
`;

// A procedural studio environment (soft boxes and strip lights in the palette) for reflections
// and refractions of the salt crystal. Directions are world space.
const G_ENV = /* glsl */`
  // Lights are placed where the grain's three visible faces mirror them (top: overhead softbox,
  // right: a tall cyan strip, front: a low magenta kicker), so highlights slide across the faces
  // as it turns. A warm key from the front catches the bevels.
  vec3 env(vec3 d){
    vec3 c = vec3(0.010, 0.008, 0.026) * (0.55 + 0.6 * max(d.y, -0.6));          // dim violet sky
    vec3 k = normalize(vec3(-0.55, 0.62, 0.56)); float kd = dot(d, k);            // warm key, high front left
    c += uGold * (2.6 * smoothstep(0.975, 0.99, kd) + 0.06 * pow(max(kd, 0.0), 6.0));
    vec3 o = normalize(vec3(-0.25, 0.72, -0.65)); float od = dot(d, o);           // overhead softbox, behind
    c += mix(uInk, uGold, 0.35) * (0.7 * smoothstep(0.972, 0.99, od) + 0.035 * pow(max(od, 0.0), 6.0));
    vec3 f = normalize(vec3(0.94, 0.06, -0.32));                                  // tall cool strip, right
    float ah = dot(normalize(d.xz + 1e-5), normalize(f.xz));
    c += uCyan * 2.0 * smoothstep(0.994, 0.9985, ah) * (1.0 - smoothstep(0.3, 0.55, abs(d.y - 0.05)));
    c += uCyan * 0.035 * pow(max(dot(d, f), 0.0), 5.0);
    vec3 r = normalize(vec3(-0.72, -0.68, -0.1));                                 // low magenta kicker, left
    c += uMag * (0.5 * smoothstep(0.965, 0.99, dot(d, r)) + 0.06 * pow(max(dot(d, r), 0.0), 6.0));
    c += uInk * 0.4 * exp(-abs(d.y - 0.30) * 70.0) * smoothstep(-0.2, 0.6, -d.z);   // a far horizon line
    c += uViolet * 0.22 * exp(-abs(d.x + 0.2) * 50.0) * smoothstep(0.0, 0.7, d.y);
    return c;
  }
`;

// Infinite-zoom texture of a (100) rock-salt face. It is mostly flat: at every decade from 100 um
// down to 1 nm a few cells grow a square hillock (stacked terraces; at the finest scales the steps
// are one ion high), and a few long cleavage steps run straight along the <100> directions. Then the
// surface ions themselves. q: position on the face in metres, fp: metres per pixel.
const G_FACE = /* glsl */`
  float stepLine(float c, float fpc, float salt, float prob){
    float hs = hash12(vec2(floor(c), salt));
    if (hs >= prob) return 0.0;
    float off = (hs / prob - 0.5) * 0.7;
    float dpx = abs(fract(c) - 0.5 - off) / fpc;            // distance to the step in pixels
    return 1.0 - smoothstep(0.35, 1.3, dpx);
  }
  vec3 facePattern(vec2 q, float fp, float t, float glint, float atoms){
    vec3 col = vec3(0.0);
    for (int i = 0; i < 6; i++) {
      float fi = float(i);
      float S = pow(10.0, -4.0 - fi);
      // a decade shows once its cells span ~100 px; decades far coarser than the view are locally flat, so skip them
      float w = smoothstep(26.0 * fp, 260.0 * fp, S) * (1.0 - smoothstep(2.0e4 * fp, 6.0e4 * fp, S));
      if (w <= 0.001) continue;
      vec3 tint = mix(mix(uCyan, uInk, 0.35), uGold, fi / 5.0);
      vec2 c = q / S + vec2(17.31, 41.13) * fi;
      vec2 id = floor(c);
      float best = -1.0; vec2 bd = vec2(0.0); float br = 1.0; float bh = 0.0;
      for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
        vec2 cid = id + vec2(float(x), float(y));
        vec3 h = hash32(cid + fi * 31.7);
        if (hash12(cid * 1.37 + fi * 5.1) > 0.36) continue;  // about one cell in three
        vec2 dd = c - (cid + 0.2 + 0.6 * h.xy);
        float r = 0.3 + 0.75 * h.z;
        float hh = 1.0 - max(abs(dd.x), abs(dd.y)) / r;
        if (hh > best) { best = hh; bd = dd; br = r; bh = h.x; }
      }
      if (best > 0.0) {
        float levels = 1.0 + floor(bh * 3.0);
        float L = best * levels;
        float fr = fract(L);
        float fw = levels / (br * S) * fp * 1.2;             // analytic derivative of L per pixel
        float edge = 1.0 - smoothstep(0.0, fw, min(fr, 1.0 - fr));
        col += w * edge * tint * 0.45;
        col += w * mix(uViolet, tint, 0.3) * 0.055 * floor(L);  // raised terraces catch a little more light
      }
      float fpc = fp / S;
      col += w * tint * 0.4 * (stepLine(c.y, fpc, fi * 7.3, 0.14) + stepLine(c.x, fpc, fi * 3.1 + 50.0, 0.1));
    }
    float wa = smoothstep(5.0 * fp, 18.0 * fp, ${A_NACL.toExponential(4)}) * atoms;
    if (wa > 0.001) {
      vec2 g = q / ${A_NACL.toExponential(4)} + 0.5; vec2 id = floor(g); vec2 f = fract(g) - 0.5;
      float cl = mod(id.x + id.y, 2.0);   // Cl where (i + j + 3) is even, matching the 3D lattice
      float r = mix(0.12, 0.2, cl);
      float d = length(f);
      col += wa * mix(uGold, uCyan, cl) * exp(-d * d / (r * r)) * 0.4;
    }
    return col;
  }
`;

// ---- small helpers -------------------------------------------------------------------------
function colorUniforms() {
  return {
    uGold: { value: palette.planck }, uCyan: { value: palette.harmonic }, uMag: { value: palette.resonance },
    uViolet: { value: palette.violet }, uEmber: { value: palette.ember }, uMint: { value: palette.mint }, uInk: { value: palette.ink },
  };
}

/** Index buffer + attributes for `strands` ribbons of `segments` segments each. */
function ribbonGeometry(segments, strands = 1) {
  const n = segments + 1, vc = n * 2 * strands;
  const aS = new Float32Array(vc), aSide = new Float32Array(vc), aId = new Float32Array(vc);
  const idx = [];
  for (let s = 0; s < strands; s++) {
    for (let i = 0; i < n; i++) {
      const v = (s * n + i) * 2;
      aS[v] = aS[v + 1] = i / segments; aSide[v] = -1; aSide[v + 1] = 1; aId[v] = aId[v + 1] = s;
      if (i < segments) idx.push(v, v + 2, v + 1, v + 1, v + 2, v + 3);   // counter-clockwise on screen (side +1 is left of the tangent)
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vc * 3), 3));
  g.setAttribute('aS', new THREE.BufferAttribute(aS, 1));
  g.setAttribute('aSide', new THREE.BufferAttribute(aSide, 1));
  g.setAttribute('aId', new THREE.BufferAttribute(aId, 1));
  g.setIndex(idx);
  return g;
}

/** Soft bokeh disc (brighter rim, like an out-of-focus highlight through a lens). */
function bokehTexture() {
  const s = 128, c = document.createElement('canvas'); c.width = c.height = s;
  const g = c.getContext('2d');
  const grd = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
  grd.addColorStop(0, 'rgba(255,255,255,0.3)');
  grd.addColorStop(0.7, 'rgba(255,255,255,0.36)');
  grd.addColorStop(0.86, 'rgba(255,255,255,0.46)');
  grd.addColorStop(0.97, 'rgba(255,255,255,0.08)');
  grd.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grd; g.fillRect(0, 0, s, s);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// Hydrogen-like radial functions R_nl (unnormalised) with rho = 2 Z r / (n a0).
function radial(n, l, rho) {
  const e = Math.exp(-rho / 2);
  if (n === 1) return e;
  if (n === 2) return l === 0 ? (2 - rho) * e : rho * e;
  return l === 0 ? (6 - 6 * rho + rho * rho) * e : rho * (4 - rho) * e;
}
/** Inverse-CDF table for the radial probability r^2 R_nl^2 (r in angstrom). */
function radialTable(n, l, Z) {
  const N = 1536, rMax = (n * n * 7 + 4) * A0 / Z;
  const cdf = new Float64Array(N + 1);
  let acc = 0;
  for (let i = 1; i <= N; i++) {
    const r = (i - 0.5) / N * rMax, R = radial(n, l, 2 * Z * r / (n * A0));
    acc += r * r * R * R; cdf[i] = acc;
  }
  for (let i = 0; i <= N; i++) cdf[i] /= acc;
  return (u) => {
    let lo = 0, hi = N;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (cdf[m] < u) lo = m; else hi = m; }
    const f = (u - cdf[lo]) / Math.max(1e-12, cdf[hi] - cdf[lo]);
    return (lo + f) / N * rMax;
  };
}

function fmtSI(z) {
  // field of view 10^-z metres with the largest SI prefix that keeps the number >= 1. Below quecto
  // (10^-30 m, the smallest prefix there is) we count the remaining powers of ten in quectometres.
  if (z > 30.02) return '10' + sup(-Math.round(z - 30)) + ' qm';
  const v = Math.pow(10, -z);
  for (let i = 0; i < SI.length; i++) {
    const u = SI[i][0], x = v / Math.pow(10, SI[i][1]);
    if (x >= 0.995 || i === SI.length - 1) return (x >= 9.95 ? x.toFixed(0) : x.toFixed(1)) + ' ' + u;
  }
  return '';
}

// ============================================================================================
export default {
  id: 'descent',
  numeral: 'I',
  title: 'Descent',
  tempo: 'Largo, sempre più piccolo',
  caption: 'You are looking at a grain of salt. Dive in: every step down is ten times smaller. Atoms, nuclei and quarks are as far as experiments reach. String theory proposes that the dive goes on for up to seventeen more powers of ten, with a <em>string</em> at the bottom.',
  fine: `<p>Each scale has its own instrument. Light microscopes stop near 200 nm. Electron and atomic force microscopes resolve single atoms, and X-ray diffraction revealed the 0.28 nm spacing of rock salt in 1913. Electron scattering and atomic spectroscopy measured the size of nuclei (chlorine-35 is about 4 fm in radius) and of the proton (0.84 fm).</p>
<p>Collisions at HERA and the LHC probe the quarks inside protons and find no size down to about 10⁻¹⁸ to 10⁻¹⁹ m. No experiment has resolved any structure smaller than that.</p>
<p>The Planck length, √(ħG/c³) ≈ 1.6 × 10⁻³⁵ m, is where quantum effects on gravity should become strong. <em>Quantum foam</em> is John Wheeler's name for the fluctuations of spacetime expected there. In his estimate their relative size grows like ℓ<sub>P</sub>/L as the length scale L shrinks; the picture lets them build up more gradually so you can see them coming. String theory places its strings near this scale, although in many models the string length is larger. String theory has no experimental confirmation.</p>
<p>The pictures simplify. Ions are drawn at about a quarter of their real size so the lattice stays visible, electron clouds are hydrogen-like orbitals with screened charges, nucleons are drawn as spheres, quark "colours" are labels for colour charge, and the flux tubes are one snapshot of a quantum system that has no fixed shape. The gold point on the crystal face only marks where the dive is going.</p>`,
  drone: { notes: [0, 7, 12, 19], opts: { gain: 0.085, cutoff: 680, wobble: 0.35 } },
  post: { bloomStrength: 0.95, bloomRadius: 0.62, bloomThreshold: 0.2, exposure: 1.0, aberration: 0.0016, grain: 0.035, vignette: 0.9 },

  create(ctx) {
    const lowQ = ctx.quality === 'low';
    const params = new URLSearchParams(location.search);
    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const audio = ctx.audio, ui = ctx.ui;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(FOV, ctx.aspect, 0.01, 3000);
    camera.position.set(0, 0, CAM_Z);
    scene.add(camera);

    const COL = colorUniforms();
    const uTime = { value: 0 };
    const uRes = { value: new THREE.Vector2(1, 1) };
    const uGlint = { value: 0 };
    const uBeacon = { value: 1 };   // the gold point on the face that marks the way in

    // root: orbit about the focus. world: scaled by U (world units per metre), shifted by focus.
    const root = new THREE.Group(); scene.add(root);
    const world = new THREE.Group(); root.add(world);
    const crystal = new THREE.Group(); crystal.rotation.copy(TILT); world.add(crystal);

    // ---- the pivot chain (metres). The string sits at the origin inside up quark Q0. -------------
    // Chlorine-35 nucleus: 35 nucleons relaxed into a ball of radius 1.2 A^(1/3) fm.
    const NUC = 35, nuc = new Float32Array(NUC * 3), nucType = new Float32Array(NUC);
    let target = 0;
    {
      const rr = rng(1735), rIn = R_NUC - 0.9;
      for (let i = 0; i < NUC; i++) {
        let x, y, w;
        do { x = rr() * 2 - 1; y = rr() * 2 - 1; w = rr() * 2 - 1; } while (x * x + y * y + w * w > 1);
        nuc[i * 3] = x * rIn; nuc[i * 3 + 1] = y * rIn; nuc[i * 3 + 2] = w * rIn;
      }
      const dmin = 1.62;
      for (let it = 0; it < 260; it++) {
        for (let i = 0; i < NUC; i++) for (let j = i + 1; j < NUC; j++) {
          const dx = nuc[j * 3] - nuc[i * 3], dy = nuc[j * 3 + 1] - nuc[i * 3 + 1], dz = nuc[j * 3 + 2] - nuc[i * 3 + 2];
          const d = Math.hypot(dx, dy, dz);
          if (d < dmin && d > 1e-6) {
            const p = (dmin - d) * 0.5 / d;
            nuc[i * 3] -= dx * p; nuc[i * 3 + 1] -= dy * p; nuc[i * 3 + 2] -= dz * p;
            nuc[j * 3] += dx * p; nuc[j * 3 + 1] += dy * p; nuc[j * 3 + 2] += dz * p;
          }
        }
        for (let i = 0; i < NUC; i++) {
          const r = Math.hypot(nuc[i * 3], nuc[i * 3 + 1], nuc[i * 3 + 2]);
          const k = Math.min(0.996, rIn / Math.max(r, 1e-6));
          nuc[i * 3] *= k; nuc[i * 3 + 1] *= k; nuc[i * 3 + 2] *= k;
        }
      }
      // the nucleon nearest the camera is the proton we dive into; 16 more protons, 18 neutrons
      for (let i = 1; i < NUC; i++) if (nuc[i * 3 + 2] > nuc[target * 3 + 2]) target = i;
      const order = [...Array(NUC).keys()].filter((i) => i !== target).sort((a, b) => Math.sin(a * 91.7) - Math.sin(b * 91.7));
      nucType[target] = 1;
      for (let k = 0; k < 16; k++) nucType[order[k]] = 1;
    }
    // Valence quarks of that proton (fm, proton frame). Q0 (up) holds the string.
    const QB = [-0.5, -0.5 + 2.0944, -0.5 + 4.1888].map((a) => new THREE.Vector3(0.38 * Math.cos(a), 0.38 * Math.sin(a), 0.1 * Math.sin(a + 1)));
    const FM = 1e-15;
    const P_c = QB[0].clone().multiplyScalar(-FM);                                     // proton centre
    const N_c = P_c.clone().sub(new THREE.Vector3(nuc[target * 3], nuc[target * 3 + 1], nuc[target * 3 + 2]).multiplyScalar(FM)); // nucleus = ion centre
    crystal.position.copy(N_c);
    const cubeCentreL = new THREE.Vector3(0, 0, SURF_K * A_NACL - GRAIN / 2);           // crystal frame
    const cubeCentreW = cubeCentreL.clone().applyEuler(TILT).add(N_c);                   // world frame

    // ---- backdrop: a faint colour-graded glow behind the subject, and far stars ----------------
    const backMat = new THREE.ShaderMaterial({
      side: THREE.BackSide, depthWrite: false,
      uniforms: { uTint: { value: new THREE.Color() } },
      vertexShader: /* glsl */`varying vec3 vD; void main(){ vD = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: /* glsl */`uniform vec3 uTint; varying vec3 vD;
        void main(){ vec3 d = normalize(vD); float c = max(-d.z, 0.0);
          float g = pow(c, 60.0) * 0.55 + pow(c, 9.0) * 0.16; float h = 0.5 + 0.5 * d.y;
          gl_FragColor = vec4(uTint * (g + 0.05 * h + 0.025), 1.0); }`,
    });
    const backdrop = new THREE.Mesh(new THREE.SphereGeometry(1500, 48, 24), backMat);
    backdrop.renderOrder = -10; scene.add(backdrop);
    const tintCols = TINTS.map(([zz, c]) => [zz, new THREE.Color(c)]);
    const stars = makeStarfield({ count: lowQ ? 1300 : 2600, radius: 80, inner: 48, size: 1.5, intensity: 0.8 });
    scene.add(stars);

    // ---- I. the grain: an analytically ray-traced refracting cube ------------------------------
    const cubeMat = new THREE.ShaderMaterial({
      transparent: true,
      uniforms: { ...COL, uTime, uGlint, uBeacon, uFade: { value: 1 }, uCamObj: { value: new THREE.Vector3() }, uRot: { value: new THREE.Matrix3() }, uEdge: { value: GRAIN }, uView: { value: 1e-3 }, uInner: { value: 1 } },
      vertexShader: /* glsl */`varying vec3 vObj; varying vec3 vN;
        void main(){ vObj = position; vN = normal; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: /* glsl */`
        ${G_COMMON}
        ${G_ENV}
        ${G_FACE}
        uniform vec3 uCamObj; uniform mat3 uRot; uniform float uTime, uFade, uEdge, uGlint, uView, uBeacon, uInner;
        varying vec3 vObj; varying vec3 vN;
        float fres(float c){ return 0.046 + 0.954 * pow(1.0 - c, 5.0); }   // n = 1.544
        // light path inside a perfect box [-0.5, 0.5]^3: up to three internal reflections
        // light path inside a perfect box [-0.5, 0.5]^3: up to three internal reflections. Where the
        // path meets the box near an edge, the bevel there catches the light (the far edges glow through).
        float traceIn(vec3 ro, vec3 rd, float ior, int ch){
          float acc = 0.0, thr = 1.0;
          vec3 ec = mix(uInk * 0.8, uGold, 0.5);
          for (int b = 0; b < 3; b++) {
            // keep every component away from zero (with its own sign) so the slab distances stay finite
            vec3 sg = step(0.0, rd) * 2.0 - 1.0;
            rd = sg * max(abs(rd), vec3(1e-4));
            vec3 tv = (sg * 0.5 - ro) / rd;
            float t = max(min(tv.x, min(tv.y, tv.z)), 0.0);
            vec3 n = t == tv.x ? vec3(-sg.x, 0.0, 0.0) : (t == tv.y ? vec3(0.0, -sg.y, 0.0) : vec3(0.0, 0.0, -sg.z));
            ro = clamp(ro + rd * t, -0.5, 0.5);
            vec3 ah = abs(ro);
            float m2 = ah.x + ah.y + ah.z - max(ah.x, max(ah.y, ah.z)) - min(ah.x, min(ah.y, ah.z));
            acc += thr * smoothstep(0.476, 0.4985, m2) * 0.2 * ec[ch];
            vec3 o = refract(rd, n, ior);
            float f = dot(o, o) > 0.0 ? fres(clamp(-dot(rd, n), 0.0, 1.0)) : 1.0;
            if (f < 1.0) acc += thr * (1.0 - f) * env(normalize(uRot * o))[ch];
            thr *= f * 0.9;
            rd = reflect(rd, n);
          }
          return acc;
        }
        void main(){
          vec3 N = normalize(vN);
          vec3 rd = normalize(vObj - uCamObj);
          float c = clamp(-dot(rd, N), 0.0, 1.0);
          float F = fres(c);
          vec3 NW = normalize(uRot * N), rdW = normalize(uRot * rd);
          vec3 refl = env(reflect(rdW, NW));
          vec3 ro = clamp(vObj, -0.4999, 0.4999);
          vec3 tr = vec3(
            traceIn(ro, refract(rd, N, 1.0 / ${IOR[0]}), ${IOR[0]}, 0),
            traceIn(ro, refract(rd, N, 1.0 / ${IOR[1]}), ${IOR[1]}, 1),
            traceIn(ro, refract(rd, N, 1.0 / ${IOR[2]}), ${IOR[2]}, 2));
          vec3 a = abs(vObj);
          float mid = a.x + a.y + a.z - max(a.x, max(a.y, a.z)) - min(a.x, min(a.y, a.z));
          float bevel = smoothstep(0.44, 0.49, mid);
          vec3 film = 0.55 + 0.45 * cos(6.28318 * (c * 1.7 + uTime * 0.02 + vec3(0.0, 0.33, 0.67)));
          vec2 q; float fid;
          if (a.z >= a.x && a.z >= a.y) { q = vObj.xy; fid = sign(vObj.z) + 1.0; }
          else if (a.x >= a.y) { q = vObj.yz; fid = sign(vObj.x) + 5.0; }
          else { q = vObj.xz; fid = sign(vObj.y) + 9.0; }
          q *= uEdge;
          float fp = length(fwidth(q));
          vec3 face = facePattern(q + (fid - 2.0) * vec2(0.37, 0.21) * uEdge, fp, uTime, uGlint, 0.0) * (1.0 - bevel);
          // micro-facet glitter: a few tiny tilted facets that flash when they mirror the key light
          vec2 gc = q / uEdge * 40.0; vec2 gid = floor(gc); vec3 gh = hash32(gid + fid * 17.0);
          vec3 mn = normalize(NW + (gh - 0.5) * 0.5);
          float sp = pow(max(dot(reflect(rdW, mn), normalize(vec3(-0.55, 0.62, 0.56))), 0.0), 320.0);
          float spot = 1.0 - smoothstep(0.03, 0.1, length(fract(gc) - 0.5 - (gh.xy - 0.5) * 0.5));
          vec3 glitter = (uGold + 0.8) * sp * spot * step(0.72, gh.z) * (2.0 + 6.0 * uGlint) * (1.0 - bevel);
          float lum = dot(refl, vec3(0.3, 0.5, 0.2));
          // terraces are etched, not painted: they show where the face mirrors something bright
          vec3 etch = face * (0.05 + 0.5 * clamp(lum * F * 3.0, 0.0, 1.0) + 0.15 * uGlint);
          vec3 body = (uViolet * 0.028 + uCyan * 0.012) * (0.6 + 0.4 * c);          // a faint milky interior
          // close to the face the eye focuses on the surface: the light seen through the crystal fades out
          vec3 col = refl * F * (1.0 + bevel * (film * 1.5 - 0.2)) * (0.4 + 0.6 * uInner) + (tr * (1.0 - F) * 0.9 + body) * uInner + etch + glitter;
          col += uGold * 0.035 * pow(1.0 - c, 3.0);
          // ringing the grain sends a band of light across it as the glint decays
          float sweep = dot(vObj, normalize(vec3(0.8, 0.55, 0.25))) - ((1.0 - uGlint) * 2.4 - 1.2);
          col += mix(uGold, uInk, 0.4) * exp(-sweep * sweep * 160.0) * uGlint * 0.55;
          // the way in: a faint gold point breathing at the centre of the face we dive into
          float bz = step(1.5, fid) * step(fid, 2.5) * (1.0 - bevel);
          float br = length(q) / uView;
          col += bz * uGold * (exp(-br * br / 0.00012) * 2.0 + exp(-br / 0.05) * 0.07) * uBeacon * (0.75 + 0.25 * sin(uTime * 1.7));
          gl_FragColor = vec4(col, uFade);
        }`,
    });
    const cube = new THREE.Mesh(new RoundedBoxGeometry(1, 1, 1, 5, 0.045), cubeMat);
    cube.position.copy(cubeCentreL); cube.scale.setScalar(GRAIN); cube.renderOrder = 1;
    crystal.add(cube);

    // ---- the crystal face, an infinite plane with the infinite-zoom terrace texture ------------
    const planeMat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, depthTest: false,
      uniforms: { ...COL, uTime, uGlint, uBeacon, uFade: { value: 0 }, uCamL: { value: new THREE.Vector3() }, uInvRot: { value: new THREE.Matrix3() },
        uPlaneZ: { value: SURF_K * A_NACL }, uField: { value: 1 }, uU: { value: 1 }, uAtoms: { value: 1 } },
      vertexShader: /* glsl */`varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
      fragmentShader: /* glsl */`
        ${G_COMMON}
        ${G_ENV}
        ${G_FACE}
        uniform vec3 uCamL; uniform mat3 uInvRot; uniform float uTime, uFade, uPlaneZ, uField, uU, uGlint, uBeacon, uAtoms;
        varying vec3 vW;
        void main(){
          vec3 rdW = normalize(vW - cameraPosition);
          vec3 rd = uInvRot * rdW;
          if (abs(rd.z) < 1e-6) discard;
          float t = (uPlaneZ - uCamL.z) / rd.z;
          if (t <= 0.0) discard;
          vec2 q = uCamL.xy + rd.xy * t;
          float fp = length(fwidth(q));
          vec3 NW = vec3(uInvRot[0][2], uInvRot[1][2], uInvRot[2][2]);
          float c = abs(rd.z);
          float F = 0.046 + 0.954 * pow(max(1.0 - c, 0.0), 5.0);
          vec3 refl = env(reflect(rdW, NW)) * F;
          float lum = dot(refl, vec3(0.3, 0.5, 0.2));
          // steps are etched into dark glass: faint on their own, bright where the face mirrors a light
          vec3 face = facePattern(q, fp, uTime, uGlint, uAtoms) * (0.9 + 2.0 * clamp(lum * 5.0, 0.0, 1.0) + 0.9 * uGlint);
          float fog = exp(-t * uU / 30.0);
          vec3 haze = mix(uViolet, uMag, 0.25) * 0.045;
          vec3 col = (refl + face + uViolet * 0.01) * fog + haze * (1.0 - fog);
          float br = length(q) / uField;
          col += uGold * (exp(-br * br / 0.00012) * 2.0 + exp(-br / 0.05) * 0.07) * uBeacon * (0.75 + 0.25 * sin(uTime * 1.7));
          // ringing the face: a ripple runs out across it from the gold point
          float rr = (br - (1.0 - uGlint) * 0.9) * 34.0;
          col += mix(uGold, uInk, 0.5) * exp(-rr * rr) * uGlint * 0.6 * fog;
          gl_FragColor = vec4(col, uFade);
        }`,
    });
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), planeMat);
    plane.position.set(0, 0, SURF_K * A_NACL); plane.renderOrder = 2; plane.frustumCulled = false;
    crystal.add(plane);

    // ---- out-of-focus neighbouring grains (bokeh) for depth in the opening shot -----------------
    const bokehTex = bokehTexture();
    const bokeh = [
      [-0.95, -0.52, 1.1, 0.5, palette.planck, 0.55], [1.05, 0.55, -1.7, 0.38, palette.harmonic, 0.4],
      [-1.35, 0.8, -2.6, 0.55, palette.violet, 0.4], [0.62, -0.7, 1.75, 0.7, palette.planck, 0.3],
      [1.7, -0.25, -3.4, 0.62, palette.ink, 0.22], [-0.45, 1.0, -1.2, 0.26, palette.planck, 0.34],
      [0.2, 0.62, 0.7, 0.3, palette.resonance, 0.2],
    ].map(([x, y, zz, s, c, k]) => {
      const m = new THREE.SpriteMaterial({ map: bokehTex, color: new THREE.Color(c).multiplyScalar(k * 0.22), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
      const sp = new THREE.Sprite(m);
      sp.position.set(x * 1e-3, y * 1e-3, zz * 1e-3).add(cubeCentreW);
      sp.scale.set(s * 1e-3, s * 1e-3, 1);
      sp.userData.k = k * 0.22; sp.userData.c = new THREE.Color(c);
      world.add(sp);
      return sp;
    });

    // ---- II. the rock-salt lattice: sphere impostors and faint bonds --------------------------
    const LX = lowQ ? 8 : 12, LZ0 = lowQ ? -12 : -17;
    const ionArr = [], bondArr = [];
    for (let k = LZ0; k <= SURF_K; k++) for (let j = -LX; j <= LX; j++) for (let i = -LX; i <= LX; i++) {
      ionArr.push(i, j, k);
      if (i < LX) bondArr.push(i, j, k, i + 1, j, k);
      if (j < LX) bondArr.push(i, j, k, i, j + 1, k);
      if (k < SURF_K) bondArr.push(i, j, k, i, j, k + 1);
    }
    const nIon = ionArr.length / 3;
    const ionIdx = new Float32Array(ionArr), ionType = new Float32Array(nIon);
    for (let n = 0; n < nIon; n++) ionType[n] = Math.abs(ionIdx[n * 3] + ionIdx[n * 3 + 1] + ionIdx[n * 3 + 2]) % 2 === 0 ? 1 : 0; // 1 = Cl-
    const G_LAT = /* glsl */`
      uniform float uTime, uUnit, uFade, uPivotFade; uniform vec4 uWave;
      // thermal motion (~0.04 spacings rms at room temperature) plus a tapped phonon wavefront
      vec3 latticeDisp(vec3 idx, out float wave){
        vec3 j = 0.035 * vec3(sin(uTime * 7.1 + idx.x * 12.9 + idx.y * 3.1), sin(uTime * 6.3 + idx.y * 7.7 + idx.z * 5.3), sin(uTime * 5.7 + idx.z * 9.1 + idx.x * 2.3));
        float age = uTime - uWave.w;
        vec3 r = idx - uWave.xyz; float R = length(r);
        float front = age * 5.0;
        float wr = (R - front) / 1.1;
        wave = exp(-wr * wr) * exp(-age * 0.5) * step(0.0, age) * step(age, 9.0);
        return j + r / max(R, 1e-3) * 0.26 * wave * sin((R - front) * 2.2);
      }`;
    const quad = new THREE.PlaneGeometry(2, 2);
    const ionGeo = new THREE.InstancedBufferGeometry();
    ionGeo.index = quad.index;
    ionGeo.setAttribute('position', quad.attributes.position);
    ionGeo.setAttribute('aIdx', new THREE.InstancedBufferAttribute(ionIdx, 3));
    ionGeo.setAttribute('aType', new THREE.InstancedBufferAttribute(ionType, 1));
    ionGeo.instanceCount = nIon;
    const latU = { uTime, uUnit: { value: 1 }, uFade: { value: 0 }, uPivotFade: { value: 1 }, uWave: { value: new THREE.Vector4(0, 0, 0, -100) } };
    const ionMat = new THREE.ShaderMaterial({
      uniforms: { ...COL, ...latU },
      vertexShader: /* glsl */`
        ${G_COMMON}
        ${G_LAT}
        attribute vec3 aIdx; attribute float aType;
        varying vec2 vUv; varying float vType, vFog, vA, vWave;
        void main(){
          float wave;
          vec3 p = aIdx + latticeDisp(aIdx, wave);
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          float r = aType > 0.5 ? 0.17 : 0.1;   // ball-and-stick: about a quarter of the ionic radii (Cl- 1.81, Na+ 1.02 angstrom)
          mv.xy += position.xy * r * uUnit;
          vUv = position.xy; vType = aType; vWave = wave;
          float dist = -mv.z;
          vFog = exp(-max(dist / uUnit - 1.2, 0.0) / 2.9);
          float isPivot = step(dot(aIdx, aIdx), 0.01);
          vA = uFade * smoothstep(0.6 * uUnit, 1.6 * uUnit, dist) * smoothstep(0.3, 1.2, dist) * mix(1.0, uPivotFade, isPivot);
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: /* glsl */`
        ${G_COMMON}
        varying vec2 vUv; varying float vType, vFog, vA, vWave;
        void main(){
          float r2 = dot(vUv, vUv); if (r2 > 1.0) discard;
          ${lowQ ? 'if (hash12(gl_FragCoord.xy) > vA) discard;' : 'if (vA < 0.004) discard;'}
          vec3 n = vec3(vUv, sqrt(1.0 - r2));
          vec3 base = vType > 0.5 ? uCyan : uGold;
          float dif = 0.3 + 0.7 * max(dot(n, normalize(vec3(-0.45, 0.6, 0.66))), 0.0);
          float spec = pow(max(dot(n, normalize(vec3(-0.22, 0.42, 0.88))), 0.0), 48.0);
          float rim = pow(max(1.0 - n.z, 0.0), 2.2);
          vec3 col = base * (0.06 + 0.45 * dif) + spec * 0.6 + base * rim * 0.6 + mix(uGold, uInk, 0.35) * vWave * 0.45;
          gl_FragColor = vec4(col * vFog, ${lowQ ? '1.0' : 'clamp(vA * 1.1, 0.0, 1.0)'});
        }`,
    });
    ionMat.alphaToCoverage = !lowQ;   // smooth fades on the 4x MSAA target; phones dither instead
    const ions = new THREE.Mesh(ionGeo, ionMat); ions.frustumCulled = false; ions.renderOrder = 3;
    const bondGeo = new THREE.BufferGeometry();
    bondGeo.setAttribute('position', new THREE.Float32BufferAttribute(bondArr, 3));
    const bondMat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { ...COL, ...latU },
      vertexShader: /* glsl */`
        ${G_COMMON}
        ${G_LAT}
        varying float vA;
        void main(){
          float w; vec3 p = position + latticeDisp(position, w);
          vec4 mv = modelViewMatrix * vec4(p, 1.0); float dist = -mv.z;
          vA = uFade * exp(-max(dist / uUnit - 1.2, 0.0) / 3.6) * smoothstep(0.4 * uUnit, 1.1 * uUnit, dist) * (0.22 + w * 0.8);
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: /* glsl */`${G_COMMON} varying float vA; void main(){ gl_FragColor = vec4(mix(uInk, uCyan, 0.45) * vA * 0.5, 1.0); }`,
    });
    const bonds = new THREE.LineSegments(bondGeo, bondMat); bonds.frustumCulled = false;
    const lattice = new THREE.Group(); lattice.scale.setScalar(A_NACL); lattice.add(ions, bonds);
    crystal.add(lattice);

    // ---- III. a chloride ion: electrons sampled from screened hydrogen-like orbitals -----------
    // Cl- is 1s2 2s2 2p6 3s2 3p6. Slater screening gives Z_eff 16.7 (1s), 12.85 (n = 2), 5.75 (n = 3).
    // [n, l, axis (-1: s orbital), Z_eff, share of the points, colour]. Shares favour the outer shells, which
    // are far larger; each colour is one orbital (the three 3p lobes: cyan x, violet y, magenta z).
    const ORB = [
      [1, 0, -1, 16.7, 0.03, '#ffd9a0'], [2, 0, -1, 12.85, 0.035, '#c9b8ff'],
      [2, 1, 0, 12.85, 0.035, '#9d7bff'], [2, 1, 1, 12.85, 0.035, '#7f8cff'], [2, 1, 2, 12.85, 0.035, '#c46cff'],
      [3, 0, -1, 5.75, 0.09, '#7fb0e8'],
      [3, 1, 0, 5.75, 0.2, '#6ff3ff'], [3, 1, 1, 5.75, 0.2, '#8b6cff'], [3, 1, 2, 5.75, 0.2, '#ff4fbf'],
    ];
    const ORB_R = ORB.map(([n, l, , Z]) => (A0 / Z) * (3 * n * n - l * (l + 1)) / 2);   // <r> in angstrom
    const NCLOUD = lowQ ? 26000 : 56000;
    const cloudPos = new Float32Array(NCLOUD * 3), cloudSeed = new Float32Array(NCLOUD), cloudOrb = new Float32Array(NCLOUD);
    {
      const rr = rng(17); let p = 0;
      const total = ORB.reduce((s, o) => s + o[4], 0);
      ORB.forEach(([n, l, axis, Z, share], oi) => {
        const inv = radialTable(n, l, Z);
        const cnt = oi === ORB.length - 1 ? NCLOUD - p : Math.round(share / total * NCLOUD);
        for (let c = 0; c < cnt && p < NCLOUD; c++, p++) {
          let x, y, w, d2;
          for (;;) {   // direction: uniform for s, |Y|^2 ~ cos^2 about the axis for p
            x = rr() * 2 - 1; y = rr() * 2 - 1; w = rr() * 2 - 1; d2 = x * x + y * y + w * w;
            if (d2 > 1 || d2 < 1e-6) continue;
            const s = 1 / Math.sqrt(d2); x *= s; y *= s; w *= s;
            if (axis < 0) break;
            const ca = axis === 0 ? x : axis === 1 ? y : w;
            if (rr() < ca * ca) break;
          }
          const r = inv(rr());
          cloudPos[p * 3] = x * r; cloudPos[p * 3 + 1] = y * r; cloudPos[p * 3 + 2] = w * r;
          cloudSeed[p] = rr(); cloudOrb[p] = oi;
        }
      });
    }
    const cloudGeo = new THREE.BufferGeometry();
    cloudGeo.setAttribute('position', new THREE.BufferAttribute(cloudPos, 3));
    cloudGeo.setAttribute('aSeed', new THREE.BufferAttribute(cloudSeed, 1));
    cloudGeo.setAttribute('aOrb', new THREE.BufferAttribute(cloudOrb, 1));
    const orbCols = ORB.map((o) => new THREE.Color(o[5]));
    const orbU = orbCols.map((c, i) => new THREE.Vector3(c.r, c.g, c.b).multiplyScalar(ORB[i][0] <= 2 ? 0.5 : 0.9));   // dense inner shells dimmer
    const cloudMat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { uTime, uPx: { value: 1.6 }, uFade: { value: 0 }, uDim: { value: 1 }, uOrb: { value: orbU } },
      vertexShader: /* glsl */`
        uniform float uTime, uPx, uFade, uDim; uniform vec3 uOrb[9];
        attribute float aSeed; attribute float aOrb;
        varying vec3 vC; varying float vA;
        void main(){
          vec3 p = position * (1.0 + 0.012 * sin(uTime * 1.3 + aSeed * 61.0));
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          float ph = fract(uTime * (0.2 + 0.32 * fract(aSeed * 7.31)) + aSeed);
          float life = smoothstep(0.0, 0.05, ph) * (1.0 - smoothstep(0.2, 0.92, ph));
          float dist = -mv.z;
          vA = life * uFade * uDim * smoothstep(0.25, 1.4, dist);
          vC = uOrb[int(aOrb + 0.5)];
          gl_PointSize = uPx * (1.5 + 2.4 * pow(fract(aSeed * 3.7), 3.0)) * clamp(9.0 / dist, 0.5, 3.0);
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: /* glsl */`
        varying vec3 vC; varying float vA;
        void main(){ vec2 p = gl_PointCoord - 0.5; float d = dot(p, p) * 4.0; if (d > 1.0) discard;
          gl_FragColor = vec4(vC * max(exp(-d * 2.4) - 0.09, 0.0) * vA, 1.0); }`,
    });
    const cloudPts = new THREE.Points(cloudGeo, cloudMat); cloudPts.frustumCulled = false;
    const cloud = new THREE.Group(); cloud.add(cloudPts);
    cloud.position.copy(N_c); cloud.scale.setScalar(1e-10); cloudPts.rotation.set(0.62, 0.48, 0.18);
    world.add(cloud);

    // Constant-pixel-size glowing points: the nucleus seen from the atom, point quarks, flashes.
    function glowPoint(color, size, intensity = 1) {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3));
      const m = new THREE.ShaderMaterial({
        transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
        uniforms: { uC: { value: new THREE.Color(color) }, uSize: { value: size }, uI: { value: intensity }, uCore: { value: 1.25 } },
        vertexShader: /* glsl */`uniform float uSize; void main(){ gl_PointSize = uSize; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
        fragmentShader: /* glsl */`uniform vec3 uC; uniform float uI, uCore;
          void main(){ vec2 p = gl_PointCoord - 0.5; float d = length(p) * 2.0;
            float core = exp(-d * d * 120.0), halo = exp(-d * d * 14.0) * 0.32 + exp(-d * 3.5) * 0.12 * (1.0 - d);
            vec3 c = (vec3(1.0, 0.96, 0.9) * core * uCore + uC * (core * 0.7 + halo)) * uI;
            gl_FragColor = vec4(c * step(d, 1.0), 1.0); }`,
      });
      const pt = new THREE.Points(g, m); pt.frustumCulled = false; pt.renderOrder = 6;
      return pt;
    }
    const nucPoint = glowPoint(palette.planck, 40, 0); nucPoint.position.copy(N_c); world.add(nucPoint);
    const flash = glowPoint(palette.ink, 70, 0); cloudPts.add(flash);
    let flashT = 99, cloudDimT = 99;   // seconds since the last position measurement

    // ---- IV. the chlorine-35 nucleus -----------------------------------------------------------
    const nucGeo = new THREE.IcosahedronGeometry(1, lowQ ? 3 : 4);
    const nucSeed = new Float32Array(NUC);
    for (let i = 0; i < NUC; i++) nucSeed[i] = i === target ? -1 : 0.1 + i * 0.37;
    nucGeo.setAttribute('aType', new THREE.InstancedBufferAttribute(nucType, 1));
    nucGeo.setAttribute('aSeed', new THREE.InstancedBufferAttribute(nucSeed, 1));
    const nucMat = new THREE.ShaderMaterial({
      uniforms: { ...COL, uTime, uKick: { value: 0 }, uFade: { value: 0 }, uTargetFade: { value: 1 }, uOthersFade: { value: 1 } },
      vertexShader: /* glsl */`
        uniform float uTime, uKick;
        attribute float aType; attribute float aSeed;
        varying vec3 vN, vV; varying float vType, vTarget, vDist;
        void main(){
          float tgt = step(aSeed, -0.5);
          // nucleons move at ~0.2 c inside the nucleus: a restless jiggle (the target is our frame)
          vec3 jig = (1.0 - tgt) * (0.07 + 0.32 * uKick) * vec3(sin(uTime * 9.1 + aSeed * 11.0), sin(uTime * 7.7 + aSeed * 23.0), sin(uTime * 8.3 + aSeed * 5.0));
          vec4 lp = instanceMatrix * vec4(position, 1.0); lp.xyz += jig;
          vec4 mv = modelViewMatrix * lp;
          vN = normalize(normalMatrix * mat3(instanceMatrix) * normal);
          vV = -mv.xyz; vType = aType; vTarget = tgt; vDist = -mv.z;
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: /* glsl */`
        ${G_COMMON}
        uniform float uFade, uTargetFade, uOthersFade, uKick;
        varying vec3 vN, vV; varying float vType, vTarget, vDist;
        void main(){
          float a = uFade * smoothstep(0.4, 1.6, vDist) * mix(uOthersFade, uTargetFade, vTarget);
          ${lowQ ? 'if (hash12(gl_FragCoord.xy) > a) discard;' : 'if (a < 0.004) discard;'}
          vec3 N = normalize(vN), V = normalize(vV);
          float ndv = clamp(dot(N, V), 0.0, 1.0);
          vec3 base = vType > 0.5 ? mix(uGold, uEmber, 0.42) : mix(uViolet, uInk, 0.2) * 0.85;
          float rim = pow(1.0 - ndv, 3.0);
          float spec = pow(max(dot(reflect(-V, N), normalize(vec3(-0.35, 0.6, 0.72))), 0.0), 60.0);
          float sss = pow(ndv, 1.5);
          vec3 col = base * (0.05 + 0.32 * sss) + base * rim * 0.8 + spec * 0.7;
          col += vTarget * uGold * 0.1 * sss;
          col += base * uKick * 0.1 * (0.4 + sss);   // a kick makes the nucleus flare for a moment
          gl_FragColor = vec4(col, ${lowQ ? '1.0' : 'clamp(a * 1.1, 0.0, 1.0)'});
        }`,
    });
    nucMat.alphaToCoverage = !lowQ;
    const nucMesh = new THREE.InstancedMesh(nucGeo, nucMat, NUC);
    {
      const m = new THREE.Matrix4(), v = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3(0.92, 0.92, 0.92);
      for (let i = 0; i < NUC; i++) { v.set(nuc[i * 3], nuc[i * 3 + 1], nuc[i * 3 + 2]); m.compose(v, q, s); nucMesh.setMatrixAt(i, m); }
    }
    nucMesh.frustumCulled = false;
    const nucleus = new THREE.Group(); nucleus.add(nucMesh);
    nucleus.position.copy(N_c); nucleus.scale.setScalar(FM);
    world.add(nucleus);

    // ---- V. a proton: valence quarks, Y-shaped gluon flux tubes, a sea of virtual pairs --------
    const proton = new THREE.Group(); proton.position.copy(P_c); proton.scale.setScalar(FM); world.add(proton);
    const memMat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
      uniforms: { ...COL, uTime, uFade: { value: 0 } },
      vertexShader: /* glsl */`varying vec3 vN, vV, vP;
        void main(){ vP = position; vec4 mv = modelViewMatrix * vec4(position, 1.0); vN = normalize(normalMatrix * normal); vV = -mv.xyz; gl_Position = projectionMatrix * mv; }`,
      fragmentShader: /* glsl */`
        ${G_COMMON}
        ${GLSL.noise}
        uniform float uTime, uFade; varying vec3 vN, vV, vP;
        void main(){
          float ndv = abs(dot(normalize(vN), normalize(vV)));
          float fr = pow(max(1.0 - ndv, 0.0), 2.4);
          float n = snoise(vP * 2.8 + vec3(0.0, uTime * 0.35, uTime * 0.2));
          vec3 c = mix(uEmber, uGold, 0.5 + 0.5 * n) * fr * (0.5 + 0.35 * n) + uMag * 0.035 * (0.5 + 0.5 * n);
          gl_FragColor = vec4(c * uFade * smoothstep(0.3, 1.5, length(vV)) * 0.75, 1.0);
        }`,
    });
    const membrane = new THREE.Mesh(new THREE.SphereGeometry(R_NUCLEON, 64, 48), memMat);
    membrane.renderOrder = 4; proton.add(membrane);

    // colour charge: red, green, blue (tinted into the palette) and their anticolours
    const QCOL = ['#ff5470', '#62ffb0', '#5a8cff'].map((c) => new THREE.Color(c));
    const ACOL = ['#54f0ff', '#ff62d8', '#ffd25a'].map((c) => new THREE.Color(c));
    const qc = [0, 1, 2];                                   // colour index carried by each quark
    const qDisp = QCOL.map((c) => c.clone());               // displayed colour (eases toward qc)
    const qNow = QB.map((v) => v.clone());                  // animated positions (fm)
    const junction = new THREE.Vector3();
    const gluon = { from: 0, to: 1, ca: 0, cb: 1, t: -1, next: 1.0 };   // the gluon in flight (t: 0..2 along two tubes)

    const qGeo = new THREE.BufferGeometry();
    const qPos = new Float32Array(9), qCol = new Float32Array(9);
    qGeo.setAttribute('position', new THREE.BufferAttribute(qPos, 3));
    qGeo.setAttribute('aCol', new THREE.BufferAttribute(qCol, 3));
    const qMat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
      uniforms: { uSize: { value: 40 }, uI: { value: 0 } },
      vertexShader: /* glsl */`uniform float uSize; attribute vec3 aCol; varying vec3 vC;
        void main(){ vC = aCol; gl_PointSize = uSize; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: /* glsl */`uniform float uI; varying vec3 vC;
        void main(){ vec2 p = gl_PointCoord - 0.5; float d = length(p) * 2.0;
          float core = exp(-d * d * 120.0), halo = exp(-d * d * 14.0) * 0.38 + exp(-d * 3.5) * 0.13 * (1.0 - d);
          gl_FragColor = vec4((vec3(1.0, 0.96, 0.9) * core * 1.4 + vC * (core * 0.7 + halo)) * uI * step(d, 1.0), 1.0); }`,
    });
    const quarks = new THREE.Points(qGeo, qMat); quarks.frustumCulled = false; quarks.renderOrder = 7;
    proton.add(quarks);

    const tubeU = {
      uTime, uRes, uFade: { value: 0 }, uPx: { value: 1 },
      uQ0: { value: qNow[0] }, uQ1: { value: qNow[1] }, uQ2: { value: qNow[2] }, uJ: { value: junction },
      uC0: { value: qDisp[0] }, uC1: { value: qDisp[1] }, uC2: { value: qDisp[2] },
      uPulse: { value: new THREE.Vector4(0, 1, -1, 0) },
    };
    const tubeMat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
      uniforms: tubeU,
      vertexShader: /* glsl */`
        ${G_RIBBON}
        uniform vec3 uQ0, uQ1, uQ2, uJ; uniform float uTime, uPx;
        vec3 curve(float s){
          float tube = floor(aId / 4.0 + 0.01), strand = aId - tube * 4.0;
          vec3 A = tube < 0.5 ? uQ0 : (tube < 1.5 ? uQ1 : uQ2);
          vec3 ax = uJ - A; float L = max(length(ax), 1e-4); vec3 T = ax / L;
          vec3 up = abs(T.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
          vec3 B1 = normalize(cross(T, up)), B2 = cross(T, B1);
          float taper = 0.3 + 0.7 * smoothstep(0.0, 0.4, s);
          float rad = strand > 2.5 ? 0.0 : 0.05 * taper;
          float ang = strand * 2.0944 + s * L * 17.0 - uTime * (2.0 + tube * 0.6);
          vec3 sway = B1 * sin(s * 3.14159) * 0.035 * sin(uTime * 1.3 + tube * 2.0) + B2 * sin(s * 6.283) * 0.02 * sin(uTime * 1.7 + tube);
          return A + ax * s + (B1 * cos(ang) + B2 * sin(ang)) * rad + sway;
        }
        void main(){
          float strand = aId - floor(aId / 4.0 + 0.01) * 4.0;
          ${ribbonMain('uPx * (strand > 2.5 ? 9.0 : 1.3)')}
        }`,
      fragmentShader: /* glsl */`
        uniform vec3 uC0, uC1, uC2; uniform float uFade; uniform vec4 uPulse;
        varying float vV, vS, vId;
        void main(){
          float tube = floor(vId / 4.0 + 0.01), strand = vId - tube * 4.0;
          vec3 qc = tube < 0.5 ? uC0 : (tube < 1.5 ? uC1 : uC2);
          vec3 c = mix(qc, vec3(1.0, 0.95, 0.88) * 0.8, smoothstep(0.25, 1.0, vS));   // colour-neutral at the junction
          float x = abs(vV);
          float prof = strand > 2.5 ? exp(-x * x * 3.0) * 0.16 : exp(-x * x * 4.0) * 0.85;
          float pos = abs(tube - uPulse.x) < 0.5 ? uPulse.z : (abs(tube - uPulse.y) < 0.5 ? 2.0 - uPulse.z : -9.0);
          float bx = (vS - pos) * 8.0;
          float bump = exp(-bx * bx) * uPulse.w;
          gl_FragColor = vec4(c * prof * (1.0 + 2.6 * bump) * uFade, 1.0);
        }`,
    });
    const tubes = new THREE.Mesh(ribbonGeometry(96, 12), tubeMat); tubes.frustumCulled = false; tubes.renderOrder = 5;
    proton.add(tubes);

    const NPAIR = lowQ ? 22 : 46;
    const seaGeo = new THREE.BufferGeometry();
    const seaPair = new Float32Array(NPAIR * 2), seaSign = new Float32Array(NPAIR * 2);
    for (let i = 0; i < NPAIR; i++) { seaPair[i * 2] = seaPair[i * 2 + 1] = i; seaSign[i * 2] = 1; seaSign[i * 2 + 1] = -1; }
    seaGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NPAIR * 6), 3));
    seaGeo.setAttribute('aPair', new THREE.BufferAttribute(seaPair, 1));
    seaGeo.setAttribute('aSign', new THREE.BufferAttribute(seaSign, 1));
    const seaMat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
      uniforms: { ...COL, uTime, uFade: { value: 0 }, uBoost: { value: 0 }, uPx: { value: 1 }, uQC: { value: QCOL }, uAC: { value: ACOL } },
      vertexShader: /* glsl */`
        ${G_COMMON}
        uniform float uTime, uFade, uBoost, uPx; uniform vec3 uQC[3], uAC[3];
        attribute float aPair, aSign; varying vec3 vC; varying float vA;
        void main(){
          // a quark and its antiquark appear together, drift apart, return and annihilate
          float L = 0.45 + 0.9 * hash11(aPair * 3.7 + 1.0);
          float tt = uTime * (1.0 + uBoost * 1.5) + hash11(aPair * 1.3) * 10.0;
          float cyc = floor(tt / L), ph = fract(tt / L);
          vec3 h = hash31(aPair * 17.1 + cyc * 3.3);
          vec3 o = h * 2.0 - 1.0; o *= 0.62 / max(1.0, length(o));
          vec3 dir = normalize(hash31(aPair * 5.7 + cyc * 1.9) * 2.0 - 1.0 + 1e-3);
          vec3 p = o + dir * sin(3.14159 * ph) * (0.05 + 0.1 * h.x) * aSign;
          float burst = exp(-ph * 30.0) + exp(-(1.0 - ph) * 30.0);
          int ci = int(floor(h.y * 2.999));
          vC = aSign > 0.0 ? uQC[ci] : uAC[ci];
          vA = (pow(max(sin(3.14159 * ph), 0.0), 0.7) * 0.7 + 1.0 * burst) * uFade * (0.6 + 0.8 * uBoost);
          gl_PointSize = uPx * (4.5 + 6.0 * burst);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
        }`,
      fragmentShader: /* glsl */`varying vec3 vC; varying float vA;
        void main(){ vec2 p = gl_PointCoord - 0.5; float d = dot(p, p) * 4.0; if (d > 1.0) discard;
          gl_FragColor = vec4((vC * exp(-d * 4.0) + exp(-d * 30.0) * 0.6) * vA, 1.0); }`,
    });
    const sea = new THREE.Points(seaGeo, seaMat); sea.frustumCulled = false; sea.renderOrder = 6;
    proton.add(sea);

    // ---- VI. the quark as a point, the edge of experiment, the probe ping ----------------------
    const qPoint = glowPoint(QCOL[0], 40, 0); root.add(qPoint);
    const ringMat = (dashed) => new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
      uniforms: { uRes, uR: { value: 1 }, uW: { value: 1.2 }, uI: { value: 0 }, uC: { value: new THREE.Color(palette.planck) } },
      vertexShader: /* glsl */`
        ${G_RIBBON}
        uniform float uR, uW;
        vec3 curve(float s){ float a = 6.2831853 * s; return vec3(cos(a), sin(a), 0.0) * uR; }
        void main(){ ${ribbonMain('uW')} }`,
      fragmentShader: /* glsl */`uniform float uI; uniform vec3 uC; varying float vV, vS;
        void main(){ float x = abs(vV); float dash = ${dashed ? 'smoothstep(0.35, 0.45, abs(fract(vS * 96.0) - 0.5) * 2.0)' : '1.0'};
          gl_FragColor = vec4(uC * exp(-x * x * 3.0) * uI * dash, 1.0); }`,
    });
    const limitRing = new THREE.Mesh(ribbonGeometry(512, 1), ringMat(true)); limitRing.frustumCulled = false;
    const limitG = new THREE.Group(); limitG.scale.setScalar(LIMIT_D / 2); limitG.add(limitRing); world.add(limitG);
    const ping = new THREE.Mesh(ribbonGeometry(256, 1), ringMat(false)); ping.frustumCulled = false; root.add(ping);
    // Decade rings: circles 10^-n m across around the quark. In the desert they are the only thing that
    // changes, so each power of ten passing by is a ring that grows out of the centre and leaves the frame.
    const decRings = [0, 1, 2].map(() => {
      const r = new THREE.Mesh(ribbonGeometry(384, 1), ringMat(true));
      r.frustumCulled = false; r.material.uniforms.uC.value.copy(palette.ink).lerp(palette.violet, 0.35);
      root.add(r); return r;
    });
    let pingT = 99;

    // ---- VII. dust motes: scale-free streaks whose speed follows the dive ------------------------
    const NMOTE = lowQ ? 260 : 560;
    const moteBase = new THREE.BufferGeometry();
    moteBase.setAttribute('position', new THREE.BufferAttribute(new Float32Array(12), 3));
    moteBase.setAttribute('aEnd', new THREE.BufferAttribute(new Float32Array([0, 0, 1, 1]), 1));
    moteBase.setAttribute('aSide', new THREE.BufferAttribute(new Float32Array([-1, 1, -1, 1]), 1));
    moteBase.setIndex([0, 2, 1, 1, 2, 3]);   // counter-clockwise on screen, like the ribbons
    const moteGeo = new THREE.InstancedBufferGeometry();
    moteGeo.index = moteBase.index;
    for (const k of ['position', 'aEnd', 'aSide']) moteGeo.setAttribute(k, moteBase.attributes[k]);
    {
      const s = new Float32Array(NMOTE * 4), rr = rng(99);
      for (let i = 0; i < NMOTE; i++) {
        const a = rr() * Math.PI * 2, r = 0.9 + 7.5 * Math.sqrt(rr());
        s[i * 4] = Math.cos(a) * r; s[i * 4 + 1] = Math.sin(a) * r * 0.75; s[i * 4 + 2] = rr(); s[i * 4 + 3] = rr();
      }
      moteGeo.setAttribute('aSeed', new THREE.InstancedBufferAttribute(s, 4));
    }
    moteGeo.instanceCount = NMOTE;
    const moteMat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { ...COL, uRes, uTravel: { value: 0 }, uSpeed: { value: 0 }, uFade: { value: 0.3 }, uPx: { value: 1 }, uTint: { value: new THREE.Color() } },
      vertexShader: /* glsl */`
        uniform vec2 uRes; uniform float uTravel, uSpeed, uFade, uPx;
        attribute float aEnd, aSide; attribute vec4 aSeed;
        varying float vA, vV, vW;
        void main(){
          float zr = fract(aSeed.z + uTravel * (0.6 + 0.8 * aSeed.w));
          float zz = mix(-60.0, 7.0, zr);
          vec3 head = vec3(aSeed.xy, zz);
          float len = clamp(uSpeed, -4.0, 4.0) * (1.0 + 1.6 * aSeed.w);
          vec3 tail = head - vec3(0.0, 0.0, len); tail.z = min(tail.z, 7.4);
          vec4 ch = projectionMatrix * modelViewMatrix * vec4(head, 1.0);
          vec4 ct = projectionMatrix * modelViewMatrix * vec4(tail, 1.0);
          vec2 d = (ch.xy / ch.w - ct.xy / ct.w) * uRes; float dl = length(d);
          vec2 dir = dl > 1e-4 ? d / dl : vec2(1.0, 0.0);
          vec4 c = aEnd > 0.5 ? ch : ct;
          float w = uPx * (0.9 + 1.1 * aSeed.w);
          c.xy += (vec2(-dir.y, dir.x) * aSide + dir * (aEnd * 2.0 - 1.0)) * w / uRes * 2.0 * c.w;
          float fog = smoothstep(-60.0, -30.0, zz) * (1.0 - smoothstep(2.5, 6.5, zz));
          vA = uFade * fog * (w * 16.0) / (w * 16.0 + dl) * (0.45 + aSeed.w);   // long streaks spread their light
          vV = aSide; vW = aSeed.w;
          gl_Position = c;
        }`,
      fragmentShader: /* glsl */`uniform vec3 uTint; varying float vA, vV, vW;
        void main(){ gl_FragColor = vec4(uTint * exp(-vV * vV * 3.0) * vA, 1.0); }`,
    });
    const motes = new THREE.Mesh(moteGeo, moteMat); motes.frustumCulled = false; motes.renderOrder = 8;
    root.add(motes);
    let travel = 0;

    // ---- VIII. quantum foam: infinite-zoom ridged noise on parallax slices ---------------------
    const foamU = {
      ...COL, uTime, uZ: { value: 30 }, uI: { value: 0 }, uCalm: { value: 0 }, uRs: { value: 1 }, uHref: { value: 1 },
      uCamL: { value: new THREE.Vector3() }, uRootInv: { value: new THREE.Matrix3() },
    };
    const foamMat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
      uniforms: foamU,
      vertexShader: /* glsl */`varying vec3 vW; void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
      fragmentShader: /* glsl */`
        ${G_COMMON}
        ${GLSL.noise}
        uniform float uZ, uI, uCalm, uRs, uTime, uHref; uniform vec3 uCamL; uniform mat3 uRootInv;
        varying vec3 vW;
        vec3 foamSlice(vec2 uv, float sd, float fp){
          float fz = fract(uZ), iz = floor(uZ);
          vec3 acc = vec3(0.0);
          for (int j = 1; j <= 3; j++) {
            float m = iz + float(j);                        // absolute decade of this octave
            float F = uHref * pow(10.0, fz - float(j));     // its feature size in world units
            float rel = F / uHref;
            float w = smoothstep(0.006, 0.035, rel) * (1.0 - smoothstep(0.25, 0.9, rel));
            // relative fluctuation grows as the scale shrinks toward the Planck length (drawn as (l_P / L)^1/2,
            // more gradual than Wheeler's l_P / L so that it can be seen coming), saturating at order one
            float amp = clamp(pow(10.0, (m - 34.8) * 0.5), 0.0, 0.8);
            if (w * amp < 0.002) continue;
            vec2 p = uv / F + hash32(vec2(m, sd)).xy * 97.0;
            float n = snoise(vec3(p, uTime * 0.45 + m * 1.7 + sd));
            float fil = pow(max(1.0 - abs(n), 0.0), 7.0);   // (pow of a negative number is undefined in GLSL)
            float hue = 0.5 + 0.5 * sin(p.x * 0.21 + p.y * 0.17 + m);
            vec3 col = mix(uViolet, uMag, hue * hue) * fil * 0.42 + uViolet * 0.03 * (0.5 + 0.5 * n);
            vec2 sc = p * 0.7; vec2 sid = floor(sc);
            float hs = hash12(sid + m * 7.0);
            float on = step(0.972, hs) * step(0.55, hash12(sid + floor(uTime * 2.5 + hs * 9.0)));
            float spx = length((fract(sc) - 0.5) * F / 0.7) / fp;
            col += mix(uGold, vec3(1.0), 0.45) * on * exp(-spx * spx * 0.3) * 1.1;
            acc += col * w * amp;
          }
          return acc;
        }
        void main(){
          vec3 rd = normalize(uRootInv * (vW - cameraPosition));
          vec3 col = vec3(0.0);
          for (int k = 0; k < ${lowQ ? 1 : 2}; k++) {
            float zk = 1.2 - float(k) * 3.4;
            float t = (zk - uCamL.z) / min(rd.z, -1e-4);
            vec2 p = (uCamL + rd * t).xy;
            float fp = length(fwidth(p));
            float r = length(p);
            float calm = (1.0 - uCalm * (1.0 - smoothstep(uRs * 0.8, uRs * 3.6, r))) * (1.0 - 0.45 * uCalm);   // the foam settles around the string
            col += foamSlice(p, float(k) * 3.1, fp) * calm * (1.0 - float(k) * 0.2) * ${lowQ ? '1.6' : '1.15'};
          }
          gl_FragColor = vec4(col * uI, 1.0);
        }`,
    });
    const foam = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), foamMat);
    foam.position.z = -1; foam.frustumCulled = false; foam.renderOrder = 9;
    camera.add(foam);

    // ---- IX. the string: a closed loop with left- and right-moving modes ------------------------
    const STR_TRAILS = 7;
    const sgeo = ribbonGeometry(lowQ ? 480 : 768, 1);
    const strGeo = new THREE.InstancedBufferGeometry();
    strGeo.index = sgeo.index;
    for (const k of ['position', 'aS', 'aSide', 'aId']) strGeo.setAttribute(k, sgeo.attributes[k]);
    strGeo.setAttribute('aLag', new THREE.InstancedBufferAttribute(new Float32Array([0, 1, 2, 3, 4, 5, 6]), 1));
    strGeo.instanceCount = STR_TRAILS;
    const AL = new THREE.Vector4(), AR = new THREE.Vector4();
    const AL0 = [0.05, 0.06, 0.04, 0.022], AR0 = [0.045, 0.055, 0.042, 0.02];   // idle amplitudes of modes n = 2..5 (level-matched)
    const plucked = new Float32Array(4);
    const strMat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
      uniforms: { ...COL, uTime, uRes, uAL: { value: AL }, uAR: { value: AR }, uW: { value: 4 }, uFade: { value: 0 }, uCore: { value: 2.2 }, uOm: { value: 3.0 } },
      vertexShader: /* glsl */`
        ${G_RIBBON}
        uniform float uTime, uW, uOm; uniform vec4 uAL, uAR;
        attribute float aLag; varying float vLag;
        // X(sigma, t): radial and out-of-plane displacement from left movers (n sigma - n w t)
        // and right movers (n sigma + n w t). Mode n vibrates at n times the fundamental.
        vec3 curve(float s){
          float sg = 6.2831853 * s, t = uTime - aLag * 0.05;
          float rr = 1.0, yy = 0.0;
          for (int i = 0; i < 4; i++) {
            float n = float(i + 2), w = n * uOm * t;
            rr += uAL[i] * cos(n * sg - w) + uAR[i] * cos(n * sg + w + 1.1 * n);
            yy += uAL[i] * sin(n * sg - w + 0.7) - uAR[i] * sin(n * sg + w + 0.4 * n);
          }
          return vec3(cos(sg) * rr, yy, sin(sg) * rr);
        }
        void main(){ vLag = aLag; ${ribbonMain('uW * (aLag > 0.5 ? 1.7 : 1.0)')} }`,
      fragmentShader: /* glsl */`
        uniform vec3 uGold; uniform float uFade, uCore; varying float vV, vS, vLag;
        void main(){
          float x = abs(vV);
          float core = exp(-x * x * 30.0), halo = exp(-x * x * 3.2);
          vec3 col = vLag < 0.5 ? vec3(1.0, 0.95, 0.86) * core * uCore + uGold * halo * 0.4
                                : uGold * halo * 0.15 * pow(1.0 - vLag / 7.0, 2.0);   // afterglow of the last few tenths of a second
          gl_FragColor = vec4(col * uFade, 1.0);
        }`,
    });
    const strMesh = new THREE.Mesh(strGeo, strMat); strMesh.frustumCulled = false; strMesh.renderOrder = 10;
    const strSpin = new THREE.Group(); strSpin.add(strMesh);
    const aura = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture(), color: new THREE.Color(palette.planck).multiplyScalar(0.0), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false }));
    aura.scale.set(4.6, 4.6, 1);
    const strG = new THREE.Group(); strG.scale.setScalar(R_STR); strG.rotation.set(0.84, 0, 0.22); strG.add(strSpin);
    const strAuraG = new THREE.Group(); strAuraG.scale.setScalar(R_STR); strAuraG.add(aura);
    world.add(strG, strAuraG);

    // ============================================================================================
    // state
    let z = Z_MIN, vz = 0, svz = 0;
    { const d = parseFloat(params.get('depth')); if (isFinite(d)) z = clamp(Math.abs(d), Z_MIN, Z_MAX); }
    let active = false, autoDive = false, flyTo = null, beginT = 0, keyDir = 0, spaceT0 = -1;
    let dragging = false, dragMoved = 0, downT = 0, userTravel = 0, hintUp = false;
    let yaw = 0, yawT = parseFloat(params.get('yaw')) || 0, pitch = 0, hoverX = 0, hoverY = 0;
    let layerIdx = -1, lastBell = -9, lastProbe = -9;
    let pendingProbe = params.has('probe'), kick = 0, seaBoost = 0, glint = 0, measured = false;
    let toldLimit = false, toldString = false, toldDesert = false, swelled = false;
    let shep = null, hum = null;
    let RO = null, CT = null, bigEl = null, subEl = null, mkEl = null, roKey = -1, mkZ = -99, byKey = '';
    let Href = 2 * CAM_Z * TAN;
    const RULER_H = 212;
    const postObj = { aberration: 0.0016, bloomStrength: 0.95, bloomRadius: 0.62 };
    const humAmps = new Float32Array(6);
    const v3 = new THREE.Vector3(), m4 = new THREE.Matrix4();
    const ray = new THREE.Raycaster(), hitPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0), ndc0 = new THREE.Vector2();

    function applyView() {
      const a = ctx.aspect;
      camera.aspect = a; camera.updateProjectionMatrix();
      Href = 2 * CAM_Z * TAN * Math.min(1, a);
      // compose the subject a little up (and, on phones, into the upper-middle of the frame)
      const sy = a < 1 ? 0.17 : 0.035, sx = a < 1 ? 0 : 0.02;
      camera.projectionMatrix.elements[8] = -sx; camera.projectionMatrix.elements[9] = -sy;
      camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
      foam.scale.set(TAN * a * 1.6, TAN * 1.6, 1);
    }
    applyView();

    const layerAt = (zz) => { let i = 0; for (let k = 0; k < LAYERS.length; k++) if (zz >= LAYERS[k].z0) i = k; return i; };
    // Dive speed in decades per second (auto-dive), smooth in z: slower where there is a lot to see,
    // fast across the seventeen empty decades of the desert.
    const spd = (zz) => 0.6 - 0.12 * smoothstep(8.2, 8.8, zz) + 0.12 * smoothstep(16, 16.8, zz)
      + 2.0 * smoothstep(18.4, 19.6, zz) * (1 - smoothstep(28.6, 29.8, zz))
      + 0.25 * smoothstep(29.6, 30.4, zz) * (1 - smoothstep(33.2, 34, zz)) - 0.1 * smoothstep(34, 34.6, zz);

    function setAuto(on) {
      autoDive = on;
      if (on) flyTo = null;
      if (CT) { CT.set('dive', on ? 'Stop' : 'Dive'); CT.set('dive', on); }
    }
    function fly(zz) {
      flyTo = clamp(zz, Z_MIN, Z_MAX); setAuto(false); beginT = 0;
      if (audio.on && Math.abs(flyTo - z) > 0.5) audio.noise({ freq: flyTo > z ? 260 : 2400, sweepTo: flyTo > z ? 2600 : 260, dur: 1.0, gain: 0.045, q: 1.6 });
      userTravel += 2;
    }

    // ---- sound: a Shepard tone tied to depth, wind and foam crackle, the string's hum --------
    function makeShepard() {
      const ac = audio.ctx;
      const out = ac.createGain(); out.gain.value = 0; out.connect(audio.out);
      const lp = ac.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 2600; lp.Q.value = 0.3; lp.connect(out);
      const oscs = [], gains = [];
      for (let i = 0; i < 8; i++) {
        const o = ac.createOscillator(); o.type = 'sine';
        const g = ac.createGain(); g.gain.value = 0;
        o.connect(g).connect(lp); o.start(); oscs.push(o); gains.push(g);
      }
      const len = ac.sampleRate * 2, buf = ac.createBuffer(1, len, ac.sampleRate), d = buf.getChannelData(0);
      for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
      const src = ac.createBufferSource(); src.buffer = buf; src.loop = true;
      const bp = ac.createBiquadFilter(); bp.type = 'bandpass'; bp.frequency.value = 300; bp.Q.value = 0.8;
      const wg = ac.createGain(); wg.gain.value = 0;
      const hp = ac.createBiquadFilter(); hp.type = 'highpass'; hp.frequency.value = 3600;
      const fg = ac.createGain(); fg.gain.value = 0;
      src.connect(bp).connect(wg).connect(audio.out);
      src.connect(hp).connect(fg).connect(audio.out);
      src.start();
      return { out, oscs, gains, src, bp, wg, fg, last: new Float32Array(8).fill(-1) };
    }
    function stopShepard() {
      if (!shep) return;
      const s = shep; shep = null;
      try {
        const t = audio.ctx.currentTime;
        s.out.gain.setTargetAtTime(0, t, 0.08); s.wg.gain.setTargetAtTime(0, t, 0.08); s.fg.gain.setTargetAtTime(0, t, 0.05);
        s.oscs.forEach((o) => o.stop(t + 0.6)); s.src.stop(t + 0.6);
        setTimeout(() => { try { s.out.disconnect(); s.wg.disconnect(); s.fg.disconnect(); } catch (e) { /* already gone */ } }, 900);
      } catch (e) { /* context closed */ }
    }
    function audioFrame(vFoam, vString) {
      if (!audio.on || !audio.ctx) {   // sound switched off: tear the sustained voices down (they come back lazily)
        if (shep) stopShepard();
        if (hum) { hum.stop(0.3); hum = null; }
        return;
      }
      if (!shep) shep = makeShepard();
      const t = audio.ctx.currentTime, x = z * 0.5, sp = Math.abs(svz);
      // eight sines an octave apart under a bell-shaped spectral envelope: diving glides them all
      // down (one octave per two decades); a partial that falls off the bottom re-enters, silent, at the top
      for (let i = 0; i < 8; i++) {
        const e = (((i - x) % 8) + 8) % 8;
        const f = 36.71 * Math.pow(2, e);
        const fr = shep.oscs[i].frequency;
        if (shep.last[i] < 0 || Math.abs(e - shep.last[i]) > 4) { fr.cancelScheduledValues(t); fr.setValueAtTime(f, t); } else fr.setTargetAtTime(f, t, 0.03);
        shep.gains[i].gain.setTargetAtTime(Math.exp(-0.5 * Math.pow((e - 3.6) / 1.2, 2)), t, 0.05);
        shep.last[i] = e;
      }
      shep.out.gain.setTargetAtTime(0.1 * smoothstep(0.04, 1.6, sp), t, 0.12);
      shep.wg.gain.setTargetAtTime(0.05 * smoothstep(0.4, 6, sp), t, 0.12);
      shep.bp.frequency.setTargetAtTime(220 + 260 * Math.min(sp, 8), t, 0.12);
      shep.fg.gain.setTargetAtTime(vFoam * (Math.random() < 0.28 ? 0.045 : 0.006), t, 0.012);
      if (!hum && vString > 0.02) hum = audio.additive(audio.freq(12), 6, { gain: 0 });
      if (hum) {
        hum.setGain(0.07 * vString, 0.25);
        humAmps[0] = 0.55; humAmps[5] = 0.04;
        for (let i = 0; i < 4; i++) humAmps[i + 1] = Math.min(0.5, (AL.getComponent(i) + AR.getComponent(i)) * 2.4);
        hum.setAmps(humAmps, 0.08);
        if (vString <= 0.001 && z < 32) { hum.stop(0.6); hum = null; }
      }
    }
    function layerBell(i) {
      if (!audio.on) return;
      const now = ctx.time;
      if (now - lastBell < 0.25 || Math.abs(svz) > 6) return;
      lastBell = now;
      audio.bell(audio.degree(4 + i, 1), { gain: 0.1, pan: i % 2 ? 0.25 : -0.25 });
    }

    // ---- interactions ---------------------------------------------------------------------------
    function probe(ndc) {
      const now = ctx.time;
      if (now - lastProbe < 0.12) return;
      lastProbe = now;
      const id = LAYERS[layerIdx < 0 ? layerAt(z) : layerIdx].id;
      if (id === 'salt' || id === 'face') {
        glint = 1;
        audio.bell(audio.degree(12, 1), { gain: 0.12, pan: (ndc ? ndc.x : 0) * 0.5 });
      } else if (id === 'lattice') {
        // a phonon: a compression wave that runs through the lattice from where you tapped
        ndc0.set(ndc ? ndc.x : 0, ndc ? ndc.y : 0);
        ray.setFromCamera(ndc0, camera);
        if (ray.ray.intersectPlane(hitPlane, v3)) { lattice.worldToLocal(v3); latU.uWave.value.set(v3.x, v3.y, v3.z, uTime.value); }
        audio.pluck(audio.degree(0, 0), { gain: 0.22, decay: 1.6, bright: 0.25 });
        audio.noise({ freq: 180, dur: 0.5, gain: 0.05, q: 1.2 });
      } else if (id === 'ion') {
        // a position measurement: one electron found at one spot, then the cloud spreads again
        let i = 0;
        for (let tries = 0; tries < 40; tries++) { i = Math.floor(Math.random() * NCLOUD); if (cloudOrb[i] >= 5) break; }
        flash.position.set(cloudPos[i * 3], cloudPos[i * 3 + 1], cloudPos[i * 3 + 2]);
        flashT = 0; cloudDimT = 0;
        audio.bell(audio.degree(9, 2), { gain: 0.09, decay: 2.2 });
        if (!measured) { measured = true; ui.toast('A measurement finds an electron at one spot. Afterwards its probability cloud spreads out again.', 4200); }
      } else if (id === 'nucleus') {
        kick = 1;
        audio.pluck(audio.degree(3, 0), { gain: 0.24, decay: 1.2, inharm: 0.002, bright: 0.4 });
      } else if (id === 'proton') {
        seaBoost = 1; if (gluon.t < 0) gluon.next = 0;
        audio.pluck(audio.degree(5, 1), { gain: 0.16, decay: 1.4, bright: 0.7 });
      } else if (id === 'string') {
        for (let i = 0; i < 4; i++) plucked[i] = Math.min(0.32, plucked[i] + (0.06 + Math.random() * 0.1) / (i + 1));
        audio.pluck(audio.freq(12), { gain: 0.26, decay: 4.2, bright: 0.6 });
        audio.pluck(audio.freq(24), { gain: 0.1, decay: 3.2, bright: 0.5, when: 0.02, pan: 0.2 });
      } else {
        pingT = 0;
        audio.bell(audio.degree(4, 2), { gain: 0.07, decay: 3.2 });
        audio.noise({ freq: 1800, sweepTo: 500, dur: 1.2, gain: 0.025, q: 3 });
        if (id === 'desert' && !toldDesert) { toldDesert = true; ui.toast('No experiment has ever reached this scale. There is no data to show.', 3600); }
      }
    }

    // ---- programme panels -------------------------------------------------------------------------
    function buildPanels() {
      const phone = ctx.width <= 760;
      const tk = (zz) => ((zz - Z_MIN) / (Z_MAX - Z_MIN) * RULER_H).toFixed(1);
      let ticks = '';
      for (let d = 3; d <= 35; d++) ticks += `<i class="d1-tk${d % 5 === 0 ? ' mj' : ''}" style="top:${tk(d)}px"></i>`;
      const lbl = [[3, 'Salt', 3], [10, 'Atom', 9.45], [14, 'Nucleus', 13.85], [18, 'Quark', 17.55], [35, 'String', 35]]
        .map(([e, n, zz]) => `<button type="button" class="d1-lb" data-z="${zz}" style="top:${tk(e)}px"><b>−${e}</b>${n}</button>`).join('');
      const html = `<style>
        .d1-wrap{display:grid;grid-template-columns:auto 1fr;gap:16px;align-items:start;margin:2px 0 10px}
        .d1-ruler{position:relative;width:84px;height:${RULER_H}px;margin:4px 0}
        .d1-ax{position:absolute;left:5px;top:0;bottom:0;width:1px;background:rgba(236,231,246,.16)}
        .d1-tk{position:absolute;left:3px;width:5px;height:1px;background:rgba(236,231,246,.3)}
        .d1-tk.mj{left:1px;width:9px;background:rgba(236,231,246,.55)}
        .d1-band{position:absolute;left:2px;width:7px;background:repeating-linear-gradient(180deg,rgba(236,231,246,.16) 0 1px,transparent 1px 4px)}
        .d1-band.fm{background:linear-gradient(180deg,rgba(139,108,255,0),rgba(139,108,255,.55))}
        .d1-nd{position:absolute;left:17px;transform:translateY(-50%);font:9px/1 var(--f-mono);letter-spacing:.14em;text-transform:uppercase;color:rgba(236,231,246,.36);white-space:nowrap}
        .d1-lb{all:unset;cursor:pointer;position:absolute;left:17px;transform:translateY(-50%);font:9.5px/1 var(--f-mono);letter-spacing:.1em;text-transform:uppercase;color:var(--ink-dim);white-space:nowrap;padding:2px 0}
        .d1-lb b{font-weight:400;color:var(--ink);margin-right:5px;letter-spacing:0;display:inline-block;min-width:3ch}
        .d1-lb:hover,.d1-lb:focus-visible{color:var(--planck)}
        .d1-mk{position:absolute;left:-4px;top:0;width:15px;height:2px;margin-top:-1px;background:var(--planck);box-shadow:0 0 10px var(--planck)}
        .d1-big{white-space:nowrap}
        .d1-big small{font-size:.48em;color:var(--ink-dim);margin-right:.1em;font-style:italic}
        .d1-sub{font:10px/1.5 var(--f-mono);letter-spacing:.12em;text-transform:uppercase;color:var(--ink-dim);margin-top:8px}
        .d1-go{all:unset;cursor:pointer;display:inline-block;margin-top:8px;font:10.5px/1.4 var(--f-mono);letter-spacing:.14em;text-transform:uppercase;color:var(--planck);border-bottom:1px solid rgba(255,196,107,.5)}
        .d1-go:hover,.d1-go:focus-visible{border-color:var(--planck)}
        @media (max-width:760px){.d1-ruler{display:none}.d1-wrap{display:block;margin:0 0 4px}.d1-sub{display:none}}
      </style>
      <div class="d1-wrap">
        <div class="d1-ruler" aria-hidden="false"><i class="d1-ax"></i>${ticks}
          <i class="d1-band" style="top:${tk(18.6)}px;height:${(tk(29.6) - tk(18.6)).toFixed(1)}px"></i>
          <i class="d1-band fm" style="top:${tk(29.6)}px;height:${(tk(35) - tk(29.6)).toFixed(1)}px"></i>
          <span class="d1-nd" style="top:${tk(25.2)}px">no data</span>${lbl}<i class="d1-mk"></i></div>
        <div><div class="big d1-big"></div><div class="d1-sub"></div></div>
      </div>`;
      const rows = [{ id: 'fov', k: 'Field of view' }, { id: 'mag', k: 'Magnification' }];
      if (!phone) rows.push({ id: 'by', k: 'Probed by' });
      RO = ui.readout({ title: '', html, rows, note: '' });
      bigEl = RO.el.querySelector('.d1-big'); subEl = RO.el.querySelector('.d1-sub'); mkEl = RO.el.querySelector('.d1-mk');
      RO.el.querySelectorAll('.d1-lb').forEach((b) => b.addEventListener('click', () => fly(parseFloat(b.dataset.z))));
      roKey = -1; mkZ = -99; byKey = '';
      const padCol = ['#ffc46b', '#6ff3ff', '#ff6a3d', '#ff5470', '#8b6cff', '#ffc46b'];
      CT = ui.controls([
        { type: 'keys', id: 'jump', label: 'Go to scale', keys: PADS.map((p, i) => ({ label: p.label, sub: p.sub, kbd: String(i + 1), color: padCol[i], onPress: () => fly(p.z) })) },
        { type: 'button', id: 'dive', label: 'Dive', kbd: 'd', onClick: () => setAuto(!autoDive) },
        { type: 'button', id: 'surface', label: 'Surface', kbd: 's', onClick: () => fly(Z_MIN) },
        { type: 'button', id: 'probe', label: 'Ring', kbd: 'p', onClick: () => probe(null) },
      ]);
      const b = CT.get('dive'); if (b) b.title = 'Key: D or Space';
    }
    function onLayer(prev, i) {
      const L = LAYERS[i];
      if (RO) {
        RO.title(L.name);
        const phone = ctx.width <= 760;
        const go = '<button type="button" class="d1-go">Play it in Movement II &rarr;</button>';
        const phoneNote = L.id === 'string' ? 'In string theory, this is the instrument. Tap it to pluck.<br>' + go : L.id === 'desert' ? L.note : '';
        RO.note(phone ? phoneNote : L.note + (L.id === 'string' ? '<br>' + go : ''));
        const gb = RO.el.querySelector('.d1-go'); if (gb) gb.addEventListener('click', () => ctx.next());
      }
      if (CT) { CT.set('probe', L.probe); CT.set('jump', L.name.replace(/^(A|An|The) /, '')); }
      if (prev >= 0) layerBell(i);
    }
    function updateReadout() {
      if (!RO) return;
      const key = Math.round(z * 40);
      if (key !== roKey) {
        roKey = key;
        let e = Math.floor(-z + 1e-9), m = Math.pow(10, -z - e);
        if (m >= 9.95) { e += 1; m = 1; }
        bigEl.innerHTML = (m > 1.04 ? `<small>${m.toFixed(1)}×</small>` : '') + `10${sup(e)}<small> m</small>`;
        RO.set('fov', fmtSI(z));
        const mg = z - Z_MIN;
        RO.set('mag', mg < 0.95 ? '× ' + Math.pow(10, mg).toFixed(1) : '× 10' + sup(Math.round(mg)));
        const left = Math.ceil(Z_MAX - z - 0.02);
        subEl.textContent = left > 0 ? `${left} power${left === 1 ? '' : 's'} of ten to go` : 'the bottom of the dive';
        const L = LAYERS[Math.max(0, layerIdx)];
        const by = L.id === 'face' && z > 6.7 ? 'electron microscope' : L.by;
        if (by !== byKey) { byKey = by; RO.set('by', by); }
      }
      if (Math.abs(z - mkZ) > 0.01) { mkZ = z; mkEl.style.transform = `translateY(${((z - Z_MIN) / (Z_MAX - Z_MIN) * RULER_H).toFixed(1)}px)`; }
    }

    // ---- the pivot chain ----------------------------------------------------------------------------
    // The view is centred on a point (metres, world frame) that slides from the grain's centre to the
    // face above our ion, to the ion, to the proton, to the up quark that holds the string. Each hand-off
    // is a logistic in log-scale, so the residual offset shrinks ten times faster than the view does and
    // the next subject drifts smoothly into the centre while we zoom.
    // Between z = 7.4 and 8.9 the crystal turns (about the ion) until its face almost faces us, so we
    // sink straight down the alternating Na/Cl columns. The face point follows it.
    const qTilt = new THREE.Quaternion().setFromEuler(TILT), qFaceOn = new THREE.Quaternion().setFromEuler(FACE_ON);
    const faceL = new THREE.Vector3(0, 0, SURF_K * A_NACL), facePointW = new THREE.Vector3();
    const PIV = [cubeCentreW, facePointW, N_c, P_c, new THREE.Vector3()];
    const PIV_M = [3.45, 8.9, 14.2, 15.1];
    const PIV_D = [0, 1, 2, 3].map((k) => PIV[k].clone().sub(PIV[k + 1]));
    const focus = new THREE.Vector3();
    function orientCrystal(zz) {
      crystal.quaternion.slerpQuaternions(qTilt, qFaceOn, smoothstep(7.4, 8.9, zz));
      facePointW.copy(faceL).applyQuaternion(crystal.quaternion).add(N_c);
      PIV_D[0].copy(cubeCentreW).sub(facePointW);
      PIV_D[1].copy(facePointW).sub(N_c);
    }
    function focusAt(zz, out) {
      out.set(0, 0, 0);
      for (let k = 0; k < 4; k++) out.addScaledVector(PIV_D[k], 1 / (1 + Math.pow(10, 2 * (zz - PIV_M[k]))));
      return out;
    }
    const band = (x, a, b, c, d) => smoothstep(a, b, x) * (1 - smoothstep(c, d, x));
    function tintAt(zz, out) {
      let k = 0;
      while (k < tintCols.length - 2 && zz > tintCols[k + 1][0]) k++;
      const a = tintCols[k], b = tintCols[k + 1];
      return out.copy(a[1]).lerp(b[1], smoothstep(a[0], b[0], zz));
    }

    // ---- per-frame scratch ---------------------------------------------------------------------------
    let zPrev = z, dragV = 0, lastMoveT = 0, demoStir = false, probeAt = -1, phoneMode = ctx.width <= 760;
    const BEGIN_DUR = 2.4, BEGIN_V = 0.15 * Math.PI / (2 * BEGIN_DUR);   // the opening push-in covers 0.15 decades
    const motionK = reduced ? 0.25 : 1;
    const hoverK = ctx.isTouch ? 0 : 1;
    const qRot = new THREE.Quaternion();
    const tint = new THREE.Color();
    const starU = stars.material.uniforms, starC0 = starU.uColor.value.clone(), starT0 = starU.uTint.value.clone();
    const cubeU = cubeMat.uniforms, planeU = planeMat.uniforms, cloudU = cloudMat.uniforms, nucU = nucMat.uniforms;
    const memU = memMat.uniforms, seaU = seaMat.uniforms, qU = qMat.uniforms, strU = strMat.uniforms, moteU = moteMat.uniforms;
    const nucPtU = nucPoint.material.uniforms, flashU = flash.material.uniforms, qPtU = qPoint.material.uniforms;
    const ringU = limitRing.material.uniforms, pingU = ping.material.uniforms;
    pingU.uC.value.copy(palette.ink);
    const aSum = new Float32Array(4);

    // debug hook (?dbg): exposes internals so a test harness can inspect or override state after update()
    const DBG = params.has('dbg') ? (window.__d1 = { after: null, get z() { return z; }, set z(v) { z = v; zPrev = v; },
      limitRing, qPoint, strMesh, tubes, foam, ping, cube, plane, ions, cloudPts, nucMesh, membrane, sea, quarks, motes, world, root, camera }) : null;

    function showHint() {
      // on phones the hint sits mid-screen across the subject, so it bows out after a few seconds
      ui.hint(ctx.isTouch ? 'Swipe up to dive · tap to probe' : 'Scroll, ↓ or hold space to dive', ctx.width <= 760 ? 6500 : 0);
      hintUp = true;
    }

    // ---- the frame -----------------------------------------------------------------------------------
    function update(dt, t) {
      uTime.value = t;
      const pr = ctx.pixelRatio, H = ctx.height;
      uRes.value.set(ctx.width * pr, H * pr);
      const px = pr * clamp(H / 900, 0.72, 1.6);        // point sprites scale with the viewport height

      // -- dive dynamics: z in decades, vz in decades per second
      const s = spd(z);
      if (probeAt >= 0 && t >= probeAt) { probeAt = -1; pendingProbe = false; probe(null); }
      if (flyTo !== null) {
        const d = flyTo - z;
        const vT = Math.sign(d) * Math.min(4.2 * s, Math.abs(d) * 2.6);
        vz += (vT - vz) * damp(5, dt);
        if (Math.abs(d) < 0.003 && Math.abs(vz) < 0.06) {
          z = flyTo; vz = 0; flyTo = null;
          if (demoStir) { demoStir = false; probe(null); }
        }
      } else if (beginT > 0) {
        beginT = Math.max(0, beginT - dt);
        vz = BEGIN_V * Math.sin(Math.PI * (1 - beginT / BEGIN_DUR));
      } else if (autoDive) {
        vz += (1.3 * s - vz) * damp(2.2, dt);
        userTravel += Math.abs(vz) * dt;
      } else if (keyDir !== 0) {
        vz += (keyDir * 1.8 * s - vz) * damp(4, dt);
        userTravel += Math.abs(vz) * dt;
      } else if (!dragging) {
        vz *= Math.exp(-2.4 * dt);
      }
      if (!dragging) z += vz * dt;
      if (z <= Z_MIN) { z = Z_MIN; if (vz < 0) vz = 0; }
      if (z >= Z_MAX) { z = Z_MAX; if (vz > 0) vz = 0; if (autoDive) setAuto(false); }
      const rate = dt > 0 ? clamp((z - zPrev) / dt, -40, 40) : 0;
      zPrev = z;
      svz += (rate - svz) * damp(5, dt);
      const speed = Math.abs(svz);

      // -- the similarity transform: world units per metre, the focus, and the orbit
      const U = Href * Math.pow(10, z);
      orientCrystal(z);
      focusAt(z, focus);
      if (!dragging) yawT *= Math.exp(-0.45 * dt);   // the orbit springs back slowly
      yaw += (yawT - yaw) * damp(5, dt);
      pitch *= Math.exp(-2 * dt);
      hoverX += (hoverK * ctx.pointer.ndc.x - hoverX) * damp(1.4, dt);
      hoverY += (hoverK * ctx.pointer.ndc.y - hoverY) * damp(1.4, dt);
      const sway = (1 - smoothstep(3.08, 3.7, z)) * 0.42 * Math.sin(t * 0.105 + 0.5) * motionK;   // the grain turning
      const ry = yaw + sway + hoverX * 0.05 + 0.016 * Math.sin(t * 0.21) * motionK;
      const rx = pitch - hoverY * 0.035 + 0.012 * Math.sin(t * 0.17 + 1.3) * motionK;
      const rz = -(z - Z_MIN) * 0.055 * (reduced ? 0.4 : 1) + 0.01 * Math.sin(t * 0.13) * motionK;   // slow roll with depth
      root.rotation.set(rx, ry, rz, 'ZXY');
      world.position.copy(focus).multiplyScalar(-U);
      world.scale.setScalar(U);
      root.updateMatrixWorld(true);

      // -- atmosphere
      tintAt(z, tint);
      backMat.uniforms.uTint.value.copy(tint);
      const fDesert = band(z, 18.4, 19.6, 28.8, 30.6);
      const fFoam = smoothstep(29.4, 33.6, z);
      const fStr = smoothstep(33.7, 34.6, z);
      stars.quaternion.copy(root.quaternion);
      stars.userData.update(t);
      const kStar = 1 - 0.7 * fDesert - 0.5 * fFoam;
      starU.uColor.value.copy(starC0).multiplyScalar(kStar);
      starU.uTint.value.copy(starT0).multiplyScalar(kStar);

      // -- I. the grain and its out-of-focus neighbours
      const fCube = 1 - smoothstep(4.32, 4.48, z);
      cube.visible = fCube > 0.001;
      if (cube.visible) {
        cubeU.uFade.value = fCube;
        cubeU.uView.value = Math.pow(10, -z);
        cubeU.uInner.value = 1 - smoothstep(3.2, 3.75, z);
        cube.worldToLocal(cubeU.uCamObj.value.copy(camera.position));
        qRot.copy(root.quaternion).multiply(crystal.quaternion);
        cubeU.uRot.value.setFromMatrix4(m4.makeRotationFromQuaternion(qRot));
      }
      glint *= Math.exp(-1.6 * dt);
      uGlint.value = glint;
      uBeacon.value = 1 - smoothstep(7.9, 8.6, z);
      const fBok = 1 - smoothstep(3.18, 3.75, z);
      for (let i = 0; i < bokeh.length; i++) {
        const sp = bokeh[i];
        sp.visible = fBok > 0.001;
        if (sp.visible) sp.material.color.copy(sp.userData.c).multiplyScalar(sp.userData.k * fBok);
      }

      // -- the crystal face (fades as the camera sinks to within a couple of spacings of it)
      let hFace = -1;
      if (z < 9.85) { crystal.worldToLocal(v3.copy(camera.position)); hFace = (v3.z - SURF_K * A_NACL) / A_NACL; }
      // the face thins out as the 3D lattice appears beneath it, so the surface ions show through
      const fPlane = smoothstep(4.02, 4.36, z) * smoothstep(0.5, 1.8, hFace) * (1 - 0.82 * smoothstep(8.35, 8.95, z)) * (1 - smoothstep(9.6, 9.8, z));
      plane.visible = fPlane > 0.001;
      if (plane.visible) {
        planeU.uFade.value = fPlane;
        planeU.uCamL.value.copy(v3);
        qRot.copy(root.quaternion).multiply(crystal.quaternion).invert();
        planeU.uInvRot.value.setFromMatrix4(m4.makeRotationFromQuaternion(qRot));
        planeU.uField.value = Math.pow(10, -z);
        planeU.uU.value = U;
        planeU.uAtoms.value = 1 - smoothstep(8.5, 9.0, z);
        plane.scale.setScalar(Math.pow(10, -z) * 300);
      }

      // -- II. the lattice
      // the neighbours dim (out of focus) while one ion blooms into its electron cloud, then fall away
      const fLat = smoothstep(8.25, 8.9, z) * (1 - 0.8 * smoothstep(9.1, 9.45, z)) * (1 - smoothstep(9.55, 9.95, z));
      lattice.visible = fLat > 0.001;
      latU.uFade.value = fLat;
      latU.uUnit.value = A_NACL * U;
      latU.uPivotFade.value = 1 - smoothstep(8.95, 9.25, z);

      // -- III. the chloride ion and the nucleus seen from it
      const fCloud = smoothstep(8.95, 9.35, z) * (1 - smoothstep(12.55, 13.15, z));
      cloud.visible = fCloud > 0.001;
      flashT += dt; cloudDimT += dt;
      if (cloud.visible) {
        cloudU.uFade.value = fCloud;
        cloudU.uPx.value = 1.5 * px;
        cloudU.uDim.value = 1 - 0.82 * Math.exp(-cloudDimT * 1.7);
        cloudPts.rotation.y = 0.48 + t * 0.025 * motionK;
        flash.visible = flashT < 3;
        flashU.uI.value = 2.6 * Math.exp(-flashT * 2.4);
        flashU.uSize.value = 64 * px;
      }
      const fNP = smoothstep(9.55, 10.2, z) * (1 - smoothstep(12.85, 13.3, z));
      nucPoint.visible = fNP > 0.001;
      nucPtU.uI.value = fNP * 1.1;
      nucPtU.uSize.value = (24 + 22 * smoothstep(10.5, 13, z)) * px;

      // -- IV. the nucleus
      const fNuc = smoothstep(12.75, 13.3, z) * (1 - smoothstep(14.6, 14.8, z));
      nucleus.visible = fNuc > 0.001;
      kick *= Math.exp(-2.2 * dt);
      nucU.uFade.value = fNuc;
      nucU.uTargetFade.value = 1 - smoothstep(14.2, 14.45, z);   // the proton we dive into hands over to its insides
      nucU.uOthersFade.value = 1 - smoothstep(13.95, 14.4, z);   // the rest of the nucleus falls away; one proton stays
      nucU.uKick.value = kick;

      // -- V. the proton
      const fP = smoothstep(14.12, 14.45, z);
      const fMem = fP * (1 - smoothstep(15.45, 16.05, z));
      const fTube = fP * (1 - smoothstep(15.75, 16.35, z));
      const fQ = fP * (1 - smoothstep(15.95, 16.4, z));
      seaBoost *= Math.exp(-0.9 * dt);
      proton.visible = fQ > 0.001;
      if (proton.visible) {
        // gluon exchange: quark A emits a gluon (colour a, anticolour b) and turns b; quark B absorbs it and turns a
        if (gluon.t < 0) {
          gluon.next -= dt * (1 + 2 * seaBoost);
          if (gluon.next <= 0) {
            gluon.from = Math.floor(Math.random() * 3);
            gluon.to = (gluon.from + 1 + Math.floor(Math.random() * 2)) % 3;
            gluon.ca = qc[gluon.from]; gluon.cb = qc[gluon.to];
            qc[gluon.from] = gluon.cb;
            gluon.t = 0;
          }
        } else {
          gluon.t += dt * 1.7;
          if (gluon.t >= 2) { qc[gluon.to] = gluon.ca; gluon.t = -1; gluon.next = 0.7 + Math.random() * 1.6; }
        }
        tubeU.uPulse.value.set(gluon.from, gluon.to, gluon.t, gluon.t < 0 ? 0 : Math.pow(Math.sin(Math.PI * gluon.t / 2), 0.35));
        const kc = damp(5, dt);
        for (let i = 0; i < 3; i++) qDisp[i].lerp(QCOL[qc[i]], kc);
        // the up quark that carries our string is the frame of reference; the other two and the junction wander
        const jig = 0.035 * (1 + 1.5 * seaBoost);
        qNow[0].copy(QB[0]);
        for (let i = 1; i < 3; i++) {
          qNow[i].set(QB[i].x + jig * Math.sin(t * (1.7 + i) + i * 2.1), QB[i].y + jig * Math.sin(t * (1.3 + 0.4 * i) + i), QB[i].z + jig * Math.sin(t * 2.3 + i * 4.0));
        }
        junction.copy(qNow[0]).add(qNow[1]).add(qNow[2]).multiplyScalar(1 / 3);
        junction.x += 0.02 * Math.sin(t * 1.1); junction.y += 0.02 * Math.sin(t * 0.9 + 1);
        for (let i = 0; i < 3; i++) {
          qPos[i * 3] = qNow[i].x; qPos[i * 3 + 1] = qNow[i].y; qPos[i * 3 + 2] = qNow[i].z;
          qCol[i * 3] = qDisp[i].r; qCol[i * 3 + 1] = qDisp[i].g; qCol[i * 3 + 2] = qDisp[i].b;
        }
        qGeo.attributes.position.needsUpdate = true;
        qGeo.attributes.aCol.needsUpdate = true;
        memU.uFade.value = fMem; membrane.visible = fMem > 0.001;
        tubeU.uFade.value = fTube; tubeU.uPx.value = px; tubes.visible = fTube > 0.001;
        qU.uI.value = fQ * 1.25; qU.uSize.value = 38 * px;
        seaU.uFade.value = fTube; seaU.uBoost.value = seaBoost; seaU.uPx.value = px; sea.visible = fTube > 0.001;
      }

      // -- VI. the quark as a point, the edge of experiment, the probe ping
      const fQP = smoothstep(15.95, 16.35, z) * (1 - smoothstep(33.9, 34.55, z));
      qPoint.visible = fQP > 0.001;
      if (qPoint.visible) {
        qPoint.position.copy(world.position);
        // in the foam the point starts to tremble: spacetime itself is no longer quiet
        const tremble = 1 + fFoam * 0.55 * Math.sin(t * 29.0) * Math.sin(t * 13.7 + 1.0);
        qPtU.uI.value = fQP * (1 - 0.15 * fDesert) * tremble;
        qPtU.uC.value.copy(qDisp[0]);
        qPtU.uSize.value = 34 * px * (1 + 0.35 * fFoam);
      }
      const fRing = smoothstep(16.4, 17.0, z) * (1 - smoothstep(18.3, 19.0, z));
      limitG.visible = fRing > 0.001;
      ringU.uI.value = fRing * 0.9; ringU.uW.value = 1.3 * pr;
      const fDec = band(z, 18.2, 19.4, 31.0, 33.0);
      for (let k = 0; k < 3; k++) {
        const r = decRings[k], n = Math.floor(z) + k;       // this ring is 10^-n m across
        const rel = Math.pow(10, z - n);                      // its diameter in view heights
        const a = fDec * smoothstep(0.035, 0.12, rel) * (1 - smoothstep(0.9, 2.2, rel));
        r.visible = a > 0.002;
        if (!r.visible) continue;
        r.position.copy(world.position);
        const ru = r.material.uniforms;
        ru.uR.value = 0.5 * Href * rel; ru.uI.value = 0.42 * a; ru.uW.value = 1.1 * pr;
      }
      pingT += dt;
      ping.visible = pingT < 2.6;
      if (ping.visible) {
        ping.position.copy(world.position);
        pingU.uR.value = 0.12 + pingT * 2.6;
        pingU.uI.value = 1.3 * Math.exp(-pingT * 1.4) * smoothstep(0, 0.08, pingT);
        pingU.uW.value = 1.6 * pr;
      }

      // -- VII. motes: scale-free dust whose streaks follow the dive
      travel += (svz * 0.2 + 0.006 * motionK) * dt;
      moteU.uTravel.value = travel;
      moteU.uSpeed.value = clamp(svz * 1.1, -4, 4);
      moteU.uFade.value = 0.05 + 0.4 * fDesert + 0.6 * smoothstep(0.3, 4, speed);
      moteU.uPx.value = 1.1 * px;
      moteU.uTint.value.copy(tint).multiplyScalar(2.2).lerp(palette.ink, 0.6).multiplyScalar(0.8);

      // -- VIII. quantum foam
      foam.visible = fFoam > 0.002;
      if (foam.visible) {
        foamU.uZ.value = z;
        foamU.uI.value = fFoam;
        foamU.uCalm.value = smoothstep(33.8, 35, z);
        foamU.uRs.value = R_STR * U;
        foamU.uHref.value = Href;
        root.worldToLocal(foamU.uCamL.value.copy(camera.position));
        qRot.copy(root.quaternion).invert();
        foamU.uRootInv.value.setFromMatrix4(m4.makeRotationFromQuaternion(qRot));
      }

      // -- IX. the string: plucks decay, the idle modes stay level-matched (left and right movers balanced)
      strG.visible = strAuraG.visible = fStr > 0.001;
      if (strG.visible) {
        for (let i = 0; i < 4; i++) {
          plucked[i] *= Math.exp(-dt * (0.55 + 0.25 * i));
          const w = 0.55 + 0.45 * Math.sin(t * (0.3 + 0.11 * i) + i);   // slow breathing of each mode
          AL.setComponent(i, AL0[i] * w + plucked[i] * 0.55);
          AR.setComponent(i, AR0[i] * (1.45 - w) * 0.75 + plucked[i] * 0.45);
          aSum[i] = AL.getComponent(i) + AR.getComponent(i);
        }
        strU.uFade.value = fStr;
        strU.uW.value = 3.6 * pr;
        strU.uCore.value = 1.5 + 1.4 * (aSum[0] + aSum[1]);
        strSpin.rotation.y += dt * 0.22 * motionK;
        aura.material.color.copy(palette.planck).multiplyScalar(0.045 * fStr);
      }

      // -- programme, sound, post
      const li = layerAt(z);
      if (li !== layerIdx) { const prev = layerIdx; layerIdx = li; onLayer(prev, li); }
      // (on phones the toast would sit on the subject, so the readout note says it instead)
      if (!toldLimit && z > 18.7 && active) { toldLimit = true; if (!phoneMode) ui.toast('You have passed the limit of every experiment. Nobody has seen what lies below.', 4400); }
      if (fStr > 0.6 && !swelled) {
        swelled = true;
        audio.swell(audio.freq(12), { gain: 0.11, attack: 2.2, dur: 4.5 });
        audio.swell(audio.freq(19), { gain: 0.05, attack: 3.0, dur: 4.0, pan: 0.3, when: 0.4 });
        if (!toldString && active && !phoneMode) { toldString = true; ui.toast(ctx.isTouch ? 'A string. Tap it to pluck it.' : 'A string. Click it, or press P, to pluck it.', 4200); }
      }
      if (z < 32) swelled = false;
      if (hintUp && userTravel > 1.2) { ui.hideHint(); hintUp = false; }
      if (active) updateReadout();
      if (active) audioFrame(fFoam * (1 - 0.8 * foamU.uCalm.value), fStr);

      postObj.aberration = 0.0016 + 0.0042 * smoothstep(0.4, 7, speed);
      postObj.bloomStrength = 0.95 + 0.12 * fFoam + 0.08 * fStr;
      postObj.bloomRadius = 0.62 - 0.2 * fStr;   // a tighter glow at the bottom keeps the loop crisp
      if (active) ctx.setPost(postObj);
      if (DBG && DBG.after) DBG.after(t);
    }

    // ---- input --------------------------------------------------------------------------------------
    function userGrab() { flyTo = null; beginT = 0; }
    function onPointerDown(ev) {
      dragging = true; dragMoved = 0; dragV = 0;
      downT = lastMoveT = performance.now();
      ctx.setCursor('grabbing');
    }
    function onPointerMove(ev) {
      if (!dragging) return;
      dragMoved += Math.abs(ev.dx) + Math.abs(ev.dy);
      if (dragMoved < 6) return;
      userGrab();
      if (autoDive) setAuto(false);
      const now = performance.now(), dts = Math.max(4, now - lastMoveT) / 1000;
      lastMoveT = now;
      const dz = -ev.dy / Math.max(320, ctx.height) * 3.4 * spd(z);   // swipe up (or drag up) dives
      z = clamp(z + dz, Z_MIN, Z_MAX); vz = 0;
      dragV += (dz / dts - dragV) * 0.35;
      userTravel += Math.abs(dz);
      yawT = clamp(yawT + ev.dx * 0.0042, -0.75, 0.75);
    }
    function onPointerUp(ev) {
      if (!dragging) return;
      dragging = false;
      ctx.setCursor('');
      const now = performance.now();
      if (dragMoved < 8 && now - downT < 450) probe(ev.ndc);
      else if (now - lastMoveT < 90) vz = clamp(dragV, -7 * spd(z), 7 * spd(z));
    }
    function onWheel(ev) {
      let d = ev.deltaY || 0;
      if (ev.meta) d = -d * 3;                        // trackpad pinch arrives as ctrl+wheel: spread fingers to dive
      d = clamp(d, -240, 240);
      const s = spd(z);
      userGrab();
      if (autoDive && d < 0) setAuto(false);
      vz = clamp(vz + d * 0.0045 * s, -7 * s, 7 * s);
      userTravel += Math.abs(d) * 0.002;
    }
    function onKey(ev) {
      const k = ev.key;
      if (k === 'ArrowDown' || k === 'ArrowUp') {
        keyDir = k === 'ArrowDown' ? 1 : -1;
        userGrab();
        if (autoDive) setAuto(false);
        return true;
      }
      if (k === ' ' || ev.code === 'Space') {
        if (!ev.repeat) {
          userGrab();
          if (autoDive) { setAuto(false); spaceT0 = -1; } else { setAuto(true); spaceT0 = performance.now(); }
        }
        return true;
      }
      return false;
    }
    function onKeyUp(ev) {
      if ((ev.key === 'ArrowDown' && keyDir > 0) || (ev.key === 'ArrowUp' && keyDir < 0)) keyDir = 0;
      if ((ev.key === ' ' || ev.code === 'Space') && spaceT0 > 0) {
        if (performance.now() - spaceT0 > 380) setAuto(false);   // held: dive only while held. Tapped: keep diving.
        spaceT0 = -1;
      }
    }

    return {
      scene, camera,
      enter() {
        active = true;
        phoneMode = ctx.width <= 760;
        flyTo = null; autoDive = false; keyDir = 0; dragging = false; spaceT0 = -1; vz = 0; zPrev = z;
        buildPanels();
        layerIdx = layerAt(z);
        onLayer(-1, layerIdx);
        updateReadout();
        const intro = document.getElementById('intro');
        if (!(intro && !intro.classList.contains('gone'))) showHint();
        if (pendingProbe) probeAt = ctx.time + 1.2;
      },
      exit() {
        active = false;
        flyTo = null; autoDive = false; keyDir = 0; dragging = false; beginT = 0; demoStir = false; probeAt = -1; spaceT0 = -1;
        vz = 0; svz = 0;
        stopShepard();
        if (hum) { hum.stop(0.5); hum = null; }
        RO = null; CT = null; bigEl = subEl = mkEl = null; hintUp = false;
        ctx.setCursor('');
      },
      update,
      resize() {
        applyView();
        const ph = ctx.width <= 760;
        if (active && ph !== phoneMode) {
          phoneMode = ph;
          buildPanels(); onLayer(-1, layerIdx < 0 ? layerAt(z) : layerIdx); updateReadout();
          if (autoDive) setAuto(true);
        }
      },
      onPointerDown, onPointerMove, onPointerUp, onWheel, onKey, onKeyUp,
      onBegin() {
        if (z < 3.3 && flyTo === null && !params.has('depth')) beginT = BEGIN_DUR;   // a gentle push-in, then hand over
        showHint();
      },
      demo() {
        // dive quickly to the proton, then stir it so its sea of virtual pairs boils
        if (z < 12.3) { z = 12.3; zPrev = z; vz = 0; }
        fly(14.45);
        demoStir = true;
      },
    };
  },
};
