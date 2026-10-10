// V. Branes. Two parallel membranes seen from the gap between them: a sea of liquid light below
// (our brane) and a translucent ceiling above (a second brane, distance d away). Open strings ride
// the sea with their ends stuck to it, closed strings lift off into the bulk, and strings stretched
// between the two branes weigh T·d. Bring the branes together and U(1) × U(1) becomes U(2).
//
// Rendering:
//  - Branes: one shared polar grid (dense near the action, sparse at the horizon) with a custom
//    shader. Height = a slow swell + ring wave packets (amplitude ∝ 1/√r, the 2D membrane law)
//    emitted by every endpoint and impact. The fragment shader recomputes the analytic gradient per
//    pixel, so ripples stay crisp regardless of mesh density, and fakes the liquid look: thin-film
//    iridescence, fresnel sheen, the *other* brane's grid reflected (two facing mirrors) and sharp
//    reflection glints of up to 16 string "lights". The smooth part of that light (a pool under each
//    string and the glow it casts on the membrane's weave) is computed per vertex.
//  - Strings: one batched ribbon mesh. Every string is a row of a float DataTexture (256 points);
//    the vertex shader fetches neighbours, expands a screen-space ribbon and the fragment shader
//    draws a white-hot core inside a coloured halo. The same mesh is drawn twice more as mirror
//    images in each brane (wobbled by the brane's own ripple gradient) for wet reflections.
//  - Endpoints are point sprites with a faint anamorphic flare; bulk dust is drawn through a
//    thin-lens model (bokeh near the lens); the nebula sky is baked once into a cubemap.
//  - CPU: endpoint heights use a JS copy of the GLSL wave function so beads sit exactly on the sea.
//
// Sound: plucks pitched (pentatonic) by where a string lands, a watery noise for its ripples, a
// sustained additive voice for the stretched strings whose pitch and brightness rise with d, a
// rising bell and air sweep when a closed string escapes, bells and a swell when the branes merge.
import * as THREE from 'three';
import { palette, makeStarfield, GLSL, clamp, lerp, smoothstep, damp, rng } from '../fx.js';

const TAU = Math.PI * 2;
const P = 256;                 // points per string row (255 segments)
const PF = 64;                 // points per field line
const RC = 2.2;                // ripple group speed (world units / s)
const RK = 7.0;                // ripple carrier wavenumber
const RIP_LIFE = 7.0;          // seconds a ripple lives
// slow swell: [kx, kz, omega, phase, amplitude]
const SWELL = [[0.35, 0.20, 0.60, 0.0, 0.050], [-0.22, 0.41, 0.47, 1.3, 0.040], [0.61, -0.53, 0.83, 2.1, 0.022]];
const D_MAX = 4;               // largest separation, in string lengths
const VS = 1.45;               // world units per string length of separation (drawn taller for legibility)
const PENTA_SEMIS = [0, 3, 5, 7, 10, 12, 15, 17, 19, 22, 24, 27, 29, 31, 34, 36];

const g4 = (x) => (Math.round(x * 10000) / 10000).toFixed(4);
// HUD labels are uppercased; symbols keep their case inside this span (d, T·d).
const sym = (s) => `<span style="text-transform:none;letter-spacing:.02em">${s}</span>`;

// ---- GLSL ------------------------------------------------------------------------------------
// The membrane height field. Must stay in step with waveHeight() below.
const WAVE_GLSL = (NR) => /* glsl */`
#define NR ${NR}
uniform vec4 uRip[NR];     // x, z, birth time, amplitude
uniform vec3 uRipC[NR];    // crest colour
uniform float uTime, uSwell;
float waveH(vec2 p, out vec2 g, out vec3 crest){
  g = vec2(0.0); crest = vec3(0.0); float h = 0.0;
  ${SWELL.map(([kx, kz, w, ph, a]) => `{ vec2 k = vec2(${g4(kx)}, ${g4(kz)}); float q = dot(k, p) - ${g4(w)} * uTime + ${g4(ph)}; float a = ${g4(a)} * uSwell; h += a * sin(q); g += a * cos(q) * k; }`).join('\n  ')}
  for (int i = 0; i < NR; i++){
    vec4 r = uRip[i];
    float age = uTime - r.z;
    if (r.w <= 0.0 || age < 0.0 || age > ${g4(RIP_LIFE)}) continue;
    vec2 d = p - r.xy; float rr = length(d) + 1e-3;
    float u = rr - ${g4(RC)} * age; float sg = 0.45 + 0.16 * age;
    if (abs(u) > 4.0 * sg) continue;
    float e = r.w * exp(-age * 0.5 - u * u / (2.0 * sg * sg)) / sqrt(1.0 + 1.2 * rr);
    float s = sin(${g4(RK)} * u), c = cos(${g4(RK)} * u);
    h += e * s;
    g += e * (${g4(RK)} * c - u / (sg * sg) * s) * d / rr;
    crest += uRipC[i] * e * max(s, 0.0);
  }
  return h;
}`;

// Things on the far side of a brane from the camera are dimmed by that brane.
const OCCL_GLSL = /* glsl */`
uniform float uUpperY, uUpperA, uLowerA;
float occl(vec3 p){
  float a = 1.0;
  if ((p.y < -0.03) != (cameraPosition.y < 0.0)) a *= 1.0 - uLowerA;
  if (uUpperA > 0.0 && ((p.y > uUpperY + 0.03) != (cameraPosition.y > uUpperY))) a *= 1.0 - uUpperA;
  return a;
}`;

const HAZE_GLSL = /* glsl */`
uniform vec3 uHaze;
vec3 hazeAt(vec3 d){ float e = abs(d.y); return uHaze * (exp(-e * 34.0) * 1.15 + 0.16 * exp(-e * 7.0) + 0.03); }`;

// String lights shared by both brane shaders: xyz position, w strength; colour in uLC.
const LIGHTS_GLSL = (NL) => /* glsl */`
#define NL ${NL}
uniform vec4 uL[NL];
uniform vec3 uLC[NL];`;

// Vertex stage: the height field, plus the smooth, low-frequency part of the string light (a pool
// under each string and the halo that lights up the membrane's weave). Those vary over ~1 world
// unit, far wider than the mesh spacing near the strings, so they are interpolated per vertex.
const BRANE_VERT = (NR, NL) => /* glsl */`
${WAVE_GLSL(NR)}
${LIGHTS_GLSL(NL)}
uniform float uDisp;
varying vec3 vW, vPool, vLit;
void main(){
  vec4 w = modelMatrix * vec4(position, 1.0);
  vec2 g; vec3 cr;
  float h = waveH(w.xz, g, cr);
  w.y += h * uDisp * (1.0 - smoothstep(35.0, 60.0, length(w.xz)));
  vW = w.xyz;
  vec3 pool = vec3(0.0), lit = vec3(0.0);
  for (int j = 0; j < NL; j++){
    vec4 L = uL[j]; if (L.w <= 0.0) continue;
    vec3 dl = L.xyz - w.xyz;
    float r2 = dot(dl.xz, dl.xz); if (r2 > 30.0) continue;
    vec3 k = uLC[j] * (L.w * exp(-abs(dl.y) * 1.1));
    pool += k * exp(-r2 * 1.6);
    lit += k * exp(-r2 * 0.3);
  }
  vPool = pool; vLit = lit;
  gl_Position = projectionMatrix * viewMatrix * w;
}`;

const BRANE_FRAG = (NR, NL) => /* glsl */`
${WAVE_GLSL(NR)}
${HAZE_GLSL}
${GLSL.iridescence}
${LIGHTS_GLSL(NL)}
uniform vec3 uTint, uGridCol, uOtherCol, uMergeCol, uHoverCol;
uniform vec4 uHover;     // x, z, strength: where a click would drop a string
uniform float uOpacity, uEmit, uFogDen, uOtherY, uOtherOn, uMerge, uEdge, uCrest, uLightK, uNearFade, uFilm, uPoolK, uFine, uGridLit;
varying vec3 vW, vPool, vLit;
float grid(vec2 q, float s, float wdt){
  vec2 c = q / s; vec2 fw = fwidth(c);
  vec2 gd = abs(fract(c - 0.5) - 0.5) / max(fw * wdt, vec2(1e-4));
  float l = 1.0 - min(min(gd.x, gd.y), 1.0);
  return l * (1.0 - smoothstep(0.05, 0.38, max(fw.x, fw.y)));
}
void main(){
  vec3 toC = cameraPosition - vW; float dist = length(toC); vec3 V = toC / dist;
  float fog = 1.0 - exp(-pow(dist * uFogDen, 2.0));
  float edge = 1.0 - smoothstep(uEdge * 0.72, uEdge, length(vW.xz));
  if (fog > 0.985) { gl_FragColor = vec4(hazeAt(-V) * edge, uOpacity * 0.96 * edge); return; }   // lost in the haze
  vec2 g; vec3 crest;
  float h = waveH(vW.xz, g, crest);
  vec3 N = normalize(vec3(-g.x, 1.0, -g.y));
  if (dot(N, V) < 0.0) N = -N;
  float ndv = clamp(dot(N, V), 0.0, 1.0);
  float F = 0.04 + 0.96 * pow(1.0 - ndv, 5.0);
  vec3 R = reflect(-V, N);
  // the surface mirrors the hazy horizon and, if it faces it, the other brane's grid
  vec3 refl = hazeAt(R);
  float toOther = uOtherY - vW.y;
  if (uOtherOn > 0.001 && R.y * toOther > 0.0) {
    float tq = toOther / R.y;
    if (tq < 90.0) {
      vec2 Q = vW.xz + R.xz * tq;
      float og = grid(Q, 1.0, 1.0) * 0.4 + grid(Q, 5.0, 1.4) * 0.7;
      refl += uOtherCol * uOtherOn * (og + 0.02) * exp(-tq * 0.04);
    }
  }
  vec3 film = iridescence(ndv * 1.3 + h * 3.0 + uTime * 0.03 + dot(vW.xz, vec2(0.021, 0.013)));
  vec3 col = uTint * (0.18 + 0.82 * F);
  col += film * F * uFilm;
  col += refl * F * 0.75;
  vec2 q = vW.xz + g * 0.4;
  float g1 = grid(q, 1.0, 1.0), g5 = grid(q, 5.0, 1.6);
  float gl = (g1 * uFine + g5) * mix(1.0, smoothstep(1.5, 9.0, dist), uNearFade);
  col += uGridCol * gl * (0.45 + 0.55 * F);
  vec3 cr = crest * uCrest;
  col += cr / (1.0 + max(cr.r, max(cr.g, cr.b)) * 1.3);   // overlapping rings saturate softly
  // light from the strings: sharp glints of each string's reflection (per pixel), soft pools and a
  // lit weave around it (per vertex)
  vec3 acc = vec3(0.0);
  for (int j = 0; j < NL; j++){
    vec4 L = uL[j]; if (L.w <= 0.0) continue;
    float rd = dot(R, normalize(L.xyz - vW));
    if (rd < 0.86) continue;
    acc += uLC[j] * (L.w * (pow(rd, 260.0) * 1.5 + pow(rd, 40.0) * 0.022));
  }
  col += (acc * (0.25 + F) + vPool * uPoolK) * uLightK;
  float gw = grid(q, 0.5, 1.0) * 0.45 + g1 * 0.8 + g5;
  col += vLit * gw * uGridLit;
  col += uMergeCol * uMerge * (0.02 + gl * 0.1 + F * 0.06);
  if (uHover.z > 0.002) {
    float hr = length(vW.xz - uHover.xy), ph = fract(uTime * 0.7);
    float ring = exp(-pow((hr - 0.45) / 0.03, 2.0)) + 0.6 * (1.0 - ph) * exp(-pow((hr - 0.45 - 0.5 * ph) / 0.04, 2.0));
    col += uHoverCol * ring * uHover.z;
  }
  col *= uEmit;
  col = mix(col, hazeAt(-V), fog);
  float a = mix((0.38 + 0.55 * F) * uOpacity, uOpacity * 0.96, fog);
  gl_FragColor = vec4(col * edge, a * edge);
}`;

// Ribbon strings. position = (point index, row, side). Row layout in uPts: P points, then two style
// texels: (r, g, b, intensity) and (world half-width, closed?, -, min half-width in px).
// mode 0 = normal, 1 = mirror image in a brane.
const RIB_VERT = (Pn, mode, NR) => /* glsl */`
${mode === 1 ? WAVE_GLSL(NR) : 'uniform float uTime;'}
${OCCL_GLSL}
uniform sampler2D uPts;
uniform vec2 uRes;
uniform float uPlaneY, uSide, uGain, uFogDen, uFall;
varying float vV, vI, vHW; varying vec3 vCol;
void main(){
  int i = int(position.x + 0.5), r = int(position.y + 0.5); float side = position.z;
  vec4 s0 = texelFetch(uPts, ivec2(${Pn}, r), 0);
  vec4 s1 = texelFetch(uPts, ivec2(${Pn + 1}, r), 0);
  vI = 0.0; vV = side; vHW = 1.0; vCol = vec3(0.0);
  if (s0.w <= 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  bool closed = s1.y > 0.5;
  int ia = i - 1, ib = i + 1;
  if (closed) { if (ia < 0) ia = ${Pn - 2}; if (ib > ${Pn - 1}) ib = 1; }
  else { ia = max(ia, 0); ib = min(ib, ${Pn - 1}); }
  vec4 c = texelFetch(uPts, ivec2(i, r), 0);
  vec3 p = c.xyz;
  vec3 pa = texelFetch(uPts, ivec2(ia, r), 0).xyz;
  vec3 pb = texelFetch(uPts, ivec2(ib, r), 0).xyz;
  float inten = s0.w * c.w;
  ${mode === 1 ? `
  float hgt = (p.y - uPlaneY) * uSide;
  if (hgt < -0.01) inten = 0.0;
  hgt = max(hgt, 0.0);
  vec2 g; vec3 cr; waveH(p.xz, g, cr);
  vec2 wob = g * (0.25 + hgt * 1.6);
  p.y = 2.0 * uPlaneY - p.y; pa.y = 2.0 * uPlaneY - pa.y; pb.y = 2.0 * uPlaneY - pb.y;
  p.xz += wob; pa.xz += wob; pb.xz += wob;
  vec3 vv = normalize(cameraPosition - vec3(p.x, uPlaneY, p.z));
  float F = 0.12 + 0.88 * pow(1.0 - abs(vv.y), 4.0);
  inten *= uGain * F * exp(-hgt * uFall);` : `
  inten *= occl(p);`}
  float dist = length(cameraPosition - p);
  inten *= exp(-max(dist - 14.0, 0.0) * uFogDen);
  vec4 cp = projectionMatrix * viewMatrix * vec4(p, 1.0);
  vec4 ca = projectionMatrix * viewMatrix * vec4(pa, 1.0);
  vec4 cb = projectionMatrix * viewMatrix * vec4(pb, 1.0);
  float asp = uRes.x / uRes.y;
  vec2 sa = ca.xy / max(abs(ca.w), 1e-4), sb = cb.xy / max(abs(cb.w), 1e-4);
  vec2 dir = (sb - sa) * vec2(asp, 1.0);
  float ln = length(dir);
  dir = ln > 1e-6 ? dir / ln : vec2(1.0, 0.0);
  vec2 nrm = vec2(-dir.y, dir.x);
  float px = s1.x * projectionMatrix[1][1] * uRes.y * 0.5 / max(cp.w, 0.05);
  float hw = clamp(max(px, s1.w), 0.5, 80.0);
  vec2 off = nrm * side * hw * 2.0 / uRes.y; off.x /= asp;
  cp.xy += off * cp.w;
  gl_Position = cp;
  vHW = hw; vCol = s0.rgb; vI = inten;
}`;

const RIB_FRAG = /* glsl */`
uniform float uCore, uWhite;
varying float vV, vI, vHW; varying vec3 vCol;
void main(){
  if (vI <= 0.001) discard;
  float a = abs(vV);
  float pd = a * vHW;
  float core = exp(-pd * pd / (uCore * uCore));
  float halo = exp(-a * a * 4.5) * (1.0 - a);
  vec3 col = vCol * halo * 0.38 + mix(vCol, vec3(1.0), uWhite) * core;
  gl_FragColor = vec4(col * vI, 1.0);
}`;

const BEAD_VERT = /* glsl */`
${OCCL_GLSL}
attribute vec4 aCol;
attribute float aSize;
uniform float uScale;
varying vec3 vC;
void main(){
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = aSize <= 0.0 ? 0.0 : clamp(aSize * uScale / max(0.1, -mv.z), 3.0, 110.0);
  vC = aCol.rgb * aCol.a * occl(position);
}`;
// An endpoint: white-hot core, coloured halo and a faint horizontal (anamorphic) flare.
const BEAD_FRAG = /* glsl */`
varying vec3 vC;
void main(){
  vec2 p = (gl_PointCoord - 0.5) * 2.0; float r = length(p);
  float core = exp(-r * r * 90.0);
  float halo = exp(-r * r * 16.0) * (1.0 - smoothstep(0.6, 1.0, r));
  float flare = exp(-abs(p.y) * 40.0) * pow(max(0.0, 1.0 - abs(p.x)), 2.0) * 0.3;
  float k = length(vC);
  gl_FragColor = vec4(vC * (halo * 0.55 + core * 1.1 + flare) + vec3(core * 0.6 * k), 1.0);
}`;

// Dust that lives *between* the branes: its height is a fraction of d, so the bulk visibly
// flattens when the branes come together. Each mote is drawn through a thin-lens camera focused
// on the strings: its circle of confusion grows as |1/z - 1/focus|, and its light spreads over that
// disc, so motes near the lens become large faint bokeh and the frame gains a foreground.
const DUST_VERT = /* glsl */`
${OCCL_GLSL}
attribute float aSeed;
uniform float uTime, uD, uScale, uAlpha, uFocus, uCoc;
varying vec3 vC; varying float vRing;
void main(){
  float s = aSeed;
  vec3 p = position;
  p.x += sin(uTime * 0.05 + s * 31.0) * 0.8;
  p.z += cos(uTime * 0.043 + s * 17.0) * 0.8;
  p.y = mix(0.06, max(uD - 0.06, 0.06), position.y) + sin(uTime * 0.27 + s * 9.0) * 0.04 * min(uD, 1.0);
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * mv;
  float z = max(0.05, -mv.z);
  float big = step(0.965, fract(s * 7.31));
  float sharp = clamp((0.9 + 2.2 * big) * uScale * 9.0 / max(0.2, z), 0.8, 7.0);
  float coc = uCoc * uScale * abs(1.0 / z - 1.0 / uFocus);
  float sz = clamp(max(sharp, coc), 0.8, 90.0 * uScale + 4.0);
  gl_PointSize = sz;
  float tw = 0.55 + 0.45 * sin(uTime * (0.4 + s * 1.7) + s * 50.0);
  vec3 base = mix(vec3(0.42, 0.36, 0.62), vec3(0.75, 0.72, 0.85), fract(s * 3.3));
  vec3 c = mix(base * 0.45, vec3(0.45, 1.1, 1.25), big);
  float fog = exp(-max(z - 10.0, 0.0) * 0.05);
  // light conserved over the blur disc, softened so the nearest motes stay visible as bokeh
  float spread = pow(sharp / sz, 1.35);
  vRing = smoothstep(6.0, 24.0, sz);
  vC = c * tw * fog * uAlpha * occl(p) * spread * (1.0 + 2.5 * vRing) * smoothstep(0.25, 0.9, z);
}`;
const DUST_FRAG = /* glsl */`
varying vec3 vC; varying float vRing;
void main(){
  vec2 p = gl_PointCoord - 0.5; float d = length(p);
  float soft = smoothstep(0.5, 0.0, d); soft *= soft;
  float disc = smoothstep(0.5, 0.44, d) * (0.55 + 0.45 * smoothstep(0.2, 0.47, d));   // lens bokeh: flat disc, brighter rim
  gl_FragColor = vec4(vC * mix(soft, disc, vRing), 1.0);
}`;

const SKY_VERT = /* glsl */`
varying vec3 vD;
void main(){ vD = position; vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0); gl_Position = p.xyww; }`;
const SKY_FRAG = /* glsl */`
${HAZE_GLSL}
${GLSL.noise}
uniform vec3 uVoid, uNeb1, uNeb2;
uniform float uTime;
varying vec3 vD;
void main(){
  vec3 d = normalize(vD);
  float n = fbm(d * 2.2 + vec3(0.0, 0.0, uTime * 0.004));
  float m = fbm(d * 4.5 + vec3(7.1, 0.0, -uTime * 0.003));
  float up = smoothstep(-0.05, 0.6, d.y);
  vec3 neb = (uNeb1 * smoothstep(0.0, 0.7, n) + uNeb2 * smoothstep(0.15, 0.8, m) * 0.6) * up;
  gl_FragColor = vec4(uVoid + hazeAt(d) + neb, 1.0);
}`;

// ---- CPU helpers ------------------------------------------------------------------------------
/** A brane's ripple sources, shared by reference with every material that draws its waves. */
class RippleBank {
  constructor(n) {
    this.n = n;
    this.rip = Array.from({ length: n }, () => new THREE.Vector4(0, 0, -100, 0));
    this.col = Array.from({ length: n }, () => new THREE.Vector3());
  }
  /** Reuse the slot whose ripple is oldest. */
  push(x, z, t, amp, color) {
    let best = 0, bestAge = -1;
    for (let i = 0; i < this.n; i++) {
      const r = this.rip[i];
      const age = r.w <= 0 ? 1e9 : t - r.z;
      if (age > bestAge) { bestAge = age; best = i; }
    }
    this.rip[best].set(x, z, t, amp);
    this.col[best].set(color.r, color.g, color.b);
  }
  clear() { for (const r of this.rip) r.set(0, 0, -100, 0); }
}

/** JS twin of waveH() in WAVE_GLSL (height only). */
function waveHeight(x, z, t, bank, swell) {
  let h = 0;
  for (let i = 0; i < SWELL.length; i++) {
    const s = SWELL[i];
    h += s[4] * swell * Math.sin(s[0] * x + s[1] * z - s[2] * t + s[3]);
  }
  const rip = bank.rip;
  for (let i = 0; i < bank.n; i++) {
    const r = rip[i];
    const age = t - r.z;
    if (r.w <= 0 || age < 0 || age > RIP_LIFE) continue;
    const dx = x - r.x, dz = z - r.y;
    const rr = Math.sqrt(dx * dx + dz * dz) + 1e-3;
    const u = rr - RC * age, sg = 0.45 + 0.16 * age;
    if (Math.abs(u) > 4 * sg) continue;
    h += r.w * Math.exp(-age * 0.5 - (u * u) / (2 * sg * sg)) / Math.sqrt(1 + 1.2 * rr) * Math.sin(RK * u);
  }
  return h;
}

/** Ring grid for the branes: rings bunch up near the centre where the strings live. */
function polarPlane(R, rings, segs) {
  const pos = new Float32Array((1 + rings * segs) * 3);
  let k = 3;
  for (let i = 1; i <= rings; i++) {
    const r = R * Math.pow(i / rings, 1.7);
    for (let j = 0; j < segs; j++) {
      const a = (j / segs) * TAU;
      pos[k++] = Math.cos(a) * r; pos[k++] = 0; pos[k++] = Math.sin(a) * r;
    }
  }
  const idx = [];
  for (let j = 0; j < segs; j++) idx.push(0, 1 + j, 1 + ((j + 1) % segs));
  for (let i = 0; i < rings - 1; i++) {
    for (let j = 0; j < segs; j++) {
      const a = 1 + i * segs + j, b = 1 + i * segs + ((j + 1) % segs), c = a + segs, d = b + segs;
      idx.push(a, c, b, b, c, d);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setIndex(idx);
  return geo;
}

/** A batch of screen-space ribbons whose points live in a float texture (one row per string). */
class RibbonBatch {
  constructor(rows, Pn) {
    this.rows = rows; this.Pn = Pn; this.W = Pn + 2;
    this.data = new Float32Array(this.W * rows * 4);
    this.tex = new THREE.DataTexture(this.data, this.W, rows, THREE.RGBAFormat, THREE.FloatType);
    this.tex.minFilter = this.tex.magFilter = THREE.NearestFilter;
    this.tex.generateMipmaps = false;
    this.tex.needsUpdate = true;
    const nV = rows * Pn * 2;
    const meta = new Float32Array(nV * 3);
    let k = 0;
    for (let r = 0; r < rows; r++) for (let i = 0; i < Pn; i++) for (let s = -1; s <= 1; s += 2) { meta[k++] = i; meta[k++] = r; meta[k++] = s; }
    const idx = new Uint16Array(rows * (Pn - 1) * 6);
    k = 0;
    for (let r = 0; r < rows; r++) {
      for (let i = 0; i < Pn - 1; i++) {
        const a = (r * Pn + i) * 2;
        idx[k++] = a; idx[k++] = a + 1; idx[k++] = a + 2;
        idx[k++] = a + 1; idx[k++] = a + 3; idx[k++] = a + 2;
      }
    }
    this.geo = new THREE.BufferGeometry();
    this.geo.setAttribute('position', new THREE.BufferAttribute(meta, 3));
    this.geo.setIndex(new THREE.BufferAttribute(idx, 1));
    this.geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e4);
  }
  style(row, c, inten, halfW, closed, minPx) {
    const o = (row * this.W + this.Pn) * 4, d = this.data;
    d[o] = c.r; d[o + 1] = c.g; d[o + 2] = c.b; d[o + 3] = inten;
    d[o + 4] = halfW; d[o + 5] = closed ? 1 : 0; d[o + 6] = 0; d[o + 7] = minPx;
  }
  hide(row) { this.data[(row * this.W + this.Pn) * 4 + 3] = 0; }
  /** Offset of point i of a row (write x, y, z, w = per-point intensity). */
  at(row, i) { return (row * this.W + i) * 4; }
}

const easeOut3 = (x) => 1 - Math.pow(1 - clamp(x), 3);
const easeInOut = (x) => { x = clamp(x); return x * x * (3 - 2 * x); };

// ---- the movement ---------------------------------------------------------------------------------
export default {
  id: 'branes',
  numeral: 'V',
  title: 'Branes',
  tempo: 'Andante, sul ponticello',
  caption: 'Some strings are open, with two ends. Those ends cannot float free: they are stuck to membranes called <em>branes</em>. Closed loops have no ends, so they can leave. Our universe may be one brane, with only <em>gravity</em> able to escape.',
  fine: '<p>A <em>D-brane</em> is a surface where open strings may end. Their endpoints obey Dirichlet boundary conditions across the brane, so they slide along it and cannot leave. In 1995 Joseph Polchinski showed that these surfaces are dynamical objects in string theory, with mass and charge of their own. The lowest vibrations of these open strings are gauge fields like the photon: one brane carries U(1), N coincident branes carry U(N). The ripples are the brane itself moving, and those motions are open-string states too.</p>'
    + '<p>A string stretched between branes a distance d apart has mass T·d, with tension T = 1/(2πα′). With the string length ℓ<sub>s</sub> = √α′ and string mass M<sub>s</sub> = 1/√α′, that is d/2π in units of M<sub>s</sub> when d is counted in ℓ<sub>s</sub>. Its two orientations behave like the W bosons of a Higgs mechanism, with d playing the role of the Higgs field\'s value. At d = 0 they become massless and U(1) × U(1) grows into U(2).</p>'
    + '<p>Closed strings include the graviton (spin 2), and no brane holds them. In braneworld models the other forces stay on the brane while gravity also reaches into the bulk. With large extra dimensions (Arkani-Hamed, Dimopoulos and Dvali, 1998) it spreads out through them. With one warped extra dimension (Randall and Sundrum, 1999) it is concentrated away from our brane. Either could explain why gravity is about 10³⁶ times weaker than the electric force between two protons. Torsion-balance tests find Newton\'s law intact down to about 50 μm and the LHC has seen no sign of extra dimensions, so these ideas, like string theory itself, remain unconfirmed.</p>'
    + '<p>Simplified here: each brane is drawn as a 2D sheet standing in for our 3D space, the bulk shows one extra dimension instead of six, and sizes, speeds and the field-line picture are schematic.</p>',
  drone: { notes: [0, 5, 12, 17], opts: { gain: 0.085, cutoff: 1050, wobble: 0.35 } },
  post: { bloomStrength: 0.7, bloomRadius: 0.18, bloomThreshold: 0.75, exposure: 1.0, aberration: 0.0016, grain: 0.03, vignette: 0.95 },

  create(ctx) {
    const LOW = ctx.quality === 'low';
    const A = ctx.audio;
    const params = new URLSearchParams(location.search);
    const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const R = rng(5150);
    const DBG = params.get('b5dbg') || '';
    const HOVER_DBG = params.has('b5hover') ? new THREE.Vector2(...params.get('b5hover').split(',').map(Number)) : null;   // stage a mouse hover at this NDC

    // pools & sizes
    const NR_LO = LOW ? 8 : 16, NR_UP = LOW ? 6 : 10, NL = LOW ? 8 : 16;
    const N_OPEN = 14, N_UPPER = 6, N_STRETCH = 8, N_CLOSED = 6;
    const ROW_OPEN = 0, ROW_UPPER = N_OPEN, ROW_STRETCH = ROW_UPPER + N_UPPER, ROW_CLOSED = ROW_STRETCH + N_STRETCH;
    const ROWS = ROW_CLOSED + N_CLOSED;
    const N_GAUGE = LOW ? 16 : 24, N_GRAV = LOW ? 20 : 34;
    const N_BEADS = (N_OPEN + N_UPPER + N_STRETCH) * 2 + 2;

    // colours (linear)
    const C = {
      open: palette.harmonic.clone(), open2: new THREE.Color('#5fb8ff'), upper: palette.violet.clone(),
      closed: palette.resonance.clone(), ember: palette.ember.clone(), gold: palette.planck.clone(),
      u2: palette.mint.clone().lerp(palette.harmonic, 0.35),
      ripCyan: palette.harmonic.clone().multiplyScalar(0.55), ripMag: palette.resonance.clone().multiplyScalar(0.7),
      ripViolet: palette.violet.clone().multiplyScalar(0.6), ripEmber: palette.ember.clone().multiplyScalar(0.6),
      ripMint: palette.mint.clone().multiplyScalar(0.6),
    };
    const HAZE = new THREE.Color(0.06, 0.03, 0.12);

    // ---- scene, camera ----
    const scene = new THREE.Scene();
    scene.background = palette.void.clone();
    const camera = new THREE.PerspectiveCamera(40, ctx.aspect, 0.05, 400);

    // The sky (void, horizon haze and a faint fbm nebula) is baked once into a cubemap, so the frame
    // pays one texture lookup per pixel for it instead of ten noise evaluations.
    const skyRT = new THREE.WebGLCubeRenderTarget(LOW ? 128 : 256, { type: THREE.HalfFloatType, generateMipmaps: false });
    {
      const bake = new THREE.Scene();
      const mat = new THREE.ShaderMaterial({
        vertexShader: SKY_VERT, fragmentShader: SKY_FRAG, side: THREE.BackSide, depthWrite: false, depthTest: false,
        uniforms: { uHaze: { value: HAZE }, uVoid: { value: palette.void.clone() }, uTime: { value: 0 },
          uNeb1: { value: new THREE.Color(0.030, 0.012, 0.060) }, uNeb2: { value: new THREE.Color(0.045, 0.008, 0.035) } },
      });
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(10, 48, 24), mat);
      mesh.frustumCulled = false;
      bake.add(mesh);
      try { new THREE.CubeCamera(0.1, 50, skyRT).update(ctx.renderer, bake); scene.background = skyRT.texture; } catch (e) { /* keep the flat void */ }
      mesh.geometry.dispose(); mat.dispose();
    }

    const stars = makeStarfield({ count: LOW ? 1400 : 2800, radius: 150, inner: 100, size: 1.5, intensity: 1.1 });
    stars.renderOrder = -5;
    scene.add(stars);

    // ---- branes ----
    const ripLo = new RippleBank(NR_LO), ripUp = new RippleBank(NR_UP);
    const braneGeo = polarPlane(90, LOW ? 110 : 180, LOW ? 160 : 256);
    const lightsLo = { L: Array.from({ length: NL }, () => new THREE.Vector4()), C: Array.from({ length: NL }, () => new THREE.Color()) };
    const lightsUp = { L: Array.from({ length: NL }, () => new THREE.Vector4()), C: Array.from({ length: NL }, () => new THREE.Color()) };
    const time = { value: 0 };
    function braneMat(bank, lights, o) {
      return new THREE.ShaderMaterial({
        vertexShader: BRANE_VERT(bank.n, NL), fragmentShader: BRANE_FRAG(bank.n, NL),
        transparent: true, depthWrite: false, side: THREE.DoubleSide,
        blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
        uniforms: {
          uTime: time, uSwell: { value: o.swell }, uRip: { value: bank.rip }, uRipC: { value: bank.col }, uDisp: { value: 1 },
          uL: { value: lights.L }, uLC: { value: lights.C }, uHaze: { value: HAZE },
          uTint: { value: o.tint }, uGridCol: { value: o.grid }, uOtherCol: { value: o.other }, uMergeCol: { value: palette.mint.clone().multiplyScalar(0.5) },
          uOpacity: { value: o.opacity }, uEmit: { value: 1 }, uFogDen: { value: o.fog }, uOtherY: { value: 0 }, uOtherOn: { value: 1 },
          uMerge: { value: 0 }, uEdge: { value: 88 }, uCrest: { value: o.crest }, uLightK: { value: o.lightK }, uNearFade: { value: o.nearFade }, uFilm: { value: o.film }, uPoolK: { value: o.pool }, uFine: { value: o.fine }, uGridLit: { value: o.gridLit }, uHover: { value: new THREE.Vector4() }, uHoverCol: { value: o.hover },
        },
      });
    }
    const loMat = braneMat(ripLo, lightsLo, {
      swell: 1, tint: new THREE.Color(0.008, 0.016, 0.04), grid: new THREE.Color(0.016, 0.06, 0.08), other: new THREE.Color(0.12, 0.08, 0.26),
      opacity: 1, fog: 0.034, crest: 5.0, lightK: 1, nearFade: 0.55, film: 0.2, pool: 0.22, fine: 0.5, gridLit: 0.4, hover: palette.harmonic.clone().multiplyScalar(0.9),
    });
    const upMat = braneMat(ripUp, lightsUp, {
      swell: 0.8, tint: new THREE.Color(0.02, 0.012, 0.045), grid: new THREE.Color(0.1, 0.07, 0.22), other: new THREE.Color(0.03, 0.1, 0.13),
      opacity: 0.32, fog: 0.04, crest: 5.0, lightK: 0.16, nearFade: 1, film: 0.05, pool: 0.05, fine: 0.22, gridLit: 1.1, hover: palette.violet.clone().multiplyScalar(0.9),
    });
    const lower = new THREE.Mesh(braneGeo, loMat);
    const upper = new THREE.Mesh(braneGeo, upMat);
    lower.renderOrder = 2; upper.renderOrder = 4;
    lower.frustumCulled = upper.frustumCulled = false;
    scene.add(lower, upper);

    // ---- strings (ribbons) ----
    const rib = new RibbonBatch(ROWS, P);
    const res = new THREE.Vector2(1, 1);
    const occl = { uUpperY: { value: 2 }, uUpperA: { value: 0.35 }, uLowerA: { value: 0.55 } };   // shared by every occlusion-aware material
    function ribMat(batch, mode, bank, o = {}) {
      const u = {
        uPts: { value: batch.tex }, uRes: { value: res }, uTime: time, uCore: { value: 1.5 }, uWhite: { value: o.white ?? 0.62 },
        uPlaneY: { value: 0 }, uSide: { value: 1 }, uGain: { value: o.gain ?? 1 }, uFogDen: { value: 0.035 }, uFall: { value: o.fall ?? 0.5 },
        uUpperY: occl.uUpperY, uUpperA: occl.uUpperA, uLowerA: occl.uLowerA,
      };
      if (mode === 1) { u.uRip = { value: bank.rip }; u.uRipC = { value: bank.col }; u.uSwell = { value: bank === ripLo ? 1 : 0.8 }; }
      return new THREE.ShaderMaterial({
        vertexShader: RIB_VERT(batch.Pn, mode, bank ? bank.n : 1), fragmentShader: RIB_FRAG, uniforms: u,
        transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
      });
    }
    const strings = new THREE.Mesh(rib.geo, ribMat(rib, 0, null));
    const mirLo = new THREE.Mesh(rib.geo, ribMat(rib, 1, ripLo, { gain: 0.22, white: 0.12, fall: 1.0 }));
    const mirUp = new THREE.Mesh(rib.geo, ribMat(rib, 1, ripUp, { gain: 0.06, white: 0.1, fall: 2.4 }));
    mirUp.material.uniforms.uSide.value = -1;
    strings.renderOrder = 7; mirLo.renderOrder = 3; mirUp.renderOrder = 5;
    for (const m of [strings, mirLo, mirUp]) { m.frustumCulled = false; scene.add(m); }

    // field lines: a second, finer batch
    const fld = new RibbonBatch(N_GAUGE + N_GRAV, PF);
    const fieldMesh = new THREE.Mesh(fld.geo, ribMat(fld, 0, null, { white: 0.5 }));
    fieldMesh.renderOrder = 8; fieldMesh.frustumCulled = false; fieldMesh.visible = false;
    scene.add(fieldMesh);
    for (let r = 0; r < fld.rows; r++) fld.hide(r);

    // ---- beads (string endpoints) ----
    const beadPos = new Float32Array(N_BEADS * 3), beadCol = new Float32Array(N_BEADS * 4), beadSize = new Float32Array(N_BEADS);
    const beadGeo = new THREE.BufferGeometry();
    beadGeo.setAttribute('position', new THREE.BufferAttribute(beadPos, 3).setUsage(THREE.DynamicDrawUsage));
    beadGeo.setAttribute('aCol', new THREE.BufferAttribute(beadCol, 4).setUsage(THREE.DynamicDrawUsage));
    beadGeo.setAttribute('aSize', new THREE.BufferAttribute(beadSize, 1).setUsage(THREE.DynamicDrawUsage));
    const beadMat = new THREE.ShaderMaterial({
      vertexShader: BEAD_VERT, fragmentShader: BEAD_FRAG, transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
      uniforms: { uScale: { value: 1 }, uUpperY: occl.uUpperY, uUpperA: occl.uUpperA, uLowerA: occl.uLowerA },
    });
    const beads = new THREE.Points(beadGeo, beadMat);
    beads.renderOrder = 9; beads.frustumCulled = false;
    scene.add(beads);
    let beadN = 0;
    function bead(x, y, z, c, inten, size) {
      if (beadN >= N_BEADS) return;
      const i = beadN++;
      beadPos[i * 3] = x; beadPos[i * 3 + 1] = y; beadPos[i * 3 + 2] = z;
      beadCol[i * 4] = c.r; beadCol[i * 4 + 1] = c.g; beadCol[i * 4 + 2] = c.b; beadCol[i * 4 + 3] = inten;
      beadSize[i] = size;
    }

    // ---- dust between the branes ----
    const ND = LOW ? 1100 : 2400;
    const dPos = new Float32Array(ND * 3), dSeed = new Float32Array(ND);
    for (let i = 0; i < ND; i++) {
      dPos[i * 3] = (R() - 0.5) * 46; dPos[i * 3 + 1] = R(); dPos[i * 3 + 2] = 12 - R() * 52; dSeed[i] = R();
    }
    const dustGeo = new THREE.BufferGeometry();
    dustGeo.setAttribute('position', new THREE.BufferAttribute(dPos, 3));
    dustGeo.setAttribute('aSeed', new THREE.BufferAttribute(dSeed, 1));
    const dustMat = new THREE.ShaderMaterial({
      vertexShader: DUST_VERT, fragmentShader: DUST_FRAG, transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
      uniforms: { uTime: time, uD: { value: 2 }, uScale: { value: 1 }, uAlpha: { value: 1 }, uFocus: { value: 14 }, uCoc: { value: 70 }, uUpperY: occl.uUpperY, uUpperA: occl.uUpperA, uLowerA: occl.uLowerA },
    });
    const dust = new THREE.Points(dustGeo, dustMat);
    dust.renderOrder = 6; dust.frustumCulled = false;
    scene.add(dust);

    // ---- tables ----
    const SIN = new Float32Array(P * 4);          // sin(nπs), n = 1..4
    const CS = new Float32Array(P * 8);           // cos/sin(2πnσ), n = 1..4
    for (let i = 0; i < P; i++) {
      const s = i / (P - 1);
      for (let n = 0; n < 4; n++) {
        SIN[i * 4 + n] = Math.sin((n + 1) * Math.PI * s);
        CS[i * 8 + n * 2] = Math.cos((n + 1) * TAU * s); CS[i * 8 + n * 2 + 1] = Math.sin((n + 1) * TAU * s);
      }
    }

    // ---- state ----
    const st = {
      d: 2.4, dT: 2.4, dPrev: 2.4,       // separation now (eased) / target / last real separation
      dw: 2.4 * VS,                      // the same separation in world units
      upVis: 1, mk: 0, merged: false, mergeFlash: 0,
      escaped: 0, hintStage: 0,
      field: false, fieldK: 0,
      pendingCollide: 0, liftRip: 0, liftX: 0, liftZ: 0, ff: false, tOff: 0,
      entered: false, t: 0, now: 0,
    };
    if (params.has('b5d')) { st.d = st.dT = clamp(parseFloat(params.get('b5d')) || 0, 0, D_MAX); if (st.d >= 1) st.dPrev = st.d; }
    if (params.has('b5field')) { st.field = true; st.fieldK = 1; }

    const hLo = (x, z) => waveHeight(x, z, st.now, ripLo, 1);
    // the upper sheet sits at d; as it merges into ours it adopts our waves
    const hUp = (x, z) => st.dw + lerp(waveHeight(x, z, st.now, ripLo, 1), waveHeight(x, z, st.now, ripUp, 0.8), st.upVis);
    const upperBank = () => (st.upVis > 0.5 ? ripUp : ripLo);

    // open strings: ends on a brane, arc in the bulk
    function makeOpen(row, brane) {
      return {
        row, brane, alive: false, state: 0, k: 0, age: 0, fade: 1, retire: false, loud: false,
        mx: 0, mz: 0, rot: 0, span: 1, hx: 0, hz: 0, H: 0.6, lean: 0, seed: 0, wa: 0,
        ax: 0, ay: 0, az: 0, bx: 0, by: 0, bz: 0, sx: 0, sy: 0, sz: 0,
        amp: new Float32Array(4), amp0: new Float32Array(4), ph: new Float32Array(4), w1: 5.5,
        col: new THREE.Color(), inten: 2.6, flash: 0, nextRip: 1,
      };
    }
    const opens = Array.from({ length: N_OPEN }, (_, i) => makeOpen(ROW_OPEN + i, 0));
    const uppers = Array.from({ length: N_UPPER }, (_, i) => makeOpen(ROW_UPPER + i, 1));

    function makeStretch(row) {
      return {
        row, alive: false, age: 0, grow: 0, fade: 1, retire: false, connected: false, loud: false,
        lx: 0, lz: 0, hx: 0, hz: 0, oa: 0, seed: 0, wa: 0, dir: 1,
        amp: new Float32Array(3), ph: new Float32Array(3),
        x0: 0, y0: 0, z0: 0, x1: 0, y1: 0, z1: 0,
      };
    }
    const stretches = Array.from({ length: N_STRETCH }, (_, i) => makeStretch(ROW_STRETCH + i));

    function makeClosed(i) {
      return {
        row: ROW_CLOSED + i, alive: false, age: 0, life: 12,
        cx: 0, cy: 0, cz: 0, vx: 0, vy: 0, vz: 0,
        base: new Float32Array(P * 3), e1: new THREE.Vector3(), e2: new THREE.Vector3(), nrm: new THREE.Vector3(),
        rad: 0.6, q: new THREE.Quaternion(), axis: new THREE.Vector3(0, 1, 0), spin: 0, spinRate: 0.4,
        aL: new Float32Array(4), aR: new Float32Array(4), ph: new Float32Array(8), w1: 3,
      };
    }
    const closeds = Array.from({ length: N_CLOSED }, (_, i) => makeClosed(i));
    for (let r = 0; r < ROWS; r++) rib.hide(r);

    const join = { on: false, a: null, b: null, t: 0, dur: 1.35, m1x: 0, m1z: 0, m2x: 0, m2z: 0, a0: new Float32Array(4), b0: new Float32Array(4), aH0: 0, bH0: 0, aL0: 0, bL0: 0, Ht: 0.7 };

    // scratch
    const M4 = new Float32Array(4), L4 = new Float32Array(4), TH = new Float32Array(16);
    const v3 = new THREE.Vector3(), v3b = new THREE.Vector3(), look = new THREE.Vector3();
    const tmpCol = new THREE.Color(), heat = new THREE.Color();
    const ray = new THREE.Raycaster();
    const timers = [];
    const schedule = (delay, fn) => timers.push({ at: st.t + delay, fn });

    // ---- open strings ----
    function liveCount(list) { let n = 0; for (const s of list) if (s.alive && !s.retire) n++; return n; }
    function allocOpen(brane) {
      const list = brane === 0 ? opens : uppers, cap = brane === 0 ? 12 : 4;
      if (liveCount(list) >= cap) {        // retire the oldest settled string gracefully
        let old = null;
        for (const s of list) if (s.alive && !s.retire && s.state === 2 && (!old || s.age > old.age)) old = s;
        if (old) old.retire = true;
      }
      let free = null;
      for (const s of list) if (!s.alive) { free = s; break; }
      if (!free) {                        // all slots busy: reuse the most faded one
        for (const s of list) if (s.retire && s.state !== 3 && (!free || s.fade < free.fade)) free = s;
      }
      return free;
    }

    function spawnOpen(brane, x, z, user, settled = false) {
      const s = allocOpen(brane); if (!s) return null;
      const r2 = Math.hypot(x, z); if (r2 > 24) { x *= 24 / r2; z *= 24 / r2; }
      s.alive = true; s.retire = false; s.fade = 1; s.age = 0; s.loud = !!user;
      s.seed = R() * 10; s.wa = R() * TAU; s.rot = R() * Math.PI; s.span = 0.9 + R() * 0.9;
      s.mx = s.hx = x; s.mz = s.hz = z;
      s.H = 0.38 + s.span * 0.32; s.lean = (R() - 0.5) * 0.7;
      s.w1 = 4.6 + R() * 1.8;
      for (let n = 0; n < 4; n++) { s.amp0[n] = 0.022 / (n + 1); s.amp[n] = s.amp0[n]; s.ph[n] = R() * TAU; }
      if (brane === 0) { s.col.copy(C.open).lerp(C.open2, R() * 0.6); s.inten = 2.1; }
      else { s.col.copy(C.upper).lerp(C.closed, 0.12); s.inten = 1.6; }
      s.flash = 0; s.nextRip = 0.4 + R() * 2.2;
      const c = Math.cos(s.rot) * s.span * 0.5, sn = Math.sin(s.rot) * s.span * 0.5;
      s.ax = x - c; s.az = z - sn; s.bx = x + c; s.bz = z + sn;
      if (settled) { s.state = 2; s.k = 1; s.age = R() * 20; }
      else {
        s.state = 1; s.k = 0;
        s.sx = x; s.sz = z;
        s.sy = brane === 0 ? Math.min(1.7, Math.max(0.5, st.dw * 0.5)) : st.dw - Math.min(1.5, st.dw * 0.4);
      }
      return s;
    }

    function land(s) {
      s.state = 2; s.k = 1;
      const bank = s.brane === 0 ? ripLo : upperBank();
      const col = s.brane === 0 ? C.ripCyan : C.ripViolet;
      const amp = s.loud ? 0.07 : 0.05;
      bank.push(s.ax, s.az, st.now, amp, col);
      bank.push(s.bx, s.bz, st.now + 0.03, amp * 0.9, col);
      s.amp[0] = 0.15 + R() * 0.06; s.amp[1] = 0.07 + R() * 0.04; s.amp[2] = 0.04; s.amp[3] = 0.025;
      s.flash = 1;
      if (s.loud) pluckFor(s.mx, s.mz, s.brane);
    }

    function stepOpen(s, dt, t) {
      if (!s.alive) return;
      s.age += dt;
      if (s.retire) { s.fade -= dt / 0.7; if (s.fade <= 0) { s.alive = false; rib.hide(s.row); return; } }
      const up = s.brane === 1;
      if (s.state !== 3) {
        s.wa += Math.sin(t * 0.31 + s.seed * 17) * 0.7 * dt;
        s.mx += (Math.cos(s.wa) * 0.16 + (s.hx - s.mx) * 0.06) * dt;
        s.mz += (Math.sin(s.wa) * 0.16 + (s.hz - s.mz) * 0.06) * dt;
        s.rot += Math.sin(t * 0.23 + s.seed * 5) * 0.12 * dt;
        const span = s.span * (1 + 0.14 * Math.sin(t * 0.5 + s.seed * 3));
        const c = Math.cos(s.rot) * span * 0.5, sn = Math.sin(s.rot) * span * 0.5;
        s.ax = s.mx - c; s.az = s.mz - sn; s.bx = s.mx + c; s.bz = s.mz + sn;
      }
      s.ay = up ? hUp(s.ax, s.az) : hLo(s.ax, s.az);
      s.by = up ? hUp(s.bx, s.bz) : hLo(s.bx, s.bz);
      let e = 1;
      if (s.state === 1) { s.k += dt / 0.6; e = easeOut3(s.k); if (s.k >= 1) { land(s); e = 1; } }
      const kd = Math.exp(-(s.state === 3 ? 2.5 : 0.55) * dt);
      for (let n = 0; n < 4; n++) s.amp[n] = s.amp0[n] + (s.amp[n] - s.amp0[n]) * kd;
      s.flash *= Math.exp(-2.5 * dt);
      if (s.state === 2 && !s.retire) {
        s.nextRip -= dt;
        if (s.nextRip <= 0) {
          s.nextRip = 2.2 + R() * 2.8;
          const end = R() < 0.5;
          (up ? upperBank() : ripLo).push(end ? s.ax : s.bx, end ? s.az : s.bz, t, 0.035 + 0.025 * R(), up ? C.ripViolet : C.ripCyan);
        }
      }
      writeOpen(s, t, e);
    }

    function writeOpen(s, t, e) {
      const sg = s.brane === 1 ? 1 - st.upVis * 2 : 1;     // upper-brane strings hang down until the branes merge
      const dx = s.bx - s.ax, dz = s.bz - s.az, dy = s.by - s.ay;
      const L = Math.hypot(dx, dz) || 1;
      const px = -dz / L, pz = dx / L;
      const sl = Math.sin(s.lean), cl = Math.cos(s.lean);
      const ux = px * sl, uy = sg * cl, uz = pz * sl;     // arc direction
      const lx = px * cl, ly = -sg * sl, lz = pz * cl;    // lateral, orthogonal to it
      for (let n = 0; n < 4; n++) {
        const w = s.w1 * (n + 1);
        M4[n] = s.amp[n] * Math.cos(w * t + s.ph[n]);
        L4[n] = s.amp[n] * 0.6 * Math.sin(w * 1.03 * t + s.ph[n] * 1.7);
      }
      const H = s.H;
      const inten = s.inten * Math.max(0, s.fade) * (1 + s.flash * 0.9);
      // as the branes coincide every open string joins one U(2) family: their colours converge
      tmpCol.copy(s.col).lerp(C.u2, st.mk * (s.brane === 1 ? 0.75 : 0.45));
      rib.style(s.row, tmpCol, inten, s.brane === 0 ? 0.045 : 0.04, false, 2.0 * ctx.pixelRatio);
      const d = rib.data;
      const mxF = (s.ax + s.bx) * 0.5, myF = (s.ay + s.by) * 0.5, mzF = (s.az + s.bz) * 0.5;
      for (let i = 0; i < P; i++) {
        const S = i * 4, sv = i / (P - 1);
        const v = (H + M4[0]) * SIN[S] + M4[1] * SIN[S + 1] + M4[2] * SIN[S + 2] + M4[3] * SIN[S + 3];
        const la = L4[0] * SIN[S] + L4[1] * SIN[S + 1] + L4[2] * SIN[S + 2] + L4[3] * SIN[S + 3];
        let x = s.ax + dx * sv + ux * v + lx * la;
        let y = s.ay + dy * sv + uy * v + ly * la;
        let z = s.az + dz * sv + uz * v + lz * la;
        if (e < 1) {   // falling in from the bulk: a small curl that unfurls as it lands
          const x0 = s.sx + (x - mxF) * 0.12, y0 = s.sy + (y - myF) * 0.12, z0 = s.sz + (z - mzF) * 0.12;
          x = x0 + (x - x0) * e; y = y0 + (y - y0) * e; z = z0 + (z - z0) * e;
        }
        const o = rib.at(s.row, i);
        d[o] = x; d[o + 1] = y; d[o + 2] = z; d[o + 3] = 1;
      }
    }

    // ---- stretched strings ----
    function heatColor(dd, out) {
      const h = clamp(dd / 3.2);
      if (h < 0.5) out.copy(C.open).lerp(C.gold, h * 2); else out.copy(C.gold).lerp(C.ember, (h - 0.5) * 2);
      return out.lerp(C.u2, 1 - smoothstep(0.03, 0.4, dd));
    }

    function addStretched(user, x, z) {
      let s = null;
      if (liveCount(stretches) >= 6) {
        let old = null;
        for (const q of stretches) if (q.alive && !q.retire && (!old || q.age > old.age)) old = q;
        if (old) old.retire = true;
      }
      for (const q of stretches) if (!q.alive) { s = q; break; }
      if (!s) for (const q of stretches) if (q.retire && (!s || q.fade < s.fade)) s = q;
      if (!s) return null;
      if (x === undefined) { x = -3.5 + R() * 7.5; z = -6.5 + R() * 6; }
      s.alive = true; s.retire = false; s.fade = 1; s.age = 0; s.grow = user ? 0 : 1; s.connected = !user; s.loud = !!user;
      s.lx = s.hx = x; s.lz = s.hz = z; s.oa = R() * TAU; s.seed = R() * 10; s.wa = R() * TAU; s.dir = R() < 0.5 ? 1 : -1;
      for (let n = 0; n < 3; n++) { s.amp[n] = (user ? 0.12 : 0.04) / (n + 1); s.ph[n] = R() * TAU; }
      if (user) {
        ripLo.push(x, z, st.now, 0.08, C.ripEmber);
        if (audible() && st.t - snd.pluck > 0.08) {
          snd.pluck = st.t;
          const f = stretchFreq(st.d);
          A.pluck(f * 2, { gain: 0.16, bright: 0.9, decay: 2.6, partials: 12, pan: panOf(x, 0, z) });
          A.noise({ freq: 2600, q: 3, dur: 0.22, gain: 0.025, pan: panOf(x, 0, z) });
        }
      }
      return s;
    }

    function stepStretch(s, dt, t) {
      if (!s.alive) return;
      s.age += dt;
      if (s.retire) { s.fade -= dt / 0.8; if (s.fade <= 0) { s.alive = false; rib.hide(s.row); return; } }
      if (s.grow < 1) {
        s.grow = Math.min(1, s.grow + dt / 0.9);
        if (s.grow >= 1 && !s.connected) {
          s.connected = true;
          if (st.upVis > 0.3) ripUp.push(s.x1, s.z1, t, 0.08, C.ripEmber); else ripLo.push(s.x1, s.z1, t, 0.06, C.ripMint);
          if (s.loud && audible()) A.pluck(stretchFreq(st.d) * 3, { gain: 0.08, bright: 0.8, decay: 1.8, partials: 6, pan: panOf(s.x1, s.y1, s.z1) });
        }
      }
      s.wa += Math.sin(t * 0.27 + s.seed * 13) * 0.5 * dt;
      s.lx += (Math.cos(s.wa) * 0.07 + (s.hx - s.lx) * 0.05) * dt;
      s.lz += (Math.sin(s.wa) * 0.07 + (s.hz - s.lz) * 0.05) * dt;
      s.oa += Math.sin(t * 0.19 + s.seed) * 0.08 * dt;
      const dd = st.d;
      const off = lerp(1.05, 0.22, smoothstep(0.0, 0.9, dd));
      s.x0 = s.lx; s.z0 = s.lz; s.y0 = hLo(s.x0, s.z0);
      s.x1 = s.lx + Math.cos(s.oa) * off; s.z1 = s.lz + Math.sin(s.oa) * off; s.y1 = hUp(s.x1, s.z1);
      const ex = s.x1 - s.x0, ey = s.y1 - s.y0, ez = s.z1 - s.z0;
      const L = Math.hypot(ex, ey, ez) || 1e-3;
      const arcH = 0.5 * (1 - smoothstep(0.04, 0.8, dd));
      // a relativistic string's modes go as nπ/L: long strings swing slowly, short ones fast
      const w1 = clamp((Math.PI * 1.7) / Math.max(L, 0.35), 1.4, 9);
      const kd = Math.exp(-0.7 * dt);
      for (let n = 0; n < 3; n++) { const a0 = 0.075 / (n + 1); s.amp[n] = a0 + (s.amp[n] - a0) * kd; }
      // two directions across the string
      v3.set(ex / L, ey / L, ez / L);
      v3b.set(-v3.z, 0, v3.x);
      if (v3b.lengthSq() < 1e-4) v3b.set(1, 0, 0);
      v3b.normalize();
      const e1x = v3b.x, e1z = v3b.z;
      const e2x = v3.y * e1z, e2y = v3.z * e1x - v3.x * e1z, e2z = -v3.y * e1x;      // v3 × e1
      let m1 = 0, m2 = 0, m3 = 0, n1 = 0, n2 = 0, n3 = 0;
      const lenAmp = 0.6 + 0.25 * L;
      m1 = s.amp[0] * lenAmp * Math.cos(w1 * t + s.ph[0]); n1 = s.amp[0] * lenAmp * Math.sin(w1 * 0.97 * t + s.ph[0] * 2);
      m2 = s.amp[1] * lenAmp * Math.cos(2 * w1 * t + s.ph[1]); n2 = s.amp[1] * lenAmp * Math.sin(2 * w1 * 1.01 * t + s.ph[1] * 2);
      m3 = s.amp[2] * lenAmp * Math.cos(3 * w1 * t + s.ph[2]); n3 = s.amp[2] * lenAmp * Math.sin(3 * w1 * 0.99 * t + s.ph[2] * 2);
      const g = easeOut3(s.grow);
      heatColor(dd, heat);
      const inten = (1.6 + 0.6 * dd) * s.fade * (0.4 + 0.6 * g);
      rib.style(s.row, heat, inten, 0.02 + 0.015 * dd, false, (1.6 + 0.4 * dd) * ctx.pixelRatio);
      let pp = (t * 0.42 + s.seed) % 1; if (s.dir < 0) pp = 1 - pp;
      const d = rib.data;
      for (let i = 0; i < P; i++) {
        const S = i * 4, sv = (i / (P - 1));
        const a = SIN[S] * m1 + SIN[S + 1] * m2 + SIN[S + 2] * m3;
        const b = SIN[S] * n1 + SIN[S + 1] * n2 + SIN[S + 2] * n3;
        const u = sv * g;
        const o = rib.at(s.row, i);
        d[o] = s.x0 + ex * u + e1x * a + e2x * b;
        d[o + 1] = s.y0 + ey * u + arcH * SIN[S] + e2y * b;
        d[o + 2] = s.z0 + ez * u + e1z * a + e2z * b;
        const q = sv - pp;
        d[o + 3] = 0.85 + 1.3 * Math.exp(-(q * q) / 0.0035);
      }
    }

    // ---- closed strings ----
    function allocClosed() {
      let c = null;
      for (const q of closeds) if (!q.alive) { c = q; break; }
      if (!c) for (const q of closeds) if (!c || q.age > c.age) c = q;
      return c;
    }
    function launchClosed(c, up) {
      c.alive = true; c.age = 0; c.life = 12 + R() * 2; c.spin = 0;
      c.vx = (R() - 0.5) * 0.3; c.vz = -0.16 - R() * 0.2;
      c.vy = up ? 0.24 + R() * 0.08 : -(0.3 + R() * 0.1);
      c.axis.set(R() - 0.5, R() * 0.6 + 0.2, R() - 0.5).normalize();
      c.spinRate = (0.25 + R() * 0.35) * (R() < 0.5 ? -1 : 1);
      c.w1 = 2.4 + R() * 1.2;
      for (let n = 0; n < 4; n++) {
        c.aL[n] = n === 0 ? 0 : 0.085 / n; c.aR[n] = n === 0 ? 0 : 0.07 / n;
        c.ph[n] = R() * TAU; c.ph[n + 4] = R() * TAU;
      }
    }
    // A closed loop born from two open strings whose ends have met. Its shape starts as exactly
    // the two arcs (copied from the ribbon texture) and relaxes into a vibrating ring.
    function formClosed() {
      const a = join.a, b = join.b, t = st.now;
      join.on = false;
      const c = allocClosed();
      const d = rib.data, base = c.base;
      let cx = 0, cy = 0, cz = 0;
      for (let j = 0; j < P; j++) {
        const half = P / 2;
        const o = j < half ? rib.at(a.row, Math.round(j * (P - 1) / (half - 0.5))) : rib.at(b.row, Math.round((j - half) * (P - 1) / (half - 1)));
        base[j * 3] = d[o]; base[j * 3 + 1] = d[o + 1]; base[j * 3 + 2] = d[o + 2];
        if (j < P - 1) { cx += d[o]; cy += d[o + 1]; cz += d[o + 2]; }
      }
      cx /= P - 1; cy /= P - 1; cz /= P - 1;
      let rad = 0;
      for (let j = 0; j < P; j++) {
        base[j * 3] -= cx; base[j * 3 + 1] -= cy; base[j * 3 + 2] -= cz;
        rad += Math.hypot(base[j * 3], base[j * 3 + 1], base[j * 3 + 2]);
      }
      c.rad = Math.max(0.5, (rad / P) * 1.25);
      // The ring it relaxes into faces the lens, tipped back a little, so the new loop reads as a
      // loop. Its phase follows the joined arcs (point 0 stays near point 0), so nothing twists.
      v3.set(camera.position.x - cx, 0, camera.position.z - cz).normalize();
      c.nrm.set(v3.x * 0.9, 0.44, v3.z * 0.9).normalize();
      c.e1.set(base[0], base[1], base[2]).addScaledVector(c.nrm, -(base[0] * c.nrm.x + base[1] * c.nrm.y + base[2] * c.nrm.z));
      if (c.e1.lengthSq() < 1e-6) c.e1.set(-v3.z, 0, v3.x);
      c.e1.normalize();
      c.e2.crossVectors(c.nrm, c.e1);
      const q = P >> 2;
      if (base[q * 3] * c.e2.x + base[q * 3 + 1] * c.e2.y + base[q * 3 + 2] * c.e2.z < 0) c.e2.negate();
      c.nrm.crossVectors(c.e1, c.e2);
      c.cx = cx; c.cy = cy; c.cz = cz;
      launchClosed(c, true);
      c.axis.copy(c.e1); c.spinRate = (0.04 + R() * 0.03) * (R() < 0.5 ? -1 : 1);   // a slow tumble
      a.alive = false; b.alive = false; rib.hide(a.row); rib.hide(b.row);
      ripLo.push(join.m1x, join.m1z, t, 0.12, C.ripMag);
      ripLo.push(join.m2x, join.m2z, t + 0.02, 0.12, C.ripMag);
      st.liftRip = 0.45; st.liftX = cx; st.liftZ = cz;
      st.escaped++;
      ctx.ui.toast('A closed string left the brane. Gravitons are closed strings, so gravity could leak away.', 4600);
      if (st.hintStage === 1 && st.entered) { st.hintStage = 2; ctx.ui.hint(hintText(2), 8000); }
      if (audible()) {
        const pan = panOf(cx, cy, cz);
        A.bell(A.degree(5, 1), { gain: 0.11, pan });
        A.bell(A.degree(8, 1), { gain: 0.09, pan, when: 0.14 });
        A.bell(A.degree(10, 1), { gain: 0.08, pan, when: 0.3 });
        A.noise({ freq: 380, sweepTo: 5200, q: 0.9, dur: 2.6, gain: 0.05, attack: 0.7, pan });
        snd.bell = st.t;
      }
    }

    function stepClosed(c, dt, t) {
      if (!c.alive) return;
      c.age += dt;
      if (c.age > c.life) { c.alive = false; rib.hide(c.row); return; }
      const morph = easeInOut(c.age / 1.5);
      const lift = smoothstep(0.3, 1.8, c.age);
      const below = c.cy < st.dw;
      c.cx += c.vx * lift * dt; c.cy += c.vy * lift * dt; c.cz += c.vz * lift * dt;
      // nothing holds a closed string: it passes through the other brane too, barely disturbing it
      if (below && c.cy >= st.dw && st.upVis > 0.5) {
        ripUp.push(c.cx, c.cz, t, 0.05, C.ripMag);
        if (audible() && st.t - snd.bell > 0.3) { snd.bell = st.t; A.bell(A.degree(12, 1), { gain: 0.035, decay: 2.5, pan: panOf(c.cx, c.cy, c.cz) }); }
      }
      c.spin += c.spinRate * lift * dt;
      c.q.setFromAxisAngle(c.axis, c.spin);
      const kd = Math.exp(-0.25 * dt);
      for (let n = 1; n < 4; n++) {
        const w = c.w1 * (n + 1);
        c.aL[n] = 0.03 / n + (c.aL[n] - 0.03 / n) * kd; c.aR[n] = 0.025 / n + (c.aR[n] - 0.025 / n) * kd;
        // left movers cos(kσ - ωt), right movers cos(kσ + ωt): independent waves round a closed loop
        TH[n * 4] = Math.cos(w * t + c.ph[n]); TH[n * 4 + 1] = Math.sin(w * t + c.ph[n]);
        TH[n * 4 + 2] = Math.cos(w * t + c.ph[n + 4]); TH[n * 4 + 3] = Math.sin(w * t + c.ph[n + 4]);
      }
      const inten = (2.6 + 1.2 * Math.exp(-c.age * 0.4)) * smoothstep(0, 0.15, c.age) * (1 - smoothstep(c.life - 3.5, c.life, c.age));
      rib.style(c.row, C.closed, inten, 0.05, true, 2.0 * ctx.pixelRatio);
      const d = rib.data, base = c.base, e1 = c.e1, e2 = c.e2, nr = c.nrm, rad = c.rad;
      for (let i = 0; i < P; i++) {
        const K = i * 8, co = CS[K], si = CS[K + 1];
        const rx = e1.x * co + e2.x * si, ry = e1.y * co + e2.y * si, rz = e1.z * co + e2.z * si;
        let wl = 0, wr = 0;
        for (let n = 1; n < 4; n++) {
          const ck = CS[K + n * 2], sk = CS[K + n * 2 + 1];
          wl += c.aL[n] * (ck * TH[n * 4] + sk * TH[n * 4 + 1]);
          wr += c.aR[n] * (ck * TH[n * 4 + 2] - sk * TH[n * 4 + 3]);
        }
        const bx = base[i * 3], by = base[i * 3 + 1], bz = base[i * 3 + 2];
        v3.set(
          bx + (rx * rad - bx) * morph + (rx * wl + nr.x * wr) * morph,
          by + (ry * rad - by) * morph + (ry * wl + nr.y * wr) * morph,
          bz + (rz * rad - bz) * morph + (rz * wl + nr.z * wr) * morph,
        ).applyQuaternion(c.q);
        const o = rib.at(c.row, i);
        d[o] = c.cx + v3.x; d[o + 1] = c.cy + v3.y; d[o + 2] = c.cz + v3.z; d[o + 3] = 1;
      }
    }

    // ---- collisions: two open strings join end to end ----
    function collide(fromUser = true) {
      if (join.on) return;
      const live = [];
      for (const s of opens) if (s.alive && !s.retire && s.state === 2) live.push(s);
      if (live.length < 2) {
        const bx = -0.5 + R() * 2, bz = -3 + R() * 2;
        for (let i = live.length; i < 2; i++) spawnOpen(0, bx + i * 1.6, bz + (R() - 0.5), fromUser);
        st.pendingCollide = 0.85;
        return;
      }
      let best = Infinity, a = null, b = null;
      for (let i = 0; i < live.length; i++) for (let j = i + 1; j < live.length; j++) {
        const dd = Math.hypot(live[i].mx - live[j].mx, live[i].mz - live[j].mz);
        if (dd < best) { best = dd; a = live[i]; b = live[j]; }
      }
      if (best > 4.5) { spawnOpen(0, a.mx + 1.5, a.mz + 0.4, fromUser); st.pendingCollide = 0.85; return; }
      // pair the ends so that the two strings meet tip to tip: a.B with b.A and a.A with b.B
      const dStraight = Math.hypot(a.bx - b.ax, a.bz - b.az) + Math.hypot(a.ax - b.bx, a.az - b.bz);
      const dCross = Math.hypot(a.bx - b.bx, a.bz - b.bz) + Math.hypot(a.ax - b.ax, a.az - b.az);
      if (dCross < dStraight) {
        let tx = b.ax; b.ax = b.bx; b.bx = tx; tx = b.az; b.az = b.bz; b.bz = tx;
      }
      let m1x = (a.bx + b.ax) * 0.5, m1z = (a.bz + b.az) * 0.5, m2x = (a.ax + b.bx) * 0.5, m2z = (a.az + b.bz) * 0.5;
      const cx = (m1x + m2x) * 0.5, cz = (m1z + m2z) * 0.5;
      let vx = m1x - m2x, vz = m1z - m2z, vl = Math.hypot(vx, vz);
      if (vl < 1e-3) { vx = 1; vz = 0; vl = 1; }
      const len = clamp(vl, 1.0, 1.7);
      vx /= vl; vz /= vl;
      m1x = cx + vx * len * 0.5; m1z = cz + vz * len * 0.5; m2x = cx - vx * len * 0.5; m2z = cz - vz * len * 0.5;
      Object.assign(join, { on: true, a, b, t: 0, m1x, m1z, m2x, m2z, aH0: a.H, bH0: b.H, aL0: a.lean, bL0: b.lean, Ht: len * 0.62 });
      join.a0[0] = a.ax; join.a0[1] = a.az; join.a0[2] = a.bx; join.a0[3] = a.bz;
      join.b0[0] = b.ax; join.b0[1] = b.az; join.b0[2] = b.bx; join.b0[3] = b.bz;
      a.state = 3; b.state = 3;
      if (audible()) A.noise({ freq: 900, sweepTo: 1800, q: 2, dur: 1.2, gain: 0.02, attack: 0.9, pan: panOf(cx, 0.4, cz) });
    }

    function stepJoin(dt) {
      if (!join.on) return;
      const a = join.a, b = join.b;
      if (!a.alive || !b.alive || a.state !== 3 || b.state !== 3) { join.on = false; if (a.alive) a.state = 2; if (b.alive) b.state = 2; return; }
      join.t += dt;
      const k = easeInOut(join.t / join.dur);
      a.ax = lerp(join.a0[0], join.m2x, k); a.az = lerp(join.a0[1], join.m2z, k);
      a.bx = lerp(join.a0[2], join.m1x, k); a.bz = lerp(join.a0[3], join.m1z, k);
      b.ax = lerp(join.b0[0], join.m1x, k); b.az = lerp(join.b0[1], join.m1z, k);
      b.bx = lerp(join.b0[2], join.m2x, k); b.bz = lerp(join.b0[3], join.m2z, k);
      // both arcs lean the same way relative to their own direction, which tilts them to opposite
      // sides in the world (their directions are reversed), so the two arcs open into a loop
      a.H = lerp(join.aH0, join.Ht, k); b.H = lerp(join.bH0, join.Ht, k);
      a.lean = lerp(join.aL0, 0.95, k); b.lean = lerp(join.bL0, 0.95, k);
      a.mx = (a.ax + a.bx) * 0.5; a.mz = (a.az + a.bz) * 0.5; b.mx = (b.ax + b.bx) * 0.5; b.mz = (b.az + b.bz) * 0.5;
    }

    // ---- sound ----
    const snd = { pluck: -1, water: -1, bell: -1 };
    const audible = () => A.on && !st.ff;
    let voice = null, audioAcc = 0;
    function panOf(x, y, z) {
      v3b.set(x, y, z).project(camera);
      return clamp(v3b.x * 0.8, -0.9, 0.9);
    }
    /** Mass-energy as pitch: rises with d, nudged most of the way onto the pentatonic scale. */
    function stretchFreq(dd) {
      const s = 12 + dd * 4.5;
      let q = PENTA_SEMIS[0];
      for (const p of PENTA_SEMIS) if (Math.abs(p - s) < Math.abs(q - s)) q = p;
      return A.freq(s + (q - s) * 0.85);
    }
    function pluckFor(x, z, brane) {
      if (!audible() || st.t - snd.pluck < 0.08) return;
      snd.pluck = st.t;
      const deg = clamp(Math.round(((x + 7) / 14) * 9), 0, 9);
      const pan = panOf(x, 0, z);
      if (brane === 0) A.pluck(A.degree(deg, z < -6 ? 0 : 1), { gain: 0.21, decay: 2.8, bright: 0.5, partials: 9, pan });
      else A.pluck(A.degree(deg, 2), { gain: 0.13, decay: 3.2, bright: 0.7, partials: 7, inharm: 0.0018, pan });
      if (st.t - snd.water > 0.15) {
        snd.water = st.t;
        A.noise({ freq: 550 + R() * 500, sweepTo: 230, q: 1.4, dur: 1.2, gain: 0.024, pan });
      }
    }
    function stepAudio(dt) {
      audioAcc += dt;
      if (audioAcc < 0.066) return;
      audioAcc = 0;
      let n = 0;
      for (const s of stretches) if (s.alive && !s.retire && s.connected) n++;
      if (!voice && A.on && n > 0) voice = A.additive(stretchFreq(st.d), 6, { gain: 0 });
      if (!voice) return;
      const g = n === 0 ? 0 : (0.045 + 0.012 * Math.min(n, 4)) * smoothstep(0.02, 0.45, st.d);
      voice.setGain(g, 0.25);
      voice.setFreq(stretchFreq(st.d), 0.15);
      const h = st.d / D_MAX;
      for (let k = 0; k < 6; k++) {
        // sul ponticello: the hotter (heavier) the string, the more its upper partials shimmer
        const shimmer = k >= 3 ? 0.75 + 0.25 * Math.sin(st.t * (4.1 + k) + k) : 1;
        voice.setAmp(k, 0.5 * Math.pow(k + 1, -(1.6 - 0.75 * h)) * shimmer, 0.12);
      }
    }

    // ---- lights the branes receive from the strings ----
    function fillLights() {
      let n = 0, m = 0;
      const LL = lightsLo.L, LC = lightsLo.C, UL = lightsUp.L, UC = lightsUp.C;
      const fk = st.fieldK;
      if (fk > 0.01 && n < NL) { LL[n].set(FS.x, hLo(FS.x, FS.z) + 0.3, FS.z, 1.1 * fk); LC[n++].copy(C.gold); }
      for (const s of stretches) {
        if (!s.alive) continue;
        heatColor(st.d, heat);
        const k = s.fade * easeOut3(s.grow) * (0.7 + 0.12 * st.d);
        if (n < NL) { LL[n].set(s.x0, s.y0 + 0.35, s.z0, k); LC[n++].copy(heat); }
        if (m < NL && st.upVis > 0.02) { UL[m].set(s.x1, s.y1 - 0.35, s.z1, k); UC[m++].copy(heat); }
      }
      for (const c of closeds) {
        if (!c.alive) continue;
        const near = 1 - smoothstep(0.4, 2.6, Math.abs(c.cy));
        if (near > 0.01 && n < NL) { LL[n].set(c.cx, c.cy, c.cz, 0.45 * near); LC[n++].copy(C.closed); }
      }
      for (const s of opens) {
        if (!s.alive || n >= NL) continue;
        LL[n].set((s.ax + s.bx) * 0.5, (s.ay + s.by) * 0.5 + s.H * 0.55, (s.az + s.bz) * 0.5, 0.55 * s.fade * (1 + s.flash) * (s.state === 1 ? s.k : 1));
        LC[n++].copy(s.col);
      }
      for (const s of uppers) {
        if (!s.alive) continue;
        const lx = (s.ax + s.bx) * 0.5, ly = (s.ay + s.by) * 0.5, lz = (s.az + s.bz) * 0.5;
        const k = 0.5 * s.fade * (1 + s.flash) * (s.state === 1 ? s.k : 1);
        if (st.upVis > 0.02 && m < NL) { UL[m].set(lx, ly - s.H * 0.55, lz, k); UC[m++].copy(s.col); }
        else if (st.upVis <= 0.02 && n < NL) { LL[n].set(lx, ly + s.H * 0.55, lz, k); LC[n++].copy(s.col); }
      }
      for (; n < NL; n++) LL[n].w = 0;
      for (; m < NL; m++) UL[m].w = 0;
    }

    // ---- field lines: a source with charge and mass ----
    const FS = { x: -1.3, z: -3.3 };
    const GD = new Float32Array(N_GRAV * 3);         // gravity line directions (Fibonacci sphere)
    for (let j = 0; j < N_GRAV; j++) {
      const yy = 1 - (2 * (j + 0.5)) / N_GRAV, r = Math.sqrt(1 - yy * yy), a = j * 2.39996323;
      GD[j * 3] = Math.cos(a) * r; GD[j * 3 + 1] = yy; GD[j * 3 + 2] = Math.sin(a) * r;
    }
    function stepField(dt, t) {
      st.fieldK += ((st.field ? 1 : 0) - st.fieldK) * damp(2.4, dt);
      const k = st.fieldK;
      fieldMesh.visible = k > 0.004;
      if (!fieldMesh.visible) return;
      const sx = FS.x, sz = FS.z, sy = hLo(sx, sz);
      const reach = 3.6 * (0.25 + 0.75 * k);
      const d = fld.data;
      for (let j = 0; j < N_GAUGE; j++) {
        const a = (j / N_GAUGE) * TAU + 0.13, ca = Math.cos(a), sa = Math.sin(a);
        fld.style(j, C.open, 1.25 * k, 0.01, false, 1.1 * ctx.pixelRatio);
        for (let i = 0; i < PF; i++) {
          const u = i / (PF - 1), rr = 0.22 + u * reach;
          const x = sx + ca * rr, z = sz + sa * rr;
          const o = fld.at(j, i);
          d[o] = x; d[o + 1] = hLo(x, z) + 0.035; d[o + 2] = z;
          d[o + 3] = (0.3 + 1.7 * Math.pow(Math.max(0, Math.sin(rr * 3.0 - t * 3.2)), 10)) * (1 - u * u);
        }
      }
      for (let j = 0; j < N_GRAV; j++) {
        const gx = GD[j * 3], gy = GD[j * 3 + 1], gz = GD[j * 3 + 2];
        const row = N_GAUGE + j;
        fld.style(row, C.closed, 0.9 * k, 0.01, false, 1.1 * ctx.pixelRatio);
        for (let i = 0; i < PF; i++) {
          const u = i / (PF - 1), rr = 0.22 + u * reach;
          const o = fld.at(row, i);
          d[o] = sx + gx * rr; d[o + 1] = sy + 0.06 + gy * rr; d[o + 2] = sz + gz * rr;
          d[o + 3] = (0.3 + 1.7 * Math.pow(Math.max(0, Math.sin(rr * 3.0 - t * 3.2 + 1.1)), 10)) * (1 - u * u);
        }
      }
      fld.tex.needsUpdate = true;
      bead(sx, sy + 0.06, sz, C.gold, 2.4 * k, 0.75);
    }

    // ---- camera rig: orbit around a target that sits between the branes ----
    const rig = { yaw: -0.16, pitch: 0.085, R: 14, yawT: -0.16, pitchT: 0.085, RT: 14, R0: 14, px: 0, tx: 0.3, tz: -2.2 };
    function stepCamera(dt, t) {
      const k = damp(4, dt);
      rig.yaw += (rig.yawT - rig.yaw) * k;
      rig.pitch += (rig.pitchT - rig.pitch) * k;
      rig.R += (rig.RT - rig.R) * k;
      const target = ptr.down || ctx.isTouch ? rig.px : ctx.pointer.ndc.x * 0.03;
      rig.px += (target - rig.px) * damp(1.5, dt);
      const amp = reduceMotion ? 0.2 : 1;
      const yaw = rig.yaw + rig.px + amp * 0.05 * Math.sin(t * 0.07);
      const pitch = rig.pitch + amp * 0.008 * Math.sin(t * 0.11 + 1);
      const Rr = rig.R * (1 + amp * 0.02 * Math.sin(t * 0.05 + 2));
      const ty = clamp(st.dw * 0.34, 0.35, 1.4);
      // stay between the branes while they are apart; free to rise once they have merged
      const maxY = lerp(st.dw - 0.25, 8, smoothstep(1.2, 0.55, st.dw));
      const y = clamp(ty + Rr * Math.sin(pitch), 0.16, Math.max(0.16, maxY));
      const sp = clamp((y - ty) / Rr, -0.99, 0.99), cp = Math.sqrt(1 - sp * sp);
      camera.position.set(rig.tx + Rr * cp * Math.sin(yaw), ty + Rr * sp, rig.tz + Rr * cp * Math.cos(yaw));
      look.set(rig.tx, ty, rig.tz);
      camera.lookAt(look);
    }

    // ---- input ----
    const ptr = { down: false, moved: false, mode: 0, sx: 0, sy: 0, d0: 0, hit: 0, hx: 0, hz: 0, over: 0, ox: 0, oz: 0 };   // mode 0 none, 1 orbit, 2 lift; over = brane under the cursor
    const hov = { lo: 0, up: 0, grab: 0 };
    /** 0 = nothing, 1 = our brane, 2 = the upper brane. Hit point lands in ptr.hx/hz. */
    function pick(ndc) {
      ray.setFromCamera(ndc, camera);
      const o = ray.ray.origin, dv = ray.ray.direction;
      let tl = Infinity, tu = Infinity;
      if (dv.y < -1e-5 && o.y > 0) tl = -o.y / dv.y;
      if (st.upVis > 0.5 && Math.abs(dv.y) > 1e-5 && dv.y * (st.dw - o.y) > 0) tu = (st.dw - o.y) / dv.y;
      const tt = Math.min(tl, tu);
      if (!(tt < 70)) return 0;
      ptr.hx = o.x + dv.x * tt; ptr.hz = o.z + dv.z * tt;
      return tu < tl ? 2 : 1;
    }

    function setD(v, src) {
      st.dT = clamp(v, 0, D_MAX);
      if (st.dT < 0.04) st.dT = 0;
      if (st.dT >= 1) st.dPrev = st.dT;     // what Separate returns to (small passing values are not worth keeping)
      if (src === 'drag') sliderDirty = true;
      else if (src !== 'slider' && ui) ui.set('d', st.dT);
      const coin = st.dT === 0 ? 'Separate' : 'Coincide';
      if (ui && coin !== coinShown) { coinShown = coin; ui.set('coin', coin); }    // not on every drag move
    }
    let coinShown = '';
    let sliderDirty = false;
    function toggleCoincide() { if (st.dT > 0) setD(0); else setD(st.dPrev >= 1 ? st.dPrev : 2.4); }
    function setField(v) {
      st.field = !!v;
      if (ui) ui.set('field', st.field);
      if (st.field) {
        ctx.ui.toast('Field lines of a charged mass. Cyan force lines stay on the brane. Magenta gravity lines also spread into the bulk.', 5200);
        if (A.on) A.swell(A.degree(2, 1), { gain: 0.07, attack: 0.5, dur: 1.4 });
      }
    }
    /** The hint for each stage of the walkthrough: drop a string, collide, bring the branes together. */
    function hintText(stage) {
      if (stage === 0) return ctx.isTouch ? 'Tap the brane to drop a string' : 'Click the brane to drop a string';
      if (stage === 1) return ctx.isTouch ? 'Tap Collide to join two strings' : 'Press C to collide two strings';
      return ctx.isTouch ? 'Drag the upper brane down' : 'Drag the upper brane down, or press ↓';
    }
    function userSpawn(brane, x, z) {
      const s = spawnOpen(brane, x, z, true);
      if (!s) return;
      if (st.hintStage === 0 && st.entered) { st.hintStage = 1; ctx.ui.hint(hintText(1), 8000); }
    }
    function spawnRandom() { userSpawn(0, -4 + R() * 8, -7 + R() * 8); }

    // ---- UI ----
    let ro = null, ui = null, roAcc = 1;
    const NOTE_BASE = 'Gravity is about 10³⁶ times weaker than the electric force between two protons. If gravitons leak into the bulk, this may be why.';
    const NOTE_FIELD = 'Cyan lines stay on the brane. Magenta gravity lines also spread into the bulk, so they thin out faster with distance.';
    const NOTE_MERGED = 'At d = 0 the stretched strings are massless: the photon-like fields of both branes and the W-like strings join one U(2) family.';
    // phones get one-line versions so the readout leaves room for the stage
    const NOTE_SHORT = { base: 'Gravity is 10³⁶ times weaker than the electric force. The bulk may be why.', field: 'Gauge lines stay on the brane. Gravity spreads into the bulk.', merged: 'Massless stretched strings: U(2).' };
    function refreshReadout() {
      if (!ro) return;
      let n = 0;
      for (const s of opens) if (s.alive && !s.retire) n++;
      if (st.upVis < 0.05) for (const s of uppers) if (s.alive && !s.retire) n++;
      ro.set('open', String(n));
      ro.set('closed', String(st.escaped));
      const dd = st.d < 0.005 ? 0 : st.d;
      ro.set('d', dd.toFixed(2) + ' ℓₛ');
      ro.set('m', dd === 0 ? '0 · massless' : (dd / TAU).toFixed(3) + ' Mₛ');
      ro.set('g', st.merged ? 'U(2)' : 'U(1) × U(1)', st.merged ? 'gold' : '');
      const narrow = ctx.width <= 760;
      if (st.fieldK > 0.5) ro.note(narrow ? NOTE_SHORT.field : NOTE_FIELD);
      else if (st.merged) ro.note(narrow ? NOTE_SHORT.merged : NOTE_MERGED);
      else ro.note(narrow ? NOTE_SHORT.base : NOTE_BASE);
    }

    function onMerge() {
      st.mergeFlash = 1;
      ctx.ui.toast('The branes coincide. Stretched strings are massless now, and U(1) × U(1) has become U(2).', 5200);
      for (const s of stretches) if (s.alive) ripLo.push(s.x0, s.z0, st.now, 0.07, C.ripMint);
      ripLo.push(rig.tx, rig.tz, st.now, 0.1, C.ripMint);
      if (st.hintStage < 3) { st.hintStage = 3; ctx.ui.hideHint(); }
      if (audible() && st.t - snd.bell > 0.3) {
        snd.bell = st.t;
        A.bell(A.degree(5, 1), { gain: 0.09 });
        A.bell(A.degree(7, 1), { gain: 0.08, when: 0.09, pan: -0.3 });
        A.bell(A.degree(9, 1), { gain: 0.07, when: 0.18, pan: 0.3 });
        A.swell(A.degree(0, 1), { gain: 0.07, attack: 0.4, dur: 1.6 });
      }
    }

    // ---- opening tableau: already a film still before any input ----
    // spans roughly across the line of sight, so every arc reads in profile
    [[0.2, -1.4, 0.25], [2.9, -0.4, -0.35], [2.3, -5.8, 0.1], [-2.8, -8.6, -0.2], [5.6, -9.8, 0.4]].forEach(([x, z, rot]) => { const s = spawnOpen(0, x, z, false, true); if (s) s.rot = rot; });
    [[-4.4, -7.0], [3.9, -7.2]].forEach(([x, z]) => spawnOpen(1, x, z, false, true));
    addStretched(false, 1.5, -3.0); addStretched(false, -2.2, -10.2);
    {
      const c = closeds[N_CLOSED - 1], rad = 0.55;
      c.e1.set(1, 0, 0); c.e2.set(0, 0.97, -0.24).normalize(); c.nrm.crossVectors(c.e1, c.e2); c.rad = rad;   // facing the lens
      for (let i = 0; i < P; i++) {
        const co = CS[i * 8], si = CS[i * 8 + 1];
        c.base[i * 3] = (c.e1.x * co + c.e2.x * si) * rad; c.base[i * 3 + 1] = (c.e1.y * co + c.e2.y * si) * rad; c.base[i * 3 + 2] = (c.e1.z * co + c.e2.z * si) * rad;
      }
      c.cx = -0.3; c.cy = 1.0; c.cz = -5.6;
      launchClosed(c, true);
      c.age = 2.0; c.vy = 0.2; c.life = 22; c.axis.set(0.15, 1, 0.1).normalize(); c.spinRate = 0.09;
    }

    // ---- per frame ----
    let lastBeads = 0;
    // The simulation proper (no camera, DOM or audio), so demo() can fast-forward it.
    function simulate(dt) {
      const t = st.now;
      st.t += dt;
      for (let i = timers.length - 1; i >= 0; i--) {
        if (st.t >= timers[i].at) { const fn = timers[i].fn; timers.splice(i, 1); fn(); }
      }
      if (st.pendingCollide > 0) { st.pendingCollide -= dt; if (st.pendingCollide <= 0) collide(false); }
      if (st.liftRip > 0) { st.liftRip -= dt; if (st.liftRip <= 0) ripLo.push(st.liftX, st.liftZ, t, 0.15, C.ripMag); }
      // separation: eased toward its target, with the gauge group read off the real value
      st.d += (st.dT - st.d) * damp(3.2, dt);
      if (Math.abs(st.dT - st.d) < 0.002) st.d = st.dT;
      st.dw = st.d * VS;
      st.upVis = smoothstep(0.3, 0.8, st.d);
      st.mk = 1 - smoothstep(0.0, 0.35, st.d);
      if (!st.merged && st.d < 0.015) { st.merged = true; onMerge(); }
      else if (st.merged && st.d > 0.08) st.merged = false;
      st.mergeFlash *= Math.exp(-1.2 * dt);
      stepJoin(dt);
      for (let i = 0; i < N_OPEN; i++) stepOpen(opens[i], dt, t);
      for (let i = 0; i < N_UPPER; i++) stepOpen(uppers[i], dt, t);
      if (join.on && join.t >= join.dur) formClosed();
      for (let i = 0; i < N_STRETCH; i++) stepStretch(stretches[i], dt, t);
      for (let i = 0; i < N_CLOSED; i++) stepClosed(closeds[i], dt, t);
    }
    function fastForward(sec) {
      st.ff = true;
      const h = 1 / 30, t0 = ctx.time;
      for (let k = 0; k < sec / h; k++) { st.tOff += h; st.now = t0 + st.tOff; simulate(h); }
      st.ff = false;
    }

    function update(dt, t) {
      st.now = t + st.tOff; time.value = st.now;
      simulate(dt);
      t = st.now;
      const mergeK = (1 - smoothstep(0.0, 0.35, st.d)) * 0.6 + st.mergeFlash * 1.2;
      upper.position.y = st.dw;
      upper.visible = st.upVis > 0.01;
      upMat.uniforms.uOpacity.value = 0.32 * st.upVis;
      upMat.uniforms.uEmit.value = 0.55 * st.upVis * (1 + 0.55 * hov.grab);
      loMat.uniforms.uOtherY.value = st.dw; loMat.uniforms.uOtherOn.value = st.upVis;
      upMat.uniforms.uOtherY.value = 0; upMat.uniforms.uOtherOn.value = 0.7;
      loMat.uniforms.uMerge.value = mergeK;
      occl.uUpperY.value = st.dw; occl.uUpperA.value = 0.38 * st.upVis;
      mirUp.visible = st.upVis > 0.05;
      mirUp.material.uniforms.uPlaneY.value = st.dw;
      dustMat.uniforms.uD.value = st.dw;
      dustMat.uniforms.uAlpha.value = smoothstep(0.04, 0.4, st.d);
      dustMat.uniforms.uFocus.value = rig.R;
      // hover feedback: a ring where a click would land, and the upper brane brightens under the hand
      if (HOVER_DBG && !ptr.down) { ptr.over = pick(HOVER_DBG); ptr.ox = ptr.hx; ptr.oz = ptr.hz; }
      const hk = damp(8, dt), over = ptr.down ? 0 : ptr.over;
      hov.lo += ((over === 1 ? 1 : 0) - hov.lo) * hk;
      hov.up += ((over === 2 ? 1 : 0) - hov.up) * hk;
      hov.grab += ((ptr.mode === 2 || over === 2 ? 1 : 0) - hov.grab) * damp(5, dt);
      if (over) { const u = over === 1 ? loMat.uniforms.uHover.value : upMat.uniforms.uHover.value; u.x = ptr.ox; u.y = ptr.oz; }
      loMat.uniforms.uHover.value.z = hov.lo;
      upMat.uniforms.uHover.value.z = hov.up * st.upVis;

      rib.tex.needsUpdate = true;

      // endpoints
      beadN = 0;
      const pr = ctx.pixelRatio;
      for (let i = 0; i < N_OPEN + N_UPPER; i++) {
        const s = i < N_OPEN ? opens[i] : uppers[i - N_OPEN];
        if (!s.alive) continue;
        const k = s.fade * (s.state === 1 ? 0.5 + 0.5 * s.k : 1) * (1 + s.flash * 1.5);
        const o0 = rib.at(s.row, 0), o1 = rib.at(s.row, P - 1), d = rib.data;
        const sz = (s.brane === 0 ? 0.5 : 0.42) * (1 + s.flash * 0.6);
        tmpCol.copy(s.col).lerp(C.u2, st.mk * (s.brane === 1 ? 0.75 : 0.45));
        bead(d[o0], d[o0 + 1], d[o0 + 2], tmpCol, 2.2 * k, sz);
        bead(d[o1], d[o1 + 1], d[o1 + 2], tmpCol, 2.2 * k, sz);
      }
      heatColor(st.d, heat);
      for (let i = 0; i < N_STRETCH; i++) {
        const s = stretches[i];
        if (!s.alive) continue;
        bead(s.x0, s.y0, s.z0, heat, 2.4 * s.fade, 0.52 + 0.06 * st.d);
        if (s.grow > 0.97) bead(s.x1, s.y1, s.z1, heat, 2.4 * s.fade, 0.52 + 0.06 * st.d);
      }
      stepField(dt, t);
      for (let i = beadN; i < lastBeads; i++) beadSize[i] = 0;
      lastBeads = beadN;
      beadGeo.attributes.position.needsUpdate = true;
      beadGeo.attributes.aCol.needsUpdate = true;
      beadGeo.attributes.aSize.needsUpdate = true;

      fillLights();
      stepCamera(dt, t);
      stars.userData.update(t);
      stepAudio(dt);

      if (DBG) {   // ?b5dbg=ulbmsdf hides layers, for staging screenshots
        if (DBG.includes('u')) upper.visible = false;
        if (DBG.includes('l')) lower.visible = false;
        if (DBG.includes('b')) beads.visible = false;
        if (DBG.includes('m')) { mirLo.visible = false; mirUp.visible = false; }
        if (DBG.includes('s')) strings.visible = false;
        if (DBG.includes('d')) dust.visible = false;
        if (DBG.includes('S')) scene.background = null;
        if (DBG.includes('f')) stars.visible = false;
        if (DBG.includes('L')) for (const v of lightsUp.L) v.w = 0;
        if (DBG.includes('K')) for (const v of lightsLo.L) v.w = 0;
      }

      roAcc += dt;
      if (roAcc > 0.12) {
        roAcc = 0;
        refreshReadout();
        if (sliderDirty && ui) { sliderDirty = false; ui.set('d', st.dT); }
      }
    }

    function resize(w, h) {
      const asp = w / Math.max(1, h), pr = ctx.pixelRatio;
      let R0 = 14;
      if (asp < 1) {
        // portrait: a taller lens, pulled back, with the stage pushed below the readout and above
        // the bottom sheet of controls
        const k = clamp((asp - 0.45) / 0.55);
        camera.fov = lerp(58, 46, k);
        camera.setViewOffset(w, h, 0, -h * lerp(0.11, 0.03, k), w, h);
        R0 = lerp(16.5, 14.5, k);
      } else {
        camera.fov = 40;
        camera.setViewOffset(w, h, -w * 0.035, 0, w, h);
      }
      if (Math.abs(rig.RT - rig.R0) < 0.01) { rig.RT = R0; if (!st.entered) rig.R = R0; }     // keep the user's own zoom
      rig.R0 = R0;
      camera.updateProjectionMatrix();
      res.set(w * pr, h * pr);
      const core = 1.15 * Math.max(1, pr * 0.85);
      strings.material.uniforms.uCore.value = core;
      mirLo.material.uniforms.uCore.value = core * 1.4;
      mirUp.material.uniforms.uCore.value = core * 1.4;
      fieldMesh.material.uniforms.uCore.value = core * 0.85;
      beadMat.uniforms.uScale.value = camera.projectionMatrix.elements[5] * h * pr * 0.5;
      dustMat.uniforms.uScale.value = (h * pr) / 1080;
    }
    resize(ctx.width, ctx.height);

    return {
      scene, camera,
      enter() {
        st.entered = true;
        ro = ctx.ui.readout({
          title: 'Braneworld',
          rows: [
            { id: 'open', k: 'Open, on our brane', v: '0', cls: 'cyan' },
            { id: 'closed', k: 'Closed, escaped', v: '0', cls: 'mag' },
            { id: 'd', k: `Separation ${sym('d')}`, v: '' },
            { id: 'm', k: `Stretched ${sym('T·d')}`, v: '', cls: 'gold' },
            { id: 'g', k: 'Gauge group', v: '' },
          ],
          note: ctx.width <= 760 ? NOTE_SHORT.base : NOTE_BASE,
        });
        ui = ctx.ui.controls([
          { type: 'slider', id: 'd', label: `Separation ${sym('d')}`, min: 0, max: D_MAX, step: 0.01, value: st.dT,
            format: (v) => (v < 0.005 ? '0 · coincident' : v.toFixed(2) + ' ℓₛ'), onInput: (v) => setD(v, 'slider') },
          { type: 'button', id: 'collide', label: 'Collide', kbd: 'c', onClick: () => collide(true) },
          { type: 'button', id: 'stretch', label: 'Stretch', kbd: 's', onClick: () => addStretched(true) },
          { type: 'button', id: 'coin', label: st.dT === 0 ? 'Separate' : 'Coincide', kbd: 'd', onClick: toggleCoincide },
          { type: 'toggle', id: 'field', label: 'Field lines', value: st.field, kbd: 'f', onChange: setField },
        ]);
        coinShown = st.dT === 0 ? 'Separate' : 'Coincide';
        roAcc = 1;
        refreshReadout();
        // on a return visit, pick the walkthrough up where it was left
        if (st.hintStage < 3) ctx.ui.hint(hintText(st.hintStage), st.hintStage === 0 ? 0 : 8000);
        // a few ripples already spreading, so the sea is alive from the first frame
        const t = ctx.time + st.tOff;     // the simulation clock (demo() may have run it ahead)
        let fresh = false;
        for (const r of ripLo.rip) if (r.w > 0 && t - r.z < RIP_LIFE) fresh = true;
        if (!fresh) {
          for (let i = 0; i < 4; i++) {
            const s = opens[i];
            if (s.alive) ripLo.push(i % 2 ? s.ax : s.bx, i % 2 ? s.az : s.bz, t - 0.4 - i * 0.7, 0.07, C.ripCyan);
          }
          for (const s of stretches) if (s.alive) ripLo.push(s.lx, s.lz, t - 1.8, 0.05, C.ripEmber);
        }
      },
      exit() {
        st.entered = false;
        if (voice) { voice.stop(0.6); voice = null; }
        timers.length = 0;
        st.pendingCollide = 0;
        ptr.down = false; ptr.mode = 0; ptr.over = 0;
        ctx.setCursor('');
        ro = null; ui = null;
      },
      update,
      resize,
      onPointerDown(ev) {
        ptr.down = true; ptr.moved = false; ptr.mode = 0; ptr.sx = ev.x; ptr.sy = ev.y; ptr.d0 = st.dT;
        ptr.hit = pick(ev.ndc);
        ctx.setCursor(ptr.hit === 2 ? 'grabbing' : '');
      },
      onPointerMove(ev) {
        if (!ptr.down) {
          ptr.over = ev.pointerType === 'mouse' ? pick(ev.ndc) : 0;
          ptr.ox = ptr.hx; ptr.oz = ptr.hz;
          ctx.setCursor(ptr.over === 2 ? 'grab' : '');
          return;
        }
        if (!ptr.moved && Math.abs(ev.x - ptr.sx) + Math.abs(ev.y - ptr.sy) > 6) {
          ptr.moved = true;
          ptr.mode = ptr.hit === 2 && !ev.shift ? 2 : 1;
          ctx.setCursor('grabbing');
        }
        if (ptr.mode === 1) {
          rig.yawT -= ev.dx * 0.0052;
          rig.pitchT = clamp(rig.pitchT + ev.dy * 0.0035, -0.3, 1.15);
        } else if (ptr.mode === 2) {
          setD(ptr.d0 - ((ev.y - ptr.sy) / Math.max(200, ctx.height)) * 6.5, 'drag');
          rig.yawT -= ev.dx * 0.003;
        }
      },
      onPointerUp(ev) {
        if (ptr.down && !ptr.moved && ptr.hit) userSpawn(ptr.hit === 2 ? 1 : 0, ptr.hx, ptr.hz);
        ptr.down = false; ptr.mode = 0;
        ptr.over = ev.pointerType === 'mouse' ? pick(ev.ndc) : 0;
        ptr.ox = ptr.hx; ptr.oz = ptr.hz;
        ctx.setCursor(ptr.over === 2 ? 'grab' : '');
      },
      onWheel(ev) {
        rig.RT = clamp(rig.RT * Math.exp(ev.deltaY * 0.0012), 4, 24);
      },
      onKey(ev) {
        if (ev.key === ' ') { if (!ev.repeat) spawnRandom(); return true; }
        if (ev.key === 'ArrowUp') { setD(st.dT + 0.2, 'key'); return true; }
        if (ev.key === 'ArrowDown') { setD(st.dT - 0.2, 'key'); return true; }
        return false;
      },
      demo() {
        // Headless captures only see a few seconds, so play the signature moves and fast-forward:
        // six fresh strings have landed, two are stretched across the gap, and the closest pair has
        // already joined into a closed loop that is lifting off into the bulk.
        if (join.on) { join.on = false; if (join.a && join.a.alive) join.a.state = 2; if (join.b && join.b.alive) join.b.state = 2; }
        st.pendingCollide = 0; timers.length = 0;
        for (const s of opens) if (s.alive && !s.retire) { s.retire = true; s.fade = Math.min(s.fade, 0.35); }
        for (const s of stretches) if (s.alive && !s.retire) { s.retire = true; s.fade = Math.min(s.fade, 0.35); }
        for (const s of uppers) if (s.alive && !s.retire) { s.retire = true; s.fade = Math.min(s.fade, 0.35); }
        for (const c of closeds) if (c.alive) c.life = Math.min(c.life, c.age + 3.5);
        st.ff = true;
        // staggered in depth so no two arcs overlap and the loop does not sit in a symmetric frame
        // of stretched strings; a portrait lens sees a narrower slice, so the layout closes in
        const kx = ctx.aspect < 1 ? 0.58 : 1;
        const u = spawnOpen(1, 5.2 * kx, -8.4, true); if (u) u.rot = 0.2;
        const spots = [[-1.0, -3.4], [0.45, -3.9], [-4.3, -1.0], [4.1, 0.2], [5.0, -8.0], [-1.9, -7.4], [-6.5, -12.0]];
        for (const [x, z] of spots) { const s = spawnOpen(0, x * kx, z, true); if (s) s.rot = (R() - 0.5) * 0.6; }   // spans across the view, so arcs read in profile
        addStretched(true, 2.3 * kx, -5.6);
        addStretched(true, -3.6 * kx, -10.5);
        setD(2.6, 'demo');
        schedule(0.75, () => collide(true));
        st.ff = false;
        fastForward(3.6);
        refreshReadout();
      },
    };
  },
};
