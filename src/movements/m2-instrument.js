// II. The Instrument. One string alone in the void, played like an instrument.
//
// Physics model (visual units, but every relation is the real one):
//  - The string shape is X(σ,t) = base curve + Σₙ₌₁⁸ modes, each with two transverse polarisations
//    (in-plane "radial" and out-of-plane "z") and, for the closed loop, a cos nσ and a sin nσ standing wave.
//    Each of those 32 coordinates is an exact harmonic oscillator with ωₙ = n·ω₁ (rotated in phase space
//    every frame, so there is no integration drift).
//  - Integer occupation numbers kₙ are the truth. Mode n holds kₙ quanta of energy n·ε. The classical
//    amplitudes are relaxed toward that energy, so a pluck is literally rounded into quanta, and decay
//    happens as discrete quantum jumps, each one emitting a massless closed (or open) string.
//  - Level N = Σ n·kₙ. Open: α′M² = N, J_max = N+1. Closed: α′M² = 4N, J_max = 2N+2.
//  - Zero-point motion: each ghost string is a sample of the ground-state wavefunction, every mode
//    coordinate an Ornstein–Uhlenbeck process with stationary spread ∝ √(ħ/n).
//  - A pluck released from rest splits into two half-height pulses running opposite ways at dσ/dt = ω₁
//    (d'Alembert; the modes already do this because ωₙ = n·ω₁). Two light sparks ride those characteristics:
//    round the loop for a closed string, and reflecting without inversion off the free ends of an open one.
//
// Rendering: every string (hero, motion-trail ghosts, quantum-fuzz ghosts, the pad echo) is ONE instanced
// ribbon draw. The vertex shader evaluates X(σ) and its tangent from a float DataTexture of mode coefficients
// (one row per string), extrudes a screen-space ribbon, and the fragment shader paints a white-hot core inside
// a halo coloured by whichever mode dominates locally. A second draw mirrors the string and its trails into a
// black lacquered stage as a soft, glossy blur. Emitted quanta are a second instanced ribbon (tiny loop +
// tapered streak). Depth: nebula with a horizon band, stars, dust, foreground bokeh and a faint air glow.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { palette, hdr, modeColor, glowTexture, makeStarfield, GLSL, clamp, lerp, smoothstep, damp, sup } from '../fx.js';

// ---- constants -------------------------------------------------------------------------------
const NM = 8;                      // vibration modes n = 1..8
const TAU = Math.PI * 2;
const F_REF = 146.83;              // D3: the fundamental at reference tension T0
const OMEGA_REF = TAU * 0.62;      // visual angular frequency of mode 1 at T0 (rad/s)
const EPS = 0.40;                  // energy of one quantum of mode 1 (visual units); mode n: n·EPS
const R_REF = 1.45;                // loop radius at T0 (scales as ℓ_s ∝ 1/√T)
const OPEN_LEN_REF = 5.0;          // open-string length at T0
const FLOOR_GAP = 0.35;            // the lacquered stage sits this far below the loop
const GAMMA = 0.04;                // emission rate per quantum per unit n (1/s): high modes decay faster
const KMAX = 12;                   // occupation cap per mode
const FUZZ_S = 0.11;               // zero-point spread of mode 1 at ħ = 1 (world units at T0)
const MAX_Q = 14;                  // pool of emitted massless quanta
const LHC_GEV = 1.4e4;             // LHC collision energy, ~14 TeV
const MS_GEV = 1e17;               // assumed string scale at T0
const NOTE = ['C', 'C♯', 'D', 'E♭', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'B♭', 'B'];
const SNAP_SEMIS = [7, 10, 12, 15, 17]; // A2 C3 D3 F3 G3 (D minor pentatonic): where tension snaps
// The HUD uppercases control labels; Greek letters sit in a span that keeps their case (α, not Α).
const GK = (t) => `<span style="text-transform:none;letter-spacing:0">${t}</span>`;
const ORD = ['', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th'];

const noteName = (f) => { const m = Math.round(69 + 12 * Math.log2(f / 440)); return NOTE[((m % 12) + 12) % 12] + (Math.floor(m / 12) - 1); };
// the same, marked ≈ when the pitch is more than a quarter-tone-ish (25 cents) off the tempered note
// (the 7th harmonic is 31 cents flat of any equal-tempered note)
const noteLabel = (f) => { const x = 69 + 12 * Math.log2(f / 440); return (Math.abs(x - Math.round(x)) > 0.25 ? '≈' : '') + noteName(f); };

// Gaussian sampler (Box–Muller with a spare) — no allocation.
let gSpare = 0, gHas = false;
function gauss() {
  if (gHas) { gHas = false; return gSpare; }
  let u = 0; while (u === 0) u = Math.random();
  const v = Math.random(), m = Math.sqrt(-2 * Math.log(u));
  gSpare = m * Math.sin(TAU * v); gHas = true;
  return m * Math.cos(TAU * v);
}

// ---- shaders ---------------------------------------------------------------------------------
// The base curve morphs from a circle (θ = 2π) to a straight segment (θ → 0) while keeping its centroid
// at the origin: an arc of angle θ and length L. s ∈ [0,1]; the loop's seam (s = 0 ≡ 1) sits at the top,
// which is where it breaks open.
const BASE_GLSL = /* glsl */`
  uniform float uLen, uTheta;
  // p(s), the outward normal nr(s), and the tangent tb = dp/ds (|tb| = L). dnr/ds = (θ/L)·tb.
  void baseCurve(float s, out vec3 p, out vec3 nr, out vec3 tb){
    float u = s - 0.5;
    if (uTheta < 1e-3) { p = vec3(u * uLen, 0.0, 0.0); nr = vec3(0.0, -1.0, 0.0); tb = vec3(uLen, 0.0, 0.0); return; }
    float rho = uLen / uTheta, a = uTheta * u, h = 0.5 * uTheta;
    float ca = cos(a), sa = sin(a);
    p = vec3(rho * sa, rho * (1.0 - ca) - rho * (1.0 - sin(h) / h), 0.0);
    nr = vec3(sa, -ca, 0.0);
    tb = uLen * vec3(ca, sa, 0.0);
  }
`;

const MIRROR_GLSL = /* glsl */`
  uniform float uMirror, uFloorY, uTime, uRipple;
  vec3 mirrorP(vec3 p){
    float depth = max(0.0, p.y - uFloorY);
    vec3 q = vec3(p.x, 2.0 * uFloorY - p.y, p.z);
    float w = sin(p.z * 3.1 + uTime * 1.7 + p.x * 1.3) + 0.6 * sin(p.x * 5.3 - uTime * 2.3);
    q.x += w * (0.012 + 0.03 * uRipple) * (0.4 + depth);
    q.z += w * 0.02 * depth;
    return q;
  }
`;

const STRING_VS = /* glsl */`
  precision highp float;
  uniform sampler2D uCoef;
  uniform float uOpenBasis, uPR, uSatK, uFade;
  uniform vec2 uRes;
  uniform vec3 uBase, uFuzzCol;
  uniform vec3 uModeCol[8];
  uniform vec4 uPulse[2];     // per pluck: σ₀ (as s), distance run (s units), strength, unused
  attribute float aS, aSide;
  attribute float iRow, iAlpha, iSoft;
  attribute vec4 iStyle;      // half-width px, core gain, halo gain, tint toward the quantum-cloud colour
  varying float vSide, vA, vHalfW, vCore, vHalo, vHeat, vEnd, vPulse, vSoft;
  varying vec3 vCol;
  ${BASE_GLSL}
  ${MIRROR_GLSL}
  // X(s) and its exact tangent dX/ds, in one pass over the eight modes
  vec3 shape(float s, int row, out vec3 colAcc, out float wAcc, out vec3 T){
    vec3 p, nr, tb; baseCurve(s, p, nr, tb);
    float k = uOpenBasis > 0.5 ? 3.14159265 : 6.28318531, sig = k * s;
    float dr = 0.0, dz = 0.0, ddr = 0.0, ddz = 0.0, wHue = 0.0; colAcc = vec3(0.0); wAcc = 0.0;
    for (int n = 0; n < 8; n++) {
      vec4 c = texelFetch(uCoef, ivec2(n, row), 0);
      float fn = float(n + 1), cs = cos(fn * sig), sn = sin(fn * sig);
      float r = c.x * cs + c.y * sn, z = c.z * cs + c.w * sn;
      dr += r; dz += z;
      ddr += fn * (c.y * cs - c.x * sn); ddz += fn * (c.w * cs - c.z * sn);
      float e = fn * fn * (r * r + z * z);   // local energy density of mode n
      float e3 = e * e * e;                  // cubed: wherever one mode dominates locally, its colour wins
      colAcc += uModeCol[n] * e3; wAcc += e; wHue += e3;
    }
    colAcc /= max(wHue, 1e-18);
    float lum = dot(colAcc, vec3(0.2126, 0.7152, 0.0722));
    colAcc = max(mix(vec3(lum), colAcc, 1.25), 0.0);
    T = tb * (1.0 + uTheta / max(uLen, 1e-4) * dr) + nr * (ddr * k) + vec3(0.0, 0.0, ddz * k);
    return p + nr * dr + vec3(0.0, 0.0, dz);
  }
  // a spark: a hot head and a short tail trailing behind it (dir = direction of travel in s)
  float spark(float x, float dir){
    float u = x * dir;
    return exp(-u * u * 1033.0) + (u < 0.0 ? 0.42 * exp(u * 14.0) : 0.0);   // head σ = 0.022, tail 0.07
  }
  float pulses(float s){
    float acc = 0.0;
    for (int i = 0; i < 2; i++) {
      vec4 q = uPulse[i];
      if (q.z < 1e-3) continue;
      if (uOpenBasis > 0.5) {
        // free ends reflect a pulse without inverting it: unfold the segment onto a circle of
        // circumference 2 (σ ∈ [−π, π)); the point s is seen at both x = s and its mirror image x = −s
        for (int m = 0; m < 2; m++) {
          float x = m == 0 ? s : -s;
          float a = mod(x - q.x - q.y + 1.0, 2.0) - 1.0, b = mod(x - q.x + q.y + 1.0, 2.0) - 1.0;
          acc += q.z * (spark(a, 1.0) + spark(b, -1.0));
        }
      } else {
        float a = fract(s - q.x - q.y + 0.5) - 0.5, b = fract(s - q.x + q.y + 0.5) - 0.5;
        acc += q.z * (spark(a, 1.0) + spark(b, -1.0));
      }
    }
    return acc;
  }
  void main(){
    // invisible rows (trails at rest, the cloud at ħ = 0, an idle echo) skip all shape work: clipped away
    if (iAlpha < 1e-4) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); vSide = 0.0; vA = 0.0; vHalfW = 1.0; vCore = 0.0; vHalo = 0.0; vHeat = 0.0; vEnd = 0.0; vPulse = 0.0; vSoft = 0.0; vCol = vec3(0.0); return; }
    int row = int(iRow + 0.5);
    vec3 colAcc, T; float wAcc;
    vec3 P = shape(aS, row, colAcc, wAcc, T);
    vec3 P1 = P + T * (0.05 / max(length(T), 1e-5));
    // an open string's ends: the glow tapers to nothing and the endpoint beads carry the light
    float endK = uTheta < 6.28 ? smoothstep(0.0, 0.04, min(aS, 1.0 - aS)) : 1.0;
    float fade = 1.0, hw = iStyle.x, core = iStyle.y, soft = iSoft;
    if (uMirror > 0.5) {
      // the reflection: a soft, glossy blur of the nearest part of the string only
      fade = exp(-max(0.0, P.y - uFloorY) * 1.6) * uFade;
      P = mirrorP(P); P1 = mirrorP(P1);
      hw *= 2.5; core = 0.0; soft = 1.0;
    }
    mat4 vp = projectionMatrix * viewMatrix;
    vec4 c = vp * vec4(P, 1.0), c1 = vp * vec4(P1, 1.0);
    vec2 t = (c1.xy / c1.w - c.xy / c.w) * uRes;
    t /= max(length(t), 1e-6);
    float halfW = hw * uPR;
    c.xy += vec2(-t.y, t.x) * aSide * halfW * 2.0 / uRes * c.w;
    gl_Position = c;
    // never let the reflection reach the programme text along the bottom of the frame
    if (uMirror > 0.5) fade *= smoothstep(-0.95, -0.62, c.y / c.w);
    vSide = aSide; vHalfW = halfW; vSoft = soft;
    vCore = core;
    vHalo = iStyle.z;
    float sat = wAcc / (wAcc + uSatK);
    vCol = mix(mix(uBase, colAcc, sat), uFuzzCol, iStyle.w);
    vHeat = sat;
    vPulse = row == 0 ? pulses(aS) : 0.0;
    // the string keeps its core to the very tip; every other row fades out entirely at the ends
    if (row == 0) { vEnd = endK; vA = iAlpha * fade; } else { vEnd = 1.0; vA = iAlpha * fade * endK; }
  }
`;

// Filament profile in device pixels: a narrow white-hot core (the only part bright enough to bloom),
// a saturated halo in the local mode colour kept under the tone-mapper's shoulder so it stays coloured,
// and a faint wide glow that gives the line body without smearing it. Soft rows (the quantum haze, the
// reflection) trade the tight halo for the wide glow. Sparks add a white-gold flare that rides the string.
const STRING_FS = /* glsl */`
  precision highp float;
  uniform float uPR;
  varying float vSide, vA, vHalfW, vCore, vHalo, vHeat, vEnd, vPulse, vSoft;
  varying vec3 vCol;
  void main(){
    float d = abs(vSide) * vHalfW;
    float cw = 0.5 * uPR + 0.25;
    float core = exp(-d * d / (2.0 * cw * cw));
    float halo = exp(-d / (1.7 * uPR + 0.5));
    float sg = vHalfW * 0.36;
    float glow = exp(-d * d / (2.0 * sg * sg));
    float edge = 1.0 - smoothstep(0.8, 1.0, abs(vSide));
    vec3 hot = mix(vec3(1.0, 0.94, 0.86), vCol, 0.16 + 0.3 * vHeat);
    vec3 col = hot * core * vCore + vCol * vHalo * (halo * (1.0 - 0.8 * vSoft) + glow * (0.14 + 0.36 * vSoft)) * vEnd;
    col += vec3(1.0, 0.74, 0.38) * vPulse * (core * 2.4 * (1.0 - vSoft) + halo * 1.4 + glow * 0.45) * vEnd;
    gl_FragColor = vec4(col * vA * edge, 1.0);
  }
`;

// Emitted massless quanta. Each one is two instances: its body (a tiny tumbling closed loop, or a tiny
// open segment with a wiggle) and a straight streak along its recent path that tapers to nothing.
// kind: 0 loop body, 1 open body, 2 loop streak, 3 open streak.
const QUANTA_VS = /* glsl */`
  precision highp float;
  uniform float uPR;
  uniform vec2 uRes;
  attribute float aS, aSide;
  attribute vec3 iPos, iE1, iE2;
  attribute vec4 iP;          // radius, alpha, kind, phase
  varying float vSide, vA, vHalfW, vKind, vS;
  ${MIRROR_GLSL}
  uniform float uFade;
  vec3 qc(float s){
    float r = iP.x;
    if (iP.z < 0.5) { float a = 6.2831853 * s; return iPos + r * (cos(a) * iE1 + sin(a) * iE2); }
    if (iP.z < 1.5) return iPos + (s - 0.5) * 2.6 * r * iE1 + iE2 * (0.28 * r * sin(3.14159265 * s) * sin(iP.w));
    return iPos - s * iE1;      // streak: iE1 is the displacement covered over the streak's length
  }
  void main(){
    float ds = 1.0 / 200.0;
    float s0 = aS - ds, s1 = aS + ds;
    if (iP.z > 0.5) { s0 = max(s0, 0.0); s1 = min(s1, 1.0); }
    vec3 P = qc(aS), P0 = qc(s0), P1 = qc(s1);
    // never let a quantum flare up in front of the lens
    float fade = smoothstep(3.0, 6.0, distance(P, cameraPosition));
    if (uMirror > 0.5) { fade *= exp(-max(0.0, P.y - uFloorY) * 1.8) * uFade; P = mirrorP(P); P0 = mirrorP(P0); P1 = mirrorP(P1); }
    mat4 vp = projectionMatrix * viewMatrix;
    vec4 c = vp * vec4(P, 1.0), c0 = vp * vec4(P0, 1.0), c1 = vp * vec4(P1, 1.0);
    vec2 t = (c1.xy / c1.w - c0.xy / c0.w) * uRes; t /= max(length(t), 1e-6);
    float halfW = (iP.z > 1.5 ? 3.2 : 4.5) * uPR;
    c.xy += vec2(-t.y, t.x) * aSide * halfW * 2.0 / uRes * c.w;
    gl_Position = c;
    vSide = aSide; vHalfW = halfW; vKind = iP.z; vA = iP.y * fade; vS = aS;
  }
`;
const QUANTA_FS = /* glsl */`
  precision highp float;
  uniform float uPR;
  uniform vec3 uColLoop, uColOpen;
  varying float vSide, vA, vHalfW, vKind, vS;
  void main(){
    float d = abs(vSide) * vHalfW, cw = 0.5 * uPR + 0.3;
    float core = exp(-d * d / (2.0 * cw * cw)), halo = exp(-d / (1.5 * uPR + 0.4));
    vec3 c = mix(uColLoop, uColOpen, mod(vKind, 2.0));
    float a = vA, k = 1.35;
    if (vKind > 1.5) { float f = 1.0 - vS; a *= f * f * f; k = 0.9; }
    gl_FragColor = vec4((vec3(1.0, 0.95, 0.9) * core * k + c * halo * 0.75) * a * (1.0 - smoothstep(0.75, 1.0, abs(vSide))), 1.0);
  }
`;

// Lacquered stage floor: near-black with a brushed grain, a pool of the string's light, and
// ripple rings that run outward whenever a note is struck.
const FLOOR_VS = /* glsl */`
  varying vec3 vW;
  void main(){ vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }
`;
const FLOOR_FS = /* glsl */`
  precision highp float;
  uniform vec3 uPool, uFog, uCam;
  uniform float uTime, uPoolR;
  uniform vec4 uRip[6];      // xyz colour * strength, w = age (s); age < 0 means unused
  varying vec3 vW;
  ${GLSL.noise}
  void main(){
    vec2 xz = vW.xz; float r = length(xz);
    vec3 V = normalize(uCam - vW);
    float fres = pow(1.0 - clamp(V.y, 0.0, 1.0), 5.0);
    float grain = snoise(vec3(xz.x * 0.6, xz.y * 2.0, 1.7)) * 0.5 + 0.5;
    // the brushed grain fades with distance, so it never piles up into scanlines near the horizon
    float gk = mix(1.0, 0.85 + 0.3 * grain, exp(-r * 0.15));
    vec3 col = vec3(0.0045, 0.0036, 0.0095) * gk + vec3(0.020, 0.012, 0.045) * fres;
    float pool = exp(-r * r / (uPoolR * uPoolR));
    col += uPool * pool * (0.75 + 0.5 * grain);
    for (int i = 0; i < 6; i++) {
      vec4 rp = uRip[i];
      if (rp.w < 0.0) continue;
      float rr = rp.w * 2.6 + 0.6;
      float u = (r - rr) / (0.10 + 0.05 * rp.w);
      float ring = exp(-u * u) * exp(-rp.w * 1.5) * exp(-rr * 0.35);
      col += rp.xyz * ring * (0.7 + 0.3 * grain) * smoothstep(0.0, 0.25, rp.w);
    }
    float fog = 1.0 - exp(-r * 0.05);
    col = mix(col, uFog, smoothstep(0.0, 1.0, fog));
    gl_FragColor = vec4(col, 1.0);
  }
`;

// Distant nebula on a back-facing sphere: deep violet haze, kept well under the bloom threshold
// and darkened toward the horizon so the floor dissolves into it without a seam.
const NEBULA_VS = /* glsl */`
  varying vec3 vDir;
  void main(){ vDir = position; vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0); gl_Position = p.xyww; }
`;
const NEBULA_FS = /* glsl */`
  precision highp float;
  uniform float uTime;
  uniform vec3 uHorizon;
  varying vec3 vDir;
  ${GLSL.noise}
  void main(){
    vec3 d = normalize(vDir);
    float n = snoise(d * 1.5 + vec3(0.0, 0.0, uTime * 0.006)) * 0.55 + snoise(d * 3.3 + 7.0) * 0.3 + snoise(d * 7.1 - 3.0) * 0.15;
    n = n * 0.5 + 0.5;
    float lift = smoothstep(-0.06, 0.35, d.y);
    float bb = (d.y - 0.22) * 2.4;
    float band = exp(-bb * bb);
    vec3 a = vec3(0.010, 0.006, 0.030), b = vec3(0.040, 0.010, 0.040), g = vec3(0.012, 0.022, 0.050);
    vec3 col = mix(a, b, smoothstep(0.35, 0.85, n));
    col = mix(col, g, smoothstep(0.55, 0.95, snoise(d * 2.1 - 11.0) * 0.5 + 0.5) * 0.6);
    col *= (0.35 + 1.25 * n * n) * (0.35 + 0.9 * band) * lift;
    // a low band of light on the horizon: the far edge of the stage, where the floor's fog dissolves
    float hz = exp(-abs(d.y) * 26.0) * 0.75 + exp(-abs(d.y) * 6.0) * 0.25;
    col += uHorizon * hz * (0.75 + 0.5 * n);
    gl_FragColor = vec4(col, 1.0);
  }
`;

// Out-of-focus motes between the lens and the stage. Each point's blur disc grows with its distance from
// the focal plane (the string), and its brightness falls as the light spreads over the disc.
const BOKEH_VS = /* glsl */`
  attribute float aSeed;
  uniform float uTime, uFocus, uScale;
  uniform vec2 uHero;         // the hero's centre in NDC: motes fade out over it
  varying float vA, vSeed;
  void main(){
    vec3 p = position;
    p.x += sin(uTime * 0.05 + aSeed * 31.0) * 0.35;
    p.y += sin(uTime * 0.07 + aSeed * 17.0) * 0.25;
    p.z += cos(uTime * 0.04 + aSeed * 23.0) * 0.35;
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    float z = max(0.2, -mv.z);
    float coc = abs(z - uFocus) / z;
    float px = uScale * (3.0 + 150.0 * coc);
    gl_PointSize = clamp(px, 2.0, 130.0 * uScale);
    vA = (0.55 + 0.45 * sin(uTime * (0.2 + aSeed * 0.3) + aSeed * 60.0)) * clamp(34.0 / (min(px / uScale, 130.0) + 6.0), 0.0, 1.0)
       * smoothstep(0.6, 2.2, z) * smoothstep(0.8 * uFocus, 0.55 * uFocus, z);
    vSeed = aSeed;
    gl_Position = projectionMatrix * mv;
    vec2 nd = gl_Position.xy / gl_Position.w - uHero;
    vA *= smoothstep(0.25, 0.45, length(nd)) * smoothstep(-0.75, -0.55, gl_Position.x / gl_Position.w);   // and off the movement rail
  }
`;
const BOKEH_FS = /* glsl */`
  precision highp float;
  uniform vec3 uC0, uC1, uC2;
  uniform float uGain;
  varying float vA, vSeed;
  void main(){
    float r = length(gl_PointCoord - 0.5) * 2.0;
    if (r > 1.0) discard;
    float disc = smoothstep(1.0, 0.86, r);
    float rim = smoothstep(0.7, 0.95, r) * disc;
    float h = fract(vSeed * 7.31);
    vec3 c = h < 0.5 ? uC0 : (h < 0.8 ? uC1 : uC2);
    gl_FragColor = vec4(c * (0.6 * disc + 0.3 * rim) * vA * uGain, 1.0);
  }
`;

// ---- ribbon geometry helpers -------------------------------------------------------------------
function ribbonGeometry(seg) {
  const n = seg + 1, s = new Float32Array(n * 2), side = new Float32Array(n * 2), idx = [];
  for (let i = 0; i < n; i++) {
    s[i * 2] = s[i * 2 + 1] = i / seg; side[i * 2] = -1; side[i * 2 + 1] = 1;
    if (i < seg) { const a = i * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
  }
  const g = new THREE.InstancedBufferGeometry();
  g.setAttribute('aS', new THREE.BufferAttribute(s, 1));
  g.setAttribute('aSide', new THREE.BufferAttribute(side, 1));
  g.setIndex(idx);
  return g;
}

export default {
  id: 'instrument',
  numeral: 'II',
  title: 'The Instrument',
  tempo: 'Allegro, pizzicato',
  caption: 'Drag the string to pluck it, or press keys 1–8. Each note adds a <em>quantum</em> of vibration. Their harmonic numbers add up to the level, which sets the string’s <em>mass</em> and its highest <em>spin</em>. The silent ground state of a closed string is a <em>graviton</em>.',
  fine: '<p>The particle card uses the superstring’s mass formula. Add up the harmonic number of every quantum to get the level <em>N</em>. An open string then has α′M² = N and a closed string α′M² = 4N, so its mass is √N or 2√N times the string scale M<sub>s</sub>. The highest spin at each level climbs with N (N + 1 open, 2N + 2 closed): the leading <em>Regge trajectory</em>. A closed string must carry as much vibration running one way round the loop as the other (<em>level matching</em>). Standing waves always do. On the loop, each quantum you add stands for a matched pair, one wave running each way, which is why a closed string at level N has α′M² = 4N. Tension T ∝ 1/α′ raises every pitch and mass together (M<sub>s</sub> ∝ √T).</p>'
    + '<p>Every particle we know, from electrons and quarks to photons, would live in the massless level. They are told apart by how the string sits in the hidden dimensions of Movement III. The higher notes you play here are real predictions of heavier particles, but if M<sub>s</sub> is near 10¹⁷ GeV they lie about ten trillion times beyond the Large Hadron Collider. String theory has no experimental confirmation.</p>'
    + '<p>A quantum string never sits still. Each mode keeps a zero-point jitter with spread ∝ √(ħ/n), which the ghost strings sample. Summed over all modes, without the cut-off at eight used here, the string’s apparent size grows logarithmically the finer you look.</p>'
    + '<p>Simplified: this is the bosonic part of the superstring in light-cone gauge, with two of the eight transverse directions drawn and the fermions of supersymmetry left out. The open string’s ends are free to move. The pitches keep the string’s true frequency ratios, but a string at 10¹⁷ GeV would ring about 127 octaves higher. The slow motion, colours and emission timing are artistic. Real decay rates depend on couplings this picture leaves out.</p>',
  drone: { notes: [0, 12, 19], opts: { gain: 0.055, cutoff: 650, wobble: 0.25 } },
  post: { bloomStrength: 0.75, bloomRadius: 0.32, bloomThreshold: 0.45, exposure: 1.0, aberration: 0.0016, grain: 0.034, vignette: 0.92 },

  create(ctx) {
    const low = ctx.quality === 'low';
    const SEG = low ? 512 : 840;          // ribbon segments per string
    const NT = low ? 7 : 14;              // motion-trail ghosts (closely spaced so they blur, not stack)
    const NF = low ? 16 : 26;             // quantum-fuzz ghosts
    const ROWS = 1 + NT + NF + 1;         // row 0 = the string, then trails, quantum ghosts, and the pad echo
    const ECHO = ROWS - 1;
    const TSTEP = low ? 2 : 1;            // history frames (1/60 s) between trail ghosts
    const HN = NT * TSTEP + 1;            // history ring length
    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const params = new URLSearchParams(location.search);
    // a small portrait frame needs a tighter bloom, or the loop dissolves into haze
    const POST_LAND = { bloomStrength: 0.75, bloomRadius: 0.32 }, POST_PORTRAIT = { bloomStrength: 0.62, bloomRadius: 0.24 };

    const scene = new THREE.Scene();
    scene.background = palette.void.clone();
    const camera = new THREE.PerspectiveCamera(36, ctx.aspect, 0.05, 900);
    camera.position.set(0, 1.7, 9.6);

    // ---- state ------------------------------------------------------------------------------------
    // Coefficient layout per mode n (index (n-1)*4 + c): c = 0 radial·cos nσ, 1 radial·sin nσ, 2 z·cos nσ, 3 z·sin nσ.
    const Q = new Float32Array(NM * 4), V = new Float32Array(NM * 4), H = new Float32Array(NM * 4);
    const TMP = new Float32Array(NM * 4);
    const k = new Int32Array(NM);            // occupation numbers kₙ
    const kProv = new Int32Array(NM);        // provisional quanta while a pluck is held
    const E = new Float32Array(NM);          // classical energy per mode
    const Edisp = new Float32Array(NM);      // energy shown by the spectrum (includes a held pluck)
    const heroFz = new Float32Array(NM * 4);
    const fz = new Float32Array(NF * NM * 4);
    const hist = new Float32Array(HN * NM * 4);
    const row0 = new Float32Array(NM * 4);
    const amps = new Float32Array(NM);
    let histHead = 0, histAcc = 0;

    const st = {
      open: false,         // basis currently used by the modes
      openTarget: false,   // what the user asked for
      morph: 0,            // 0 loop .. 1 segment
      amp: 1,              // global amplitude during the open/closed swap
      swap: 0,             // 0 idle, 1 fading out, 2 fading in
      T: 1, f1: F_REF, omega1: OMEGA_REF, R: R_REF, openLen: OPEN_LEN_REF,
      hbar: 0.25, floorY: -(R_REF + FLOOR_GAP),
      draining: false,
      theta: TAU, len: TAU * R_REF,
      emitCool: 0, lastEmitSound: -1, lastPadSound: -1,
      interacted: false, toastEmitClosed: false, toastEmitOpen: false, toastZero: false, toastOpen: false, toastGround: false,
      wasExcited: false, phoneZoom: 0, energyVis: 0, N: 0, poolI: 0, flash: 0, kick: 0,
    };
    const echo = { on: false, n: 1, age: 0, wr: 1, wz: 0, co: 1, so: 0 };
    const postKick = { bloomStrength: 0.75 };

    // ---- textures & shared uniforms ----------------------------------------------------------------
    const coefData = new Float32Array(ROWS * NM * 4);
    const coefTex = new THREE.DataTexture(coefData, NM, ROWS, THREE.RGBAFormat, THREE.FloatType);
    coefTex.needsUpdate = true;
    const modeCols = []; for (let n = 1; n <= NM; n++) modeCols.push(modeColor(n));
    const res = new THREE.Vector2(1, 1);
    const U = {
      uCoef: { value: coefTex }, uLen: { value: st.len }, uTheta: { value: TAU }, uOpenBasis: { value: 0 },
      uFloorY: { value: -(R_REF + FLOOR_GAP) }, uTime: { value: 0 }, uPR: { value: ctx.pixelRatio }, uRes: { value: res },
      uBase: { value: palette.planck.clone() }, uModeCol: { value: modeCols }, uSatK: { value: 0.035 }, uRipple: { value: 0 },
      uFuzzCol: { value: palette.violet.clone().lerp(palette.ink, 0.35) },
      uPulse: { value: [new THREE.Vector4(0, 0, 0, 0), new THREE.Vector4(0, 0, 0, 0)] },
    };
    const pulseT0 = [0, 0], pulseA = [0, 0]; let pulseHead = 0;
    const baseClosed = palette.planck.clone().lerp(new THREE.Color('#ff9a5c'), 0.15);
    const baseOpen = palette.harmonic.clone().lerp(palette.planck, 0.18);

    function stringMaterial(mirror) {
      return new THREE.ShaderMaterial({
        uniforms: { ...U, uMirror: { value: mirror ? 1 : 0 }, uFade: { value: 0.22 } },
        vertexShader: STRING_VS, fragmentShader: STRING_FS,
        transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
      });
    }

    // ---- the strings: one instanced draw (+ its reflection) ------------------------------------------
    const sGeo = ribbonGeometry(SEG);
    const iRow = new Float32Array(ROWS), iAlpha = new Float32Array(ROWS), iStyle = new Float32Array(ROWS * 4), iSoft = new Float32Array(ROWS);
    const style = (r, w, core, halo, tint) => { iStyle[r * 4] = w; iStyle[r * 4 + 1] = core; iStyle[r * 4 + 2] = halo; iStyle[r * 4 + 3] = tint; };
    for (let r = 0; r < ROWS; r++) iRow[r] = r;
    style(0, 11, 2.1, 0.78, 0); iAlpha[0] = 1;                                // the string: white-hot core
    for (let i = 1; i <= NT; i++) style(i, 6, 0, 0.9, 0);                     // trails: colour only, no core
    for (let j = 0; j < NF; j++) style(1 + NT + j, 4.5, 0.42, 0.75, 0.22);   // quantum ghosts: fine silk threads
    style(ECHO, 8, 0.8, 1.1, 0);                                              // echo of a pad's pure harmonic
    const aAlpha = new THREE.InstancedBufferAttribute(iAlpha, 1); aAlpha.setUsage(THREE.DynamicDrawUsage);
    sGeo.setAttribute('iRow', new THREE.InstancedBufferAttribute(iRow, 1));
    sGeo.setAttribute('iAlpha', aAlpha);
    // the quantum ghosts restyle every frame with ħ (silk threads at low ħ, a soft haze at high ħ)
    const aStyle = new THREE.InstancedBufferAttribute(iStyle, 4); aStyle.setUsage(THREE.DynamicDrawUsage);
    const aSoft = new THREE.InstancedBufferAttribute(iSoft, 1); aSoft.setUsage(THREE.DynamicDrawUsage);
    sGeo.setAttribute('iStyle', aStyle);
    sGeo.setAttribute('iSoft', aSoft);
    sGeo.instanceCount = ROWS;
    // the reflection only carries the string and its trails (the quantum ghosts would be invisible at that
    // fade anyway), so it shares every attribute but draws fewer instances
    const mGeo = new THREE.InstancedBufferGeometry();
    mGeo.setIndex(sGeo.index);
    for (const key in sGeo.attributes) mGeo.setAttribute(key, sGeo.attributes[key]);
    mGeo.instanceCount = 1 + NT;
    const strings = new THREE.Mesh(sGeo, stringMaterial(false));
    const stringsMirror = new THREE.Mesh(mGeo, stringMaterial(true));
    for (const m of [strings, stringsMirror]) { m.frustumCulled = false; m.renderOrder = 2; scene.add(m); }
    stringsMirror.renderOrder = 1;

    // ---- emitted quanta ----------------------------------------------------------------------------
    const QT = 2;                              // instances per quantum: body + streak
    const QI = MAX_Q * QT;
    const qGeo = ribbonGeometry(64);
    const qPosA = new Float32Array(QI * 3), qE1A = new Float32Array(QI * 3), qE2A = new Float32Array(QI * 3), qPA = new Float32Array(QI * 4);
    const qAttr = {
      iPos: new THREE.InstancedBufferAttribute(qPosA, 3), iE1: new THREE.InstancedBufferAttribute(qE1A, 3),
      iE2: new THREE.InstancedBufferAttribute(qE2A, 3), iP: new THREE.InstancedBufferAttribute(qPA, 4),
    };
    for (const key in qAttr) { qAttr[key].setUsage(THREE.DynamicDrawUsage); qGeo.setAttribute(key, qAttr[key]); }
    qGeo.instanceCount = QI;
    const qU = { uColLoop: { value: hdr(palette.resonance, 1.6) }, uColOpen: { value: hdr(palette.harmonic, 1.5) } };
    const quantaMat = (mirror) => new THREE.ShaderMaterial({
      uniforms: { uPR: U.uPR, uRes: U.uRes, uFloorY: U.uFloorY, uTime: U.uTime, uRipple: U.uRipple, uMirror: { value: mirror ? 1 : 0 }, uFade: { value: 0.3 }, ...qU },
      vertexShader: QUANTA_VS, fragmentShader: QUANTA_FS,
      transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    });
    const quanta = new THREE.Mesh(qGeo, quantaMat(false));
    const quantaMirror = new THREE.Mesh(qGeo, quantaMat(true));
    for (const m of [quanta, quantaMirror]) { m.frustumCulled = false; m.renderOrder = 3; scene.add(m); }
    // per-quantum simulation state
    const qs = {
      on: new Uint8Array(MAX_Q), pos: new Float32Array(MAX_Q * 3), vel: new Float32Array(MAX_Q * 3),
      e1: new Float32Array(MAX_Q * 3), b: new Float32Array(MAX_Q * 3), c: new Float32Array(MAX_Q * 3),
      age: new Float32Array(MAX_Q), life: new Float32Array(MAX_Q), r: new Float32Array(MAX_Q),
      kind: new Uint8Array(MAX_Q), spin: new Float32Array(MAX_Q), phase: new Float32Array(MAX_Q),
    };
    let qNext = 0;

    // ---- endpoint beads, hover marker ------------------------------------------------------------------
    const glow = glowTexture();
    const sprite = (color, size, opacity = 1) => {
      const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: glow, color, transparent: true, opacity, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending }));
      s.scale.setScalar(size); s.renderOrder = 4; scene.add(s); return s;
    };
    const beads = [sprite(hdr(palette.harmonic, 1.25), 0.26), sprite(hdr(palette.harmonic, 1.25), 0.26)];
    const beadCores = [sprite(hdr(palette.ink, 3.2), 0.075), sprite(hdr(palette.ink, 3.2), 0.075)];
    const beadsMirror = [sprite(hdr(palette.harmonic, 0.8), 0.32, 0.3), sprite(hdr(palette.harmonic, 0.8), 0.32, 0.3)];
    // the string lights the air around it: one huge, very faint glow behind it (kept under the bloom threshold)
    const air = sprite(palette.planck.clone(), 9, 0.05); air.renderOrder = 0;
    const hover = sprite(hdr(palette.planck, 2.6), 0.42, 0);
    // release flash: a brief anamorphic star streak (a long horizontal and a short vertical line of light)
    const flash = sprite(hdr(palette.ink, 4), 0.6, 0); flash.visible = false;
    const flashV = sprite(hdr(palette.ink, 4), 0.6, 0); flashV.visible = false;
    const hoverCore = sprite(hdr(palette.ink, 3), 0.12, 0);

    // ---- stage: floor, nebula, stars, dust -------------------------------------------------------------
    const ripU = []; for (let i = 0; i < 6; i++) ripU.push(new THREE.Vector4(0, 0, 0, -1));
    const horizon = new THREE.Color(0.024, 0.0125, 0.038);   // linear; far below the bloom threshold
    const floorMat = new THREE.ShaderMaterial({
      uniforms: {
        uPool: { value: new THREE.Color(0, 0, 0) }, uFog: { value: horizon.clone().multiplyScalar(0.85) }, uCam: { value: camera.position },
        uTime: U.uTime, uPoolR: { value: 2.6 }, uRip: { value: ripU },
      },
      vertexShader: FLOOR_VS, fragmentShader: FLOOR_FS,
    });
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(900, 900), floorMat);
    floor.rotation.x = -Math.PI / 2; floor.position.y = st.floorY; floor.renderOrder = -2;
    scene.add(floor);
    let ripHead = 0;

    // drawn after the opaque floor at the far plane (xyww) with depth testing, so early-z skips every
    // pixel the floor already covers
    const nebulaMat = new THREE.ShaderMaterial({ uniforms: { uTime: U.uTime, uHorizon: { value: horizon } }, vertexShader: NEBULA_VS, fragmentShader: NEBULA_FS, side: THREE.BackSide, depthWrite: false, depthTest: true });
    const nebula = new THREE.Mesh(new THREE.SphereGeometry(120, 48, 24), nebulaMat);
    nebula.renderOrder = -1; nebula.frustumCulled = false; scene.add(nebula);

    const stars = makeStarfield({ count: low ? 1500 : 3000, radius: 110, inner: 60, size: 1.7, intensity: 1.5 });
    const dust = makeStarfield({ count: low ? 300 : 600, radius: 16, inner: 6, size: 1.3, intensity: 0.9, tint: palette.planck, tintAmount: 0.3 });
    scene.add(stars, dust);

    // foreground bokeh: a loose ring of motes around the stage, so some always drift between lens and string
    const NB = low ? 10 : 18;
    const bPos = new Float32Array(NB * 3), bSeed = new Float32Array(NB);
    for (let i = 0; i < NB; i++) {
      const a = Math.PI / 2 + ((i + Math.random()) / NB - 0.5) * 2.2, r = 4.6 + Math.random() * 4.2;
      bPos[i * 3] = Math.cos(a) * r; bPos[i * 3 + 1] = -1.2 + Math.random() * 3.6; bPos[i * 3 + 2] = Math.sin(a) * r;
      bSeed[i] = Math.random();
    }
    const bGeo = new THREE.BufferGeometry();
    bGeo.setAttribute('position', new THREE.BufferAttribute(bPos, 3));
    bGeo.setAttribute('aSeed', new THREE.BufferAttribute(bSeed, 1));
    const bokehMat = new THREE.ShaderMaterial({
      uniforms: {
        uTime: U.uTime, uFocus: { value: 9 }, uScale: { value: 1 }, uGain: { value: 1 },
        uC0: { value: palette.violet.clone().multiplyScalar(0.15) }, uC1: { value: palette.planck.clone().lerp(palette.ink, 0.45).multiplyScalar(0.08) },
        uC2: { value: palette.harmonic.clone().multiplyScalar(0.09) }, uHero: { value: new THREE.Vector2() },
      },
      vertexShader: BOKEH_VS, fragmentShader: BOKEH_FS,
      transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
    });
    const bokehBase = [0, 1, 2].map((j) => bokehMat.uniforms['uC' + j].value.clone());
    const bokeh = new THREE.Points(bGeo, bokehMat);
    bokeh.frustumCulled = false; bokeh.renderOrder = 5; scene.add(bokeh);

    // ---- camera & controls ----------------------------------------------------------------------------
    const controls = new OrbitControls(camera, ctx.canvas);
    controls.enabled = false;
    controls.enableDamping = true; controls.dampingFactor = 0.08;
    controls.enablePan = false; controls.rotateSpeed = 0.55; controls.zoomSpeed = 0.6;
    controls.minPolarAngle = 0.62; controls.maxPolarAngle = 1.52;
    controls.target.set(0, 0, 0);
    let lastUser = -10, orbiting = false;
    controls.addEventListener('start', () => { orbiting = true; lastUser = ctx.time; });
    controls.addEventListener('end', () => { orbiting = false; lastUser = ctx.time; });
    let fitPrev = 0;

    // ---- scratch objects (no allocation in the frame loop) ----------------------------------------------
    const _p = new THREE.Vector3(), _n = new THREE.Vector3(), _v = new THREE.Vector3(), _w = new THREE.Vector3();
    const _hit = new THREE.Vector3(), _dir = new THREE.Vector3(), _off = new THREE.Vector3();
    const _plane = new THREE.Plane(), _ray = new THREE.Raycaster(), _sph = new THREE.Spherical();
    const _col = new THREE.Color();

    // ---- string geometry on the CPU (mirrors the vertex shader; used for picking, beads, emission) --------
    function baseCurve(s, outP, outN) {
      const u = s - 0.5, th = st.theta, L = st.len;
      if (th < 1e-3) { outP.set(u * L, 0, 0); outN.set(0, -1, 0); return; }
      const rho = L / th, a = th * u, h = 0.5 * th;
      outP.set(rho * Math.sin(a), rho * (1 - Math.cos(a)) - rho * (1 - Math.sin(h) / h), 0);
      outN.set(Math.sin(a), -Math.cos(a), 0);
    }
    const sigScale = () => (st.open ? Math.PI : TAU);
    function shapeAt(s, coef, out) {
      baseCurve(s, out, _n);
      const sig = sigScale() * s;
      let dr = 0, dz = 0;
      for (let n = 1; n <= NM; n++) {
        const i = (n - 1) * 4, cs = Math.cos(n * sig), sn = Math.sin(n * sig);
        dr += coef[i] * cs + coef[i + 1] * sn; dz += coef[i + 2] * cs + coef[i + 3] * sn;
      }
      out.addScaledVector(_n, dr); out.z += dz;
      return out;
    }

    // ---- energies & quanta ------------------------------------------------------------------------------
    const omega = (n) => n * st.omega1;
    function modeEnergy(n, q, v) {
      const i = (n - 1) * 4, w = omega(n);
      let e = 0;
      for (let c = 0; c < 4; c++) { const qq = q[i + c], vv = v ? v[i + c] : 0; e += 0.5 * (vv * vv + w * w * qq * qq); }
      return e;
    }
    const levelOf = (kk) => { let N = 0; for (let n = 1; n <= NM; n++) N += n * kk[n - 1]; return N; };

    // A Gaussian bump of height A at σ₀, projected onto modes 1..8 analytically:
    //   closed:  aₙ = (A w√(2π)/π) e^{−n²w²/2} (cos nσ₀, sin nσ₀)
    //   open:    aₙ = (2A w√(2π)/π) e^{−n²w²/2} cos nσ₀        (free ends: cosine series)
    // The n = 0 term (centre-of-mass shift) is dropped, and the series is normalised so its peak
    // sits exactly under the finger.
    function bump(s, Ar, Az, w, out) {
      const sig0 = sigScale() * s, open = st.open;
      const C = (open ? 2 : 1) * w * Math.sqrt(TAU) / Math.PI;
      let peak = 0;
      for (let n = 1; n <= NM; n++) { const g = C * Math.exp(-0.5 * n * n * w * w); peak += open ? g * Math.cos(n * sig0) ** 2 : g; }
      peak = Math.max(peak, 0.35);
      for (let n = 1; n <= NM; n++) {
        const g = C * Math.exp(-0.5 * n * n * w * w) / peak, i = (n - 1) * 4;
        const cs = Math.cos(n * sig0), sn = open ? 0 : Math.sin(n * sig0);
        out[i] = Ar * g * cs; out[i + 1] = Ar * g * sn; out[i + 2] = Az * g * cs; out[i + 3] = Az * g * sn;
      }
    }
    const bumpWidth = () => (st.open ? 0.32 : 0.38);

    // Add one standing wave of mode n with the energy of q quanta, random polarisation and orientation.
    // A pad keeps its polarisation within ±34° of the loop's plane: the out-of-plane direction points at the
    // lens, so a wave polarised along it would be almost invisible.
    const lastWave = { wr: 1, wz: 0, co: 1, so: 0 };
    function addWave(n, quanta, phase, inPlane) {
      const i = (n - 1) * 4, w = omega(n), a = Math.sqrt(2 * quanta * n * EPS) / w;
      const pol = inPlane ? (Math.random() - 0.5) * 1.2 + (Math.random() < 0.5 ? 0 : Math.PI) : Math.random() * TAU;
      const pr = Math.cos(pol), pz = Math.sin(pol) * 0.85;
      const pl = Math.hypot(pr, pz) || 1;
      const ori = st.open ? 0 : Math.random() * TAU, co = Math.cos(ori), so = st.open ? 0 : Math.sin(ori);
      const cq = Math.cos(phase), sv = -Math.sin(phase) * w;
      const wr = pr / pl, wz = pz / pl;
      lastWave.wr = wr; lastWave.wz = wz; lastWave.co = co; lastWave.so = so;
      Q[i] += a * wr * co * cq; Q[i + 1] += a * wr * so * cq; Q[i + 2] += a * wz * co * cq; Q[i + 3] += a * wz * so * cq;
      V[i] += a * wr * co * sv; V[i + 1] += a * wr * so * sv; V[i + 2] += a * wz * co * sv; V[i + 3] += a * wz * so * sv;
    }
    function reseed() {
      Q.fill(0); V.fill(0);
      for (let n = 1; n <= NM; n++) if (k[n - 1] > 0) addWave(n, k[n - 1], Math.random() * TAU);
    }
    function dominant(kk) { let best = 0, bn = 0; for (let n = 1; n <= NM; n++) { const e = n * kk[n - 1]; if (e > best) { best = e; bn = n; } } return bn; }

    // ---- sound helpers ----------------------------------------------------------------------------------
    let voice = null, bed = null, audioAcc = 0, bedBuf = null, bedCtx = null;
    function startBed() {
      const a = ctx.audio, ac = a.ctx;
      if (!ac) return null;
      if (!bedBuf || bedCtx !== ac) {
        // gently pink noise, 2 s loop
        const len = ac.sampleRate * 2; bedBuf = ac.createBuffer(1, len, ac.sampleRate); bedCtx = ac;
        const d = bedBuf.getChannelData(0); let b0 = 0, b1 = 0;
        for (let i = 0; i < len; i++) { const x = Math.random() * 2 - 1; b0 = 0.97 * b0 + 0.03 * x; b1 = 0.6 * b1 + 0.4 * x; d[i] = (b1 - b0 * 0.6) * 0.9; }
      }
      const src = ac.createBufferSource(); src.buffer = bedBuf; src.loop = true;
      const bp = ac.createBiquadFilter(); bp.type = 'bandpass'; bp.frequency.value = 3800; bp.Q.value = 0.45;
      const g = ac.createGain(); g.gain.value = 0;
      src.connect(bp).connect(g).connect(a.out);
      src.start();
      src.onended = () => { try { src.disconnect(); bp.disconnect(); g.disconnect(); } catch (e) { /* already gone */ } };
      return { src, bp, g };
    }
    function stopBed() {
      if (!bed) return;
      const ac = ctx.audio.ctx;
      try { const t = ac.currentTime; bed.g.gain.cancelScheduledValues(t); bed.g.gain.setTargetAtTime(0, t, 0.12); bed.src.stop(t + 0.7); } catch (e) { /* context closed */ }
      bed = null;
    }
    function stopVoices() { if (voice) { voice.stop(0.6); voice = null; } stopBed(); }

    // ---- floor ripples (a struck note runs out across the stage) ---------------------------------------
    function ripple(color, strength) {
      const r = ripU[ripHead]; ripHead = (ripHead + 1) % ripU.length;
      r.set(color.r * strength, color.g * strength, color.b * strength, 0);
      U.uRipple.value = Math.min(1.5, U.uRipple.value + strength * 2);
    }

    // ---- emission of a massless quantum (decay by one quantum of mode n) --------------------------------
    function emit(n) {
      // leave from where mode n is moving the string the most
      const i0 = (n - 1) * 4, sig = sigScale();
      let bestS = 0.25, best = -1;
      for (let j = 0; j < 40; j++) {
        const s = (j + 0.5) / 40, cs = Math.cos(n * sig * s), sn = Math.sin(n * sig * s);
        const r = row0[i0] * cs + row0[i0 + 1] * sn, z = row0[i0 + 2] * cs + row0[i0 + 3] * sn;
        const m = r * r + z * z + Math.random() * 1e-4;
        if (m > best) { best = m; bestS = s; }
      }
      shapeAt(bestS, row0, _p); baseCurve(bestS, _v, _n);
      const q = qNext; qNext = (qNext + 1) % MAX_Q;
      qs.on[q] = 1; qs.age[q] = 0; qs.life[q] = 2.6 + Math.random() * 1.2;
      qs.kind[q] = st.open ? 1 : 0; qs.r[q] = (st.open ? 0.09 : 0.085) * (st.R / R_REF);
      qs.spin[q] = (Math.random() - 0.5) * 5; qs.phase[q] = Math.random() * TAU;
      // closed: leave outward; open: either side of the segment. Never dive steeply into the stage.
      _dir.copy(_n).multiplyScalar(st.open && Math.random() < 0.5 ? -0.9 : 0.9);
      _dir.x += (Math.random() - 0.5) * 0.7; _dir.y += (Math.random() - 0.2) * 0.6; _dir.z += (Math.random() - 0.5) * 1.3;
      _dir.y = Math.max(_dir.y, -0.22);
      _dir.normalize();
      // keep it from flying at the lens: cap the component pointing back toward the camera
      camera.getWorldDirection(_w);
      const toward = -_dir.dot(_w);
      if (toward > 0) { _dir.addScaledVector(_w, toward).normalize(); }
      const sp = 1.9 + Math.random() * 0.9;
      qs.pos[q * 3] = _p.x; qs.pos[q * 3 + 1] = _p.y; qs.pos[q * 3 + 2] = _p.z;
      qs.vel[q * 3] = _dir.x * sp; qs.vel[q * 3 + 1] = _dir.y * sp; qs.vel[q * 3 + 2] = _dir.z * sp;
      // orientation frame, built from the camera's axes: e1 and b span the screen plane (so a loop reads as a
      // ring, never a bar seen edge-on) and c points at the lens; the body wobbles between b and c as it flies
      const me = camera.matrixWorld.elements, ang = Math.random() * TAU, ca = Math.cos(ang), sa = Math.sin(ang);
      for (let x = 0; x < 3; x++) {
        const right = me[x], up = me[4 + x];
        qs.e1[q * 3 + x] = right * ca + up * sa;
        qs.b[q * 3 + x] = -right * sa + up * ca;
        qs.c[q * 3 + x] = me[8 + x];
      }
      if (st.open) {
        // a photon-like open quantum flies along its own length, so it reads as a dash of light
        camera.getWorldDirection(_w);
        _v.copy(_dir).addScaledVector(_w, -_dir.dot(_w));
        if (_v.lengthSq() > 1e-6) {
          _v.normalize(); _w.cross(_v).normalize();
          qs.e1[q * 3] = _v.x; qs.e1[q * 3 + 1] = _v.y; qs.e1[q * 3 + 2] = _v.z;
          qs.b[q * 3] = _w.x; qs.b[q * 3 + 1] = _w.y; qs.b[q * 3 + 2] = _w.z;
        }
      }

      const a = ctx.audio, now = ctx.time;
      if (a.on && now - st.lastEmitSound > 0.09) {
        st.lastEmitSound = now;
        a.pluck(a.degree(14 + Math.floor(Math.random() * 6), 0), { gain: 0.055, decay: 1.5, bright: 0.85, partials: 4, pan: clamp(_p.x / 4, -0.8, 0.8) });
      }
      const flag = st.open ? 'toastEmitOpen' : 'toastEmitClosed';
      if (!st[flag]) {
        st[flag] = true;
        const after = levelOf(k), before = after + n, what = st.open ? 'a <em>photon-like quantum</em>' : 'a <em>graviton</em>';
        ctx.ui.toast(ctx.isTouch ? `Radiated ${what}: level ${before} to ${after}.` : `Your string radiated ${what} and fell from level ${before} to level ${after}.`, 4200);
      }
    }

    // 0 inside the readout, the controls and the toast band (CSS px rectangles, refreshed about once a second)
    const panelRects = [];
    let rectsAge = 9;
    function refreshRects() {
      panelRects.length = 0;
      const cr = ctx.canvas.getBoundingClientRect();
      for (const el of [readout && readout.el, ctrls && ctrls.get('pads') && ctrls.get('pads').closest('.controls')]) {
        if (!el) continue;
        const r = el.getBoundingClientRect();
        if (r.width > 0) panelRects.push(r.left - cr.left, r.top - cr.top, r.right - cr.left, r.bottom - cr.top);
      }
      const w = ctx.width;
      panelRects.push(w * 0.5 - 280, 0, w * 0.5 + 280, ctx.aspect < 1 ? 0 : 92);
    }
    function screenMask(x, y, z) {
      _v.set(x, y, z).project(camera);
      const sx = (_v.x * 0.5 + 0.5) * ctx.width, sy = (-_v.y * 0.5 + 0.5) * ctx.height;
      let m = 1;
      for (let i = 0; i < panelRects.length; i += 4) {
        const dx = Math.max(panelRects[i] - sx, sx - panelRects[i + 2], 0), dy = Math.max(panelRects[i + 1] - sy, sy - panelRects[i + 3], 0);
        m = Math.min(m, smoothstep(0, 40, Math.hypot(dx, dy)));
      }
      return m;
    }

    const STREAK_T = 0.32;   // the streak covers the last 0.32 s of flight
    function updateQuanta(dt) {
      for (let q = 0; q < MAX_Q; q++) {
        const ib = q * QT, is = ib + 1;
        if (!qs.on[q]) { qPA[ib * 4 + 1] = 0; qPA[is * 4 + 1] = 0; continue; }
        qs.age[q] += dt;
        const u = qs.age[q] / qs.life[q];
        if (u >= 1) { qs.on[q] = 0; qPA[ib * 4 + 1] = 0; qPA[is * 4 + 1] = 0; continue; }
        const p = q * 3;
        qs.pos[p] += qs.vel[p] * dt; qs.pos[p + 1] += qs.vel[p + 1] * dt; qs.pos[p + 2] += qs.vel[p + 2] * dt;
        qs.phase[q] += qs.spin[q] * dt;
        const ph = qs.phase[q], wob = 0.85 * Math.sin(ph), cph = Math.cos(wob), sph = Math.sin(wob);
        // fades with age, and with distance from the stage so a quantum dissolves before it reaches the programme
        const px = qs.pos[p], py = qs.pos[p + 1], pz = qs.pos[p + 2];
        const dist = Math.hypot(px, py, pz) / (st.R / R_REF);
        const fade = smoothstep(0, 0.05, u) * (1 - smoothstep(0.4, 1, u)) * (1 - smoothstep(2.6, 4.4, dist)) * screenMask(px, py, pz);
        // body: e1 fixed, e2 rocking about it by up to ±49° (a wobble, never edge-on)
        const ob = ib * 3;
        qPosA[ob] = qs.pos[p]; qPosA[ob + 1] = qs.pos[p + 1]; qPosA[ob + 2] = qs.pos[p + 2];
        qE1A[ob] = qs.e1[p]; qE1A[ob + 1] = qs.e1[p + 1]; qE1A[ob + 2] = qs.e1[p + 2];
        qE2A[ob] = qs.b[p] * cph + qs.c[p] * sph; qE2A[ob + 1] = qs.b[p + 1] * cph + qs.c[p + 1] * sph; qE2A[ob + 2] = qs.b[p + 2] * cph + qs.c[p + 2] * sph;
        qPA[ib * 4] = qs.r[q]; qPA[ib * 4 + 1] = fade; qPA[ib * 4 + 2] = qs.kind[q]; qPA[ib * 4 + 3] = ph * 2.0;
        // streak: from the body back along the path it has actually flown (never longer than its age)
        const os = is * 3, back = Math.min(qs.age[q], STREAK_T);
        qPosA[os] = qs.pos[p]; qPosA[os + 1] = qs.pos[p + 1]; qPosA[os + 2] = qs.pos[p + 2];
        qE1A[os] = qs.vel[p] * back; qE1A[os + 1] = qs.vel[p + 1] * back; qE1A[os + 2] = qs.vel[p + 2] * back;
        qPA[is * 4] = 0; qPA[is * 4 + 1] = fade * 0.55; qPA[is * 4 + 2] = 2 + qs.kind[q]; qPA[is * 4 + 3] = 0;
      }
      for (const key in qAttr) qAttr[key].needsUpdate = true;
    }

    // ---- actions ------------------------------------------------------------------------------------------
    let readout = null, ctrls = null;
    const grab = { on: false, s: 0, Ar: 0, Az: 0, vr: 0, vz: 0, lastT: 0, p0: new THREE.Vector3(), nr: new THREE.Vector3() };

    function noteInteraction() {
      if (!st.interacted) { st.interacted = true; ctx.ui.hideHint(); }
      st.draining = false;
    }

    // Round the energy a pluck added to (or took from) each mode into whole quanta. Only the change is
    // rounded, so a mode that is still relaxing after a pad press keeps the quanta it already has. Returns ΔN.
    const Ebefore = new Float32Array(NM), env = new Float32Array(NM).fill(0.35);
    function quantize() {
      const before = levelOf(k);
      for (let n = 1; n <= NM; n++) {
        const dE = modeEnergy(n, Q, V) - Ebefore[n - 1], dk = Math.round(dE / (n * EPS));
        k[n - 1] = clamp(k[n - 1] + dk, 0, KMAX);
        if (dk > 0) env[n - 1] = 1;
      }
      return levelOf(k) - before;
    }

    // Release a held bump: it becomes initial displacement (plus the finger's velocity) of modes 1..8.
    function release(flick) {
      const w = bumpWidth() * (1 - 0.45 * flick);   // a fast flick is a sharper pluck: more high harmonics
      for (let n = 1; n <= NM; n++) Ebefore[n - 1] = modeEnergy(n, Q, V);
      bump(grab.s, grab.Ar, grab.Az, w, H);
      for (let i = 0; i < NM * 4; i++) Q[i] += H[i];
      bump(grab.s, grab.vr, grab.vz, w, TMP);
      for (let i = 0; i < NM * 4; i++) V[i] += TMP[i] * 0.55;
      H.fill(0);
      const pulled = Math.hypot(grab.Ar, grab.Az);
      const dN = quantize();
      const N = levelOf(k);
      const a = ctx.audio;
      if (N > 0) {
        const str = clamp(dN / 18, 0.15, 1);
        a.pluck(st.f1, { gain: 0.1 + 0.15 * str, decay: 2.6, bright: 0.3 + 0.55 * flick, partials: 8, pan: clamp(grab.p0.x / 4, -0.7, 0.7) });
        const dn = dominant(k) || 1;
        ripple(modeCols[dn - 1], 0.05 + 0.08 * str);
        shapeAt(grab.s, row0, _p); flash.position.copy(_p); flashV.position.copy(_p);
        flash.material.color.copy(modeCols[dn - 1]).lerp(palette.ink, 0.55).multiplyScalar(4);
        flashV.material.color.copy(flash.material.color);
        st.flash = 0.6 + 0.4 * str; st.kick = Math.max(st.kick, 0.5 + 0.5 * str);
        // d'Alembert: the released bump splits into two half-height pulses that run apart at dσ/dt = ω₁
        const pq = U.uPulse.value[pulseHead]; pulseT0[pulseHead] = ctx.time; pulseA[pulseHead] = 0.55 + 0.6 * str;
        pq.set(grab.s, 0, pulseA[pulseHead], 0); pulseHead = (pulseHead + 1) % 2;
      } else if (pulled > 0.04 * st.R && !st.toastZero) {
        st.toastZero = true;
        ctx.ui.toast('Too gentle: that pluck rounded to <em>zero quanta</em>. A string can only hold whole quanta.', 3800);
      }
    }

    function pickString(x, y, radius) {
      // nearest point on the string to the pointer, in screen pixels
      const w = ctx.width, h = ctx.height, M = 180;
      let best = radius * radius, bs = -1;
      for (let j = 0; j < M; j++) {
        const s = st.morph > 0.5 ? j / (M - 1) : j / M;
        shapeAt(s, row0, _p).project(camera);
        const px = (_p.x * 0.5 + 0.5) * w, py = (-_p.y * 0.5 + 0.5) * h;
        const d = (px - x) * (px - x) + (py - y) * (py - y);
        if (d < best) { best = d; bs = s; }
      }
      return bs;
    }

    function addQuantum(n) {
      noteInteraction();
      const i = n - 1;
      if (k[i] >= KMAX) { ctx.ui.toast(`Mode ${n} is full here (${KMAX} quanta).`, 1600); return; }
      k[i]++; env[i] = 1;
      addWave(n, 1, 0, true);
      // echo: a large, fading ghost of the pure harmonic flares out of the string in its own colour,
      // always in the loop's plane so its n waves are easy to count
      echo.on = true; echo.n = n; echo.age = 0;
      echo.wr = 1; echo.wz = 0; echo.co = lastWave.co; echo.so = lastWave.so;
      const a = ctx.audio, now = ctx.time;
      if (now - st.lastPadSound > 0.08) {
        st.lastPadSound = now;
        a.pluck(st.f1 * n, { gain: 0.2, decay: 2.4 - n * 0.12, bright: 0.45, partials: 3, pan: (n - 4.5) * 0.08 });
      }
      ripple(modeCols[i], 0.09);
    }

    function setOpen(open) {
      if (open === st.openTarget) return;
      noteInteraction();
      if (grab.on) cancelGrab();
      hoverS = -1;   // the shape is about to change: drop the marker until the pointer moves again
      st.openTarget = open;
      st.swap = 1;
      ctrls && ctrls.set('topo', open ? 'open' : 'closed');
      const a = ctx.audio;
      a.noise({ freq: open ? 500 : 2600, sweepTo: open ? 2600 : 500, dur: 0.7, gain: 0.035, q: 1.2 });
      a.bell(st.f1 * (open ? 3 : 2), { gain: 0.08 });
      if (open && !st.toastOpen) {
        st.toastOpen = true;
        ctx.ui.toast('Open string: its massless state is now spin 1, a <em>photon-like</em> particle. Its ends are free to move.', 4600);
      }
    }

    function applyTension(T) {
      st.T = T;
      st.f1 = F_REF * Math.sqrt(T);
      st.omega1 = OMEGA_REF * Math.sqrt(T);
      st.R = R_REF / Math.sqrt(T);
      st.openLen = OPEN_LEN_REF / Math.sqrt(T);
      refreshPads();
    }
    // Snap the tension so the fundamental lands on a D-minor-pentatonic note.
    function snapTension() {
      let best = 1, bd = 1e9;
      for (const s of SNAP_SEMIS) { const T = (ctx.audio.freq(s) / F_REF) ** 2; const d = Math.abs(Math.log(T / st.T)); if (d < bd) { bd = d; best = T; } }
      applyTension(best);
      ctrls && ctrls.set('tension', best);
    }
    // Keyboard on the tension slider: step from one snap note to the next instead of by 0.01
    // (a native step would fire 'change' and snap straight back).
    function tensionKey(e) {
      const up = e.key === 'ArrowRight' || e.key === 'ArrowUp' || e.key === 'PageUp', dn = e.key === 'ArrowLeft' || e.key === 'ArrowDown' || e.key === 'PageDown';
      if (!up && !dn && e.key !== 'Home' && e.key !== 'End') return;
      e.preventDefault();
      const Ts = SNAP_SEMIS.map((x) => (ctx.audio.freq(x) / F_REF) ** 2).filter((T) => T >= 0.5 && T <= 2);
      let i = 0;
      for (let j = 1; j < Ts.length; j++) if (Math.abs(Math.log(Ts[j] / st.T)) < Math.abs(Math.log(Ts[i] / st.T))) i = j;
      i = e.key === 'Home' ? 0 : e.key === 'End' ? Ts.length - 1 : clamp(i + (up ? 1 : -1), 0, Ts.length - 1);
      applyTension(Ts[i]);
      ctrls && ctrls.set('tension', Ts[i]);
    }
    function setHbar(h) { st.hbar = h; }

    function dampAll() {
      noteInteraction();
      if (grab.on) cancelGrab();
      hoverS = -1;   // the shape is about to change: drop the marker until the pointer moves again
      k.fill(0);
      st.draining = true;
      ctx.audio.noise({ freq: 1600, sweepTo: 260, dur: 0.55, gain: 0.04, q: 0.7 });
    }

    function cancelGrab() {
      grab.on = false; H.fill(0);
      controls.enabled = active;
      ctx.setCursor('');
    }

    // ---- input ---------------------------------------------------------------------------------------------
    let active = false;
    // event time (s), so flick speed does not depend on how many pointer events share one frame
    const evTime = (ev) => ((ev && ev.original && ev.original.timeStamp) || performance.now()) / 1000;
    let touchN = 0;   // touches landed since the pointer went down: a second finger never starts a pluck
    function pointerToAmp(ev) {
      _ray.setFromCamera(ev.ndc, camera);
      if (!_ray.ray.intersectPlane(_plane, _hit)) return false;
      _hit.sub(grab.p0);
      let ar = _hit.dot(grab.nr), az = _hit.z;
      const m = Math.hypot(ar, az), MAXP = 1.4 * st.R / R_REF;
      if (m > 1e-6) { const s = MAXP * Math.tanh(m / MAXP) / m; ar *= s; az *= s; }
      const now = evTime(ev), dtm = Math.max(1 / 240, now - grab.lastT);
      grab.vr = lerp(grab.vr, (ar - grab.Ar) / dtm, 0.45);
      grab.vz = lerp(grab.vz, (az - grab.Az) / dtm, 0.45);
      grab.lastT = now;
      grab.Ar = ar; grab.Az = az;
      return true;
    }

    function onPointerDown(ev) {
      if (ev.pointerType === 'touch') touchN++;
      if (st.swap || grab.on) return;      // a second finger never steals a pluck in progress
      if (touchN > 1) return;              // ...nor turns a pinch into a pluck
      const radius = ev.pointerType === 'touch' ? 58 : 40;
      const s = pickString(ev.x, ev.y, radius);
      if (s < 0) { lastUser = ctx.time; return; }
      noteInteraction();
      controls.enabled = false;
      grab.on = true; grab.s = s; grab.Ar = grab.Az = grab.vr = grab.vz = 0; grab.lastT = evTime(ev);
      shapeAt(s, row0, grab.p0);
      baseCurve(s, _v, grab.nr);
      camera.getWorldDirection(_dir);
      _plane.setFromNormalAndCoplanarPoint(_dir, grab.p0);
      ctx.setCursor('grabbing');
      ctx.audio.noise({ freq: 5200, dur: 0.06, gain: 0.02, q: 3 });
    }
    function onPointerMove(ev) {
      if (grab.on) {
        if (pointerToAmp(ev)) bump(grab.s, grab.Ar, grab.Az, bumpWidth(), H);
        return;
      }
      if (ctx.pointer.down || st.swap) return;
      const s = pickString(ev.x, ev.y, ev.pointerType === 'touch' ? 58 : 40);
      hoverS = s;
      ctx.setCursor(s >= 0 ? 'grab' : '');
    }
    function onPointerUp(ev) {
      touchN = 0;
      if (!grab.on) { lastUser = ctx.time; return; }
      // a pause before letting go kills the flick
      const idle = Math.max(0, evTime(ev) - grab.lastT);
      const decay = Math.exp(-idle * 14);
      grab.vr *= decay; grab.vz *= decay;
      const speed = Math.hypot(grab.vr, grab.vz);
      const flick = smoothstep(1.5, 9, speed);
      const vmax = 6 * st.R / R_REF;
      if (speed > vmax) { grab.vr *= vmax / speed; grab.vz *= vmax / speed; }
      release(flick);
      grab.on = false;
      controls.enabled = active;
      // the released string moves away from the cursor: re-pick on the next frames instead of leaving a stale marker
      hoverS = -1; hoverDirty = ev && ev.pointerType !== 'touch';
      ctx.setCursor('');
      lastUser = ctx.time;
    }
    let hoverS = -1, hoverDirty = false, hoverAcc = 0;

    const PHI0 = 1.40;   // resting polar angle: eye about 10° above the stage, so the floor reads as a receding mirror
    // ---- camera framing ------------------------------------------------------------------------------------
    // Landscape layout: the hero prefers to sit a little right of centre, but must stay between the movement
    // rail (left) and the column of panels (right, ~376 px). On narrower screens it slides left first and
    // only shrinks when there is no room. layout.xc is the hero's screen x in CSS px.
    const layout = { xc: 0, ox: NaN, oy: NaN };
    function fitDistance() {
      const w = Math.max(1, ctx.width);
      const asp = clamp(ctx.aspect || 1, 0.25, 4), tv = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)), portrait = asp < 1;
      const sz = Math.pow(st.R / R_REF, 0.35);              // let tension visibly resize the string, but keep it framed
      const extH = lerp(R_REF, OPEN_LEN_REF * 0.5, st.morph) * sz;
      const extV = lerp(R_REF, OPEN_LEN_REF * 0.2, st.morph) * sz;
      // portrait: until the first pluck the loop is sized so the phone hint (fixed by the shell at 46% height)
      // sits inside its hollow; after that it grows, and the open string spans most of the width
      const pz = st.phoneZoom;
      const fv = portrait ? lerp(0.215, 0.24, pz) : 0.41, fh = portrait ? lerp(lerp(0.46, 0.53, pz), 0.8, st.morph) : lerp(0.56, 0.4, st.morph);
      let d = Math.max(extV / (tv * fv), extH / (tv * asp * fh));
      if (portrait) { layout.xc = w / 2; return d; }
      const g = clamp(0.024 * w, 16, 36), panelL = w - g - 376, railR = g + 170;
      const avail = Math.max(60, (panelL - railR) / 2 - 24);
      const ext = extH * 1.08;                               // perspective margin
      d = Math.max(d, ext / (tv * asp * (avail / (w / 2))));
      const hw = ext / (d * tv * asp) * (w / 2);
      const lo = railR + 24 + hw;
      layout.xc = clamp(0.522 * w, lo, Math.max(lo, panelL - 24 - hw));
      return d;
    }
    function applyOffset() {
      const w = Math.max(1, ctx.width), h = Math.max(1, ctx.height), portrait = w / h < 1;
      const ox = portrait ? 0 : -(layout.xc - w / 2), oy = (portrait ? 0.03 : 0.085) * h;
      if (Math.abs(ox - layout.ox) < 0.5 && Math.abs(oy - layout.oy) < 0.5 && camera.view && camera.view.fullWidth === w && camera.view.fullHeight === h) return;
      layout.ox = ox; layout.oy = oy;
      camera.setViewOffset(w, h, ox, oy, w, h);
    }
    function applyView() {
      fitDistance();
      applyOffset();
      if (active) ctx.setPost(ctx.aspect < 1 ? POST_PORTRAIT : POST_LAND);
    }
    applyView();
    camera.position.set(0, 0, 1).applyAxisAngle(_w.set(1, 0, 0), -(Math.PI / 2 - PHI0)).multiplyScalar(fitDistance());
    camera.lookAt(0, 0, 0);
    fitPrev = fitDistance();
    controls.minDistance = fitPrev * 0.55; controls.maxDistance = fitPrev * 1.9;
    controls.update();

    function updateCamera(dt, t) {
      const fd = fitDistance();
      controls.target.y = reduced ? 0 : 0.06 * Math.sin(t * 0.27);
      _off.copy(camera.position).sub(controls.target);
      if (Math.abs(fd - fitPrev) > 1e-5) {
        _off.multiplyScalar(fd / fitPrev);
        controls.minDistance = fd * 0.55; controls.maxDistance = fd * 1.9;
        fitPrev = fd;
      }
      applyOffset();
      if (!orbiting && !grab.on && t - lastUser > 3.5) {
        // idle: a slow sway around the instrument, never a full turn (edge-on would hide the loop)
        _sph.setFromVector3(_off);
        const amp = reduced ? 0.05 : 0.34;
        const thT = amp * Math.sin(t * 0.075) + (reduced ? 0 : 0.06 * Math.sin(t * 0.19));
        const phT = PHI0 + (reduced ? 0 : 0.03 * Math.sin(t * 0.11));
        let d = thT - _sph.theta; d = Math.atan2(Math.sin(d), Math.cos(d));
        const kk = damp(0.35, dt);
        _sph.theta += d * kk; _sph.phi += (phT - _sph.phi) * kk;
        _off.setFromSpherical(_sph);
      }
      camera.position.copy(controls.target).add(_off);
      controls.update();
    }

    // ---- readout -----------------------------------------------------------------------------------------------
    const SPEC_HTML = '<div style="display:grid;grid-template-columns:repeat(8,1fr);gap:5px;align-items:end;height:42px;margin:2px 0 3px;border-bottom:1px solid rgba(236,231,246,.16)">'
      + modeCols.map((c) => `<i class="m2-bar" style="display:block;height:0;min-height:1px;border-radius:1px 1px 0 0;background:#${c.getHexString()};box-shadow:0 0 9px -2px #${c.getHexString()}"></i>`).join('')
      + '</div><div style="display:grid;grid-template-columns:repeat(8,1fr);gap:5px;text-align:center;font-size:9px;letter-spacing:.04em;color:rgba(236,231,246,.5);margin:0 0 6px">'
      + modeCols.map((c, i) => `<span style="color:#${c.getHexString()}">${i + 1}</span>`).join('') + '</div>';
    let narrow = false, bars = [], barH = new Float32Array(NM), barScale = -1, lastN = -1, lastNote = '', lastPadF = 0, lastTitle = '';
    const EQ = new Float32Array(NM * 4);

    function updateReadout() {
      // while a pluck is held, show what it would become if released now
      let kk = k;
      if (grab.on) {
        for (let i = 0; i < NM * 4; i++) EQ[i] = Q[i] + H[i];
        for (let n = 1; n <= NM; n++) {
          Edisp[n - 1] = modeEnergy(n, EQ, V);
          kProv[n - 1] = clamp(k[n - 1] + Math.round((Edisp[n - 1] - modeEnergy(n, Q, V)) / (n * EPS)), 0, KMAX);
        }
        kk = kProv;
      } else for (let n = 0; n < NM; n++) Edisp[n] = E[n];
      const N = levelOf(kk);
      st.N = N;
      if (!readout) return;
      const closed = !st.openTarget;
      const dn = dominant(kk);
      const sig = ((N * 9 + dn) * 2 + (grab.on ? 1 : 0)) * 2 + (closed ? 1 : 0);
      if (sig !== lastSig || st.T !== lastT) { lastSig = sig; lastT = st.T; writeText(N, closed, dn); }
      updateSpectrum();
    }
    let lastSig = -1, lastT = -1;
    function writeText(N, closed, dn) {
      const title = N === 0 ? (closed ? 'Graviton' : 'Photon-like gauge boson') : 'Massive string state';
      if (title !== lastTitle) { readout.title(title); lastTitle = title; }
      readout.set('N', N + (grab.on ? ' · held' : ''));
      const mf = closed ? 2 : 1;
      readout.set('mass', N === 0 ? '0 · massless' : `${(mf * Math.sqrt(N)).toFixed(2)} Mₛ`);
      readout.set('spin', String(closed ? 2 * N + 2 : N + 1));
      readout.set('dom', dn ? `${ORD[dn]} · ${noteLabel(st.f1 * dn)}` : 'none');
      readout.set('pitch', `${st.f1.toFixed(1)} Hz · ${noteName(st.f1)}`);   // the fundamental; the voice sounds every harmonic
      let note;
      if (N === 0) note = narrow ? 'Massless: moves at light speed.' : closed ? 'Massless: travels at the speed of light. Spin 2. The same level also holds the dilaton and the B-field.' : 'Massless: travels at the speed of light. Spin 1, like the photon.';
      else {
        const gev = mf * Math.sqrt(N) * MS_GEV * Math.sqrt(st.T);
        const ex = Math.floor(Math.log10(gev)), man = gev / 10 ** ex;
        const ratio = gev / LHC_GEV / 1e12;
        const r = ratio >= 1000 ? `${(ratio / 1000).toFixed(ratio >= 10000 ? 0 : 1)} quadrillion` : `${ratio >= 100 ? Math.round(ratio / 10) * 10 : Math.round(ratio)} trillion`;
        note = narrow ? `≈ ${man.toFixed(1)} × 10${sup(ex)} GeV, ${r} × the LHC.` : `≈ ${man.toFixed(1)} × 10${sup(ex)} GeV: about ${r} times what the Large Hadron Collider reaches.`;
      }
      if (note !== lastNote) { readout.note(note); lastNote = note; }
      if (ctrls && N !== lastN) { ctrls.set('pads', `level N = ${N}`); lastN = N; }
    }
    function updateSpectrum() {
      // spectrum: bar height = energy in units of the level it contributes; stripes mark single quanta
      let mx = 0; for (let n = 0; n < NM; n++) mx = Math.max(mx, Edisp[n] / EPS);
      const scale = Math.max(8, Math.ceil(mx / 4) * 4);
      const restyle = scale !== barScale; barScale = scale;
      for (let n = 1; n <= NM && n <= bars.length; n++) {
        const b = bars[n - 1], h = Math.min(42, 42 * (Edisp[n - 1] / EPS) / scale);
        if (Math.abs(h - barH[n - 1]) > 0.3) { b.style.height = h.toFixed(1) + 'px'; barH[n - 1] = h; }
        if (restyle) {
          const u = n * 42 / scale, c = '#' + modeCols[n - 1].getHexString();
          b.style.background = u >= 3.5 ? `repeating-linear-gradient(to top, ${c} 0, ${c} ${(u - 1.2).toFixed(2)}px, rgba(5,4,10,.75) ${(u - 1.2).toFixed(2)}px, rgba(5,4,10,.75) ${u.toFixed(2)}px)` : c;
        }
      }
    }
    function refreshPads() {
      if (!ctrls || Math.abs(lastPadF - st.f1) < 0.01) return;
      lastPadF = st.f1;
      const el = ctrls.get('pads'); if (!el) return;
      [...el.children].forEach((b, i) => { const sm = b.querySelector('small'); if (sm) sm.textContent = noteLabel(st.f1 * (i + 1)); });
    }

    // ---- debug staging params (harmless when absent) ------------------------------------------------------------
    if (params.has('m2open')) { st.open = st.openTarget = true; st.morph = 1; U.uOpenBasis.value = 1; }
    if (params.has('m2hbar')) st.hbar = clamp(parseFloat(params.get('m2hbar')) || 0, 0, 1);
    if (params.has('m2T')) applyTension(clamp(parseFloat(params.get('m2T')) || 1, 0.5, 2));
    if (params.has('m2pads')) { params.get('m2pads').split(',').forEach((x) => { const n = parseInt(x, 10); if (n >= 1 && n <= NM) { k[n - 1]++; } }); reseed(); st.interacted = true; }

    let hoverA = 0;

    // ---- frame loop ------------------------------------------------------------------------------------------------
    function update(dt, t) {
      U.uTime.value = t;
      U.uPR.value = ctx.pixelRatio;
      ctx.renderer.getDrawingBufferSize(res);

      // 1. open/closed swap: fade the modes out, change basis, re-seed the same quanta, fade in
      if (st.swap === 1) {
        st.amp = Math.max(0, st.amp - dt / 0.22);
        if (st.amp <= 0) {
          st.open = st.openTarget; U.uOpenBasis.value = st.open ? 1 : 0;
          reseed(); fz.fill(0); heroFz.fill(0); hist.fill(0);
          for (const q of U.uPulse.value) q.z = 0;
          st.swap = 2;
        }
      } else if (st.swap === 2) {
        st.amp = Math.min(1, st.amp + dt / 0.5);
        if (st.amp >= 1) st.swap = 0;
      }
      const mt = st.openTarget ? 1 : 0;
      st.morph += (mt - st.morph) * damp(2.4, dt);
      if (Math.abs(mt - st.morph) < 2e-4) st.morph = mt;
      const m = st.morph, me = m * m * (3 - 2 * m);
      st.theta = TAU * (1 - m);
      st.len = lerp(TAU * st.R, st.openLen, me);
      U.uTheta.value = st.theta; U.uLen.value = st.len;
      U.uBase.value.copy(baseClosed).lerp(baseOpen, me);

      // 2. exact harmonic evolution of all 32 mode coordinates, then relax toward the quantised energy
      let Etot = 0;
      for (let n = 1; n <= NM; n++) {
        const i = (n - 1) * 4, w = omega(n), c = Math.cos(w * dt), s = Math.sin(w * dt);
        let e = 0;
        for (let j = 0; j < 4; j++) {
          const q = Q[i + j], v = V[i + j];
          const q2 = q * c + (v / w) * s, v2 = -q * w * s + v * c;
          Q[i + j] = q2; V[i + j] = v2;
          e += 0.5 * (v2 * v2 + w * w * q2 * q2);
        }
        const target = k[n - 1] * n * EPS;
        let f = 1;
        if (target === 0) f = Math.exp(-(st.draining ? 5.5 : 2.2) * dt);
        else if (e < 1e-9) { addWave(n, k[n - 1], Math.random() * TAU); e = target; }
        else { const r = Math.sqrt(target / e); f = 1 + (r - 1) * damp(r < 1 ? 3.2 : 5, dt); }
        if (f !== 1) for (let j = 0; j < 4; j++) { Q[i + j] *= f; V[i + j] *= f; }
        E[n - 1] = e * f * f; Etot += E[n - 1];
      }
      if (st.draining && Etot < 1e-4) st.draining = false;

      // 3. decay by emission: each quantum of mode n leaves at rate GAMMA·n
      st.emitCool -= dt;
      if (!grab.on && !st.draining && st.swap === 0 && st.emitCool <= 0) {
        let rate = 0; for (let n = 1; n <= NM; n++) rate += GAMMA * n * k[n - 1];
        if (rate > 0 && Math.random() < 1 - Math.exp(-rate * dt)) {
          let x = Math.random() * rate, n = 1;
          for (; n < NM; n++) { x -= GAMMA * n * k[n - 1]; if (x <= 0) break; }
          if (k[n - 1] > 0) { k[n - 1]--; emit(n); st.emitCool = 0.14; }
        }
      }

      // 4. zero-point motion: Ornstein–Uhlenbeck jitter with stationary spread ∝ √(ħ/n)
      // (an open string has one coefficient per polarisation per mode where a loop has two, so √2 keeps the
      // same ħ looking equally fuzzy)
      const sc = st.R / R_REF, hb = st.hbar, openK = st.open ? Math.SQRT2 : 1;
      for (let n = 1; n <= NM; n++) {
        const th = 1.5 + 0.9 * n, a = Math.exp(-th * dt), b = Math.sqrt(1 - a * a);
        const sb = FUZZ_S * sc * openK * Math.sqrt(hb / n) * b, i = (n - 1) * 4;
        for (let g = 0; g <= NF; g++) {
          const arr = g === NF ? heroFz : fz, o = g === NF ? i : g * NM * 4 + i, amp = g === NF ? sb * 0.3 : sb;
          arr[o] = arr[o] * a + amp * gauss(); arr[o + 2] = arr[o + 2] * a + amp * gauss();
          if (st.open) { arr[o + 1] = 0; arr[o + 3] = 0; } else { arr[o + 1] = arr[o + 1] * a + amp * gauss(); arr[o + 3] = arr[o + 3] * a + amp * gauss(); }
        }
      }

      // 5. coefficient rows: 0 = the string, then motion trails from history, then the quantum ghosts
      const A = st.amp;
      for (let i = 0; i < NM * 4; i++) row0[i] = (Q[i] + H[i]) * A + heroFz[i];
      for (let i = 0; i < NM * 4; i++) coefData[i] = row0[i];
      histAcc += dt;
      while (histAcc >= 1 / 60) { histAcc -= 1 / 60; histHead = (histHead + 1) % HN; hist.set(row0, histHead * NM * 4); }
      for (let r = 1; r <= NT; r++) {
        const h = ((histHead - r * TSTEP) % HN + HN) % HN, src = h * NM * 4, dst = r * NM * 4;
        for (let i = 0; i < NM * 4; i++) coefData[dst + i] = hist[src + i];
      }
      for (let g = 0; g < NF; g++) {
        const dst = (1 + NT + g) * NM * 4, src = g * NM * 4;
        for (let i = 0; i < NM * 4; i++) coefData[dst + i] = (Q[i] + H[i]) * A + fz[src + i];
      }
      const er = ECHO * NM * 4;
      for (let i = 0; i < NM * 4; i++) coefData[er + i] = row0[i];
      if (echo.on) {
        echo.age += dt;
        if (echo.age > 2.4) echo.on = false;
        const j = er + (echo.n - 1) * 4, ea = 0.3 * (st.R / R_REF) * Math.cos(omega(echo.n) * echo.age) * A;
        coefData[j] += ea * echo.wr * echo.co; coefData[j + 1] += ea * echo.wr * echo.so;
        coefData[j + 2] += ea * echo.wz * echo.co; coefData[j + 3] += ea * echo.wz * echo.so;
      }
      coefTex.needsUpdate = true;

      // 6. per-string brightness: trails only show when the string moves; the cloud scales with ħ
      let Eh = 0; if (grab.on) Eh = Math.hypot(grab.Ar, grab.Az) * 3;
      const motion = smoothstep(0, 2.5 * EPS, Etot * A * A + Eh);
      st.energyVis += (motion - st.energyVis) * damp(4, dt);
      const ta = (low ? 0.12 : 0.075) * st.energyVis;
      for (let r = 1; r <= NT; r++) iAlpha[r] = ta * Math.pow(1 - (r - 1) / NT, 1.2);
      // quantum ghosts: fine silk threads at low ħ that widen and soften into one haze as ħ rises
      // (alpha falls as they widen, so the cloud's total brightness stays level)
      const fa = (0.36 * Math.sqrt(hb) + 0.25 * hb) / Math.sqrt(NF) * smoothstep(0, 0.04, hb) / (1 + 1.5 * hb);
      const gw = 4.5 + 9 * hb, gc = 0.42 * (1 - 0.85 * hb), gt = 0.22 - 0.12 * hb, gs = smoothstep(0.15, 0.7, hb);
      for (let g = 0; g < NF; g++) {
        const r = 1 + NT + g;
        iAlpha[r] = fa * (0.65 + 0.35 * ((g * 0.618) % 1));
        iStyle[r * 4] = gw; iStyle[r * 4 + 1] = gc; iStyle[r * 4 + 3] = gt; iSoft[r] = gs;
      }
      if (hb !== lastGhostH) { lastGhostH = hb; aStyle.needsUpdate = true; aSoft.needsUpdate = true; }
      iAlpha[0] = 1;
      iAlpha[ECHO] = echo.on ? 0.75 * Math.exp(-echo.age / 0.7) * smoothstep(0, 0.05, echo.age) : 0;
      aAlpha.needsUpdate = true;
      // the two pluck pulses run apart along the string: a loop is 1 in s, so they go ω₁/2π loops per second;
      // an open string (σ ∈ [0, π]) carries them at ω₁/π
      for (let j = 0; j < 2; j++) {
        const q = U.uPulse.value[j];
        if (q.z <= 0) continue;
        const age = t - pulseT0[j];
        q.y = age * st.omega1 / (st.open ? Math.PI : TAU);
        q.z = pulseA[j] * Math.exp(-age / 1.4) * A * smoothstep(0, 0.04, age);
        if (age > 6) q.z = 0;
      }

      // 7. emitted quanta
      updateQuanta(dt);

      // 8. endpoint beads (open string), hover / grab marker
      const bo = smoothstep(0.55, 0.95, st.morph);
      for (let j = 0; j < 2; j++) {
        const vis = bo > 0.002;
        beads[j].visible = beadCores[j].visible = beadsMirror[j].visible = vis;
        if (!vis) continue;
        shapeAt(j, row0, _p);
        beads[j].position.copy(_p); beadCores[j].position.copy(_p);
        beadsMirror[j].position.set(_p.x, 2 * st.floorY - _p.y, _p.z);
        const pulse = 1 + 0.12 * Math.sin(t * 3.1 + j * 2) + 0.3 * st.energyVis;
        beads[j].material.opacity = bo; beadCores[j].material.opacity = bo; beadsMirror[j].material.opacity = 0.28 * bo;
        beads[j].scale.setScalar(0.32 * pulse * Math.sqrt(sc)); beadCores[j].scale.setScalar(0.1 * Math.sqrt(sc));
      }
      // keep the hover marker on the nearest point of the moving string (about ten times a second)
      hoverAcc += dt;
      if (active && !grab.on && !ctx.pointer.down && !ctx.isTouch && !st.swap && (hoverDirty || (hoverS >= 0 && hoverAcc > 0.1))) {
        hoverAcc = 0; hoverDirty = false;
        hoverS = pickString(ctx.pointer.x, ctx.pointer.y, 40);
        ctx.setCursor(hoverS >= 0 ? 'grab' : '');
      }
      if (!ctx.pointer.down) touchN = 0;
      let hv = 0;
      if (grab.on) { shapeAt(grab.s, row0, _p); hv = 1; }
      else if (hoverS >= 0 && !ctx.isTouch && active) { shapeAt(hoverS, row0, _p); hv = 0.5; }
      if (hv > 0) { hover.position.copy(_p); hoverCore.position.copy(_p); }
      hoverA += (hv - hoverA) * damp(12, dt);
      hover.material.opacity = hoverA; hoverCore.material.opacity = hoverA;
      hover.visible = hoverCore.visible = hoverA > 0.01;
      hover.scale.setScalar(0.42 * (1 + 0.15 * Math.sin(t * 6)) * (grab.on ? 1.25 : 1));
      // pluck flash: a burst of light where the string was let go, plus a brief swell of the bloom
      if (st.flash > 0.004) {
        st.flash *= Math.exp(-dt / 0.12);
        const L = (1.4 + (1 - st.flash) * 1.2) * Math.sqrt(sc);
        flash.visible = flashV.visible = true; flash.material.opacity = st.flash; flashV.material.opacity = 0.7 * st.flash;
        flash.scale.set(L, L / 12, 1); flashV.scale.set(L / 20, L * 0.4, 1);
      } else if (flash.visible) { flash.visible = flashV.visible = false; st.flash = 0; }
      if (st.kick > 0 && active) {
        st.kick *= Math.exp(-5 * dt);
        if (st.kick < 0.002) st.kick = 0;
        postKick.bloomStrength = (ctx.aspect < 1 ? POST_PORTRAIT : POST_LAND).bloomStrength + 0.4 * st.kick;
        ctx.setPost(postKick);
      }

      // 9. stage: light pool, ripples, sky
      const dn = dominant(k);
      _col.copy(U.uBase.value);
      if (dn) _col.lerp(modeCols[dn - 1], 0.5);
      const pi = 0.025 + 0.045 * st.energyVis + 0.015 * hb;
      st.poolI += (pi - st.poolI) * damp(3, dt);
      floorMat.uniforms.uPool.value.copy(_col).multiplyScalar(st.poolI);
      air.material.color.copy(_col);
      air.material.opacity = 0.75 * st.poolI;
      air.scale.setScalar(lerp(7.5, 9.5, st.morph) * Math.sqrt(sc));
      air.position.set(0, 0, -0.6);
      floorMat.uniforms.uPoolR.value = lerp(2.4, 3.4, st.morph) * Math.sqrt(sc);
      st.floorY = -(lerp(st.R, 0.42 * sc, me) + FLOOR_GAP); U.uFloorY.value = st.floorY; floor.position.y = st.floorY;
      for (const r of ripU) if (r.w >= 0) { r.w += dt; if (r.w > 3.2) r.w = -1; }
      U.uRipple.value *= Math.exp(-1.6 * dt);
      stars.userData.update(t); dust.userData.update(t);
      dust.rotation.y = t * 0.006;

      // 10. camera (and the bokeh's focal plane, which sits on the string)
      updateCamera(dt, t);
      bokehMat.uniforms.uFocus.value = camera.position.length();
      bokehMat.uniforms.uGain.value = 0.6 + 0.8 * st.energyVis;
      _v.set(0, 0, 0).project(camera); bokehMat.uniforms.uHero.value.set(_v.x, _v.y);
      if (dn !== lastBokehDn) {
        lastBokehDn = dn;
        for (let j = 0; j < 3; j++) { const u = bokehMat.uniforms['uC' + j].value; u.copy(bokehBase[j]); if (dn) u.lerp(modeCols[dn - 1].clone().multiplyScalar(bokehBase[j].r + bokehBase[j].g + bokehBase[j].b), 0.45); }
      }
      st.phoneZoom += ((st.interacted ? 1 : 0) - st.phoneZoom) * damp(1.2, dt);
      rectsAge += dt; if (rectsAge > 1 && active) { rectsAge = 0; refreshRects(); }
      bokehMat.uniforms.uScale.value = res.y / 1080;

      // 11. sound: one additive voice whose partials are √(energy) of each mode, plus the zero-point hiss
      const au = ctx.audio;
      if (au.on && active) {
        if (!voice) { voice = au.additive(st.f1, NM, { gain: 0.15 }); lastVoiceF = st.f1; }
        if (!bed) bed = startBed();
      } else if (voice || bed) stopVoices();
      for (let n = 0; n < NM; n++) env[n] += (0.35 - env[n]) * damp(0.4, dt);
      audioAcc += dt;
      if (audioAcc >= 1 / 30 && (voice || bed)) {
        audioAcc = 0;
        let sum = 0;
        // timbre = √(energy) per mode; each partial rings up when its mode is struck and settles over ~3 s
        for (let n = 0; n < NM; n++) { amps[n] = 0.3 * Math.sqrt(E[n] * A * A / EPS) * env[n]; sum += amps[n]; }
        if (sum > 1) for (let n = 0; n < NM; n++) amps[n] /= sum;
        if (voice) {
          voice.setAmps(amps, 0.05);
          if (Math.abs(lastVoiceF - st.f1) > 0.01) { voice.setFreq(st.f1, 0.06); lastVoiceF = st.f1; }
          _sph.setFromVector3(_off); voice.setPan(Math.sin(_sph.theta) * 0.35);
        }
        if (bed && au.ctx) bed.g.gain.setTargetAtTime(0.05 * Math.pow(hb, 1.3), au.ctx.currentTime, 0.25);
      }

      // 12. readout and discoveries
      updateReadout();
      if (st.N > 0 && !grab.on) st.wasExcited = true;
      if (st.wasExcited && st.N === 0 && !grab.on && !st.openTarget && !st.toastGround && Etot < 0.5 * EPS) {
        st.toastGround = true; st.wasExcited = false;
        ctx.ui.toast('Silent again. A closed string with no quanta is massless and has spin 2: a <em>graviton</em>.', 4200);
      }
    }
    let lastVoiceF = 0, lastGhostH = -1, lastBokehDn = -1;

    // ---- panels --------------------------------------------------------------------------------------------------------
    function buildPanels() {
      narrow = ctx.width <= 760 || ctx.height < 740;   // phones and short windows: a compact card
      const rows = [
        { id: 'N', k: 'Level N', v: '0' },
        { id: 'mass', k: 'Mass', v: '0 · massless', cls: 'gold' },
        { id: 'spin', k: 'Max spin', v: '2' },
      ];
      // compact: the readout shares a small screen with the string and the controls, so keep it short
      if (!narrow) rows.push({ id: 'dom', k: 'Dominant harmonic', v: 'none' }, { id: 'pitch', k: 'Fundamental', v: '', cls: 'cyan' });
      readout = ctx.ui.readout({ title: 'Graviton', html: SPEC_HTML, rows, note: '' });
      bars = [...readout.el.querySelectorAll('.m2-bar')];
      barH.fill(-1); barScale = -1; lastN = -1; lastNote = ''; lastTitle = ''; lastPadF = 0; lastSig = -1; lastT = -1;
      ctrls = ctx.ui.controls([
        {
          type: 'keys', id: 'pads', label: ctx.isTouch ? 'Add a quantum' : 'Add a quantum · keys 1–8',
          keys: modeCols.map((c, i) => ({ label: String(i + 1), sub: noteLabel(st.f1 * (i + 1)), color: '#' + c.getHexString(), kbd: String(i + 1), onPress: () => addQuantum(i + 1) })),
        },
        {
          type: 'segmented', id: 'topo', label: ctx.isTouch ? 'String' : 'String · O', value: st.openTarget ? 'open' : 'closed',
          options: [{ value: 'closed', label: 'Closed loop' }, { value: 'open', label: 'Open string' }],
          onChange: (v) => setOpen(v === 'open'),
        },
        {
          type: 'slider', id: 'tension', label: `Tension T ∝ 1/${GK('α′')}`, min: 0.5, max: 2, step: 0.01, value: st.T,
          format: (v) => `×${v.toFixed(2)} · ${noteName(F_REF * Math.sqrt(v))}`,
          onInput: (v) => applyTension(v),
        },
        {
          type: 'slider', id: 'hbar', label: `Quantum fuzz ${GK('ħ')}`, min: 0, max: 1, step: 0.01, value: st.hbar,
          format: (v) => v.toFixed(2), onInput: (v) => setHbar(v),
        },
        { type: 'button', id: 'damp', label: ctx.isTouch ? 'Damp' : 'Damp · Space', kbd: ' ', onClick: () => dampAll() },
      ]);
      // pointer release snaps (keyboard steps are intercepted first, so 'change' only follows a drag)
      const tin = ctrls.get('tension');
      if (tin) { tin.addEventListener('change', snapTension); tin.addEventListener('keydown', tensionKey); }
      rectsAge = 9;
      refreshPads();
    }

    return {
      scene, camera,
      enter() {
        active = true;
        controls.enabled = true;
        lastUser = ctx.time;
        buildPanels();
        ctx.setPost(ctx.aspect < 1 ? POST_PORTRAIT : POST_LAND);
        // touch: short enough to sit inside the loop on a phone
        if (!st.interacted) ctx.ui.hint(ctx.isTouch ? 'Drag to pluck' : 'Drag the string to pluck · keys 1–8');
        else ctx.ui.hint(ctx.isTouch ? 'Drag to pluck' : 'Drag the string · 1–8 · O opens it', 3500);
      },
      exit() {
        active = false;
        st.kick = 0;
        if (grab.on) cancelGrab();
        controls.enabled = false;
        orbiting = false;
        stopVoices();
        ctx.setCursor('');
        hoverS = -1;
        readout = null; ctrls = null; bars = [];
      },
      update,
      resize() { applyView(); },
      onPointerDown, onPointerMove, onPointerUp,
      onKey(ev) {
        if (ev.key === 'o' || ev.key === 'O') { if (!ev.repeat) setOpen(!st.openTarget); return true; }
        return false;
      },
      demo() {
        noteInteraction();
        setHbar(0.35); if (ctrls) ctrls.set('hbar', 0.35);
        // a programmatic pluck at σ = 0.3 of the string, released with a little flick
        const s = 0.3, sc = st.R / R_REF;
        grab.s = s; shapeAt(s, row0, grab.p0); baseCurve(s, _v, grab.nr);
        grab.Ar = 0.85 * sc; grab.Az = 0.4 * sc; grab.vr = 2.0 * sc; grab.vz = 1.5 * sc;
        release(0.3);
        addQuantum(2); addQuantum(5);
      },
    };
  },
};
