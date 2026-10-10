// Shared visual vocabulary: palette, glow sprites, star/dust fields and GLSL snippets.
// Movements import nothing from each other; everything shared lives here.
import * as THREE from 'three';

// ---- palette ---------------------------------------------------------------
// Linear-space colours. Multiply by >1 (see hdr()) for anything that should bloom.
export const palette = {
  void: new THREE.Color('#05040a'),
  ink: new THREE.Color('#ece7f6'),
  planck: new THREE.Color('#ffc46b'),     // gold: the accent, "the string itself"
  harmonic: new THREE.Color('#6ff3ff'),   // cyan: low modes, open strings, light
  resonance: new THREE.Color('#ff4fbf'),  // magenta: high modes, closed strings, gravity
  violet: new THREE.Color('#8b6cff'),     // the hidden dimensions
  ember: new THREE.Color('#ff6a3d'),      // heat, tension, infinities
  mint: new THREE.Color('#7dffc4'),       // "habitable", resolution, success
  deep: new THREE.Color('#120d2a'),       // fog / far background tint
};

const SPECTRUM = ['#6ff3ff', '#5fb8ff', '#8b6cff', '#c45cff', '#ff4fbf', '#ff6a8a', '#ff9a5c', '#ffc46b'].map((c) => new THREE.Color(c));
/** Colour for harmonic/mode number n (1-based). Cyan -> violet -> magenta -> gold, then cycles. */
export function modeColor(n) {
  const i = ((Math.max(1, Math.round(n)) - 1) % SPECTRUM.length + SPECTRUM.length) % SPECTRUM.length;
  return SPECTRUM[i].clone();
}
/** Smooth spectrum lookup, t in [0,1]. */
export function spectrum(t) {
  const x = Math.min(0.9999, Math.max(0, t)) * (SPECTRUM.length - 1);
  const i = Math.floor(x);
  return SPECTRUM[i].clone().lerp(SPECTRUM[i + 1], x - i);
}
/** Same colour pushed above 1.0 so UnrealBloom picks it up. */
export function hdr(color, intensity = 2) {
  return new THREE.Color(color).multiplyScalar(intensity);
}
/** CSS hex for a THREE.Color (for matching DOM accents to the scene). */
export function css(color) { return '#' + new THREE.Color(color).getHexString(); }

// ---- textures --------------------------------------------------------------
let _glow;
/** Soft round sprite (white, premultiplied falloff). Use with AdditiveBlending. */
export function glowTexture() {
  if (_glow) return _glow;
  const s = 128, c = document.createElement('canvas'); c.width = c.height = s;
  const g = c.getContext('2d');
  const grd = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
  grd.addColorStop(0, 'rgba(255,255,255,1)');
  grd.addColorStop(0.18, 'rgba(255,255,255,0.75)');
  grd.addColorStop(0.45, 'rgba(255,255,255,0.18)');
  grd.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grd; g.fillRect(0, 0, s, s);
  _glow = new THREE.CanvasTexture(c);
  _glow.colorSpace = THREE.SRGBColorSpace;
  return _glow;
}

// ---- star / dust fields ------------------------------------------------------
/**
 * Twinkling point field on a spherical shell (or filled ball if inner=0).
 * opts: count (2500), radius (60), inner (0.6 * radius), size (1.6 px at 1080p), color (ink), intensity (1.2)
 * Returns THREE.Points; call points.userData.update(time) each frame for twinkle.
 */
export function makeStarfield(opts = {}) {
  const { count = 2500, radius = 60, inner = radius * 0.6, size = 1.6, color = palette.ink, intensity = 1.2, tint = palette.violet, tintAmount = 0.35 } = opts;
  const pos = new Float32Array(count * 3), seed = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const u = Math.random() * 2 - 1, th = Math.random() * Math.PI * 2;
    const r = inner + (radius - inner) * Math.cbrt(Math.random());
    const s = Math.sqrt(1 - u * u);
    pos.set([r * s * Math.cos(th), r * u, r * s * Math.sin(th)], i * 3);
    seed[i] = Math.random();
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('seed', new THREE.BufferAttribute(seed, 1));
  const mat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: {
      uTime: { value: 0 }, uSize: { value: size }, uColor: { value: new THREE.Color(color).multiplyScalar(intensity) },
      uTint: { value: new THREE.Color(tint).multiplyScalar(intensity) }, uTintAmt: { value: tintAmount },
      uScale: { value: window.innerHeight / 1080 * Math.min(window.devicePixelRatio, 2) },
    },
    vertexShader: /* glsl */`
      attribute float seed; uniform float uTime, uSize, uScale; varying float vA; varying float vS;
      void main(){
        vec4 mv = modelViewMatrix * vec4(position,1.0);
        float tw = 0.55 + 0.45 * sin(uTime * (0.6 + seed * 2.2) + seed * 40.0);
        vA = tw * (0.35 + 0.65 * fract(seed * 7.13)); vS = seed;
        gl_PointSize = uSize * uScale * (0.6 + 1.6 * pow(fract(seed * 3.7), 3.0)) * (40.0 / max(1.0, -mv.z)) ;
        gl_PointSize = clamp(gl_PointSize, 0.5, 6.0 * uScale + 1.0);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: /* glsl */`
      uniform vec3 uColor, uTint; uniform float uTintAmt; varying float vA; varying float vS;
      void main(){
        vec2 p = gl_PointCoord - 0.5; float d = length(p);
        float a = smoothstep(0.5, 0.0, d); a *= a;
        vec3 c = mix(uColor, uTint, step(1.0 - uTintAmt, fract(vS * 13.7)));
        gl_FragColor = vec4(c * a * vA, 1.0);
      }`,
  });
  const pts = new THREE.Points(geo, mat);
  pts.frustumCulled = false;
  pts.userData.update = (t) => { mat.uniforms.uTime.value = t; };
  return pts;
}

// ---- GLSL snippets -----------------------------------------------------------
export const GLSL = {
  /** float snoise(vec3) in [-1,1] (Ashima/Stefan Gustavson simplex noise, MIT). */
  noise: /* glsl */`
    vec3 mod289(vec3 x){return x-floor(x*(1.0/289.0))*289.0;}
    vec4 mod289(vec4 x){return x-floor(x*(1.0/289.0))*289.0;}
    vec4 permute(vec4 x){return mod289(((x*34.0)+10.0)*x);}
    vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}
    float snoise(vec3 v){
      const vec2 C=vec2(1.0/6.0,1.0/3.0); const vec4 D=vec4(0.0,0.5,1.0,2.0);
      vec3 i=floor(v+dot(v,C.yyy)); vec3 x0=v-i+dot(i,C.xxx);
      vec3 g=step(x0.yzx,x0.xyz); vec3 l=1.0-g; vec3 i1=min(g.xyz,l.zxy); vec3 i2=max(g.xyz,l.zxy);
      vec3 x1=x0-i1+C.xxx; vec3 x2=x0-i2+C.yyy; vec3 x3=x0-D.yyy;
      i=mod289(i);
      vec4 p=permute(permute(permute(i.z+vec4(0.0,i1.z,i2.z,1.0))+i.y+vec4(0.0,i1.y,i2.y,1.0))+i.x+vec4(0.0,i1.x,i2.x,1.0));
      float n_=0.142857142857; vec3 ns=n_*D.wyz-D.xzx;
      vec4 j=p-49.0*floor(p*ns.z*ns.z); vec4 x_=floor(j*ns.z); vec4 y_=floor(j-7.0*x_);
      vec4 x=x_*ns.x+ns.yyyy; vec4 y=y_*ns.x+ns.yyyy; vec4 h=1.0-abs(x)-abs(y);
      vec4 b0=vec4(x.xy,y.xy); vec4 b1=vec4(x.zw,y.zw);
      vec4 s0=floor(b0)*2.0+1.0; vec4 s1=floor(b1)*2.0+1.0; vec4 sh=-step(h,vec4(0.0));
      vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy; vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
      vec3 p0=vec3(a0.xy,h.x); vec3 p1=vec3(a0.zw,h.y); vec3 p2=vec3(a1.xy,h.z); vec3 p3=vec3(a1.zw,h.w);
      vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
      p0*=norm.x; p1*=norm.y; p2*=norm.z; p3*=norm.w;
      vec4 m=max(0.5-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.0); m=m*m;
      return 105.0*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
    }
    float fbm(vec3 p){ float a=0.5, s=0.0; for(int i=0;i<5;i++){ s+=a*snoise(p); p*=2.03; a*=0.5; } return s; }
  `,
  /** vec3 iridescence(float t): thin-film style hue ramp in the piece's palette (t any real). */
  iridescence: /* glsl */`
    vec3 iridescence(float t){
      vec3 a=vec3(0.55,0.45,0.75), b=vec3(0.45,0.4,0.35), c=vec3(1.0,1.0,1.0), d=vec3(0.0,0.18,0.42);
      return a + b*cos(6.28318*(c*t+d));
    }
  `,
};

// ---- small maths helpers ------------------------------------------------------
export const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
export const lerp = (a, b, t) => a + (b - a) * t;
export const smoothstep = (a, b, x) => { const t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t); };
/** Frame-rate independent exponential approach: x += (target - x) * damp(rate, dt). */
export const damp = (rate, dt) => 1 - Math.exp(-rate * dt);
/** Superscript digits for scientific labels, e.g. sup(-35) -> "⁻³⁵". */
export function sup(n) {
  const map = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '.': '·' };
  return String(n).split('').map((ch) => map[ch] ?? ch).join('');
}
/** Seeded PRNG (mulberry32) so procedural scenes look the same every visit. */
export function rng(seed = 1) {
  let a = seed >>> 0;
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
