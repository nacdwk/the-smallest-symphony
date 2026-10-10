# IV. Worldsheets — `src/movements/m4-worldsheet.js`

numeral `IV` · title `Worldsheets` · tempo `Scherzo, legato` ("smoothly connected")

## Job
Show why strings tame the infinities of quantum gravity. A point particle traces a line through spacetime; when two
particles interact, their lines meet at a sharp point, and that point is where gravity's calculations blow up. A
string traces a **surface** (its worldsheet); when two strings join, the surface is a smooth "pair of pants" with no
special point. Different observers even disagree about *where* the joining happened. The user discovers this by
slicing spacetime themselves.

## Scene
- Time runs **upward**. A raymarched (SDF) worldsheet: two tubes rising from below (two closed strings) that merge
  smoothly (polynomial smooth-min) into one tube above (one string). Optionally a gentle sway so it feels alive.
  Material: glassy iridescent membrane (fresnel rim, thin-film colour from `GLSL.iridescence`), faint horizontal
  contour lines every Δt (like a topographic map of time), subtle inner glow. Raymarch inside a bounding box mesh,
  ≤ 96 steps (48 on low quality), soft shadows not needed.
- **The "now" slice**: a translucent plane cutting through spacetime. Where it intersects the worldsheet, draw a
  bright gold contour (the strings as they exist at that instant). Below the crotch the contour is two loops; above it
  is one loop. Render the contour in the SDF shader (|plane distance| < ε near the surface) so it is perfectly smooth.
- **Observer tilt**: a slider (and shift+drag) tilts the slicing plane (a different observer's "now"). Mark with a
  pulsing point the place where, for this observer, the two loops first touch (the saddle point of the plane's height
  function on the surface). As you tilt, the point slides around the crotch: there is no unique place or time where the
  interaction happened. Compute it on the CPU: for the current plane normal, bisection on the plane offset h, at each
  h sample the SDF on a ~64×64 grid in the plane and test whether the cross-section is connected (flood fill); the
  bridge cell at the critical h gives the point. Recompute only when tilt changes (debounced).
- **Particles mode** (segmented control): the same process for point particles, drawn as two thin luminous worldlines
  meeting at one vertex and continuing as one line. The vertex flares `ember` with a pulsing "∞" label (CSS2D or a sprite)
  and tilting the slice never moves it: every observer agrees on the point, which is exactly the problem.
- Optional third option **Quantum loop**: a worldsheet with a handle (a tube that splits and rejoins), showing that
  quantum corrections are surfaces with holes.

## Interaction
Drag vertically on the scene (or the **Time** slider, or ↑/↓ consumed) moves the slice through time. Tilt slider and
shift+drag tilt it. Segmented control `Strings / Particles / Loop` (keys 1/2/3). Orbit with drag on empty space +
slow idle orbit. `demo()`: strings mode, slice positioned just above the crotch with a visible tilt so the gold
contour and the saddle marker both show; support debug param `slice=<-1..1>`.

## Readout
Mode, the time of the slice, number of strings in the slice (2 → 1, computed from your connectivity test), and for
strings "Interaction point: depends on observer" with the current marker coordinates; for particles "Interaction
point: (x, t) fixed for everyone" and "Gravity at this vertex: diverges".

## Sound
The slice is a chord: two tones (the two strings) below the crotch that glide together into one tone above it
(sustained `additive` voices whose pitch follows the loop circumferences). A soft bell when the topology changes.
Particles mode: a harsher, buzzing interval at the vertex (still in key, but with a beating detune) to sound
"infinite". drone e.g. `[0, 7, 15]`.

## Caption / fine print
- caption: A moving point draws a line through spacetime; a moving string draws a surface. When particles collide,
  their lines meet at a sharp point. When strings join, the surface stays smooth. Drag to slice through time.
- fine: worldlines vs worldsheets; why point interactions make quantum gravity non-renormalizable; the worldsheet
  has no preferred point, so string perturbation theory avoids those ultraviolet infinities (shown finite in every
  case worked out, with a general proof still an active topic); worldsheets with h handles are h-loop corrections;
  Lorentz observers slice spacetime differently (relativity of simultaneity).

## Look
Glass, gold slice, ember vertex. A clean, scientific-instrument elegance; contour lines give it a cartographic
beauty. Strong rim light, deep background starfield.
