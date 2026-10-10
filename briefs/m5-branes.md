# V. Branes — `src/movements/m5-branes.js`

numeral `V` · title `Branes` · tempo `Andante, sul ponticello`

## Job
Show D-branes and the braneworld idea: open strings have their ends stuck on membranes (branes); closed strings
(gravitons) are free to leave. Our whole universe might be a 3-dimensional brane floating in a higher-dimensional
space, which could explain why gravity is so much weaker than the other forces.

## Scene
- A vast shimmering membrane (our "D3-brane", drawn as a 2D sheet) stretching to the horizon: translucent, iridescent,
  faint grid, gently undulating (vertex shader waves) with **ripples that spread from every string endpoint and every
  impact**. Fog swallows the far edge. Above and below is the dark "bulk" with faint drifting dust.
- **Open strings**: glowing arcs whose two endpoints are beads stuck to the brane, sliding around on it, the arc above
  the brane vibrating. Colour `harmonic` cyan. They are the photons, gluons, electrons and quarks of a braneworld.
- **Closed strings**: small glowing loops (`resonance` magenta). When created they can lift off the brane and drift
  away into the bulk, fading: gravity leaking out.
- **A second brane**, parallel, above the first (separation d). Strings stretched *between* the two branes: their
  mass is tension × length, so as d grows they glow hotter (cyan → `ember`), thicker, and their tone rises. At d = 0
  (coincident branes) those stretched strings become massless and the readout shows the gauge symmetry enhance from
  U(1) × U(1) to U(2) (the stringy version of the Higgs mechanism, run in reverse).

## Interaction
- Click/tap the brane: spawn an open string there (its endpoints land, ripples, a plucked note pitched by where you
  clicked, pentatonic).
- **Collide** button (`c`): two nearby open strings join end to end into a closed loop, which detaches and floats away
  into the bulk (ripples, a rising bell, toast "A closed string left the brane: that is gravity leaking away").
- **Separation slider** d (and dragging the upper brane vertically). Button **Stretch** (`s`) adds a string between the branes.
- Orbit drag on empty space, wheel zoom, slow idle drift.
- `demo()`: spawn ~6 open strings, 2 stretched strings, set d to a mid value, trigger one collide so a closed loop is mid-flight.

## Readout
Open strings on the brane, closed strings escaped, separation d (in string lengths), stretched-string mass = T·d
(in M_s units), gauge group (U(1)×U(1) or U(2) when d ≈ 0), and a note line about gravity's weakness.

## Sound
Open strings pluck on spawn (pentatonic by position), stretched strings hold a sustained tone whose pitch rises with
d (quantised gently), closed-string escape is a rising bell + airy noise sweep. Ripples are faint watery noise.
drone e.g. `[0, 5, 12, 17]` (open fourths, glassy).

## Caption / fine print
- caption: Some strings are open, with two ends. Those ends cannot float free: they are stuck to membranes called
  branes. Closed loops have no ends, so they can leave. Our universe may be one brane, with only gravity able to escape.
- fine: D-branes (Dirichlet boundary conditions; Polchinski 1995); open strings give gauge fields, N coincident branes
  give U(N); separated branes give massive W-like strings (mass = tension × distance); braneworld scenarios (large extra
  dimensions, Randall–Sundrum) as possible, untested explanations for gravity's weakness; the brane here is drawn 2D
  but stands for our 3D space.

## Look
The membrane is the star: a sheet of liquid light seen at a low grazing angle, with reflections of the strings faked
by additive glow, horizon fog, stars. Elegant, oceanic, slow.
