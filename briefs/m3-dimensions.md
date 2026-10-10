# III. Hidden Dimensions — `src/movements/m3-dimensions.js`

numeral `III` · title `Hidden Dimensions` · tempo `Adagio misterioso`

## Job
Superstring theory is only consistent in 10 spacetime dimensions; the six extra ones must be curled up too small
to see. Make that intuitive and gorgeous in three connected stages (a `segmented` control, keys `1` `2` `3`), each
a camera move inside one continuous scene, not a hard cut.

### Stage 1 — "A curled dimension" (the garden hose + T-duality)
- A long luminous tube stretching across the frame (a 2D surface: one long dimension × one circle of radius R).
  From far away it reads as a 1D line; as the camera dollies in (scroll), the circular dimension appears.
- A closed string lives on it. Two kinds of motion: **momentum** n (the loop slides around the circle; quantised,
  energy ∝ n/R) and **winding** w (the loop wraps around the circle w times; energy ∝ w·R/α′). Show a wound string
  as a glowing coil wrapping the tube w times; show momentum as a loop orbiting the tube with a travelling wave.
- **Radius slider R** (log scale, e.g. 0.25…4 in units of √α′). Readout lists the lowest energies for
  (n, w) = (1,0), (0,1), (1,1)… using E_KK = n/R, E_wind = w·R (units α′ = 1). Show the T-dual radius 1/R as a ghost
  tube. Key insight: the list of energies at R and at 1/R is identical with n ↔ w swapped. At R = 1 (self-dual) the
  two tones meet. Nothing can be smaller than the string scale in a meaningful way.
- Sound: two sustained tones, momentum pitch ∝ 1/R and winding pitch ∝ R (mapped into a pleasant range); they cross
  at the self-dual radius. Moving the slider through R = 1 should be an audible "aha".

### Stage 2 — "Six dimensions, folded" (Calabi–Yau)
- The hero: a Calabi–Yau cross-section rendered as an iridescent, translucent, softly glowing surface with fine
  contour lines, orbitable (OrbitControls with damping), breathing slowly.
- Use Hanson's construction for the Fermat curve z₁ⁿ + z₂ⁿ = 1: for patches k₁, k₂ ∈ {0..n−1}, with complex
  z = x + iy, x ∈ [0, π/2], y ∈ [−a, a] (a ≈ 1.0–1.6):
  z₁ = e^{2πik₁/n} · cos(z)^{2/n}, z₂ = e^{2πik₂/n} · sin(z)^{2/n} (complex cos/sin/power, principal branch),
  project to 3D as (Re z₁, Re z₂, cos α·Im z₁ + sin α·Im z₂) with α ≈ π/4 (animate α slowly for a living shape).
  n² patches, each a parametric grid (e.g. 40×40), merged into one BufferGeometry with good normals.
- **Degree slider n = 3…6** rebuilds the shape with a smooth crossfade. Readout: genus of this curve g = (n−1)(n−2)/2
  ("holes"), and the physics: in the simplest constructions the number of particle families = |χ|/2 of the six-dimensional
  Calabi–Yau; the quintic threefold has χ = −200 → 100 families; our universe has 3, so physicists look for shapes with
  |χ| = 6. Make clear the visual is a 2D slice of a 6D shape.
- A glowing string loop wraps one of the surface's cycles and travels around it (sample a closed curve on the surface,
  e.g. constant y over one patch boundary chain, or a loop around a "hole").
- Sound: the shape hums a chord whose notes come from n (more holes, richer chord).

### Stage 3 — "At every point of space"
- Pull far back: a lattice of ordinary 3D space (a faint grid receding into fog) where **every grid point carries a
  tiny Calabi–Yau** (InstancedMesh of a low-poly version, each slowly rotating with a different phase). As you
  hover/click one it grows to show it is the same shape. Line: "Every point in your room would hide one of these."

## Interaction summary
Segmented stage control (1/2/3), stage-specific sliders (R in stage 1, n in stage 2), winding/momentum buttons in
stage 1 (`w` adds a winding, `m` adds momentum), orbit drag, scroll zoom. `demo()`: go to stage 2 with n = 5 and the
string loop visible (that is the money shot); support debug param `stage=1|2|3` for testing.

## Caption / fine print
- caption: Superstring theory only works in ten dimensions. We see four, so six must be curled up smaller than we can
  detect. Their shape decides which particles and forces exist.
- fine: why 10 (quantum consistency: the conformal anomaly cancels only for D = 10 for superstrings, 26 for bosonic);
  Kaluza–Klein idea; T-duality as a genuinely stringy fact; Calabi–Yau manifolds preserve supersymmetry; the Hanson
  picture is a 2D real slice of a complex surface; number of families from Euler characteristic (standard embedding of
  heterotic strings); there are vast numbers of candidate shapes (link to Movement VI).

## Look
Violet is this movement's colour (`palette.violet`) with cyan/magenta iridescence. Glassy, mysterious, slow.
drone e.g. `[0, 10, 15, 22]` (minor-seventh colour).
