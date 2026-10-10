# VI. The Landscape — `src/movements/m6-landscape.js`

numeral `VI` · title `The Landscape` · tempo `Finale, maestoso`

## Job
The finale. String theory seems to allow an enormous number of stable ways to curl up its extra dimensions (often
quoted as ~10⁵⁰⁰). Each is a **vacuum**: a possible universe with its own particles, forces and dark energy. Picture
them as valleys in a vast landscape. The user drops universes into it and watches each one settle into its own
valley, with its own laws. Then the piece closes with an honest, moving coda.

## Scene
- A vast nocturnal terrain seen from a slow, high, drifting camera: a procedural heightfield with many basins of
  different depths (fbm + domain warp + explicit craters/basins). Rendered as glowing **topographic contour lines**
  (iso-lines in the fragment shader) over a dark surface, height-tinted (deep valleys `planck` gold, ridges `violet`),
  mist pooling in valleys, stars above, the far terrain fading into fog. Hundreds of faint dots mark local minima.
- Compute the heightfield once on the CPU (e.g. 256×256 Float32Array) → DataTexture for vertex displacement and
  shading, and reuse the same array (bilinear sampling) for the physics so marbles and terrain agree exactly.

## Interaction
- **Click/tap the terrain to drop a universe**: a bright marble falls, rolls downhill with inertia and friction
  (gradient of the heightfield), and settles in a local minimum. Then a translucent **bubble universe** inflates from
  the valley (iridescent sphere, expanding and fading to a stable size) and a card describes its laws.
- Each valley's laws are deterministic from its id (seeded `rng`): dark energy Λ (large positive / tiny positive like
  ours / negative), large dimensions (usually 4, sometimes not), particle families (0–5), forces present, and a verdict:
  **"Could support chemistry and stars"** (mint) only for a narrow band (tiny positive Λ, 4 large dimensions, ≥ 1 family,
  EM present), otherwise a plain reason ("Λ too large: space expands too fast for galaxies to form") in `ember`.
- **Tunnel** button (`t`): the most recent universe quantum-tunnels to a neighbouring lower valley (a dashed glowing
  arc, a new bubble nucleates there, the old one collapses). Real physics: vacuum decay by bubble nucleation.
- **Rain** button (`r`): drop 12 universes at once across the view (spectacle).
- Counter: "Universes you have made: 7 · Vacua in the landscape: ~10⁵⁰⁰".
- **Coda** button (or automatically after ~8 universes): the camera rises high, all bubbles glow, the music swells,
  and the readout shows the closing text (below) with a **Play again** button that calls `ctx.engine.goTo(0)`.
- Orbit/pan with drag on empty space, wheel zoom, slow idle drift.
- `demo()`: rain ~8 universes so several are rolling/settled and one card is shown; support debug param `coda` to jump to the coda.

## Coda text (readout note, plain and honest)
String theory has not been tested by experiment. Its strings are too small for any collider we can imagine, and
the landscape makes sharp predictions hard. It is still the most developed idea we have for joining quantum
mechanics and gravity, and it has already changed mathematics and our understanding of black holes. Somewhere
below every particle in your body, if it is right, a string is playing. (Edit for rhythm; keep it accurate.)

## Sound
Each settling plays a chord built from the valley depth (deeper = lower root), bubbles inflate with a `swell`,
rolling marbles a soft filtered-noise glide, tunnelling a descending bell gliss. The coda: a slow layered swell of the
whole pentatonic chord, then silence. drone e.g. `[0, 7, 12, 15, 19]` (fullest of the piece).

## Caption / fine print
- caption: String theory allows a vast number of ways to fold its hidden dimensions, perhaps 10⁵⁰⁰. Each is a
  possible universe with its own laws. Click to drop a universe into the landscape and see where it settles.
- fine: flux compactifications and moduli stabilisation (KKLT 2003, Bousso–Polchinski 2000); where 10⁵⁰⁰ comes from
  (counts of flux vacua) and that it is an estimate; the cosmological constant problem and the anthropic argument as a
  contested proposal; the "swampland" programme questioning whether such de Sitter vacua exist at all; eternal inflation
  and bubble nucleation (Coleman–De Luccia).

## Look
Night flight over a luminous topographic map. Gold valleys, violet ridges, mist, bubbles like soap films catching
light. Grand and calm.
