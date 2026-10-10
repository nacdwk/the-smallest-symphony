# I. Descent — `src/movements/m1-descent.js`

numeral `I` · title `Descent` · tempo `Largo, sempre più piccolo` (Italian: "ever smaller")

## Job
The opener, and the scene that sits (dimmed) behind the intro card. A continuous powers-of-ten dive from
a grain of salt to the Planck length, ending on a single vibrating string. It must make the user *feel* how
absurdly small 10⁻³⁵ m is. This is the first impression of the whole piece: it has to be breathtaking.

## The journey (one continuous variable `depth` = log10 of the view width in metres, from -3 to -35)
| depth | layer | what it looks like |
|---|---|---|
| -3 | **A grain of salt** (1 mm) | A faceted cubic crystal with bevelled, glinting edges, slowly turning in the void, light caustics/sparkles on its faces, refractive-looking (fake it with fresnel + iridescence). |
| -9.5 | **The crystal lattice** (0.28 nm spacing) | A cubic grid of alternating sodium (small, gold) and chloride (larger, cyan) ions, receding into fog in every direction, bonds as faint lines. We fly into it. |
| -10 | **One atom** (10⁻¹⁰ m) | An electron probability cloud: tens of thousands of additive points sampled from hydrogen-like orbital densities (mix 1s, 2p, 3d lobes), shimmering; a tiny bright nucleus at the centre. |
| -14 | **The nucleus** (10⁻¹⁴ m) | A jiggling cluster of 35 nucleons (chlorine-35: 17 protons warm gold/ember, 18 neutrons cool violet/ink), glossy glowing spheres. |
| -15 | **A proton** (1.7×10⁻¹⁵ m) | Three valence quarks (two up, one down) as bright orbs coloured red/green/blue colour charge (tint them toward the palette), joined by twisting gluon flux tubes; a boiling sea of virtual quark–antiquark pairs popping in and out. |
| -18 | **A quark** | Still a point at the limit of our best experiments (≤ 10⁻¹⁸ m). A single bright point with a faint halo, and nothing else. |
| -18 → -33 | **The desert** | Seventeen orders of magnitude where we have no data. Near-empty dark with sparse drifting motes streaking past (speed ∝ dive speed). As depth passes -30 the void itself begins to froth: **quantum foam**, spacetime fluctuating (fbm noise turbulence, ripples, sparks), intensifying toward -35. |
| -35 | **A string** (≈ 1.6×10⁻³⁵ m) | Out of the foam a single closed loop of light resolves: gold, white-hot core, vibrating in a few harmonics, softly humming. The foam calms around it. The readout invites the user to the next movement ("This is the instrument. Movement II lets you play it."). |

Implementation idea (infinite zoom): camera stays fixed; each layer i has native size 10^L_i and is drawn at
scale s_i = 10^(L_i − depth) (times a base size). Layers fade in when s_i is small (they emerge at the centre of
the previous layer, e.g. the nucleus sits at the centre of the atom) and fade out as they grow past the camera
(s_i ≳ 4). Only update/draw layers whose s_i is in range. Add streaking motes and a subtle radial zoom blur
feel (you can drive `ctx.setPost({ aberration })` with dive speed for a warp effect).

## Interaction
- Scroll wheel / trackpad, vertical drag, and ↓/↑ (consume them) change depth with inertia. Holding Space (or the
  **Dive** button) auto-dives smoothly; tapping it again stops. A **Surface** button flies back to -3.
- Horizontal drag orbits the current layer a little (yaw/pitch, damped, springs back slowly).
- `onBegin()`: start a slow, gentle auto-dive for ~2 s then hand control to the user, with hint "Scroll or hold space to dive".
- `demo()`: jump/dive quickly to depth ≈ -15 (the proton) so the screenshot shows a rich layer. Support a debug param
  `depth=<n>` to start at any depth for testing.

## Readout (top-right)
- A big scale number like `10⁻¹⁵ m` (use `sup()`), the layer name, one plain sentence about it, and the
  magnification from the start (`× 10¹²`). Include a small vertical scale ruler (HTML ticks for -3…-35) with a moving marker.
- In the desert: "No experiment has seen anything in this range." At -35: the famous comparison:
  *if an atom were blown up to the size of the observable universe, a string would be about the height of a tall tree*
  (atom 10⁻¹⁰ m, Planck length 1.6×10⁻³⁵ m, observable universe 8.8×10²⁶ m → ≈ 140 m). Facts must be correct.

## Sound
- A **Shepard tone** tied to the dive: ~8 sine partials an octave apart with a bell-shaped gain over log-frequency,
  continuously gliding *down* while you dive in (up when you rise), speed ∝ dive velocity, nearly silent when still.
  It is the auditory version of an infinite zoom. Build it on `audio.ctx` → `audio.out`, lazily, and stop it in `exit()`.
- A soft `bell` from the pentatonic scale when you cross into each new layer; a long `swell` when the string appears;
  the string itself hums (quiet `additive` voice whose partial amplitudes follow its vibration).
- drone: low and spacious, e.g. `[0, 7, 12, 19]`.

## Caption / fine print
- caption (≤ 60 words): something like: You are holding a grain of salt. Dive in. Every step down is ten times smaller. Atoms, nuclei and quarks are as far as experiments reach. String theory says the dive continues for seventeen more powers of ten, and at the bottom there is a string.
- fine: what each scale is measured by (electron microscopes, particle accelerators), the experimental limit on quark size (~10⁻¹⁸–10⁻¹⁹ m), why the Planck length (√(ħG/c³) ≈ 1.6×10⁻³⁵ m) is where quantum gravity matters, and that string length could be larger than the Planck length in some models.

## Post
Slightly stronger bloom near the end (the string), more aberration while diving fast.
