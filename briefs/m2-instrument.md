# II. The Instrument — `src/movements/m2-instrument.js`

numeral `II` · title `The Instrument` · tempo `Allegro, pizzicato` ("plucked")

## Job
The centrepiece. One string, alone in the void, that the user plays like an instrument. What they **hear is the
string's actual spectrum**: each vibration mode n sounds as harmonic n (additive synthesis driven by the mode
amplitudes), so the sound and the picture are the same data. It must feel tactile, musical and alive, and it must
teach the real core idea: the way a string vibrates determines which particle it is and how heavy it is.

## The string
- Closed string (a loop, default) or open string (a segment with two glowing endpoint beads). Toggle `O`.
- Shape: X(σ, t) = base curve + Σₙ₌₁⁸ displacement modes, each with two transverse polarisations (in-plane radial and
  out-of-plane), evolving with frequency ωₙ = n·ω₁ (standing waves for open; for closed, left- and right-movers can be
  shown as counter-rotating patterns — standing waves are fine when they are excited equally). Light damping.
- Render: a luminous filament (white-hot core, coloured halo, ≥ 512 segments, smooth, a few px wide plus bloom). Colour
  along the string can shift with local mode content (`modeColor(n)`). Add motion trails (ghost copies of recent shapes
  at decaying opacity) so the vibration envelope reads as a glowing shimmer.
- **Quantum fuzz** slider (ħ, 0…1): a cloud of 16–32 faint ghost strings with zero-point jitter (each mode gets random
  Gaussian amplitude ∝ √(ħ/n), re-sampled continuously) around the classical string. At 0 it is a crisp classical string;
  at 1 it is a nebulous probability cloud. Even with no notes played, the quantum string never stops trembling.

## Interaction
- **Pluck by dragging** any point of the string (nearest point in screen space within ~40 px; cursor `grab`/`grabbing`).
  Drag pulls a smooth Gaussian bump that follows the pointer; release decomposes the bump into modes 1–8 (project onto
  cos nσ / sin nσ for each polarisation) and it rings. Fast flicks = brighter tone.
- **Harmonic pads 1–8** (`keys` control, keyboard 1–8, each pad coloured `modeColor(n)`): add one quantum to mode n
  (excite a pure standing wave n with random polarisation) and play harmonic n.
- `O` toggles open/closed; a **Tension** slider (string tension T ∝ 1/α′) scales pitch and the mass scale (M_s ∝ √T);
  ħ slider for the fuzz; **Damp** button (`space`) silences the string by draining all modes smoothly.
- Mouse wheel / pinch: zoom gently. Slow idle orbit of the camera; dragging empty space orbits.
- **Decay by emission**: as an excited string loses energy, occasionally emit a tiny closed loop (a massless quantum) that
  flies off and fades, with a soft high pluck and a toast the first time ("Your string just radiated a graviton"; for an
  open string "a photon-like quantum").
- `demo()`: pluck the string programmatically (e.g. a bump at σ = 0.3) and add quanta to modes 2 and 5, ħ ≈ 0.35.

## The particle card (readout, top-right) — must be physically right
Keep integer occupation numbers kₙ (a pluck rounds energies into quanta; decay removes quanta). Level N = Σ n·kₙ.
- Title: closed & N=0 → **Graviton** (spin 2; the same level also holds the dilaton and the B-field). open & N=0 →
  **Photon-like gauge boson** (spin 1). N ≥ 1 → **Massive string state** (level N).
- Rows: Level N · Mass (in units of the string scale M_s: open √N, closed 2√N, i.e. α′M² = N open, α′M² = 4N closed,
  massless at N = 0) · Max spin (leading Regge trajectory: open N+1, closed 2N+2) · Dominant harmonic · Pitch (Hz).
- A live **spectrum**: 8 bars coloured `modeColor(n)` showing energy per mode (custom HTML in the readout).
- Note line: at N = 0 "Massless: travels at the speed of light". At N ≥ 1 "≈ 10¹⁷ GeV × …: about ten trillion times
  what the Large Hadron Collider reaches" (be careful with the arithmetic: LHC ~1.4×10⁴ GeV).

## Sound
- One sustained `additive` voice with 8 partials at f₁·n, partial amplitudes = √(mode energy), so the timbre is the
  shape. f₁ = a pentatonic note around D3–D4, scaled by √T from the tension slider (quantise to the scale when the
  slider is released). Plus `pluck` transients on release / pad hit.
- Zero-point fuzz adds a faint airy noise bed proportional to ħ (`audio.noise` loops or a quiet filtered noise voice).
- drone: sparse, e.g. `[0, 12, 19]`, low gain so the string is the soloist.

## Caption / fine print
- caption: Drag the string to pluck it, or press keys 1–8. Each note you add is a quantum of vibration, and the
  total sets the string's mass and spin. The silent ground state of a closed string is a graviton.
- fine: the mass formula and Regge trajectories in words; level matching for closed strings; that all known
  particles (electrons, quarks, photons) sit in the massless level and differ by how the string sits in the extra
  dimensions (Movement III), not by higher notes; the higher notes are real predictions but far too heavy for any
  collider; zero-point motion and that a quantum string's apparent size grows (logarithmically) the finer you look;
  that these spectra are those of the superstring in light-cone gauge, simplified (we ignore fermions/supersymmetry).
