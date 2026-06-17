# What's Inside a Black Hole? Three Ways We Might Actually Find Out

## The short version

For fifty years, physicists have known that Einstein's equations predict their own breakdown: at the center of every black hole sits a singularity — a point where density becomes infinite and the mathematics stops making sense. The most natural fix, dating back to James Bardeen in 1968, is to replace the singularity with a smooth, finite-density core. The idea is elegant, but it raises an uncomfortable question: if black holes have fuzzy cores instead of singularities, how would we ever know?

A new computational analysis answers that question systematically — and the answer is more structured than anyone expected. Of the three observational channels examined, one is **constrained today** by gravitational-wave data from the neutron-star merger GW170817. Another will be testable in the 2030s when the Einstein Telescope and Cosmic Explorer come online. And the third — the much-discussed search for gravitational-wave echoes — turns out to be a false lead: echoes don't test for a regular core inside a black hole. They test for the absence of a black hole entirely.

Along the way, the calculations reveal a surprising coincidence (the echo threshold matches the black hole's extremality condition exactly), a three-way taxonomy of how different core models hide from observation, and a concrete, falsifiable connection to dark matter.

---

## The problem, in plain English

Black holes are the universe's best-kept secrets. Once you cross the event horizon, the roles of space and time swap: moving toward the center becomes as inevitable as moving toward tomorrow. Einstein's equations handle this just fine — right up until they don't. At the center, the math blows up into an infinity. Everyone agrees this means Einstein's theory is incomplete there, but nobody agrees on what replaces it.

The oldest and simplest proposal is the "regular black hole": keep everything outside the horizon unchanged, but swap the singularity for a de Sitter-like core — a region of negative pressure that pushes back, preventing the infinite crunch. Versions of this idea were put forward by Bardeen (1968), Dymnikova (1992), and Hayward (2006). They are genuine solutions of Einstein's equations, but they require exotic matter with $p = -\rho$ — the same equation of state that drives cosmic inflation. Nobody knows if such matter exists. But nobody has ruled it out either.

The new work doesn't try to derive the core from first principles. Instead, it asks a simpler question: **if such cores exist, at what threshold would existing or near-future experiments detect them?** It's a computational survey — run the numbers, see what falls out, and be honest about what we do and don't know.

---

## Three channels to catch a fuzzy core

The paper examines three independent routes to detection. Each asks a different question of the same hypothesis.

### Channel 1: Ringdown — listening to how a black hole rings

When two black holes merge, the remnant "rings" like a struck bell, emitting gravitational waves at specific frequencies called quasinormal modes (QNMs). A regular core — even a tiny one — shifts those frequencies by a calculable amount. The question is whether the shift is big enough to measure.

The answer depends dramatically on *which kind* of regular core Nature chose. Using a WKB approximation (a standard quantum-mechanical technique adapted to black hole perturbations), the analysis computed the fractional frequency shift $|\delta f/f|$ for all three major regular-black-hole models:

| Core type | Shift formula | Detectability |
|-----------|--------------|---------------|
| **Hayward** | $0.049 \times (L/M)^2$ | Subtle — polynomial suppression |
| **Bardeen** | $0.139 \times (e/M)^2$ | 2.8× larger than Hayward |
| **Dymnikova** | $\approx 0$ | Invisible — exponential suppression |

This is a genuinely new result: not all regular black holes are created equal. Hayward cores are subtly detectable. Bardeen cores are more detectable. And Dymnikova cores are radio-silent — the shift is exponentially suppressed, falling below any conceivable detector threshold for all physically reasonable core sizes. The word "regular" alone doesn't tell you whether you can find it. You need to know which specific model Nature picked.

The catch: even the most optimistic shift (Bardeen at the largest physically plausible core) is only about 1.4% — roughly seven times smaller than LIGO's current ringdown precision. **Current detectors can't tell a regular black hole from Schwarzschild.**

But next-generation detectors can. The Einstein Telescope and Cosmic Explorer, expected in the 2030s, will reach a precision of roughly 0.01%, enough to detect a Hayward core as small as $L/M \approx 0.045$ — about 2% of the black hole radius. Stacking signals from hundreds of events could push this even lower.

**Spin complicates the picture.** LIGO's black holes spin. The analysis extended the WKB calculation to slow rotation, but for GW150914 — the first detected merger, whose remnant spun at 67% of the maximum — the slow-rotation approximation breaks down completely. The spin-corrected frequencies are numerical artefacts, not physical predictions. A proper treatment requires solving the full Teukolsky equation on a regular black hole background, which remains an open problem. For now, the static (non-spinning) result is the trustworthy bound: GW150914 cannot constrain $L/M$.

**Isospectrality survives.** In standard relativity, the two gravitational-wave polarizations from a Schwarzschild black hole ring at exactly the same frequency — a property called isospectrality. Could a regular core break this degeneracy, giving us a second, independent observable? The calculation says no: axial and polar modes split by less than 0.12% even for large cores. Isospectrality holds at the precision accessible to current methods. This is a negative result, but a non-trivial consistency check.

**What's new relative to earlier work.** Earlier papers (Toshmatov et al. 2017, Flachi & Lemos 2013, Li & Bambi 2013) computed individual QNM frequencies for regular black holes. The new analysis does three things they didn't: it fits the leading-order scaling coefficient directly from the WKB data for all three metrics in a single unified framework; it reveals that the three core types fall into qualitatively different suppression classes (polynomial, power-law, exponential); and it connects the QNM thresholds to specific detector capabilities and to the echo-extremality coincidence described below.

### Channel 2: Tides — squeezing a neutron star with a hidden core

**This is the strongest result in the paper** because it connects to data that already exists. The same negative-pressure fluid that smooths out a black hole's interior can also live inside neutron stars. If neutron stars harbor a hidden core, the core's negative pressure makes the star more compact — which changes both the maximum possible mass and how much the star stretches in a gravitational wave (its tidal deformability).

Using the SLy equation of state — a standard nuclear-physics model calibrated to laboratory data — the computer ran 394 different star configurations with varying core densities, sizes, and central pressures. The results:

- **Pure SLy matter** produces a maximum neutron star mass of 2.05 solar masses, consistent with the heaviest known pulsar (PSR J0740+6620, at $2.08 \pm 0.07$).
- **Add even a trace of the hidden core** — central density above $3 \times 10^{11}$ g/cm³, about one-thousandth of nuclear density — and the maximum mass drops below the observed pulsar mass, **ruling out that core density at 1-sigma confidence**.
- **The tidal deformability $\Lambda$** — how much the neutron star stretches — drops dramatically: from 557 (pure SLy) to as low as 54 for a dense core. GW170817 already restricts $\Lambda$ at 1.4 solar masses to the range 70–580. The lower bound excludes core densities above $3 \times 10^{13}$ g/cm³, complementing the NICER mass constraint at a different density scale.

A measurement of $\Lambda$ below roughly 200 for a 1.4-solar-mass neutron star whose companion exceeds 2.0 solar masses would be unexplained by any standard nuclear equation of state — but would be a **natural prediction** of the hidden-core model. This is a falsifiable claim using data that either exists or will exist within the decade.

**Post-merger frequency.** When two neutron stars merge, the remnant oscillates at a characteristic frequency $f_2$. The hidden core stiffens the equation of state, shifting $f_2$ upward. For the SLy-allowed core densities, the shift is +7% to +10% — at the edge of what the Einstein Telescope will measure. For higher core densities (allowed only with a stiffer nuclear EOS), the shifts reach +27%, well above detectability. This gives a second, independent test, albeit one that requires disentangling EOS effects from core effects.

**Causality passes.** A sweep of all 394 configurations found zero violations of the rule that sound cannot travel faster than light. The hidden core actually *reduces* the sound speed relative to pure nuclear matter — the two-fluid model is internally consistent even without a microphysical derivation.

### Channel 3: Echoes — a false lead (but an instructive one)

Gravitational-wave echoes — time-delayed pulses from partial reflection between the photon sphere and the core surface — have been widely discussed as a signature of regular black holes. The calculation reveals a problem so clean it's almost beautiful.

Echo time delays can be computed exactly from the black hole metric. When you run the numbers, echo delays become LIGO-resolvable only when the core size $L$ exceeds $0.77\,M$ — which **coincides exactly with the point where the black hole ceases to have a horizon at all.** This is the Hayward extremality condition $L_{\rm ext} = 4M/(3\sqrt{3}) \approx 0.770\,M$, and it turns out to be identical for the Bardeen model too ($e_{\rm ext} = 4M/(3\sqrt{3})$).

In plain English: echoes don't test for a regular core inside a black hole. They test for the absence of a black hole entirely. If you see echoes, the object has no event horizon — which is a far more radical claim than merely having a regular interior. This corrects a genuine confusion in the literature and reframes what echo searches actually constrain.

The result is likely generic: the integral that determines echo timing diverges logarithmically near any horizon, meaning resolvable echoes require near-extremal or horizonless conditions for all metrics, not just Hayward.

---

## The dark matter connection: black holes that never die

If black holes have regular cores, Stephen Hawking's famous calculation gets a new ending. Ordinarily, a black hole's Hawking temperature rises as it shrinks, diverging to infinity at the endpoint. With a regular core, the temperature goes to zero instead. The black hole never fully evaporates — it leaves behind a cold, stable remnant.

The remnant mass can be calculated exactly: $M_{\rm remnant} \approx 1.3\,L$. What this means depends entirely on what $L$ is:

- If $L$ is the **Planck length** ($10^{-35}$ m): the remnant is a Planck-scale speck — the standard, unexciting expectation.
- If $L$ is the **QCD scale** ($10^{-15}$ m, roughly the size of a proton): the remnant weighs about $10^{15}$ grams — **squarely in the mass range where primordial black holes could be the dark matter.**
- If $L$ is any larger: existing microlensing surveys would already have seen the remnants. Data from EROS, OGLE, and MACHO push the remnant mass below $10^{18}$ g, corresponding to $L$ below $10^{-14}$ m.

This is the single most provocative number in the entire analysis: a direct, calculable bridge between regular black hole physics and the dark matter problem, requiring no additional assumptions beyond Hawking's evaporation formalism and the existence of a regular core.

---

## The falsifiability table

The analysis provides something rare in theoretical physics: a concrete table of what measurement, from which detector, on what timeline, would falsify each claim. Entries marked ✓ are constrained by data that already exists.

| What to look for | Today | What's needed | Detector | When | What it would exclude |
|---|---|---|---|---|---|
| **Ringdown shift (Hayward)** | No constraint | $\Delta f/f \sim 10^{-4}$ | ET/CE (stacked) | 2030s | $L/M \gtrsim 0.045$ |
| **Ringdown shift (Bardeen)** | No constraint | $\Delta f/f \sim 10^{-4}$ | ET/CE (stacked) | 2030s | $e/M \gtrsim 0.027$ |
| **Tidal deformability** ✓ | $70 < \Lambda < 580$ | $\Lambda$ to 1% | LIGO A+ / ET | **Now** / Late 2020s | $\rho_0 \gtrsim 3\times10^{13}$ g/cm³ |
| **Post-merger $f_2$** | No measurement | $f_2$ to 10% | ET | 2030s | $\rho_0 \gtrsim 10^{12}$ g/cm³ (stiffer EOS needed) |
| **Neutron star mass** ✓ | $M > 2.08\,M_\odot$ | Better radius | NICER / STROBE-X | **Now** | $\rho_0 \gtrsim 3\times10^{10}$ g/cm³ (1σ) |
| **Echoes** | Ruled out unless horizonless | SNR threshold | LIGO A+ | **Now** | Horizonless objects |
| **Remnant dark matter** | $M_{\rm rem} < 10^{18}$ g | Sub-lunar lensing | LSST / Roman | Late 2020s | $L \gtrsim 10^{-14}$ m |
| **Causality** ✓ | $c_s^2 \leq 1$ (0/394 violations) | N/A | Theory | **Now** | Model is internally consistent |

**The tidal channel is the only one actively constraining the model right now.** The ringdown channel is the most theoretically robust but awaits next-generation detectors. The echo channel turns out to be a false lead — it tests for horizon absence, not core regularity.

---

## What we know and what we don't

The paper draws a bright line between what is established, what is speculative, and what is simply unknown:

**Established physics used:** Einstein's general relativity, the SLy nuclear equation of state (calibrated to laboratory data), the Iyer-Will WKB method for black hole perturbations, Hawking's evaporation formalism, LIGO/Virgo GW150914 and GW170817 data, NICER pulsar mass measurements, and existing microlensing surveys.

**Speculative hypothesis tested:** that black hole and neutron star interiors contain a finite, negative-pressure core described by the Hayward, Bardeen, or Dymnikova metrics.

**Not provided:** a derivation of the core from quantum gravity or any microphysical theory; a formation mechanism (how does gravitational collapse produce a regular core instead of a singularity?); a prediction for the core length scale $L$ or density $\rho_0$ from first principles; a resolution of the active/passive diffeomorphism question; a stability analysis under non-spherical perturbations; a full Kerr (spinning) treatment — the slow-rotation expansion breaks down for realistic astrophysical spins.

The analysis is honest about these gaps. It doesn't claim to explain what's inside a black hole. It claims that *if* a regular core is there, *here is what it would take to detect it* — and, crucially, *here is what measurement would prove it wrong.*

---

## The bottom line

Nobody has proven that black hole singularities are replaced by smooth cores. What the computational analysis offers is a systematic answer to the question: **if such cores exist, what would it take to detect them?**

The answer is surprisingly structured. One channel (tidal deformability) is constrained today by GW170817. One channel (ringdown) will be constrained in the 2030s. One channel (echoes) was a false lead — a category error corrected by a clean mathematical coincidence. And the Dymnikova result adds an important nuance: some "regular" cores are observationally invisible, meaning the label "regular black hole" is not enough to make a claim testable. You need to specify which metric.

The single most provocative number is $M_{\rm remnant} \approx 1.3\,L$. If $L$ is set by the QCD scale — roughly the size of a proton — then cold remnants from evaporated primordial black holes would naturally fall in the mass window where they could be the dark matter. This connection requires no additional assumptions beyond the core's existence and Hawking's evaporation formalism. It is direct, calculable, and falsifiable.

In science, a framework that tells you exactly what measurement would prove it wrong is a framework worth taking seriously — even if it's not yet a framework anyone has proven right.
