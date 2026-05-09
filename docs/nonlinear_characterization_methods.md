# Nonlinear System Characterization Methods

A reference survey of industry-standard and research-grade techniques for
identifying and characterizing nonlinear dynamic systems, with pointers to
foundational papers.  The goal is to provide grounding for choosing which
test types to add to the Numen characterization framework.

---

## Why nonlinear characterization is different from linear

A linear system satisfies superposition: its response at frequency f₁+f₂
is simply the sum of its responses at f₁ and f₂ separately.  A nonlinear
system does not.  This creates the core problem: a standard FRF (transfer
function magnitude and phase) is amplitude-dependent and tells you nothing
about *how* nonlinear the system is, *what kind* of nonlinearity it has, or
*where* in the state space the nonlinearity lives.

The methods below are grouped by the fundamental experimental paradigm they
use.  The currently-implemented Numen tests are noted; the rest are candidates
for implementation.

---

## 1. Single-tone harmonic analysis

### 1.1 Stepped sine / discrete frequency sweep  *(implemented)*

Apply a sinusoid at fixed frequency f, wait for steady state, measure the
*fundamental* component of the response (via lock-in or DFT).  Repeat across
a frequency grid.  The result is an amplitude-dependent FRF: the apparent
resonance frequency and peak height shift with excitation level, and hysteresis
(jump phenomena) appears near resonance in hardening or softening systems.

The amplitude sweep test (also implemented) drives the same physics orthogonally:
fixed frequency, varying amplitude.

### 1.2 Total Harmonic Distortion (THD) sweep  *(not yet implemented)*

For each stepped-sine frequency, instead of measuring only the fundamental,
measure all harmonic components H₁, H₂, H₃, … at 2f, 3f, …  Then:

```
THD = sqrt(H₂² + H₃² + H₄² + ...) / H₁
```

THD as a function of frequency localises the nonlinearity: a Duffing spring
produces strong 3rd harmonic near resonance; a quadratic (asymmetric) stiffness
produces strong 2nd harmonic; Coulomb friction produces a characteristic 3f
signature.  The *order* of the first significant harmonic identifies the dominant
nonlinearity type.

**References:**
- Worden, K. & Tomlinson, G. R. (2001). *Nonlinearity in Structural Dynamics:
  Detection, Identification and Modelling.* Institute of Physics Publishing.
  (ISBN 0-7503-0356-5 — the canonical textbook for structural nonlinear dynamics.)
- Ewins, D. J. (2000). *Modal Testing: Theory, Practice and Application* (2nd ed.).
  Research Studies Press.  Chapter 8 covers nonlinear effects on FRFs.

---

## 2. Two-tone / multi-tone tests and intermodulation distortion  *(not yet implemented)*

### 2.1 Theory

Apply two simultaneous sinusoids at incommensurate frequencies f₁ and f₂:

```
u(t) = A₁ sin(2πf₁t) + A₂ sin(2πf₂t)
```

The output of an n-th order nonlinear system contains cross-product (intermodulation)
components at all frequencies of the form:

```
f_IM = |m·f₁ ± n·f₂|,   m+n = total order
```

So a cubic nonlinearity generates third-order intermodulation products at:

```
2f₁ - f₂,   2f₂ - f₁   (lower sidebands — most diagnostically useful)
2f₁ + f₂,   2f₂ + f₁   (upper sidebands)
```

A quadratic nonlinearity generates second-order products at f₁+f₂ and |f₁−f₂|.

### 2.2 What it tells you

- The *existence* of IM products confirms nonlinearity (a linear system produces
  none).
- The *order* of the first non-zero product identifies the leading nonlinearity
  order (2nd for asymmetric, 3rd for symmetric stiffness/damping).
- The *slope* of IM product amplitude vs. input amplitude on a log-log plot gives
  the nonlinearity order: a cubic term produces 3:1 slope for the 3rd-order
  products, 1:1 for the fundamental.
- **Third-order intercept point (IP3):** extrapolate the fundamental (slope 1:1)
  and the 3rd-order IM product (slope 3:1) on a log-log plot; their intersection
  is IP3.  A higher IP3 means the nonlinearity only becomes significant at larger
  amplitudes.  Originally an RF/microwave metric, IP3 is increasingly used in
  MEMS and acoustic transducer characterization.
- Placing f₁ and f₂ near resonance reveals whether the intermodulation is driven
  by the nonlinear restoring force, damping, or inertia.

### 2.3 Choosing f₁ and f₂

Standard practice:
- Both tones near resonance: f₁ ≈ f₀ − Δ, f₂ ≈ f₀ + Δ, with Δ small.
  The IM products at 2f₁−f₂ ≈ f₀−2Δ and 2f₂−f₁ ≈ f₀+2Δ are in-band.
- One tone at resonance, one away: isolates whether the nonlinearity is
  resonance-amplified.
- Both tones far from resonance: tests the "hard" nonlinearity without resonance
  gain (useful for Duffing oscillators with ε small).

Ensure f₁/f₂ is irrational (e.g. f₂ = 1.0836·f₁) so harmonics don't fall on
IM products.

### 2.4 References

- Schoukens, J., Pintelon, R., & Dobrowiecki, T. (2002). Linear approximation of
  weakly nonlinear MIMO systems.  *Automatica*, 38(7), 1219–1228.
  DOI: [10.1016/S0005-1098(02)00009-4](https://doi.org/10.1016/S0005-1098(02)00009-4)
- Schoukens, J., Vaes, M., & Pintelon, R. (2016). Linear system identification in
  a nonlinear setting: Nonparametric analysis of the nonlinear distortions and their
  impact on the best linear approximation.  *IEEE Control Systems Magazine*, 36(3),
  38–69.  DOI: [10.1109/MCS.2016.2535918](https://doi.org/10.1109/MCS.2016.2535918)
- Gelb, A. & Vander Velde, W. E. (1968). *Multiple Input Describing Functions and
  Nonlinear System Design.* McGraw-Hill.  (Classical treatment of dual-tone methods.)
- Adams, D. E. & Allemang, R. J. (2000). A frequency domain method for estimating
  the parameters of a non-linear structural dynamic model through feedback.
  *Mechanical Systems and Signal Processing*, 14(4), 637–656.
  DOI: [10.1006/mssp.2000.1292](https://doi.org/10.1006/mssp.2000.1292)

---

## 3. Higher-order frequency response functions (HOFRFs) and Volterra series  *(not yet implemented)*

A linear system is fully characterized by its 1st-order FRF H₁(f).  For a
nonlinear system the Volterra series provides the natural generalization:

```
y(t) = ∫ h₁(τ) u(t−τ) dτ
      + ∫∫ h₂(τ₁,τ₂) u(t−τ₁) u(t−τ₂) dτ₁ dτ₂
      + ...
```

where h₁, h₂, h₃, … are the Volterra kernels.  Their Fourier transforms H₁(f₁),
H₂(f₁,f₂), H₃(f₁,f₂,f₃) are the Higher-Order FRFs (HOFRFs).

Multi-tone inputs (with carefully chosen frequencies) are used to isolate
individual HOFRFs: H₂ is measured by placing two tones and observing the output
at the sum/difference frequencies; H₃ by three tones and observing triple-product
frequencies.

In practice, only H₁, H₂, H₃ are measured (higher orders require prohibitively
many tones and long records).  H₂ diagnoses asymmetric (quadratic) nonlinearities;
H₃ diagnoses symmetric (cubic/Duffing) ones.

**References:**
- Schetzen, M. (1980). *The Volterra and Wiener Theories of Nonlinear Systems.*
  Wiley-Interscience.  (Definitive theoretical reference.)
- Worden, K., Manson, G., & Tomlinson, G. R. (1997). A harmonic probing algorithm
  for the multi-input Volterra series.  *Journal of Sound and Vibration*, 201(1),
  67–84.  DOI: [10.1006/jsvi.1996.0746](https://doi.org/10.1006/jsvi.1996.0746)
- Billings, S. A. (2013). *Nonlinear System Identification: NARMAX Methods in the
  Time, Frequency, and Spatio-Temporal Domains.* Wiley.
  (Comprehensive modern treatment; NARMAX and Volterra closely related.)

---

## 4. Backbone curves and Nonlinear Normal Modes (NNMs)  *(not yet implemented)*

### 4.1 Concept

In a linear system, natural frequencies are amplitude-independent.  In a nonlinear
system, the instantaneous frequency of free oscillation depends on energy/amplitude.
Tracing this amplitude → frequency relationship produces the **backbone curve**.

The theoretical foundation is the theory of **Nonlinear Normal Modes (NNMs)**,
which generalize linear normal modes to nonlinear systems.  An NNM is a periodic
(or quasi-periodic) orbit of the conservative nonlinear system — not just an
eigenvector.

For a hardening spring (Duffing), the backbone curve bends to the right (higher
frequency at higher amplitude); for a softening spring it bends left.  The
backbone intersects the frequency response family at each amplitude, tracing the
locus of resonance peaks across the amplitude-sweep family.

### 4.2 Experimental measurement

Two approaches:

**Free decay + Hilbert transform** (see §5): ring down from a large amplitude
and track the instantaneous frequency envelope.  Simple but affected by damping.

**Phase resonance / force appropriation**: apply a force that exactly cancels
damping so the system is on its conservative backbone.  The signature condition
is 90° phase lag between force and response (as in linear resonance testing).
Phase-lock loop (PLL) control or stepped-phase control is used to walk up the
backbone.  This is the gold standard for NNM identification.

**Numerical continuation + simulation**: in simulation (as opposed to experiment),
the shooting method + pseudo-arc-length continuation can trace the backbone
exactly.  The Numen `discrete_frequency_sweep` at many amplitudes implicitly
sweeps near the backbone but does not trace it.

### 4.3 References

- Shaw, S. W. & Pierre, C. (1993). Normal modes for non-linear vibratory systems.
  *Journal of Sound and Vibration*, 164(1), 85–124.
  DOI: [10.1006/jsvi.1993.1198](https://doi.org/10.1006/jsvi.1993.1198)
  (Defines NNMs on invariant manifolds — the canonical theoretical paper.)
- Kerschen, G., Peeters, M., Golinval, J.-C., & Vakakis, A. F. (2009). Nonlinear
  normal modes, Part I: A useful framework for the structural dynamicist.
  *Mechanical Systems and Signal Processing*, 23(1), 170–194.
  DOI: [10.1016/j.ymssp.2008.04.002](https://doi.org/10.1016/j.ymssp.2008.04.002)
- Peeters, M., Viguié, R., Sérandour, G., Kerschen, G., & Golinval, J.-C. (2009).
  Nonlinear normal modes, Part II: Toward a practical computation using numerical
  continuation techniques.  *Mechanical Systems and Signal Processing*, 23(1),
  195–216.
  DOI: [10.1016/j.ymssp.2008.04.003](https://doi.org/10.1016/j.ymssp.2008.04.003)
- Peeters, M., Kerschen, G., & Golinval, J. C. (2011). Dynamic testing of
  nonlinear vibrating structures using nonlinear normal modes.  *Journal of Sound
  and Vibration*, 330(3), 486–509.
  DOI: [10.1016/j.jsv.2010.08.028](https://doi.org/10.1016/j.jsv.2010.08.028)
  (Experimental realization of NNM identification via phase resonance.)
- Renson, L., Gonzalez-Buelga, A., Barton, D. A. W., & Neild, S. A. (2016).
  Robust identification of backbone curves using control-based continuation.
  *Journal of Sound and Vibration*, 367, 145–158.
  DOI: [10.1016/j.jsv.2015.12.035](https://doi.org/10.1016/j.jsv.2015.12.035)
  (Control-based continuation — applicable to simulation as well as experiment.)

---

## 5. Free decay and Hilbert transform analysis  *(not yet implemented)*

### 5.1 Method

Apply a step or impulse to bring the system to a large initial condition, then
release it and let it ring down.  From the free-decay time series:

1. Compute the analytic signal via Hilbert transform: `z(t) = x(t) + j·H[x(t)]`
2. The instantaneous amplitude is `A(t) = |z(t)|`
3. The instantaneous phase is `φ(t) = arg(z(t))`
4. The instantaneous frequency is `f(t) = (1/2π) dφ/dt`
5. The instantaneous damping ratio is estimated from the log-decrement of `A(t)`

Plotting f(t) vs A(t) traces the backbone curve; plotting the damping ratio vs A(t)
reveals amplitude-dependent damping (viscoelastic, Coulomb, aeroelastic, etc.).

For clean results: bandpass filter around the mode of interest before the Hilbert
transform, and use EMD (Empirical Mode Decomposition) if multiple modes are present.

### 5.2 References

- Feldman, M. (2011). *Hilbert Transform Applications in Mechanical Vibration.*
  Wiley.  (Comprehensive treatment of HT-based instantaneous frequency/damping.)
- Feldman, M. (1994). Non-linear system vibration analysis using the Hilbert
  transform — I: Free vibration analysis method 'FREEVIB'.  *Mechanical Systems
  and Signal Processing*, 8(2), 119–127.
  DOI: [10.1006/mssp.1994.1008](https://doi.org/10.1006/mssp.1994.1008)
- Staszewski, W. J. (1997). Identification of non-linear systems using multi-scale
  ridges and skeletons of the wavelet transform.  *Journal of Sound and Vibration*,
  214(4), 639–658.
  DOI: [10.1006/jsvi.1998.1616](https://doi.org/10.1006/jsvi.1998.1616)
  (Wavelet-based alternative — more robust for lightly damped systems.)

---

## 6. Restoring Force Surface (RFS)  *(not yet implemented)*

### 6.1 Method

For a SDOF system mx'' + f(x, x') = F(t), rewrite as:

```
f(x, x') = F(t) - m·x''
```

From measured (or simulated) displacement x(t) and the applied force F(t),
compute x'(t) by differentiation and x''(t) by double differentiation.  Then
scatter-plot f vs (x, x') as a 3D surface — the **restoring force surface**.

For a linear-viscous + linear-stiffness system the surface is a tilted plane.
Deviations from the plane reveal the nonlinear structure directly:
- Curvature along the x axis → nonlinear stiffness (Duffing = cubic bowl)
- Curvature along the x' axis → nonlinear damping (quadratic drag = parabola)
- Cross-coupling → coupled stiffness-damping nonlinearity

The method is nonparametric — it does not require a model form to be assumed.

### 6.2 References

- Masri, S. F. & Caughey, T. K. (1979). A nonparametric identification technique
  for nonlinear dynamic problems.  *Journal of Applied Mechanics*, 46(2), 433–447.
  DOI: [10.1115/1.3424568](https://doi.org/10.1115/1.3424568)
  (Original paper — one of the most cited in experimental nonlinear dynamics.)
- Worden, K. (1990). Data processing and experiment design for the restoring force
  surface method, part I: Integration and differentiation of measured time data.
  *Mechanical Systems and Signal Processing*, 4(4), 295–319.
  DOI: [10.1016/0888-3270(90)90010-I](https://doi.org/10.1016/0888-3270(90)90010-I)

---

## 7. NARMAX — nonlinear black-box system identification  *(not yet implemented)*

NARMAX (Nonlinear Auto-Regressive Moving Average with Exogenous inputs) is a
discrete-time polynomial model:

```
y(k) = F[y(k−1), …, y(k−ny), u(k−1), …, u(k−nu), e(k−1), …, e(k−ne)] + e(k)
```

where F is a polynomial (or more generally a nonlinear function) of past outputs,
past inputs, and past noise.  The NARMAX identification problem is: given a
broadband input/output record, find the polynomial order and coefficients that
best explain the data.

NARMAX sits between Volterra (too many parameters for high order) and
nonparametric RFS (no model structure at all).  The Error Reduction Ratio (ERR)
algorithm selects which model terms are significant — a structured automatic
feature selection.

In simulation this is most useful as a validation tool: identify a NARMAX model
from simulation data and compare its predictions to held-out runs.

**References:**
- Billings, S. A. (2013). *Nonlinear System Identification: NARMAX Methods in the
  Time, Frequency, and Spatio-Temporal Domains.* Wiley.
  DOI: [10.1002/9781118535561](https://doi.org/10.1002/9781118535561)
  (The definitive modern textbook — covers NARMAX, Volterra, HOFRFs, wavelet.)
- Leontaritis, I. J. & Billings, S. A. (1985). Input-output parametric models for
  non-linear systems.  *International Journal of Control*, 41(2), 303–344.
  DOI: [10.1080/0020718508961129](https://doi.org/10.1080/0020718508961129)
  (Original NARMAX paper.)

---

## 8. Bispectrum and higher-order spectra  *(not yet implemented)*

For a random-excitation experiment, the bispectrum B(f₁, f₂) is the 2D Fourier
transform of the third-order cumulant of the output.  It is non-zero only if
quadratic phase coupling exists between components at f₁, f₂, and f₁+f₂ — which
is the signature of a quadratic nonlinearity.

The **bicoherence** is a normalized version (range 0–1) that separates nonlinear
coupling from noise.  Peaks in the bicoherence at (f₁, f₂) confirm that energy
at f₁+f₂ is produced by nonlinear interaction, not independent excitation.

Higher-order spectra (HOS) extend this: the trispectrum is sensitive to cubic
nonlinearities.

In simulation, bispectrum analysis is applied to the output of a broadband random
excitation run.  No special input signal design is needed — any broadband input works,
though random white noise gives the flattest baseline.

**References:**
- Kim, Y. C. & Powers, E. J. (1979). Digital bispectral analysis and its
  applications to nonlinear wave interactions.  *IEEE Transactions on Plasma
  Science*, 7(2), 120–131.
  DOI: [10.1109/TPS.1979.4317207](https://doi.org/10.1109/TPS.1979.4317207)
  (Original bispectrum paper for nonlinear systems.)
- Collis, W. B., White, P. R., & Hammond, J. K. (1998). Higher-order spectra:
  The bispectrum and trispectrum.  *Mechanical Systems and Signal Processing*,
  12(3), 375–394.
  DOI: [10.1006/mssp.1997.0145](https://doi.org/10.1006/mssp.1997.0145)

---

## 9. Phase portraits and Poincaré maps  *(partially implemented via phase portrait plot; Poincaré not yet)*

For periodic forcing u(t) = A·sin(Ωt), the state space trajectory (x, ẋ) is a
closed orbit in steady state for a periodic response.  Stroboscopic sampling at
the forcing period T = 1/Ω produces the **Poincaré section** — a single point for
a periodic response, a finite set for a subharmonic response, and a strange
attractor for a chaotic response.

This is the primary diagnostic tool for:
- Detecting period-doubling bifurcations (onset of chaos)
- Confirming subharmonic resonances (1/2, 1/3 order)
- Identifying quasiperiodic orbits

**References:**
- Thompson, J. M. T. & Stewart, H. B. (2002). *Nonlinear Dynamics and Chaos*
  (2nd ed.). Wiley.  (Standard graduate textbook; Part II covers Poincaré maps
  and bifurcations in detail.)
- Nayfeh, A. H. & Balachandran, B. (1995). *Applied Nonlinear Dynamics.*  Wiley.
  DOI: [10.1002/9783527617548](https://doi.org/10.1002/9783527617548)

---

## 10. Nonlinear subspace identification (NSI) and grey-box methods

The class of subspace identification methods (N4SID, MOESP, …) produces linear
state-space models from input-output data.  NSI extends this by allowing a known
nonlinear functional form in the state equation, then fitting its coefficients.

This is grey-box: the physicist specifies the nonlinearity *type* (Duffing,
van der Pol, quadratic damping, …) and the algorithm fits the coefficients from
broadband data in a single shot — no stepping, no sweeping.

NSI is the method recommended by Noël & Kerschen (2017) when the nonlinearity
structure is known in advance.

**References:**
- Noël, J.-P. & Kerschen, G. (2017). Nonlinear system identification in structural
  dynamics: 10 years of progress.  *Mechanical Systems and Signal Processing*, 83,
  2–35.
  DOI: [10.1016/j.ymssp.2016.07.020](https://doi.org/10.1016/j.ymssp.2016.07.020)
  (Excellent review — covers RFS, NARMAX, NSI, NNMs, and best-practice guidance.)
- Paduart, J., Lauwers, L., Swevers, J., Smolders, K., Schoukens, J., & Pintelon,
  R. (2010). Identification of nonlinear systems using polynomial nonlinear state
  space models.  *Automatica*, 46(4), 647–656.
  DOI: [10.1016/j.automatica.2010.01.001](https://doi.org/10.1016/j.automatica.2010.01.001)

---

## Summary table

| Method | Input signal | What it reveals | Complexity |
|---|---|---|---|
| Stepped sine + THD | Single tone, sweep f | Harmonic order, nonlinearity type | Low |
| **Two-tone / IMD** | Two tones | IM order, IP3, coupling between modes | Low |
| HOFRF (multi-tone) | N tones | Full nth-order kernel | Medium |
| Free decay + HT | Impulse/step | Backbone curve, amplitude-dep. damping | Medium |
| Backbone tracing | Phase resonance or continuation | NNM locus, energy-amplitude relationship | High |
| Restoring force surface | Broadband or swept | Nonparametric force surface | Medium |
| NARMAX | Broadband random | Black-box polynomial model | High |
| Bispectrum | Broadband random | Quadratic/cubic phase coupling | High |
| Phase portrait / Poincaré | Periodic forcing, sweep amplitude | Period-doubling, chaos onset, subharmonics | Low–Medium |
| NSI | Broadband random | Grey-box coefficient fit | High |

**Recommended starting point for simulation-based characterization:**
Two-tone IMD (§2) and free-decay backbone (§5) give the highest information-per-test
ratio for the effort required, and both are straightforward to implement in Numen's
existing simulation-and-FFT pipeline.
