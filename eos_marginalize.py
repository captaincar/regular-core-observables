#!/usr/bin/env python3
"""EOS Bayesian marginalization for NICER rho0 bound.

Computes the marginalized chi-sector density bound by weighting
4 EOSs (SLy, H4, APR4, MS1b) by their likelihood given PSR J0740+6620
(M = 2.08 +/- 0.07 Msun) and GW170817 tidal constraints.

Method:
  P(EOS | data) ∝ P(data | EOS) × P(EOS)
  P(data | EOS) = Φ((M_max - 2.08)/0.07)  [Gaussian CDF]
  Prior choices: uniform, GW170817-Λ-filtered, SLy+APR4 only

Output: marginalized ρ₀ bound at 1σ (M_max >= 2.01) and 2σ (>= 1.94).
"""
import numpy as np
from scipy.stats import norm

# Observational constraints
M_PSR = 2.08
SIGMA_PSR = 0.07
NICER_1S = 2.01
NICER_2S = 1.94
LAMBDA_GW170817_MAX = 580

# M_max(ρ₀) from eos_comparison.py (r_c=2 km, n=2, golden-section refined)
rho0_vals = np.array([0, 1e10, 3e10, 1e11, 3e11, 1e12, 3e12, 1e13, 3e13, 1e14])
eos_data = {
    'SLy':  {'M_pure': 2.050, 'Gamma1': 3.005, 'Lambda14': 531,
             'M_by_rho': [2.050, 2.024, 2.012, 1.994, 1.972, 1.937, 1.892, 1.788, 1.702, 1.495]},
    'H4':   {'M_pure': 2.037, 'Gamma1': 2.909, 'Lambda14': 1200,
             'M_by_rho': [2.037, 2.029, 2.029, 2.028, 2.024, 2.005, 1.964, 1.877, 1.724, 1.440]},
    'APR4': {'M_pure': 2.200, 'Gamma1': 2.830, 'Lambda14': 280,
             'M_by_rho': [2.200, 2.161, 2.145, 2.121, 2.091, 2.043, 1.987, 1.905, 1.695, 1.591]},
    'MS1b': {'M_pure': 2.800, 'Gamma1': 3.456, 'Lambda14': 2400,
             'M_by_rho': [2.800, 2.742, 2.716, 2.678, 2.626, 2.544, 2.439, 2.286, 2.131, 1.841]},
}


def find_bound(M_by_rho, threshold):
    """Highest ρ₀ where M_max >= threshold (from discrete grid)."""
    best = None
    for rho0, mm in zip(rho0_vals, M_by_rho):
        if mm >= threshold:
            best = rho0
    return best


def marginalize(prior_weights, label):
    """Compute marginalized ρ₀ bound for a given prior over EOSs."""
    names = list(eos_data.keys())
    
    # PSR likelihood
    likes = {n: norm.cdf(eos_data[n]['M_pure'], M_PSR, SIGMA_PSR) for n in names}
    
    # Posterior = prior × likelihood
    posterior = {}
    total = 0
    for i, n in enumerate(names):
        posterior[n] = prior_weights[i] * likes[n]
        total += posterior[n]
    for n in posterior:
        posterior[n] /= total

    print(f"\n{'='*60}")
    print(f"Prior: {label}")
    print(f"{'='*60}")
    print(f"  {'EOS':>6s}  {'M_pure':>7s}  {'L(1.4)':>7s}  {'prior':>6s}  {'likelih':>7s}  {'poster':>7s}")
    for i, n in enumerate(names):
        d = eos_data[n]
        lam_str = f"{d['Lambda14']}" if d['Lambda14'] <= LAMBDA_GW170817_MAX else f"{d['Lambda14']}*"
        print(f"  {n:>6s}  {d['M_pure']:7.3f}  {lam_str:>7s}  {prior_weights[i]:6.3f}  {likes[n]:7.4f}  {posterior[n]:7.4f}")
    print("  * = excluded by GW170817 Lambda(1.4) < 580")

    for thresh, thresh_label in [(NICER_1S, "1sig"), (NICER_2S, "2sig")]:
        wb = sum(posterior[n] * (find_bound(eos_data[n]['M_by_rho'], thresh) or 0) for n in names)
        print(f"  Marginalized rho0 ({thresh_label}, M_max >= {thresh}): {wb:.1e} g/cm3")
    
    return posterior


def main():
    print("=" * 60)
    print("EOS BAYESIAN MARGINALIZATION: NICER rho0 bound")
    print(f"PSR J0740+6620: M = {M_PSR} +/- {SIGMA_PSR} Msun")
    print(f"GW170817: Lambda(1.4) < {LAMBDA_GW170817_MAX} (90% CI)")
    print("=" * 60)

    # Three prior scenarios
    marginalize([0.25, 0.25, 0.25, 0.25],
                "Uniform (all 4 EOSs)")

    marginalize([1/3, 1/3, 1/3, 0.0],
                "GW170817-consistent only (exclude MS1b, Lambda~2400)")

    marginalize([0.5, 0.0, 0.5, 0.0],
                "SLy + APR4 only (GW170817-consistent, covers range)")

    # Summary
    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")
    print(f"  SLy-only 1sig bound:  3.0e10 g/cm3  [most conservative]")
    print(f"  SLy-only 2sig bound:  3.0e11 g/cm3")
    print(f"  Marginalized (SLy+APR4): ~8e11 g/cm3  [GW170817-consistent]")
    print(f"  Marginalized (all 4):   ~1e13 g/cm3  [MS1b-inflated]")
    print(f"")
    print(f"  The chi-sector rho0 bound is EOS-dominated. The SLy value is the")
    print(f"  conservative floor; marginalization over viable EOSs relaxes it")
    print(f"  by 1-2 orders of magnitude. A proper constraint requires a")
    print(f"  continuous EOS prior, not 4 discrete points.")


if __name__ == '__main__':
    main()
