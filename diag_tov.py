#!/usr/bin/env python3
"""
Quick diagnostic: test the Hayward metric for a few L/M values and
print key quantities.
"""
import numpy as np

def hayward_m(r, M, L):
    return M * r**3 / (r**3 + 2.0 * M * L**2)

def hayward_f(r, M, L):
    return 1.0 - 2.0 * hayward_m(r, M, L) / np.asarray(r)

def hayward_rho(r, M, L):
    r3 = np.asarray(r)**3
    denom = r3 + 2.0 * M * L**2
    dm_dr = 6.0 * M**2 * L**2 * np.asarray(r)**2 / denom**2
    with np.errstate(divide='ignore', invalid='ignore'):
        result = dm_dr / (4.0 * np.pi * np.asarray(r)**2)
    result = np.asarray(result)
    result[~np.isfinite(result)] = 0.0
    return float(result) if result.ndim == 0 else result

M = 1.0
L_vals = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
ell = 3

print(f"{'L/M':>8s}  {'rho_c [M^-2]':>14s}  {'f_min':>10s}  "
      f"{'V_peak':>12s}  {'dw/w0(local)':>14s}  {'echo_ms':>12s}")
print("-" * 80)

for L in L_vals:
    rho_c = hayward_rho(1e-6, M, L)

    r_range = np.logspace(-3, 2, 2000)
    f_vals = hayward_f(r_range, M, L)
    f_min = float(f_vals.min())

    V = hayward_f(r_range, M, L) * (ell*(ell+1)/r_range**2
        - 6*hayward_m(r_range, M, L)/r_range**3
        + 8*np.pi*hayward_rho(r_range, M, L))
    V_valid = V > 0
    if V_valid.any():
        V_peak = float(V[V_valid].max())
    else:
        V_peak = np.nan

    r_photon = 3.0 * M
    delta_V = 8.0 * np.pi * hayward_rho(r_photon, M, L) * hayward_f(r_photon, M, L)
    V_schw = (1 - 2*M/r_photon) * (ell*(ell+1)/r_photon**2 - 6*M/r_photon**3)
    frac_local = float(delta_V / V_schw)

    # Echo delay: rough estimate = 2 * (r_*_photon - r_*_core) in tortoise
    # For L close to M, the core is not deep so we approximate
    # r_*_core ~ 0 for large L, r_*_photon ~ O(M)
    # For small L, r_*_core is deeply negative so delay is enormous
    # Simplified: echo_delay ~ 2 * |r_*_photon|
    r_star_photon = r_photon + 2*M*np.log(r_photon/(2*M)-1)
    echo_M = 2 * abs(r_star_photon)
    echo_ms = echo_M * 10.0 * 4.925e-6 * 1000.0

    print(f"  {L/M:5.3f}  {rho_c:14.3e}  {f_min:10.3e}  "
          f"{V_peak:12.3e}  {frac_local:14.3e}  {echo_ms:12.1f}")

print()
print("Detection thresholds:")
print("  Next-gen GW (|dw/w0| > 1e-4):    L/M > ~0.02")
print("  Observable echoes (< 100 ms):     L/M > ~0.5")
