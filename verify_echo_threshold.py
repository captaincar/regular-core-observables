#!/usr/bin/env python3
"""Verify the echo threshold L/M >= 0.77 numerically.

Computes the tortoise integral for the Hayward metric:
    Delta t_echo ≈ 2 * integral_{r_core}^{r_barrier} dr/f(r)

For a Hayward black hole:
    f(r) = 1 - 2Mr^2 / (r^3 + 2ML^2)

The echo becomes resolvable (~0.1 ms for 10 M_sun) when L/M is near extremality.
"""
import numpy as np
from scipy.integrate import simpson

G = 6.67430e-8
c_cm = 2.99792458e10
M_sun_g = 1.98847e33
M_sun_cm = G * M_sun_g / c_cm**2  # ~1.477e5 cm for 1 Msun

def hayward_f(r, M, L):
    """Hayward metric function f(r) = 1 - 2Mr^2/(r^3 + 2ML^2)."""
    return 1.0 - 2.0 * M * r**2 / (r**3 + 2.0 * M * L**2)

def hayward_fprime(r, M, L):
    """Derivative of Hayward f(r) = 2Mr(r^3 - 4ML^2)/(r^3+2ML^2)^2."""
    num = 2.0 * M * r * (r**3 - 4.0 * M * L**2)
    den = (r**3 + 2.0 * M * L**2)**2
    return num / den

def find_horizon(M, L):
    """Find outermost root of f(r)=0 via Newton iteration."""
    if L >= 4.0*M/(3.0*np.sqrt(3.0)):
        return None  # no horizon (extremal or horizonless)
    # Start from r=2M (Schwarzschild radius) and move inward
    r = 2.0 * M
    for _ in range(100):
        f = hayward_f(r, M, L)
        fp = hayward_fprime(r, M, L)
        if abs(fp) < 1e-15:
            break
        dr = -f / fp
        r += dr
        if abs(dr/r) < 1e-12:
            return r
    return r if hayward_f(r, M, L) < 1e-10 else None

def find_light_ring(M, L):
    """Find photon sphere radius from f'(r)/2 = f(r)/r."""
    r = 3.0 * M  # initial guess (Schwarzschild photon sphere)
    for _ in range(100):
        f = hayward_f(r, M, L)
        fp = hayward_fprime(r, M, L)
        # Condition: fp/2 - f/r = 0
        g = fp/2.0 - f/r
        # Numerical derivative of g
        eps = 1e-6 * r
        rp = r + eps
        fp2 = hayward_fprime(rp, M, L)
        f2 = hayward_f(rp, M, L)
        gp = fp2/2.0 - f2/rp
        dg = (gp - g) / eps
        if abs(dg) < 1e-15:
            break
        dr = -g / dg
        r += dr
        if abs(dr/r) < 1e-12:
            return r
    return r

def tortoise_echo_delay(M, L, M_msun=10.0):
    """Compute echo time delay in milliseconds.
    
    Integrates 2*int_{r_core}^{r_barrier} dr/f(r).
    r_core ≈ 0 (or the inner boundary where f->1 for horizonless case).
    r_barrier = photon sphere radius.
    """
    r_barrier = find_light_ring(M, L)
    r_horizon = find_horizon(M, L)
    
    if r_horizon is not None:
        # Has horizon — no echo possible (GWs fall into horizon)
        return None, r_horizon, r_barrier
    else:
        # No horizon — echo from core reflection
        r_core = 0.01 * M  # integrate from near-center
    
    # Logarithmic grid for integration (dense near horizon/core)
    n_pts = 20000
    r_grid = np.logspace(np.log10(r_core), np.log10(r_barrier), n_pts)
    
    integrand = 1.0 / hayward_f(r_grid, M, L)
    
    # Trapz integration
    integral = np.trapz(integrand, r_grid)
    
    # Convert geometric units to ms
    # integral is in units of M (geometric length = GM/c^2)
    # physical time = 2 * integral * (GM/c^3)
    M_phys_g = M_msun * M_sun_g
    dt_sec = 2.0 * integral * (G * M_phys_g / c_cm**3)
    dt_ms = dt_sec * 1000.0
    
    return dt_ms, r_horizon, r_barrier

def main():
    M = 1.0  # geometric units (M = 1)
    M_msun = 10.0  # 10 solar mass black hole
    
    print("=" * 70)
    print("ECHO THRESHOLD VERIFICATION — Hayward metric")
    print(f"Remnant mass: {M_msun:.0f} Msun")
    print("=" * 70)
    print(f"{'L/M':>8s}  {'r_horizon':>10s}  {'r_barrier':>10s}  {'dt_echo(ms)':>14s}  {'Status':>12s}")
    print("-" * 70)
    
    L_ext = 4.0 / (3.0 * np.sqrt(3.0))  # ~0.7698
    
    for L_M in np.linspace(0.01, 0.90, 20):
        L = L_M * M
        dt_ms, r_h, r_b = tortoise_echo_delay(M, L, M_msun)
        
        has_horizon = r_h is not None
        rh_str = f'{r_h:.4f}' if has_horizon else 'NONE'
        
        if dt_ms is None:
            print(f"{L_M:8.4f}  {rh_str:>10s}  {r_b:10.4f}  {'N/A':>14s}  {'NO ECHO':>12s}")
        elif dt_ms > 1.0:
            print(f"{L_M:8.4f}  {rh_str:>10s}  {r_b:10.4f}  {dt_ms:14.4f}  {'RESOLVABLE':>12s}")
        elif dt_ms > 0.1:
            print(f"{L_M:8.4f}  {rh_str:>10s}  {r_b:10.4f}  {dt_ms:14.4f}  {'MARGINAL':>12s}")
        else:
            print(f"{L_M:8.4f}  {rh_str:>10s}  {r_b:10.4f}  {dt_ms:14.4f}  {'unresolvable':>12s}")
    
    # Precise threshold sweep
    print(f"\n{'='*70}")
    print(f"THRESHOLD SCAN near L_ext = {L_ext:.6f}")
    print(f"{'='*70}")
    for L_M in np.linspace(L_ext - 0.05, L_ext + 0.05, 21):
        L = L_M * M
        dt_ms, r_h, r_b = tortoise_echo_delay(M, L, M_msun)
        has_horizon = r_h is not None
        
        rh_str = f'{r_h:.6f}' if has_horizon else 'NONE'
        dt_str = f'{dt_ms:.6f} ms' if dt_ms is not None else 'N/A (has horizon)'
        print(f"L/M = {L_M:.6f}: dt = {dt_str}, r_h = {rh_str}")

if __name__ == '__main__':
    main()
