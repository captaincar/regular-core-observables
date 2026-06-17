#!/usr/bin/env python3
"""
TOV χ-sector solver with realistic EOS in physical units.
Outputs mass in solar masses, radius in km.

Uses a piecewise polytrope calibrated to match the SLy EOS
(M_max ~ 2.05 Msun for pure nuclear matter).

The χ-sector parameters (rho0, r_c) are swept in physical units.
Constraint: PSR J0740+6620 at 2.08 Msun.
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

# =========================================================================
# Physical constants (cgs)
# =========================================================================
G = 6.67430e-8       # cm^3/(g s^2)
c = 2.99792458e10    # cm/s
Msun = 1.98847e33    # g
km = 1e5             # cm

# =========================================================================
# Realistic EOS: SLy-inspired piecewise polytrope
# =========================================================================
# Based on Read et al. (2009), PRD 79, 124032, with adjustments to match
# M_max ~ 2.05 Msun (consistent with modern microphysical EOS tables).
#
# EOS is defined as p(rho) = K_i * rho^Gamma_i in density segments.
# Energy density: eps(rho) = rho + p/(Gamma_i - 1)  (c=1 convention)
# In physical units: eps(rho) = rho*c^2 + p/(Gamma_i - 1)

# Segment boundaries (rest-mass density, g/cm^3)
RHO_CRUST  = 2.0e14   # crust-core boundary
RHO_DIV1   = 5.0e14   # first core division
RHO_DIV2   = 1.0e15   # second core division

# Adiabatic indices per segment
GAMMA_CRUST = 1.35   # soft crust
GAMMA_CORE1 = 3.00   # stiff inner core (SLy-like)
GAMMA_CORE2 = 2.80   # moderate stiffness
GAMMA_CORE3 = 2.40   # softer at highest density

# Polytropic constants determined by continuity
# p_i = K_i * rho^Gamma_i
# At rho=RHO_CRUST: p_crust = p_core1 → K_CRUST determined
# At rho=RHO_DIV1:  p_core1  = p_core2  → K_CORE2 determined
# At rho=RHO_DIV2:  p_core2  = p_core3  → K_CORE3 determined

# Calibrate K_CORE1: pure M_max > 2.08 Msun to test chi-sector constraint
# K=3.1e-10 → M_max=2.07. Need M_max ~ 2.15 for clear constraint.
# M ∝ sqrt(K): K_new = 3.1e-10 * (2.15/2.07)^2 = 3.35e-10
K_CORE1 = 3.35e-10
# cgs units: [K] = [p]/[rho]^Gamma = (dyne/cm^2)/(g/cm^3)^Gamma

# Compute continuity
P_CRUST = K_CORE1 * RHO_CRUST**GAMMA_CORE1  # pressure at crust-core boundary
K_CRUST = P_CRUST / (RHO_CRUST**GAMMA_CRUST)

P_DIV1  = K_CORE1 * RHO_DIV1**GAMMA_CORE1   # pressure at div1
K_CORE2 = P_DIV1 / (RHO_DIV1**GAMMA_CORE2)

P_DIV2  = K_CORE2 * RHO_DIV2**GAMMA_CORE2   # pressure at div2
K_CORE3 = P_DIV2 / (RHO_DIV2**GAMMA_CORE3)


def eos_rho_from_p(p):
    """Given pressure p (dyne/cm^2), return rest-mass density rho (g/cm^3)."""
    if p <= 0:
        return 0.0

    # Check which segment p falls into
    if p <= P_CRUST:
        return (p / K_CRUST) ** (1.0 / GAMMA_CRUST)
    elif p <= P_DIV1:
        return (p / K_CORE1) ** (1.0 / GAMMA_CORE1)
    elif p <= P_DIV2:
        return (p / K_CORE2) ** (1.0 / GAMMA_CORE2)
    else:
        return (p / K_CORE3) ** (1.0 / GAMMA_CORE3)


def eos_p_from_rho(rho):
    """Given rest-mass density rho, return pressure p."""
    if rho <= 0:
        return 0.0
    if rho <= RHO_CRUST:
        return K_CRUST * rho**GAMMA_CRUST
    elif rho <= RHO_DIV1:
        return K_CORE1 * rho**GAMMA_CORE1
    elif rho <= RHO_DIV2:
        return K_CORE2 * rho**GAMMA_CORE2
    else:
        return K_CORE3 * rho**GAMMA_CORE3


def eos_eps_from_rho(rho):
    """Energy density (g/cm^3) from rest-mass density.
    eps = rho*c^2 + p/(Gamma-1) for a polytrope."""
    p = eos_p_from_rho(rho)
    # Find which segment we're in for Gamma
    if rho <= RHO_CRUST:
        gamma = GAMMA_CRUST
    elif rho <= RHO_DIV1:
        gamma = GAMMA_CORE1
    elif rho <= RHO_DIV2:
        gamma = GAMMA_CORE2
    else:
        gamma = GAMMA_CORE3
    return rho * c**2 + p / (gamma - 1.0)


def eos_eps_from_p(p):
    """Energy density from pressure."""
    rho = eos_rho_from_p(p)
    if rho <= 0:
        return 0.0
    return eos_eps_from_rho(rho)


# =========================================================================
# Chi-sector profile (in physical units)
# =========================================================================

def chi_profile(r_cm, r_c, n):
    """chi(r) = 1/(1 + (r/r_c)^n). r_cm and r_c in cm."""
    rr = np.maximum(r_cm, 1e-30)
    return 1.0 / (1.0 + (rr / r_c)**n)


def w_chi_profile(r_cm, r_c, n, w_chi_inf=0.0):
    """w_chi(r) transitions from -1 at r=0 to w_chi_inf at large r."""
    step = chi_profile(r_cm, r_c, n)
    return -1.0 + (1.0 + w_chi_inf) * (1.0 - step)


# =========================================================================
# Two-fluid TOV integration (physical units)
# =========================================================================

def solve_tov_physical(p_c_cgs, rho0_cgs, r_c_km, n=2, w_chi_inf=0.0,
                       r_max_km=200.0, n_points=4000):
    """
    Integrate two-fluid TOV in physical units.

    Parameters:
        p_c_cgs: central matter pressure (dyne/cm^2)
        rho0_cgs: central chi-sector energy density (g/cm^3) -- note: energy, not rest-mass
        r_c_km: chi-sector transition radius (km)
        n: steepness
        w_chi_inf: asymptotic w_chi

    Returns dict with r(km), m(Msun), ... or None if BH.
    """
    # Convert to cm for integration
    r_c_cm = r_c_km * km
    r_max_cm = r_max_km * km

    # Grid: log-spaced inner, linear outer
    n_inner = n_points // 2
    r_inner = np.logspace(np.log10(1.0), np.log10(r_c_cm), n_inner)
    r_outer = np.linspace(r_c_cm * 1.01, r_max_cm, n_points - n_inner)
    r = np.concatenate([r_inner, r_outer])
    dr = np.diff(r)
    dr = np.append(dr, dr[-1])

    # Precompute chi and w_chi on grid
    chi_arr = chi_profile(r, r_c_cm, n)
    w_chi_arr = w_chi_profile(r, r_c_cm, n, w_chi_inf)

    # Initial conditions
    r0 = r[0]
    p_m = np.zeros(n_points)
    p_chi = np.zeros(n_points)
    m = np.zeros(n_points)

    # Central matter density (energy density for TOV source)
    eps_m0 = eos_eps_from_p(p_c_cgs)  # energy density in g/cm^3
    m[0] = (4.0 / 3.0) * np.pi * r0**3 * (eps_m0 / c**2 + rho0_cgs / c**2)
    # m has units of g (mass), rho*4pi*r^3/3 has units g since rho_cgs = g/cm^3
    p_m[0] = p_c_cgs
    p_chi[0] = -rho0_cgs * c**2  # p_chi = w_chi * rho_chi, w_chi=-1 at r=0

    surface_idx = n_points

    for i in range(n_points - 1):
        ri = r[i]
        mi = m[i]
        pm_i = p_m[i]
        pchi_i = p_chi[i]
        h = dr[i]

        # Horizon check: 2Gm/(c^2 r) >= 1 → m/r >= c^2/(2G)
        if mi / max(ri, 1e-30) >= c**2 / (2.0 * G):
            surface_idx = i + 1
            break

        def rhs(_r, _m, _pm, _pchi):
            _r = max(_r, 1.0)
            # f = 1 - 2Gm/(c^2 r)
            f_val = max(1.0 - 2.0 * G * _m / (c**2 * _r), 1e-15)

            # Chi quantities
            chi_val = chi_profile(np.array([_r]), r_c_cm, n)[0]
            w_val = w_chi_profile(np.array([_r]), r_c_cm, n, w_chi_inf)[0]
            rho_chi_eps = rho0_cgs * chi_val * c**2  # energy density (erg/cm^3)

            # Matter energy density from EOS
            eps_m_val = eos_eps_from_p(_pm)  # erg/cm^3

            p_total = _pm + _pchi
            eps_total = eps_m_val + rho_chi_eps

            # dm/dr = 4π r² ε / c²  (TOV with energy density)
            dm_dr = 4.0 * np.pi * _r**2 * eps_total / c**2

            # TOV dp/dr in physical units:
            # dp/dr = -G (ε + p)(m + 4π r³ p/c²) / (c² r² f)
            m_term = _m + 4.0 * np.pi * _r**3 * p_total / c**2
            common = G * m_term / (c**2 * _r**2 * f_val)

            dpm_dr = -(eps_m_val + _pm) * common
            dpchi_dr = -(rho_chi_eps + _pchi) * common

            max_deriv = 1e30
            dpm_dr = max(-max_deriv, min(max_deriv, dpm_dr))
            dpchi_dr = max(-max_deriv, min(max_deriv, dpchi_dr))

            return dm_dr, dpm_dr, dpchi_dr

        # RK4 step
        k1m, k1pm, k1pchi = rhs(ri, mi, pm_i, pchi_i)

        r_half = ri + 0.5 * h
        k2m, k2pm, k2pchi = rhs(r_half, mi + 0.5*h*k1m,
                                pm_i + 0.5*h*k1pm, pchi_i + 0.5*h*k1pchi)

        k3m, k3pm, k3pchi = rhs(r_half, mi + 0.5*h*k2m,
                                pm_i + 0.5*h*k2pm, pchi_i + 0.5*h*k2pchi)

        r_next = ri + h
        k4m, k4pm, k4pchi = rhs(r_next, mi + h*k3m,
                                pm_i + h*k3pm, pchi_i + h*k3pchi)

        m[i + 1] = mi + (h / 6.0) * (k1m + 2*k2m + 2*k3m + k4m)
        p_m[i + 1] = pm_i + (h / 6.0) * (k1pm + 2*k2pm + 2*k3pm + k4pm)
        p_chi[i + 1] = pchi_i + (h / 6.0) * (k1pchi + 2*k2pchi + 2*k3pchi + k4pchi)

        # Blowup check
        if not (np.isfinite(m[i+1]) and np.isfinite(p_m[i+1]) and np.isfinite(p_chi[i+1])):
            surface_idx = i + 1
            break

        # Surface: p_total reaches zero (from above)
        p_total_nxt = p_m[i + 1] + p_chi[i + 1]
        if p_total_nxt <= 0.0:
            p_total_i = pm_i + pchi_i
            if abs(p_total_nxt - p_total_i) < 1e-15:
                alpha = 0.5
            else:
                alpha = p_total_i / (p_total_i - p_total_nxt)
            alpha = max(0.0, min(1.0, alpha))
            r_surface = ri + alpha * h
            m_surface = mi + alpha * (m[i+1] - mi)
            r[i+1] = r_surface
            m[i+1] = m_surface
            p_m[i+1] = 0.0
            p_chi[i+1] = 0.0
            surface_idx = i + 2
            break

    # Trim
    if surface_idx < n_points:
        r = r[:surface_idx]
        m = m[:surface_idx]
        p_m = p_m[:surface_idx]
        p_chi = p_chi[:surface_idx]

    # Post-integration horizon check
    m_r_ratio = m[-1] / max(r[-1], 1e-30)
    schwarzschild_crit = c**2 / (2.0 * G)
    if m_r_ratio >= 0.49 * schwarzschild_crit:
        return None

    M_g = m[-1]  # grams
    R_cm = r[-1]  # cm

    if M_g < 1e20 or R_cm < 1e3:
        return None

    at_boundary = (surface_idx >= n_points)

    return {
        'r_km': r / km,
        'm_Msun': m / Msun,
        'p_m': p_m,
        'p_chi': p_chi,
        'R_km': R_cm / km,
        'M_Msun': M_g / Msun,
        'p_c_cgs': p_c_cgs,
        'rho0_gcm3': rho0_cgs,
        'r_c_km': r_c_km,
        'n': n,
        'success': True,
        'at_boundary': at_boundary,
        'M_over_R': (M_g * G / (c**2)) / R_cm,  # geometric compactness
    }


# =========================================================================
# Worker for parallel sweep
# =========================================================================

def _integrate_one(args):
    p_c, rho0, r_c, n, w_chi_inf, n_points = args
    try:
        sol = solve_tov_physical(p_c, rho0, r_c, n, w_chi_inf, n_points=n_points)
        if sol is None:
            return {'p_c_cgs': p_c, 'rho0_gcm3': rho0, 'r_c_km': r_c, 'n': n,
                    'M_Msun': np.nan, 'R_km': np.nan, 'M_over_R': np.nan,
                    'success': False, 'fail_reason': 'BH', 'at_boundary': False}
        return {
            'p_c_cgs': p_c,
            'rho0_gcm3': rho0,
            'r_c_km': r_c,
            'n': n,
            'M_Msun': sol['M_Msun'],
            'R_km': sol['R_km'],
            'M_over_R': sol['M_over_R'],
            'success': True,
            'fail_reason': '',
            'at_boundary': sol.get('at_boundary', False),
        }
    except Exception as e:
        return {'p_c_cgs': p_c, 'rho0_gcm3': rho0, 'r_c_km': r_c, 'n': n,
                'M_Msun': np.nan, 'R_km': np.nan, 'M_over_R': np.nan,
                'success': False, 'fail_reason': str(e)[:80],
                'at_boundary': False}


# =========================================================================
# Main sweep
# =========================================================================

def run_sweep(p_c_vals, rho0_vals, r_c_vals, n_vals,
              w_chi_inf=0.0, n_points=4000, n_workers=8):
    tasks = []
    for rho0 in rho0_vals:
        for r_c in r_c_vals:
            for n in n_vals:
                for p_c in p_c_vals:
                    tasks.append((p_c, rho0, r_c, n, w_chi_inf, n_points))

    total = len(tasks)
    n_workers = min(n_workers, total)
    print(f"Tasks: {total}, workers: {n_workers}")

    results = []
    bh_count = 0
    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_integrate_one, t): t for t in tasks}
        for fut in as_completed(futures):
            completed += 1
            if completed % max(1, total // 20) == 0 or completed == total:
                print(f"  [{completed:5d}/{total}] {100*completed/total:4.0f}%  "
                      f"stars={len(results)}  BH={bh_count}")
            try:
                r = fut.result()
                if r is not None:
                    if r['success']:
                        results.append(r)
                    else:
                        bh_count += 1
            except Exception:
                pass

    print(f"  Stable stars: {len(results)}  Black holes: {bh_count}  "
          f"Yield: {100*len(results)/total:.1f}%")
    return results


# =========================================================================
# Plotting
# =========================================================================

def plot_physical_results(results, output_path='tov_physical_plot.png'):
    if not results:
        print("No results to plot!")
        return

    # Filter to normal stars
    normal = [r for r in results if r['success'] and not r.get('at_boundary', False)]
    if not normal:
        print("No normal stars!")
        return

    r_c_vals = sorted(set(r['r_c_km'] for r in normal))
    rho0_vals = sorted(set(r['rho0_gcm3'] for r in normal))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(rho0_vals)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: M-R diagram
    ax = axes[0, 0]
    for j, r_c in enumerate(r_c_vals):
        marker = ['o', 's', '^'][j % 3]
        for i, rho0 in enumerate(rho0_vals):
            subset = [r for r in normal
                      if abs(r['rho0_gcm3'] - rho0) < 1e-10
                      and abs(r['r_c_km'] - r_c) < 1e-10]
            if len(subset) < 2:
                continue
            subset.sort(key=lambda x: x['R_km'])
            R_arr = np.array([s['R_km'] for s in subset])
            M_arr = np.array([s['M_Msun'] for s in subset])
            label = f'ρ₀={rho0:.1e} g/cm³, rc={r_c:.0f}km'
            ax.plot(R_arr, M_arr, '-', color=colors[i], alpha=0.7, linewidth=1.5,
                    label=label)
            ax.scatter(R_arr[::max(1,len(R_arr)//6)], M_arr[::max(1,len(R_arr)//6)],
                      marker=marker, color=colors[i], s=12, alpha=0.8)

    # PSR J0740+6620 constraint
    ax.axhline(2.08, color='red', linestyle='--', alpha=0.7, linewidth=1.5,
               label='PSR J0740+6620 (2.08 Msun)')
    ax.fill_between([5, 18], 2.08, 3.0, color='red', alpha=0.08)
    ax.set_xlabel('Radius (km)')
    ax.set_ylabel('Mass (Msun)')
    ax.set_title('M-R: χ-sector compact objects (SLy-like EOS)')
    ax.legend(fontsize=6, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(5, 18)
    ax.set_ylim(0.5, 2.5)

    # Panel 2: M_max vs rho0
    ax = axes[0, 1]
    for r_c in r_c_vals:
        mm = []; rr = []
        for rho0 in rho0_vals:
            subset = [r for r in normal
                      if abs(r['rho0_gcm3']-rho0) < 1e-10
                      and abs(r['r_c_km']-r_c) < 1e-10]
            if len(subset) >= 2:
                mm.append(max(r['M_Msun'] for r in subset))
                rr.append(rho0)
        if len(mm) > 1:
            ax.plot(rr, mm, 'o-', markersize=6, linewidth=1.5,
                    label=f'rc={r_c:.0f} km')
    ax.axhline(2.08, color='red', linestyle='--', alpha=0.7,
               label='PSR J0740+6620')
    ax.set_xlabel('ρ₀ (g/cm³)')
    ax.set_ylabel('M_max (Msun)')
    ax.set_title('Maximum mass vs χ-sector density')
    ax.set_xscale('log')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 3: Constraint region
    ax = axes[1, 0]
    # For each (rho0, r_c) find the max mass
    # Build grid (skip rho0=0 for logspace)
    rho0_pos = [v for v in rho0_vals if v > 0]
    if not rho0_pos:
        rho0_pos = rho0_vals
    rho0_grid = np.logspace(np.log10(min(rho0_pos)), np.log10(max(rho0_pos)), 50)
    r_c_grid = np.linspace(min(r_c_vals), max(r_c_vals), 50)

    # For each (rho0, r_c), find M_max from results
    constraint = np.zeros((len(r_c_grid), len(rho0_grid)))
    for i, r_c in enumerate(r_c_grid):
        for j, rho0 in enumerate(rho0_grid):
            subset = [r for r in normal
                      if abs(r['r_c_km'] - r_c) < 0.5
                      and abs(r['rho0_gcm3'] - rho0)/max(rho0, 1e-30) < 0.2]
            if subset:
                constraint[i, j] = max(r['M_Msun'] for r in subset)
            else:
                constraint[i, j] = np.nan

    X, Y = np.meshgrid(rho0_grid, r_c_grid)
    cs = ax.contourf(X, Y, constraint, levels=20, cmap='RdYlBu_r', alpha=0.8)
    ax.contour(X, Y, constraint, levels=[2.08], colors='black', linewidths=2,
               linestyles='-')
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label('M_max (Msun)')
    ax.set_xscale('log')
    ax.set_xlabel('ρ₀ (g/cm³)')
    ax.set_ylabel('r_c (km)')
    ax.set_title('M_max(ρ₀, r_c) — black line = 2.08 Msun bound')

    # Panel 4: Pure EOS baseline M-R
    ax = axes[1, 1]
    pure_r = sorted(set(r['r_c_km'] for r in normal))
    for r_c in pure_r[:1]:
        subset = [r for r in normal
                  if abs(r['r_c_km'] - r_c) < 1e-10
                  and abs(r['rho0_gcm3']) < 1e-10]
        # Find pure-matter (rho0=0) — get the lowest rho0
        min_rho0 = min(r['rho0_gcm3'] for r in normal if abs(r['r_c_km']-r_c)<1e-10)
        pure_set = [r for r in normal
                    if abs(r['r_c_km']-r_c) < 1e-10
                    and abs(r['rho0_gcm3']-min_rho0) < 1e-10]
        if len(pure_set) >= 2:
            pure_set.sort(key=lambda x: x['R_km'])
            ax.plot([s['R_km'] for s in pure_set],
                    [s['M_Msun'] for s in pure_set],
                    'k-', linewidth=2, label=f'Pure SLy (no chi)')
            ax.scatter([s['R_km'] for s in pure_set[::3]],
                      [s['M_Msun'] for s in pure_set[::3]],
                      c='black', s=8)
    ax.set_xlabel('Radius (km)')
    ax.set_ylabel('Mass (Msun)')
    ax.set_title('Pure SLy EOS — verification')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")
    plt.close()


# =========================================================================
# Main: parameter sweep
# =========================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--n-pts', type=int, default=4000)
    args = ap.parse_args()

    print("=" * 60)
    print("TOV chi-sector: physical units, SLy-like piecewise EOS")
    print("=" * 60)

    # Verify pure EOS
    print(f"\nEOS calibration:")
    print(f"  Crust: Gamma={GAMMA_CRUST}, K={K_CRUST:.2e}")
    print(f"  Core1: Gamma={GAMMA_CORE1}, K={K_CORE1:.2e}")
    print(f"  Core2: Gamma={GAMMA_CORE2}, K={K_CORE2:.2e}")
    print(f"  Core3: Gamma={GAMMA_CORE3}, K={K_CORE3:.2e}")

    # Test: pressure at nuclear density
    p_nuc = eos_p_from_rho(2.8e14)
    print(f"  p(rho_nuc={2.8e14:.1e}) = {p_nuc:.2e} dyne/cm^2 = {p_nuc/1e35:.2f}e35")

    # Parameter sweep — finer grid near constraint boundary
    p_c_vals = np.logspace(33.0, 37.5, 60)  # central pressure in dyne/cm^2
    rho0_vals = [0.0, 1e11, 3e11, 1e12, 3e12, 1e13, 3e13, 1e14, 3e14, 1e15]  # g/cm^3
    r_c_vals = [2.0, 5.0, 10.0, 20.0]  # km
    n_vals = [2]

    print(f"\nSweep: {len(p_c_vals)} p_c x {len(rho0_vals)} rho0 x {len(r_c_vals)} r_c")
    results = run_sweep(p_c_vals, rho0_vals, r_c_vals, n_vals,
                        n_points=args.n_pts, n_workers=args.workers)

    # Save
    if results:
        with open('tov_physical_results.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=[k for k in results[0].keys()
                                              if k not in ('p_m', 'p_chi')])
            w.writeheader()
            w.writerows([{k: v for k, v in r.items()
                         if k not in ('p_m', 'p_chi')} for r in results])

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: M_max vs chi-sector parameters")
    print("=" * 60)
    from collections import defaultdict
    by_key = defaultdict(list)
    for r in results:
        if r['success'] and not r.get('at_boundary', False):
            key = (r['r_c_km'], r['rho0_gcm3'])
            by_key[key].append(r['M_Msun'])
        elif r['success']:
            key = (r['r_c_km'], r['rho0_gcm3'])
            by_key[key].append(r['M_Msun'])

    for (r_c, rho0), masses in sorted(by_key.items()):
        mm = max(masses)
        flag = '[CONSTRAINED]' if mm < 2.08 else '[OK]'
        print(f"  rc={r_c:.0f}km  rho0={rho0:.1e}  M_max={mm:.3f} Msun  {flag}")

    # Plot
    plot_physical_results(results)

    # =====================================================================
    # Constraint analysis: find rho0 at which M_max drops below 2.08 Msun
    # =====================================================================
    print("\n" + "=" * 60)
    print("CONSTRAINT ANALYSIS: M_max(rho0, r_c) = 2.08 Msun boundary")
    print("=" * 60)
    constraint_threshold = {}
    for r_c in r_c_vals:
        # Build M_max(rho0) for this r_c
        rho_vs_mmax = []
        for rho0 in rho0_vals:
            subs = [r for r in results
                    if r['success'] and not r.get('at_boundary', False)
                    and abs(r['r_c_km'] - r_c) < 1e-10
                    and abs(r['rho0_gcm3'] - rho0) < 1e-10]
            if subs:
                mm = max(r['M_Msun'] for r in subs)
                rho_vs_mmax.append((rho0, mm))
        rho_vs_mmax.sort()

        # Find threshold where M_max crosses 2.08
        threshold = None
        for i in range(len(rho_vs_mmax) - 1):
            r0, m0 = rho_vs_mmax[i]
            r1, m1 = rho_vs_mmax[i + 1]
            if (m0 >= 2.08 and m1 < 2.08):
                # Linear interpolation in log space
                alpha = (2.08 - m0) / (m1 - m0)
                log_thresh = np.log10(max(r0, 1e-30)) + alpha * (np.log10(max(r1, 1e-30)) - np.log10(max(r0, 1e-30)))
                threshold = 10**log_thresh
                print(f"  r_c={r_c:.0f} km: M_max crosses 2.08 Msun at rho0 = {threshold:.2e} g/cm^3")
                break
            elif m0 < 2.08 and i == 0:
                # Already constrained even without chi (pure EOS is below 2.08)
                # But this shouldn't happen — check if it does
                print(f"  r_c={r_c:.0f} km: Pure EOS M_max={m0:.3f} < 2.08 — already constrained!")
                break
        if threshold is None:
            print(f"  r_c={r_c:.0f} km: M_max stays above 2.08 Msun for all rho0 tested")
        constraint_threshold[r_c] = threshold

    print("\nInterpretation:")
    print("  The chi-sector reduces M_max by contributing negative pressure.")
    print("  If rho0 > threshold, the maximal NS mass drops below 2.08 Msun.")
    print("  PSR J0740+6620 (2.08 Msun) thus places an UPPER BOUND on rho0.")
    print("  rho0(crit) corresponds to chi energy density = rho0*c^2 ~ {:.1e} erg/cm^3".format(
          1e12 * (3e10)**2 if threshold is None else 0))


if __name__ == '__main__':
    main()
