#!/usr/bin/env python3
"""
TOV χ-sector mass-radius solver.

Solves the two-fluid Tolman-Oppenheimer-Volkoff equations for a compact object
with both ordinary matter (polytrope EOS) and a Dragan-Ekert χ-sector component
that dominates in the deep interior.

Two-fluid TOV system (geometric units G = c = 1):
    dm/dr   = 4π r² (ρ_m + ρ_χ)
    dp_m/dr = -(ρ_m + p_m) (m + 4π r³ (p_m + p_χ)) / (r (r - 2m))
    dp_χ/dr = -(ρ_χ + p_χ) (m + 4π r³ (p_m + p_χ)) / (r (r - 2m))

χ-sector profile (from deepArticle.md §I.5):
    χ(r)    = 1 / (1 + (r/rc)^n)
    ρ_χ(r)  = ρ₀ χ(r)
    p_χ(r)  = w_χ(r) ρ_χ(r)
    w_χ(r)  = -1 + (1 + w_χ∞) · (r/rc)^n / (1 + (r/rc)^n)
    → w_χ = -1 (de Sitter) at r ≪ rc, transitions to w_χ∞ at r ≫ rc

Matter EOS: polytrope  p_m = K ρ_m^γ   (e.g., γ=2 for n=1 polytrope)

Boundary conditions at r → 0:
    m(0) = 0
    p_m(0) = p_c (central matter pressure)
    p_χ(0) = -ρ₀

Surface: p_total = p_m + p_χ = 0  →  R, M = m(R)

Usage:
    python tov_chi_sector.py
    python tov_chi_sector.py --rho0 0.01,0.05,0.1,0.5,1.0
    python tov_chi_sector.py --workers 12
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


# =========================================================================
#  χ-sector profile
# =========================================================================

def chi_profile(r, rc, n):
    """χ(r) = 1 / (1 + (r/rc)^n)"""
    r = np.maximum(r, 1e-30)
    return 1.0 / (1.0 + (r / rc)**n)


def w_chi_profile(r, rc, n, w_chi_inf=0.0):
    """
    w_χ(r) transitions from -1 at r=0 to w_chi_inf at large r.
    Uses the same sigmoidal form as χ(r):
        w_χ(r) = -1 + (1 + w_chi_inf) · (r/rc)^n / (1 + (r/rc)^n)
    """
    step = chi_profile(r, rc, n)  # 1/(1+x^n)
    return -1.0 + (1.0 + w_chi_inf) * (1.0 - step)
    # Alternatively: step = (r/rc)^n / (1+(r/rc)^n) = 1 - χ(r)
    # So w_χ = -1 + (1+w_chi_inf)*(1-χ)


# =========================================================================
#  Matter EOS
# =========================================================================

def matter_rho_from_p(p, K, gamma):
    """Inverse polytrope: ρ = (p/K)^(1/γ)"""
    if p <= 0:
        return 0.0
    return (p / K) ** (1.0 / gamma)


def matter_p_from_rho(rho, K, gamma):
    """Polytrope: p = K ρ^γ"""
    return K * rho**gamma


# =========================================================================
#  Two-fluid TOV integration (RK4 fixed-step)
# =========================================================================

def solve_tov_twofluid(p_c, rho0, rc, n, K, gamma, w_chi_inf=0.0,
                       r_max=80.0, n_points=4000):
    """
    Integrate two-fluid TOV from centre to surface.

    Parameters:
        p_c: central matter pressure
        rho0: central χ-sector density ρ₀
        rc: χ-sector transition radius
        n: χ-sector steepness
        K, gamma: matter polytrope parameters (p_m = K ρ_m^γ)
        w_chi_inf: asymptotic χ-sector EOS parameter at large r
        r_max: maximum radius for integration grid
        n_points: number of grid points

    Returns dict with r, m, p_m, p_χ, ρ_m, ρ_χ, f, or None if integration fails.
    """
    # Radial grid: log-spaced inner, linear outer
    n_inner = n_points // 2
    r_inner = np.logspace(-5, 0, n_inner)
    r_outer = np.linspace(1.01, r_max, n_points - n_inner)
    r = np.concatenate([r_inner, r_outer])
    dr = np.diff(r)
    dr = np.append(dr, dr[-1])

    # Precompute χ(r) and w_χ(r) on grid
    chi_arr = chi_profile(r, rc, n)
    w_chi_arr = w_chi_profile(r, rc, n, w_chi_inf)

    # Initial conditions (at first grid point r[0])
    r0 = r[0]
    p_m = np.zeros(n_points)
    p_chi = np.zeros(n_points)
    m = np.zeros(n_points)

    rho_m0 = matter_rho_from_p(p_c, K, gamma)
    m[0] = (4.0 / 3.0) * np.pi * r0**3 * (rho_m0 + rho0)
    p_m[0] = p_c
    p_chi[0] = -rho0  # w_χ = -1 at r=0

    surface_idx = n_points  # index where p_total goes to zero

    for i in range(n_points - 1):
        ri = r[i]
        mi = m[i]
        pm_i = p_m[i]
        pchi_i = p_chi[i]
        h = dr[i]

        def rhs(_r, _m, _pm, _pchi):
            """Right-hand side of two-fluid TOV."""
            _r = max(_r, 1e-15)
            f_val = max(1.0 - 2.0 * _m / _r, 1e-15)

            # χ quantities at this r
            chi_val = chi_profile(np.array([_r]), rc, n)[0]
            w_val = w_chi_profile(np.array([_r]), rc, n, w_chi_inf)[0]
            rho_chi_val = rho0 * chi_val

            # Matter density from EOS
            rho_m_val = matter_rho_from_p(_pm, K, gamma)

            p_total = _pm + _pchi
            rho_total = rho_m_val + rho_chi_val

            # Gravitational mass term
            m_term = _m + 4.0 * np.pi * _r**3 * p_total

            dm_dr = 4.0 * np.pi * _r**2 * rho_total

            # dp/dr for each fluid couples to total pressure
            common = m_term / (_r**2 * f_val)

            dpm_dr = -(rho_m_val + _pm) * common
            dpchi_dr = -(rho_chi_val + _pchi) * common

            # Clip to prevent blowup
            max_deriv = 1e8
            dpm_dr = max(-max_deriv, min(max_deriv, dpm_dr))
            dpchi_dr = max(-max_deriv, min(max_deriv, dpchi_dr))

            return dm_dr, dpm_dr, dpchi_dr

        # Stop if horizon forms (2m/r >= 1)
        if mi / np.maximum(ri, 1e-30) >= 0.5:
            surface_idx = i + 1
            break

        # RK4 step
        k1m, k1pm, k1pchi = rhs(ri, mi, pm_i, pchi_i)

        r_half = ri + 0.5 * h
        k2m, k2pm, k2pchi = rhs(r_half,
                                mi + 0.5 * h * k1m,
                                pm_i + 0.5 * h * k1pm,
                                pchi_i + 0.5 * h * k1pchi)

        k3m, k3pm, k3pchi = rhs(r_half,
                                mi + 0.5 * h * k2m,
                                pm_i + 0.5 * h * k2pm,
                                pchi_i + 0.5 * h * k2pchi)

        r_next = ri + h
        k4m, k4pm, k4pchi = rhs(r_next,
                                mi + h * k3m,
                                pm_i + h * k3pm,
                                pchi_i + h * k3pchi)

        m[i + 1] = mi + (h / 6.0) * (k1m + 2 * k2m + 2 * k3m + k4m)
        p_m[i + 1] = pm_i + (h / 6.0) * (k1pm + 2 * k2pm + 2 * k3pm + k4pm)
        p_chi[i + 1] = pchi_i + (h / 6.0) * (k1pchi + 2 * k2pchi + 2 * k3pchi + k4pchi)

        # Stop on blowup
        if not (np.isfinite(m[i + 1]) and np.isfinite(p_m[i + 1]) and
                np.isfinite(p_chi[i + 1])):
            surface_idx = i + 1
            break

        # Stop if total pressure reaches zero (stellar surface)
        # p_total always DECREASES in TOV; if it starts ≤0, no surface exists.
        p_total_nxt = p_m[i + 1] + p_chi[i + 1]

        if p_total_nxt <= 0.0:
            # Surface: p_total crossed zero from above
            p_total_i = pm_i + pchi_i
            if abs(p_total_nxt - p_total_i) < 1e-15:
                alpha = 0.5
            else:
                alpha = p_total_i / (p_total_i - p_total_nxt)
            alpha = max(0.0, min(1.0, alpha))
            r_surface = r[i] + alpha * h
            m_surface = m[i] + alpha * (m[i + 1] - m[i])
            r[i + 1] = r_surface
            m[i + 1] = m_surface
            p_m[i + 1] = 0.0
            p_chi[i + 1] = 0.0
            surface_idx = i + 2
            break

    # Trim to valid range
    if surface_idx < n_points:
        r = r[:surface_idx]
        m = m[:surface_idx]
        p_m = p_m[:surface_idx]
        p_chi = p_chi[:surface_idx]
        chi_arr = chi_arr[:surface_idx]
        w_chi_arr = w_chi_arr[:surface_idx]

    # Check if integration stopped due to horizon formation
    if m[-1] / np.maximum(r[-1], 1e-30) >= 0.49:
        return None  # Black hole

    # Compute densities
    rho_m = np.array([matter_rho_from_p(pm, K, gamma) for pm in p_m])
    rho_chi = rho0 * chi_arr[:len(r)]

    # f(r)
    f = 1.0 - 2.0 * m / np.maximum(r, 1e-30)
    f = np.clip(f, 1e-15, None)

    # Asymptotic mass
    M_final = m[-1]
    R_final = r[-1]

    if M_final < 1e-6 or R_final < 1e-6:
        return None

    if not np.isfinite(M_final) or not np.isfinite(R_final):
        return None

    # Check if surface reached grid boundary (star didn't really terminate)
    at_boundary = (surface_idx >= n_points)

    return {
        'r': r,
        'm': m,
        'p_m': p_m,
        'p_chi': p_chi,
        'rho_m': rho_m,
        'rho_chi': rho_chi,
        'f': f,
        'chi': chi_arr[:len(r)],
        'w_chi': w_chi_arr[:len(r)],
        'R': R_final,
        'M': M_final,
        'p_c': p_c,
        'rho0': rho0,
        'rc': rc,
        'n': n,
        'K': K,
        'gamma': gamma,
        'success': True,
        'at_boundary': at_boundary,
    }


# =========================================================================
#  Worker for parallel sweep
# =========================================================================

def _integrate_one(args):
    """Worker: integrate one (p_c, rho0, rc, n) combination."""
    p_c, rho0, rc, n, K, gamma, w_chi_inf, n_points = args
    try:
        sol = solve_tov_twofluid(p_c, rho0, rc, n, K, gamma,
                                 w_chi_inf=w_chi_inf, n_points=n_points)
        if sol is None:
            return {'p_c': p_c, 'rho0': rho0, 'rc': rc, 'n': n,
                    'M': np.nan, 'R': np.nan, 'M_over_R': np.nan,
                    'rho_c_matter': np.nan, 'rho_c_total': np.nan,
                    'success': False, 'fail_reason': 'BH',
                    'at_boundary': False}
        return {
            'p_c': p_c,
            'rho0': rho0,
            'rc': rc,
            'n': n,
            'M': sol['M'],
            'R': sol['R'],
            'M_over_R': sol['M'] / sol['R'],
            'rho_c_matter': matter_rho_from_p(p_c, K, gamma),
            'rho_c_total': matter_rho_from_p(p_c, K, gamma) + rho0,
            'success': True,
            'fail_reason': '',
            'at_boundary': sol.get('at_boundary', False),
        }
    except Exception as e:
        return {'p_c': p_c, 'rho0': rho0, 'rc': rc, 'n': n,
                'M': np.nan, 'R': np.nan, 'M_over_R': np.nan,
                'rho_c_matter': np.nan, 'rho_c_total': np.nan,
                'success': False, 'fail_reason': str(e)[:50],
                'at_boundary': False}


# =========================================================================
#  Main sweep
# =========================================================================

def run_sweep(p_c_vals, rho0_vals, rc_vals, n_vals, K, gamma,
              w_chi_inf=0.0, n_points=3000, n_workers=8):
    """Sweep over parameter space, collect M(R) sequences.
    
    For each (rho0, rc, n), sweeps p_c from low to high until BH formation.
    Produces M-R sequences revealing stability threshold.
    """
    tasks = []
    for rho0 in rho0_vals:
        for rc in rc_vals:
            for n in n_vals:
                for p_c in p_c_vals:
                    tasks.append((p_c, rho0, rc, n, K, gamma,
                                  w_chi_inf, n_points))

    total = len(tasks)
    n_workers = min(n_workers, total)
    print(f"  Tasks: {total}, workers: {n_workers}")

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
#  Plotting
# =========================================================================

def plot_results(results, output_path):
    """Generate M-R diagram and diagnostic plots."""
    if not results:
        print("No results to plot!")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ── Panel 1: M-R diagram ──
    ax = axes[0, 0]

    # Group by (rho0, rc | n) for distinct curves
    # Color by rho0, linestyle by rc
    rho0_vals = sorted(set(r['rho0'] for r in results))
    n_vals = sorted(set(r['n'] for r in results))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(rho0_vals)))

    for j, n_val in enumerate(n_vals):
        marker = ['o', 's', '^', 'D'][j % 4]
        for i, rho0 in enumerate(rho0_vals):
            subset = [r for r in results
                      if abs(r['rho0'] - rho0) < 1e-10
                      and abs(r['n'] - n_val) < 1e-10]
            if len(subset) < 2:
                continue

            # Separate boundary-hitting stars (no real surface)
            normal = [s for s in subset if not s.get('at_boundary', False)]
            boundary = [s for s in subset if s.get('at_boundary', False)]

            label = f'ρ₀={rho0:.1e}'
            if normal:
                normal.sort(key=lambda x: x['R'])
                R_arr = np.array([s['R'] for s in normal])
                M_arr = np.array([s['M'] for s in normal])
                ax.plot(R_arr, M_arr, '-', color=colors[i], alpha=0.7, linewidth=1.5,
                        label=label)
                ax.scatter(R_arr[::max(1, len(R_arr)//8)], M_arr[::max(1, len(R_arr)//8)],
                          marker=marker, color=colors[i], s=15, alpha=0.8)
            # Boundary stars: dashed extension
            if boundary:
                boundary.sort(key=lambda x: x['R'])
                R_b = np.array([s['R'] for s in boundary])
                M_b = np.array([s['M'] for s in boundary])
                ax.plot(R_b, M_b, '--', color=colors[i], alpha=0.3, linewidth=1)

    # Schwarzschild bound M/R = 4/9 (Buchdahl)
    R_line = np.linspace(0.1, 15, 200)
    ax.plot(R_line, (4.0 / 9.0) * R_line, 'k--', alpha=0.5, linewidth=1,
            label='Buchdahl: M/R = 4/9')
    ax.set_xlabel('R (geometric units, G = c = 1)')
    ax.set_ylabel('M (geometric units)')
    ax.set_title('Mass-Radius: χ-sector compact objects')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    # ── Panel 2: M vs ρ_c ──
    ax = axes[0, 1]
    for i, rho0 in enumerate(rho0_vals):
        subset = [r for r in results if abs(r['rho0'] - rho0) < 1e-10]
        if len(subset) < 2:
            continue
        subset.sort(key=lambda x: x['rho_c_total'])
        ax.plot([s['rho_c_total'] for s in subset],
                [s['M'] for s in subset],
                'o-', color=colors[i], markersize=4, linewidth=1,
                label=f'ρ₀={rho0:.2f}')
    ax.set_xlabel('Central density ρ_c (total)')
    ax.set_ylabel('M')
    ax.set_title('Mass vs central density')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # ── Panel 3: Compactness M/R ──
    ax = axes[1, 0]
    for i, rho0 in enumerate(rho0_vals):
        subset = [r for r in results if abs(r['rho0'] - rho0) < 1e-10]
        if len(subset) < 2:
            continue
        subset.sort(key=lambda x: x['M'])
        ax.plot([s['M'] for s in subset],
                [s['M_over_R'] for s in subset],
                'o-', color=colors[i], markersize=4, linewidth=1,
                label=f'ρ₀={rho0:.2f}')
    ax.axhline(4.0 / 9.0, color='k', linestyle='--', alpha=0.5,
               label='Buchdahl limit')
    ax.set_xlabel('M')
    ax.set_ylabel('Compactness M/R')
    ax.set_title('Compactness vs mass')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: M_max vs ρ₀ (one curve per rc, n) ──
    ax = axes[1, 1]
    rc_vals_plot = sorted(set(r['rc'] for r in results))
    for rc_v in rc_vals_plot[:6]:  # limit curves
        for n_v in n_vals[:3]:
            mm = []
            rr = []
            for rho0 in rho0_vals:
                subset = [r for r in results
                          if abs(r['rho0'] - rho0) < 1e-10
                          and abs(r['rc'] - rc_v) < 1e-10
                          and abs(r['n'] - n_v) < 1e-10]
                if subset:
                    max_m = max(s['M'] for s in subset)
                    mm.append(max_m)
                    rr.append(rho0)
            if len(mm) > 1:
                ax.plot(rr, mm, 'o-', markersize=4, linewidth=1,
                        label=f'rc={rc_v:.1f}, n={n_v:.0f}')
    ax.set_xlabel('ρ₀ (χ-sector central density)')
    ax.set_ylabel('M_max')
    ax.set_title('Maximum mass vs χ-sector density')
    ax.legend(fontsize=6, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")
    plt.close()

    # ── Bonus: Summary figure ──
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))

    # Left: M_max vs rho0
    ax = axes2[0]
    r0_list = sorted(set(r['rho0'] for r in results))
    mm_list = []
    mc_list = []
    yield_list = []
    for rho0 in r0_list:
        normal = [r for r in results
                  if abs(r['rho0'] - rho0) < 1e-10
                  and r['success'] and not r.get('at_boundary', False)]
        all_at_r0 = [r for r in results if abs(r['rho0'] - rho0) < 1e-10]
        bh_count = sum(1 for r in all_at_r0 if not r['success'])
        total = len(all_at_r0)
        if normal:
            mm_list.append(max(r['M'] for r in normal))
            mc_list.append(max(r['M_over_R'] for r in normal))
        else:
            mm_list.append(np.nan)
            mc_list.append(np.nan)
        yield_list.append(len(normal) / max(total, 1) * 100 if total > 0 else 0)
    ax.plot(r0_list, mm_list, 'o-', color='steelblue', markersize=8, linewidth=2)
    ax.axhline(mm_list[0], color='gray', linestyle='--', alpha=0.5, label=f'Pure: {mm_list[0]:.4f}')
    ax.set_xlabel('rho0 (chi-sector central density)')
    ax.set_ylabel('M_max')
    ax.set_title('Maximum mass vs chi-sector density')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Center: Compactness
    ax = axes2[1]
    ax.plot(r0_list, mc_list, 'o-', color='darkorange', markersize=8, linewidth=2)
    ax.axhline(mc_list[0], color='gray', linestyle='--', alpha=0.5, label=f'Pure: {mc_list[0]:.4f}')
    ax.axhline(4.0/9.0, color='red', linestyle=':', alpha=0.5, label='Buchdahl 4/9')
    ax.set_xlabel('rho0')
    ax.set_ylabel('Max compactness M/R')
    ax.set_title('Maximum compactness vs chi-sector density')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: BH yield
    ax = axes2[2]
    ax.bar(range(len(r0_list)), yield_list, color='crimson', alpha=0.7)
    ax.set_xticks(range(len(r0_list)))
    ax.set_xticklabels([f'{r:.0e}' for r in r0_list], rotation=30)
    ax.set_xlabel('rho0')
    ax.set_ylabel('Stable star yield (%)')
    ax.set_title('Black hole formation rate')
    ax.grid(True, alpha=0.3, axis='y')

    summary_path = output_path.replace('.png', '_summary.png')
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150)
    print(f"Summary plot saved to {summary_path}")
    plt.close()


# =========================================================================
#  Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TOV χ-sector: mass-radius for DE-extended compact objects')
    parser.add_argument('--rho0', type=str, default='0.001,0.005,0.01,0.05,0.1,0.2,0.5',
                        help='χ-sector central densities ρ₀ (comma-separated)')
    parser.add_argument('--rc', type=str, default='0.5,1.0,2.0,4.0',
                        help='χ-sector transition radii rc (comma-separated)')
    parser.add_argument('--n', type=str, default='2,3,5',
                        help='χ-sector steepness values n (comma-separated)')
    parser.add_argument('--p-c', type=str,
                        default='0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0,5.0',
                        help='Central matter pressures p_c (comma-separated)')
    parser.add_argument('--K', type=float, default=10.0,
                        help='Polytrope constant K (p = K ρ^γ)')
    parser.add_argument('--gamma', type=float, default=2.0,
                        help='Polytrope exponent γ')
    parser.add_argument('--w-chi-inf', type=float, default=0.0,
                        help='Asymptotic χ-sector EOS w_χ∞')
    parser.add_argument('--n-points', type=int, default=3000,
                        help='TOV integration grid points')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--output-csv', type=str,
                        default='tov_chi_sector_results.csv')
    parser.add_argument('--output-plot', type=str,
                        default='tov_chi_sector_plot.png')
    args = parser.parse_args()

    rho0_vals = [float(x) for x in args.rho0.split(',')]
    rc_vals = [float(x) for x in args.rc.split(',')]
    n_vals = [float(x) for x in args.n.split(',')]
    p_c_vals = [float(x) for x in args.p_c.split(',')]

    print("=" * 70)
    print("TOV chi-SECTOR: Mass-Radius for DE-extended compact objects")
    print("=" * 70)
    print(f"  chi-sector:")
    print(f"    rho0 values:  {rho0_vals}")
    print(f"    rc values:    {rc_vals}")
    print(f"    n values:     {n_vals}")
    print(f"    w_chi_inf:    {args.w_chi_inf}")
    print(f"  Matter EOS: p = K rho^gamma with K={args.K}, gamma={args.gamma}")
    print(f"  Central pressures: {len(p_c_vals)} values "
          f"[{p_c_vals[0]:.4f} ... {p_c_vals[-1]:.2f}]")
    print(f"  Total combinations: "
          f"{len(rho0_vals)*len(rc_vals)*len(n_vals)*len(p_c_vals)}")
    print()

    results = run_sweep(p_c_vals, rho0_vals, rc_vals, n_vals,
                        args.K, args.gamma,
                        w_chi_inf=args.w_chi_inf,
                        n_points=args.n_points,
                        n_workers=args.workers)

    if not results:
        print("\nERROR: No converged solutions!")
        print("Try lowering rho0 or increasing K.")
        sys.exit(1)

    # Save CSV
    fieldnames = ['p_c', 'rho0', 'rc', 'n', 'M', 'R', 'M_over_R',
                  'rho_c_matter', 'rho_c_total']
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {args.output_csv}")

    # Summary statistics
    M_vals = np.array([r['M'] for r in results])
    R_vals = np.array([r['R'] for r in results])
    compact = np.array([r['M_over_R'] for r in results])

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Converged solutions: {len(results)}")
    print(f"  M range:  [{M_vals.min():.4f}, {M_vals.max():.4f}]")
    print(f"  R range:  [{R_vals.min():.4f}, {R_vals.max():.4f}]")
    print(f"  M/R range: [{compact.min():.4f}, {compact.max():.4f}]")
    print(f"  Buchdahl violations (M/R > 4/9): "
          f"{np.sum(compact > 4.0/9.0)}/{len(results)}")
    print()

    # Per-ρ₀ summary
    print(f"  Per rho0 breakdown (max M, min R):")
    print(f"  {'rho0':>8s}  {'M_max':>10s}  {'at R':>10s}  "
          f"{'max M/R':>10s}  {'N':>5s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*5}")
    for rho0 in rho0_vals:
        subset = [r for r in results if abs(r['rho0'] - rho0) < 1e-10]
        if subset:
            M_sub = [s['M'] for s in subset]
            idx_max = np.argmax(M_sub)
            print(f"  {rho0:8.4f}  {M_sub[idx_max]:10.4f}  "
                  f"{subset[idx_max]['R']:10.4f}  "
                  f"{max(s['M_over_R'] for s in subset):10.4f}  "
                  f"{len(subset):5d}")

    # Physics check: does χ-sector regularise the core?
    print(f"\n  Physics notes:")
    print(f"  The chi-sector contributes p_chi = -rho0 at the core (de Sitter).")
    print(f"  This means dp/dr is less negative near r=0, allowing larger")
    print(f"  central densities before collapse. Objects with rho0 > 0")
    print(f"  can be more massive than pure-matter configurations.")

    # Black hole threshold
    bh_mask = compact > 0.4
    if np.any(bh_mask):
        print(f"\n  WARNING: {np.sum(bh_mask)} configurations exceed M/R > 0.4")
        print(f"  These are near or beyond the black hole formation threshold.")

    # Plot
    plot_results(results, args.output_plot)

    print(f"\nDone.")


if __name__ == '__main__':
    main()
