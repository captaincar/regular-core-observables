#!/usr/bin/env python3
"""
Tier 1, Calculations 2 & 3: Hayward regular black hole + Regge-Wheeler QNM shift.

Uses the Hayward metric as a toy regular black hole model:

    f(r) = 1 - 2Mr^2 / (r^3 + 2ML^2)

where L is the core length scale. At r >> L: Schwarzschild (f ~ 1 - 2M/r).
At r << L: de Sitter core (f ~ 1 - r^2/L^2, effective Lambda = 3/L^2).

Derives the effective stress-energy from Einstein equations, computes the
Regge-Wheeler potential, and estimates the fractional QNM shift as function
of the core scale L (in units of M). Also estimates gravitational-wave
echo delay from the core-scattered signal.

Uses multiprocessing (ProcessPoolExecutor) for parallel parameter sweeps.

Usage:
    python tov_rw_sweep.py                          # default parameter sweep
    python tov_rw_sweep.py --workers 8              # use 8 CPU cores
    python tov_rw_sweep.py --L 0.01,0.05,0.1,0.5,1.0  # specific L/M values
    python tov_rw_sweep.py --plot-only              # plot from saved CSV
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import csv
import os
import sys
import time

# ── Hayward metric functions ────────────────────────────────────────────
# f(r) = 1 - 2Mr^2/(r^3 + 2ML^2)
# m(r) = Mr^3/(r^3 + 2ML^2)   -- effective mass function
# rho = m'/(4pi r^2)          -- effective density (Einstein eqns)
# p_r = -rho                  -- Hayward metric satisfies p = -rho exactly!
# p_t from conservation law


def hayward_m(r, M, L):
    """Effective mass function."""
    return M * r**3 / (r**3 + 2.0 * M * L**2)


def hayward_rho(r, M, L):
    """Effective density rho(r) = m'(r)/(4 pi r^2)."""
    r3 = r**3
    denom = r3 + 2.0 * M * L**2
    # m' = 6 M^2 L^2 r^2 / denom^2
    dm_dr = 6.0 * M**2 * L**2 * r**2 / denom**2
    with np.errstate(divide='ignore', invalid='ignore'):
        result = dm_dr / (4.0 * np.pi * r**2)
    result[~np.isfinite(result)] = 0.0
    return result


def hayward_pr(r, M, L):
    """Radial pressure: p_r = -rho (exact for Hayward metric)."""
    return -hayward_rho(r, M, L)


def hayward_pt(r, M, L):
    """Tangential pressure from conservation law: p_t = p_r + (r/2) p_r'."""
    # Analytical derivative of rho
    r3 = r**3
    denom = r3 + 2.0 * M * L**2
    # rho = (3 M^2 L^2)/(2 pi denom^2)
    # rho' = -(9 M^2 L^2 r^2)/(pi denom^3)
    with np.errstate(divide='ignore', invalid='ignore'):
        drho_dr = -(9.0 / np.pi) * M**2 * L**2 * r**2 / denom**3
    drho_dr[~np.isfinite(drho_dr)] = 0.0
    rho = hayward_rho(r, M, L)
    return -rho - 0.5 * r * drho_dr


def hayward_f(r, M, L):
    """Metric function f(r) = 1 - 2m(r)/r."""
    return 1.0 - 2.0 * hayward_m(r, M, L) / r


# ── Build solution on radial grid ───────────────────────────────────────

def build_hayward_solution(M, L, r_max=50.0, n_points=5000):
    """
    Compute the Hayward solution on a dense radial grid.

    Returns dict with: r, m, p (radial), p_t, rho, Phi, f, success,
                       terminated_at, r_match, M_match, L.
    """
    # Dense near core, sparser at large r
    r_inner = np.logspace(-5, max(-2, np.log10(2 * L + 1e-10)), n_points // 2)
    r_outer = np.linspace(r_inner[-1] * 1.01, r_max, n_points // 2)
    r = np.concatenate([r_inner, r_outer])

    m = hayward_m(r, M, L)
    rho = hayward_rho(r, M, L)
    p_r = hayward_pr(r, M, L)
    p_t = hayward_pt(r, M, L)
    f = hayward_f(r, M, L)

    # Phi = 0 for Hayward metric (g_tt = -f, no extra redshift)
    Phi = np.zeros_like(r)

    # r_match: where Hayward is within 0.5% of Schwarzschild exterior
    f_schw = 1.0 - 2.0 * M / np.maximum(r, 2.0 * M + 1e-6)
    diff = np.abs(f - f_schw) / np.maximum(np.abs(f_schw), 1e-10)
    mask = r > 2.0 * M + 0.01
    match_idx = np.where(mask & (diff < 0.005))[0]
    r_match = float(r[match_idx[0]]) if len(match_idx) > 0 else float(r[-1])

    return {
        'r': r,
        'm': m,
        'p': p_r,
        'p_t': p_t,
        'rho': rho,
        'Phi': Phi,
        'f': f,
        'success': True,
        'terminated_at': 'p_zero',
        'r_match': r_match,
        'M_match': M,
        'L': L,
    }


# ── Tortoise coordinate ─────────────────────────────────────────────────

def compute_tortoise(r, m, Phi, f):
    """Tortoise coordinate: dr_*/dr = e^{-Phi(r)} / f(r)."""
    smooth_f = np.clip(f, 1e-15, None)
    integrand = np.exp(-Phi) / smooth_f
    r_star = cumulative_trapezoid(integrand, r, initial=0.0)
    return r, r_star


# ── Regge-Wheeler potential ─────────────────────────────────────────────

def compute_regge_wheeler_potential(sol, ell=3):
    """
    Regge-Wheeler effective potential for axial perturbations:

    V_ell = e^{2Phi} f [ell(ell+1)/r^2 - 6m/r^3 + 4pi(rho - p_r)]

    Returns r, V, r_star arrays.
    """
    r = sol['r']
    m = sol['m']
    p = sol['p']
    rho = sol['rho']
    Phi = sol['Phi']
    f = sol['f']

    with np.errstate(divide='ignore', invalid='ignore'):
        V = (np.exp(2.0 * Phi) * f *
             (ell * (ell + 1) / (r**2) - 6.0 * m / (r**3)
              + 4.0 * np.pi * (rho - p)))
    V[~np.isfinite(V)] = 0.0
    V[np.abs(V) > 1e6] = 0.0  # clip enormous values near r=0

    r_full, r_star = compute_tortoise(r, m, Phi, f)
    V_interp = interp1d(r, V, kind='linear', bounds_error=False, fill_value=0.0)
    V_on_tortoise = V_interp(r_full)

    return r_full, V_on_tortoise, r_star


def schwarzschild_rw_potential(r_vals, M, ell=3):
    """Schwarzschild Regge-Wheeler potential (reference)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        f_schw = 1.0 - 2.0 * M / np.maximum(r_vals, 2.0 * M + 1e-10)
        V_schw = f_schw * (ell * (ell + 1) / r_vals**2 - 6.0 * M / r_vals**3)
    V_schw[~np.isfinite(V_schw)] = 0.0
    V_schw[r_vals < 2.0 * M + 1e-6] = 0.0
    return V_schw


# ── QNM shift via overlap integral ──────────────────────────────────────

def compute_qnm_shift(sol, ell=3):
    """
    Fractional QNM shift via two complementary estimates.

    Method 1 (overlap): delta_omega/omega_0 ~ M * integrate(W * delta_V, dr_*)
    where W is a Gaussian peaked at the photon sphere (simulating the |psi|^2
    weighting of the QNM eigenfunction).

    Method 2 (local): delta_omega/omega_0 ~ delta_V(r_photon) / V_0(r_photon)
    where V_0 is the Schwarzschild potential at the photon sphere. This is a
    simple local estimate appropriate when the perturbation varies slowly
    near the photon sphere.

    Both methods should agree for L/M not too small. For L << M, the
    perturbation is buried deep in the core and tunnels through the angular
    momentum barrier, so Method 1 (with the correct weighting) gives a tiny
    (but potentially unreliable) shift.
    """
    r_full, V_mod, r_star = compute_regge_wheeler_potential(sol, ell)
    M = sol['M_match']

    V_schw = schwarzschild_rw_potential(r_full, M, ell)
    delta_V = V_mod - V_schw

    # --- Method 1: overlap integral with WKB normalization ---
    # Photon sphere: r = 3M, r_* = 3M + 2M ln(1/2)
    r_photon = 3.0 * M
    r_star_photon = r_photon + 2.0 * M * np.log(r_photon / (2.0 * M) - 1.0)
    sigma = M * 2.0

    W = np.exp(-0.5 * ((r_star - r_star_photon) / sigma)**2)
    norm = np.trapz(W, r_star)
    if norm > 0:
        W /= norm

    overlap = np.trapz(W * delta_V, r_star)
    # Dimensional analysis: [delta_V] = 1/M^2, [dr_*] = M, so [overlap] = 1/M.
    # QNM frequency omega_0 ~ 1/M, so delta_omega/omega_0 ~ M * overlap
    # (dimensionless). We also divide by omega_0 for the usual normalisation
    # factor in perturbation theory: delta_omega = <psi|delta_V|psi> / (2 omega_0)
    omega_0 = 0.9673 / M
    fractional_overlap = M * overlap  # dimensionless, correct scaling

    # --- Method 2: local perturbation at photon sphere ---
    # Interpolate delta_V at the photon sphere
    idx_photon = np.argmin(np.abs(r_full - r_photon))
    V_schw_photon = V_schw[idx_photon]
    V_mod_photon = V_mod[idx_photon]
    if V_schw_photon > 1e-15:
        fractional_local = (V_mod_photon - V_schw_photon) / V_schw_photon
    else:
        fractional_local = np.nan

    return {
        'overlap_integral': float(overlap),
        'fractional_shift': float(fractional_overlap),  # Method 1
        'fractional_local': float(fractional_local),    # Method 2
        'r_photon': r_photon,
        'omega_0': omega_0,
    }


# ── Echo delay estimate ─────────────────────────────────────────────────

def compute_echo_delay(sol):
    """
    GW echo delay from core scattering.

    The echo delay is the proper time for a signal to travel from the photon
    sphere to the regular core and back, computed in tortoise coordinates:

        Delta t_echo = 2 * [r_*(r_photon) - r_*(r_core)]

    Both r_* values are computed from the Hayward metric with the same
    normalisation (r_* = 0 at a common reference point). We shift so that
    r_*(r = 50M) matches the Schwarzschild value, then compute everything
    relative to that.

    Returns delay in units of M and in milliseconds (for M = 10 M_sun).
    """
    r = sol['r']
    f = sol['f']
    Phi = sol['Phi']
    M = sol['M_match']
    L = sol.get('L', 0.1)

    # --- Tortoise from Hayward metric, normalised consistently ---
    smooth_f = np.clip(f, 1e-15, None)
    integrand = np.exp(-Phi) / smooth_f
    r_star_raw = cumulative_trapezoid(integrand, r, initial=0.0)

    # Normalise: match Schwarzschild tortoise far away (at r_max)
    # Schwarzschild: r_* = r + 2M ln(r/(2M) - 1)
    r_max = r[-1]
    r_star_schw_ref = r_max + 2.0 * M * np.log(r_max / (2.0 * M) - 1.0)
    offset = r_star_schw_ref - r_star_raw[-1]
    r_star = r_star_raw + offset

    # --- Photon sphere tortoise ---
    r_photon = 3.0 * M
    r_star_interp = interp1d(r, r_star, kind='linear',
                             bounds_error=False, fill_value='extrapolate')
    r_star_photon = float(r_star_interp(r_photon))

    # --- Core surface: where density = half central ---
    rho_vals = sol['rho']
    rho_half = rho_vals[0] / 2.0
    core_indices = np.where(rho_vals >= rho_half)[0]
    r_core = float(r[core_indices[-1]]) if len(core_indices) > 0 else L
    r_star_core = float(r_star_interp(r_core))

    echo_delay_M = 2.0 * abs(r_star_photon - r_star_core)

    # Convert to ms for M = 10 M_sun
    # 1 M_sun in seconds: GM_sun/c^3 = 4.925e-6 s
    echo_delay_ms = echo_delay_M * 10.0 * 4.925e-6 * 1000.0

    return {
        'echo_delay_M': echo_delay_M,
        'echo_delay_ms': echo_delay_ms,
        'r_core': r_core,
    }


# ── Single parameter evaluation (worker task) ───────────────────────────

def evaluate_params(args_tuple):
    """
    Evaluate a single (L/M) point.

    This function is pickled and sent to worker processes via
    ProcessPoolExecutor. All heavy imports (numpy, scipy) must be
    available in the worker.
    """
    idx, L_val, M, ell = args_tuple

    try:
        sol = build_hayward_solution(M, L_val)
    except Exception:
        return {'idx': idx, 'L': L_val, 'M': M, 'success': False,
                'error': 'solution_failed'}

    try:
        qnm = compute_qnm_shift(sol, ell)
    except Exception:
        qnm = {'overlap_integral': np.nan, 'fractional_shift': np.nan,
               'fractional_local': np.nan}

    try:
        echo = compute_echo_delay(sol)
    except Exception:
        echo = {'echo_delay_M': np.nan, 'echo_delay_ms': np.nan, 'r_core': np.nan}

    return {
        'idx': idx,
        'L': L_val,
        'L_over_M': L_val / M,
        'M': M,
        'success': True,
        'r_match': sol['r_match'],
        'rho_central': float(sol['rho'][0]),
        'p_central': float(sol['p'][0]),
        'fractional_shift': qnm['fractional_shift'],
        'fractional_local': qnm['fractional_local'],
        'echo_delay_M': echo['echo_delay_M'],
        'echo_delay_ms': echo['echo_delay_ms'],
        'r_core': echo['r_core'],
    }


# ── Parameter grid builder ──────────────────────────────────────────────

def build_parameter_grid(args):
    """Build list of (L, M, ell) tasks from CLI arguments."""
    def parse_list(s):
        return [float(x) for x in s.split(',')]

    if args.L:
        L_vals = parse_list(args.L)
    else:
        L_vals = np.logspace(args.L_min, args.L_max, args.n_L)

    M_vals = parse_list(args.M) if args.M else [1.0]

    tasks = []
    idx = 0
    for M in M_vals:
        for L_val in L_vals:
            tasks.append((idx, L_val, M, args.ell))
            idx += 1
    return tasks


# ── Main sweep runner ───────────────────────────────────────────────────

def run_sweep(tasks, workers, output_csv):
    """Run parameter sweep with ProcessPoolExecutor."""
    n_total = len(tasks)
    print(f"Running {n_total} Hayward evaluations across {workers} workers...")
    print(f"  L/M range: [{min(t[1] for t in tasks):.3e}, {max(t[1] for t in tasks):.3e}]")
    print(f"  M values: {sorted(set(t[2] for t in tasks))}")
    print()

    results = []
    n_done = 0
    n_success = 0
    t_start = time.perf_counter()

    fieldnames = [
        'idx', 'L', 'L_over_M', 'M', 'success', 'error',
        'r_match', 'rho_central', 'p_central',
        'fractional_shift', 'fractional_local', 'echo_delay_M', 'echo_delay_ms', 'r_core',
    ]

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(evaluate_params, t): t for t in tasks}

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                writer.writerow(result)
                n_done += 1
                if result['success']:
                    n_success += 1

                if n_done % max(1, n_total // 20) == 0 or n_done == n_total:
                    elapsed = time.perf_counter() - t_start
                    rate = n_done / elapsed if elapsed > 0 else 0
                    pct = 100 * n_done / n_total
                    print(f"  [{n_done:5d}/{n_total}] {pct:5.1f}%  "
                          f"success={n_success}  rate={rate:.1f}/s  elapsed={elapsed:.0f}s")

    t_total = time.perf_counter() - t_start
    print(f"\nDone. {n_success}/{n_total} successful ({100*n_success/n_total:.1f}%)")
    print(f"Total time: {t_total:.1f}s  Rate: {n_total/t_total:.1f} evaluations/s")
    print(f"Results saved to: {output_csv}")

    return results


# ── Plotting ────────────────────────────────────────────────────────────

def plot_results(csv_path):
    """Generate diagnostic plots from results CSV."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plots. Install with: pip install matplotlib")
        return

    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    data = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    if not data:
        print("No data in CSV.")
        return

    successful = [d for d in data if d['success'] == 'True']
    if not successful:
        print("No successful evaluations in CSV.")
        return

    def sf(val, default=np.nan):
        """Safely convert to float."""
        try:
            return float(val) if val and str(val).strip() else default
        except (ValueError, TypeError):
            return default

    L_over_M = np.array([sf(d['L_over_M']) for d in successful])
    shift = np.array([sf(d['fractional_shift']) for d in successful])
    shift_local = np.array([sf(d.get('fractional_local', sf(d['fractional_shift'])))
                            for d in successful])
    echo_ms = np.array([sf(d['echo_delay_ms']) for d in successful])
    echo_M = np.array([sf(d['echo_delay_M']) for d in successful])
    rho_c = np.array([sf(d.get('rho_central', 0)) for d in successful])
    p_c = np.array([sf(d.get('p_central', 0)) for d in successful])
    r_core = np.array([sf(d.get('r_core', 0)) for d in successful])
    r_match = np.array([sf(d.get('r_match', 0)) for d in successful])
    M_vals = np.array([sf(d['M']) for d in successful])

    # Sort by L/M
    order = np.argsort(L_over_M)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Hayward Regular Black Hole: Regge-Wheeler Analysis', fontsize=13)

    # 1. Fractional QNM shift vs L/M
    ax = axes[0, 0]
    valid = np.isfinite(shift_local)
    ax.plot(L_over_M[order & valid], np.abs(shift_local[order & valid]),
            'o-', markersize=3, color='steelblue')
    ax.axhline(1e-4, color='red', linestyle='--', alpha=0.5,
               label='Next-gen GW threshold (|dw/w0| = 1e-4)')
    # Power-law guide: |dw/w0| ~ 0.18 (L/M)^2
    L_fit = np.logspace(-2, 1, 20)
    ax.plot(L_fit, 0.18 * L_fit**2, 'k--', alpha=0.3,
            label='~0.18 (L/M)^2 guide')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Core scale L / M')
    ax.set_ylabel('|dw/w0|')
    ax.set_title('Fractional QNM shift vs. core scale')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 2. Echo delay vs L/M
    ax = axes[0, 1]
    valid = np.isfinite(echo_ms) & (echo_ms > 0)
    ax.plot(L_over_M[order & valid], echo_ms[order & valid],
            's-', markersize=3, color='darkorange')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Core scale L / M')
    ax.set_ylabel('Echo delay (ms, for M=10 M_sun)')
    ax.set_title('GW echo delay vs. core scale')
    ax.axhline(100.0, color='green', linestyle='--', alpha=0.5,
               label='Observability upper bound (~100 ms)')
    ax.axhspan(100.0, ax.get_ylim()[1] if ax.get_ylim()[1] > 100 else 1e6,
               alpha=0.05, color='gray', label='Unobservable (too long/shallow)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 3. Central density and pressure vs L/M
    ax = axes[1, 0]
    ax.plot(L_over_M[order], rho_c[order], 'o-', markersize=3,
            color='steelblue', label='rho_central [M^-2]')
    ax.plot(L_over_M[order], np.abs(p_c[order]), 's-', markersize=3,
            color='darkorange', label='|p_central| [M^-2]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Core scale L / M')
    ax.set_ylabel('M^-2')
    ax.set_title('Central density and |pressure|')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. r_core and r_match vs L/M (structure chart)
    ax = axes[1, 1]
    ax.plot(L_over_M[order], r_core[order], 'o-', markersize=3,
            color='steelblue', label='r_core [M]')
    ax.plot(L_over_M[order], r_match[order], 's-', markersize=3,
            color='darkorange', label='r_match [M]')
    ax.axhline(2.0, color='gray', linestyle='--', alpha=0.3, label='r = 2M (Schw. horizon)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Core scale L / M')
    ax.set_ylabel('Radius [M]')
    ax.set_title('Structure: core radius and matching radius')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = csv_path.replace('.csv', '.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.close()


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Hayward regular BH + Regge-Wheeler QNM analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--workers', '-w', type=int, default=os.cpu_count(),
                        help=f'Worker processes (default: CPU count = {os.cpu_count()})')
    parser.add_argument('--output', '-o', default='hayward_results.csv',
                        help='Output CSV file (default: hayward_results.csv)')
    parser.add_argument('--L', help='Comma-separated L/M values (e.g. 0.01,0.1,1.0)')
    parser.add_argument('--M', default='1.0',
                        help='Comma-separated M values in solar masses (default: 1.0)')
    parser.add_argument('--L-min', type=float, default=-2.0,
                        help='Log10 min L/M (default: -2.0 -> L/M = 0.01)')
    parser.add_argument('--L-max', type=float, default=1.0,
                        help='Log10 max L/M (default: 1.0 -> L/M = 10.0)')
    parser.add_argument('--n-L', type=int, default=30,
                        help='Number of L/M points in logspace (default: 30)')
    parser.add_argument('--ell', type=int, default=3,
                        help='Angular harmonic l (default: 3)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip computation, only plot from existing CSV')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting')

    args = parser.parse_args()

    if args.plot_only:
        plot_results(args.output)
        return

    tasks = build_parameter_grid(args)
    if not tasks:
        print("No tasks generated. Check parameter ranges.")
        sys.exit(1)

    results = run_sweep(tasks, args.workers, args.output)

    # Print summary
    successful = [r for r in results if r.get('success')]
    if successful:
        shifts = [abs(r.get('fractional_local', r.get('fractional_shift', np.nan)))
                  for r in successful
                  if np.isfinite(r.get('fractional_local', r.get('fractional_shift', np.nan)))]
        if shifts:
            print(f"\nSummary (local perturbation at photon sphere):")
            print(f"  Max |dw/w0| = {max(shifts):.3e}")
            print(f"  Min |dw/w0| = {min(shifts):.3e}")
            if max(shifts) > 1e-4:
                detectable = sum(1 for s in shifts if abs(s) > 1e-4)
                print(f"  Detectable by next-gen GW: {detectable}/{len(shifts)} points")

        echo_delays = [r['echo_delay_ms'] for r in successful
                       if np.isfinite(r['echo_delay_ms']) and r['echo_delay_ms'] > 0]
        if echo_delays:
            print(f"  Max echo delay = {max(echo_delays):.1f} ms (M=10 M_sun)")
            print(f"  Min echo delay = {min(echo_delays):.1f} ms (M=10 M_sun)")

    if not args.no_plot:
        plot_results(args.output)


if __name__ == '__main__':
    main()
