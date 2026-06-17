#!/usr/bin/env python3
"""
TOV EOS sweep: Self-consistent TOV integration with variable w(r) equation of state.

Sweeps over a family of EOS transition profiles w(r) that go from de Sitter
core (w = -1) to vacuum/dust at large r, integrates the TOV equations
self-consistently, computes the Regge-Wheeler QNM potential, and compares
fractional QNM shifts to the Hayward baseline.

This addresses the reviewer's request: "Solve the TOV system numerically with
chi(r) determined self-consistently."

Key question: Are the QNM shifts from the article (computed from the Hayward
ansatz) robust against variations in the EOS transition profile?

Usage:
    python tov_eos_sweep.py
    python tov_eos_sweep.py --samples 200
    python tov_eos_sweep.py --plot-only
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


# =========================================================================
#  TOV integrator (RK4 fixed-step)
# =========================================================================


def solve_tov(w_func, rho_central, M_target, r_max=80.0, n_points=3000):
    """
    Integrate TOV equations with a given w(r) function using RK4 fixed-step.

    Returns dict with r, m, p, rho, Phi, f, w arrays, or None if integration fails.
    """
    # Radial grid: log-spaced for core resolution, linear for exterior
    r_inner = np.logspace(-4, 0, n_points // 2)
    r_outer = np.linspace(1.01, r_max, n_points // 2)
    r = np.concatenate([r_inner, r_outer])
    dr = np.diff(r)
    dr = np.append(dr, dr[-1])

    # Prepare w on the grid
    w_arr = np.array([w_func(ri) for ri in r])

    # Initial conditions at first grid point
    r0 = r[0]
    w0 = w_arr[0]
    p = np.zeros(n_points)
    m = np.zeros(n_points)
    Phi = np.zeros(n_points)

    p[0] = w0 * rho_central
    m[0] = (4.0 / 3.0) * np.pi * r0**3 * rho_central
    Phi[0] = 0.0

    # RK4 integration
    for i in range(n_points - 1):
        ri = r[i]
        mi = m[i]
        pi = p[i]
        Phii = Phi[i]
        h = dr[i]
        wi = w_arr[i]

        def rhs(_r, _m, _p, _Phi, _w):
            _r = max(_r, 1e-15)
            fv = max(1.0 - 2.0 * _m / _r, 1e-15)
            if abs(_w) > 1e-10:
                rv = _p / _w
                if rv < 0:
                    rv = 0.0  # rho must be non-negative
            else:
                rv = 0.0
            rho_plus_p = rv + _p
            m_term = _m + 4.0 * np.pi * _r**3 * _p
            dm = 4.0 * np.pi * _r**2 * rv
            # Guard against overflow: clip large derivatives
            dp_raw = -rho_plus_p * m_term / (_r**2 * fv)
            dPhi_raw = m_term / (_r**2 * fv)
            # Cap derivative magnitudes to prevent blowup
            max_deriv = 1e6
            dp = max(-max_deriv, min(max_deriv, dp_raw))
            dPhi = max(-max_deriv, min(max_deriv, dPhi_raw))
            return dm, dp, dPhi

        # k1
        k1m, k1p, k1Phi = rhs(ri, mi, pi, Phii, wi)

        # k2
        r_half = ri + 0.5 * h
        w_half = w_func(r_half)
        k2m, k2p, k2Phi = rhs(r_half, mi + 0.5*h*k1m, pi + 0.5*h*k1p,
                              Phii + 0.5*h*k1Phi, w_half)

        # k3
        k3m, k3p, k3Phi = rhs(r_half, mi + 0.5*h*k2m, pi + 0.5*h*k2p,
                              Phii + 0.5*h*k2Phi, w_half)

        # k4
        r_next = ri + h
        w_next = w_arr[i + 1] if i + 1 < n_points else w_func(r_next)
        k4m, k4p, k4Phi = rhs(r_next, mi + h*k3m, pi + h*k3p,
                              Phii + h*k3Phi, w_next)

        m[i+1] = mi + (h/6.0) * (k1m + 2*k2m + 2*k3m + k4m)
        p[i+1] = pi + (h/6.0) * (k1p + 2*k2p + 2*k3p + k4p)
        Phi[i+1] = Phii + (h/6.0) * (k1Phi + 2*k2Phi + 2*k3Phi + k4Phi)

        # Stop if something blew up
        if not (np.isfinite(m[i+1]) and np.isfinite(p[i+1])):
            break

        # Stop if pressure goes positive (surface)
        if p[i+1] >= 0:
            p[i+1:] = 0.0
            break

    # Compute rho from EOS
    rho = np.where(np.abs(w_arr) > 1e-10, p / w_arr, 0.0)
    rho = np.clip(rho, 0.0, None)

    # f(r)
    f = 1.0 - 2.0 * m / np.maximum(r, 1e-30)
    f = np.clip(f, 1e-15, None)

    # Scale to target mass
    M_asymptotic = m[-1]
    if M_asymptotic < 0.01 * M_target or not np.isfinite(M_asymptotic):
        return None

    scale = M_target / M_asymptotic
    m *= scale
    rho *= scale
    p *= scale

    return {
        'r': r,
        'm': m,
        'p': p,
        'rho': rho,
        'Phi': Phi,
        'f': f,
        'w': w_arr,
        'success': True,
    }


# =========================================================================
#  Regge-Wheeler potential and QNM shift
# =========================================================================

def compute_rw_potential(sol, ell=3):
    """Regge-Wheeler effective potential from TOV solution dict."""
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
    V[V > 1e6] = 0.0

    return r, V


def compute_qnm_shift(sol, M, ell=3):
    """Fractional QNM shift via local perturbation at the photon sphere."""
    r, V = compute_rw_potential(sol, ell)

    # Schwarzschild reference potential
    r_schw = np.logspace(np.log10(2.1 * M), 2, 500)
    f_schw = 1.0 - 2.0 * M / r_schw
    V_schw = f_schw * (ell * (ell + 1) / r_schw**2 - 6.0 * M / r_schw**3)
    V_schw[~np.isfinite(V_schw)] = 0.0

    # Peak values at photon sphere (r = 3M)
    r_photon = 3.0 * M
    idx_photon = np.argmin(np.abs(r - r_photon))
    idx_schw_photon = np.argmin(np.abs(r_schw - r_photon))

    V_mod_peak = V[idx_photon]
    V_schw_peak = V_schw[idx_schw_photon]

    if V_schw_peak < 1e-15:
        return np.nan

    return float((V_mod_peak - V_schw_peak) / V_schw_peak)


# =========================================================================
#  Hayward baseline (for comparison)
# =========================================================================

def hayward_qnm_shift(L_over_M, ell=3):
    """Hayward QNM shift: |dw/w0| approx 0.18 (L/M)^2."""
    return 0.18 * (L_over_M)**2


# =========================================================================
#  EOS transition profiles
# =========================================================================

def make_sigmoid_step(x, n):
    """Smooth step: x^n / (1 + x^n), goes from 0 to 1."""
    xn = x ** n
    return xn / (1.0 + xn)


def make_w_profile(rc, n, w_inf):
    """
    Build w(r) function: de Sitter core (w=-1) transitioning to w_inf at large r.

    Parameters:
        rc: transition radius
        n: steepness
        w_inf: asymptotic w at large r (0 = dust/vacuum, small positive = radiation-like)
    """
    def w_func(r):
        r = max(r, 1e-15)
        step = make_sigmoid_step(r / rc, n)
        return -1.0 + (1.0 + w_inf) * step
    return w_func


# =========================================================================
#  Single integration worker (for multiprocessing)
# =========================================================================

def _integrate_one(args):
    """Worker function: integrate TOV for one parameter combination."""
    L_over_M, rc, n, w_inf, M, n_points = args

    w_func = make_w_profile(rc, n, w_inf)
    rho_central = 3.0 / (8.0 * np.pi * L_over_M**2)

    sol = solve_tov(w_func, rho_central, M, r_max=80.0, n_points=n_points)

    if sol is None:
        return None
    if sol['m'][-1] < 0.5 * M or sol['m'][-1] > 1.5 * M:
        return None

    shift = compute_qnm_shift(sol, M)
    if not np.isfinite(shift):
        return None

    shift_hayward = hayward_qnm_shift(L_over_M)

    w_arr = sol['w']
    r_arr = sol['r']
    core_mask = w_arr < -0.5
    r_core = r_arr[core_mask][-1] if np.any(core_mask) else rc

    return {
        'L_over_M': L_over_M,
        'rc': rc,
        'n': n,
        'w_inf': w_inf,
        'shift': shift,
        'shift_hayward': shift_hayward,
        'r_core': r_core,
        'M_final': sol['m'][-1],
        'rho_central': rho_central,
        'delta_shift': shift - shift_hayward,
    }


# =========================================================================
#  Sweep
# =========================================================================

def run_sweep(L_vals, rc_vals, n_vals, w_inf_vals, M=1.0, n_points=2000,
              n_workers=8):
    """
    Sweep over EOS parameters, solve TOV for each, compute QNM shift.
    Uses multiprocessing for parallel integration.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Build parameter list
    tasks = []
    for L_over_M in L_vals:
        for rc in rc_vals:
            for n in n_vals:
                for w_inf in w_inf_vals:
                    tasks.append((L_over_M, rc, n, w_inf, M, n_points))

    total = len(tasks)
    print(f"  Tasks: {total}, workers: {n_workers}")

    results = []
    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_integrate_one, t): t for t in tasks}
        for fut in as_completed(futures):
            completed += 1
            if completed % 25 == 0:
                print(f"  Progress: {completed}/{total}")
            try:
                r = fut.result()
                if r is not None:
                    results.append(r)
            except Exception as e:
                print(f"  Worker error: {e}")

    print(f"  Completed: {len(results)}/{total} converged")
    return results


# =========================================================================
#  Plotting
# =========================================================================

def plot_results(results, output_path):
    """Generate diagnostic plots."""
    if not results:
        print("No results to plot!")
        return

    L_vals = sorted(set(r['L_over_M'] for r in results))
    w_inf_vals = sorted(set(r['w_inf'] for r in results))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # ── Panel 1: QNM shift vs L/M (all points, colored by w_inf) ──
    ax = axes[0, 0]
    L_arr = np.array([r['L_over_M'] for r in results])
    shift_arr = np.array([r['shift'] for r in results])
    w_arr = np.array([r['w_inf'] for r in results])
    sc = ax.scatter(L_arr, shift_arr, c=w_arr, cmap='coolwarm', alpha=0.5, s=8)
    L_fine = np.linspace(0.05, 0.9, 100)
    ax.plot(L_fine, hayward_qnm_shift(L_fine), 'k-', linewidth=2, label='Hayward: $0.18(L/M)^2$')
    ax.set_xlabel('$L/M$')
    ax.set_ylabel(r'QNM shift $|\delta\omega/\omega_0|$')
    ax.set_title('QNM shift: TOV self-consistent vs Hayward')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label='$w_\infty$')

    # ── Panel 2: Deviation from Hayward vs L/M ──
    ax = axes[0, 1]
    delta_arr = np.array([r['delta_shift'] for r in results])
    ax.scatter(L_arr, delta_arr, c=w_arr, cmap='coolwarm', alpha=0.5, s=8)
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('$L/M$')
    ax.set_ylabel('$\delta\omega - \delta\omega_{\mathrm{Hayward}}$')
    ax.set_title('Deviation from Hayward prediction')
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Deviation histogram ──
    ax = axes[0, 2]
    ax.hist(delta_arr, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('$\delta\omega - \delta\omega_{\mathrm{Hayward}}$')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of deviations')

    # ── Panel 4: r_core vs L/M ──
    ax = axes[1, 0]
    r_core_arr = np.array([r['r_core'] for r in results])
    ax.scatter(L_arr, r_core_arr, c=w_arr, cmap='coolwarm', alpha=0.5, s=8)
    ax.plot(L_fine, L_fine, 'k--', alpha=0.5, label='$r_{\mathrm{core}} = L$')
    ax.set_xlabel('$L/M$')
    ax.set_ylabel('Effective core radius $r_{\mathrm{core}}/M$')
    ax.set_title('Core radius vs input scale')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 5: Shift sensitivity to steepness n ──
    ax = axes[1, 1]
    n_vals = sorted(set(r['n'] for r in results))
    for w_inf in w_inf_vals:
        subset = [r for r in results if abs(r['w_inf'] - w_inf) < 0.01 and abs(r['L_over_M'] - 0.3) < 0.02]
        if len(subset) > 3:
            n_arr = [r['n'] for r in subset]
            s_arr = [r['shift'] for r in subset]
            order = np.argsort(n_arr)
            ax.plot(np.array(n_arr)[order], np.array(s_arr)[order], 'o-',
                    markersize=4, label=f'$w_\infty={w_inf:.1f}$')
    hw = hayward_qnm_shift(0.3)
    ax.axhline(hw, color='k', linestyle=':', alpha=0.5, label=f'Hayward ({hw:.4f})')
    ax.set_xlabel('Steepness $n$')
    ax.set_ylabel('$|\delta\omega/\omega_0|$')
    ax.set_title('Shift vs transition steepness ($L/M=0.3$)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Panel 6: Summary — mean shift and spread vs L/M ──
    ax = axes[1, 2]
    L_bins = sorted(set(round(r['L_over_M'], 2) for r in results))
    means = []
    lows = []
    highs = []
    for Lb in L_bins:
        shifts = [r['shift'] for r in results if abs(r['L_over_M'] - Lb) < 0.015]
        if len(shifts) > 2:
            means.append(np.mean(shifts))
            lows.append(np.percentile(shifts, 16))
            highs.append(np.percentile(shifts, 84))
        else:
            means.append(np.nan)
            lows.append(np.nan)
            highs.append(np.nan)

    ax.fill_between(L_bins, lows, highs, alpha=0.3, color='steelblue',
                    label='68% spread (EOS variation)')
    ax.plot(L_bins, means, 'b-', linewidth=2, label='Mean TOV shift')
    ax.plot(L_fine, hayward_qnm_shift(L_fine), 'k-', linewidth=2, label='Hayward: $0.18(L/M)^2$')
    ax.set_xlabel('$L/M$')
    ax.set_ylabel('$|\delta\omega/\omega_0|$')
    ax.set_title('Spread vs Hayward prediction')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.close()


# =========================================================================
#  Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TOV EOS sweep: self-consistent QNM shifts vs Hayward')
    parser.add_argument('--samples', type=int, default=150,
                        help='Approximate number of parameter combinations')
    parser.add_argument('--L', type=str, default='0.1,0.2,0.3,0.4,0.5,0.6,0.7',
                        help='L/M values (comma-separated)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only plot from saved CSV')
    parser.add_argument('--output-csv', type=str,
                        default='tov_eos_sweep_results.csv')
    parser.add_argument('--output-plot', type=str,
                        default='tov_eos_sweep_plot.png')
    parser.add_argument('--n-points', type=int, default=2000,
                        help='TOV integration grid points')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel worker processes')
    args = parser.parse_args()

    if args.plot_only:
        if not os.path.exists(args.output_csv):
            print(f"CSV not found: {args.output_csv}")
            sys.exit(1)
        results = []
        with open(args.output_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append({k: float(v) for k, v in row.items()})
        plot_results(results, args.output_plot)
        return

    L_vals = [float(x) for x in args.L.split(',')]

    # Build parameter grid with ~samples total points
    # Distribute samples across dimensions
    n_L = len(L_vals)
    n_per_L = args.samples // n_L
    # Allocate: rc (~4), n (~3), w_inf (~3)
    n_rc = min(4, n_per_L)
    n_n = min(3, max(2, n_per_L // n_rc))
    n_w = max(1, n_per_L // (n_rc * n_n))

    rc_vals = np.linspace(0.15, 1.5, n_rc)
    n_vals = np.linspace(1.5, 6.0, n_n)
    w_inf_vals = np.linspace(-0.1, 0.3, n_w)

    print("=" * 70)
    print("TOV EOS SWEEP: Self-consistent QNM shifts")
    print("=" * 70)
    print(f"  L/M values: {L_vals}")
    print(f"  rc values:  {[f'{x:.2f}' for x in rc_vals]}")
    print(f"  n values:   {[f'{x:.1f}' for x in n_vals]}")
    print(f"  w_inf:      {[f'{x:.2f}' for x in w_inf_vals]}")
    print(f"  Total combinations: {len(L_vals)*len(rc_vals)*len(n_vals)*len(w_inf_vals)}")
    print()

    results = run_sweep(L_vals, rc_vals, n_vals, w_inf_vals,
                        M=1.0, n_points=args.n_points,
                        n_workers=args.workers)

    if not results:
        print("ERROR: No converged solutions!")
        sys.exit(1)

    # Save CSV
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {args.output_csv}")

    # Summary statistics
    shifts = np.array([r['shift'] for r in results])
    deltas = np.array([r['delta_shift'] for r in results])

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Converged solutions: {len(results)}")
    print(f"  Mean QNM shift:      {np.mean(shifts):.5f}")
    print(f"  Median QNM shift:    {np.median(shifts):.5f}")
    print(f"  Std dev:             {np.std(shifts):.5f}")
    print(f"  Mean deviation from Hayward: {np.mean(deltas):.5f}")
    print(f"  RMS deviation from Hayward:  {np.sqrt(np.mean(deltas**2)):.5f}")
    print(f"  68% range: [{np.percentile(shifts, 16):.5f}, {np.percentile(shifts, 84):.5f}]")

    # Per-L summary
    print(f"\n  Per-L/M breakdown:")
    print(f"  {'L/M':>6s}  {'Mean shift':>12s}  {'Hayward':>10s}  {'16%':>10s}  {'84%':>10s}  {'N':>5s}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*5}")
    for L_val in L_vals:
        subset = [r['shift'] for r in results if abs(r['L_over_M'] - L_val) < 0.015]
        if subset:
            m = np.mean(subset)
            lo = np.percentile(subset, 16)
            hi = np.percentile(subset, 84)
            hw = hayward_qnm_shift(L_val)
            print(f"  {L_val:6.2f}  {m:12.5f}  {hw:10.5f}  {lo:10.5f}  {hi:10.5f}  {len(subset):5d}")

    # Plot
    plot_results(results, args.output_plot)
    print()


if __name__ == '__main__':
    main()
