#!/usr/bin/env python3
"""
Tier 1, Calculation 1: Superluminal observer angle dynamics in 1+1D.

Models a superluminal "observer" as a spacelike geodesic in the Schwarzschild
spacetime (equatorial plane, effectively 2+1D with azimuthal coordinate phi).

The key observable is alpha(r) — the local angle between the trajectory
and the radial direction, measured in the orthonormal frame of a static
observer:

    tan alpha = (angular velocity) / (radial velocity)
              = |ell| sqrt(f) / (r |dr/dlambda|)

where f(r) = 1 - 2M/r, ell = conserved angular momentum,
(dr/dlambda)^2 = eps^2 + f(r)(1 - ell^2/r^2)  for spacelike geodesics.

Physics questions:
  1. Gravitational "straightening": does alpha -> 0 at the horizon?
  2. Can superluminal signals escape from inside the BH?
  3. Impact parameter dependence of the deflection angle
  4. Bound "orbits" inside the horizon for superluminal signals

Uses multiprocessing for parameter sweeps over (v_inf, impact_parameter).

Usage:
    python theta_dynamics.py                       # default sweep
    python theta_dynamics.py --workers 8           # 8 CPU cores
    python theta_dynamics.py --v-inf 1.1,2,10,100  # specific v_inf values
    python theta_dynamics.py --plot-only           # plot from saved CSV
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import csv
import os
import sys
import time

# ── Schwarzschild metric ────────────────────────────────────────────────

def f_schw(r, M=1.0):
    """Schwarzschild metric function f(r) = 1 - 2M/r."""
    return 1.0 - 2.0 * M / np.maximum(r, 1e-12)


# ── Spacelike geodesic equations ────────────────────────────────────────
#
# Constants of motion:
#   eps = f(r) dt/dlambda     (energy parameter)
#   ell = r^2 dphi/dlambda    (angular momentum)
#
# Radial equation (spacelike, kappa = +1):
#   (dr/dlambda)^2 = eps^2 + f(r)(1 - ell^2/r^2)
#
# Angle alpha in local orthonormal frame:
#   tan(alpha) = |ell| sqrt(f) / (r |dr/dlambda|)
#
# Asymptotic speed: v_inf = sqrt(1 + 1/eps^2) > 1
# Impact parameter:  b = ell / sqrt(eps^2 + 1)


def dr_dlambda_sq(r, eps, ell, M=1.0):
    """Squared radial derivative for spacelike geodesic."""
    f = f_schw(r, M)
    return eps**2 + f * (1.0 - ell**2 / r**2)


def local_angle_alpha(r, drdl_sq, ell, M=1.0):
    """
    Local angle alpha between trajectory and radial direction.
    tan(alpha) = |ell| sqrt(f) / (r |dr/dlambda|)
    Returns alpha in radians.
    """
    if drdl_sq <= 0:
        return np.pi / 2.0  # turning point: purely angular
    f = f_schw(r, M)
    if f <= 0:
        # Inside horizon: alpha is not well-defined for static observer
        # Return the analytic continuation
        return np.nan
    tan_alpha = abs(ell) * np.sqrt(f) / (r * np.sqrt(drdl_sq))
    return np.arctan(tan_alpha)


def eps_from_vinf(v_inf):
    """Convert asymptotic speed to energy parameter epsilon."""
    if v_inf <= 1.0:
        raise ValueError(f"Superluminal requires v_inf > 1, got {v_inf}")
    return 1.0 / np.sqrt(v_inf**2 - 1.0)


# ── Analytical escape boundary ───────────────────────────────────────────

def analytical_b_esc(eps, M=1.0):
    """
    Compute the critical impact parameter b_esc analytically.

    For spacelike geodesics in Schwarzschild:
      (dr/dλ)² = ε² + f(r)(1 - ℓ²/r²),  f = 1 - 2M/r

    The turning point condition (dr/dλ)² = 0 at the effective potential
    peak gives:
      ε² = (r-2M)² / (r(3M-r))    →  solve for r
      b²  = r³ / (4M-r)           →  b_esc

    b_esc is the MAXIMUM impact parameter that results in CAPTURE
    (particle falls into the horizon). For b > b_esc, the particle
    encounters a turning point and escapes.

    The absolute maximum is 3√3 M ≈ 5.196 M, reached as ε → ∞ (v → c⁺).
    """
    if eps <= 0:
        # v → ∞: b_esc → 2M
        return 2.0 * M

    eps_sq = eps**2
    # Solve ε² = (r-2)²/(r(3-r)) for r ∈ (2M, 3M).
    # Rearrange: ε² r(3-r) = (r-2)²
    #            ε²(3r - r²) = r² - 4r + 4
    #            ε²·3r - ε²·r² - r² + 4r - 4 = 0
    #            (1+ε²)r² - (4+3ε²)r + 4 = 0  [with M=1]
    #
    # With general M: use dimensionless variable x = r/M.
    # ε²·x(3-x) = (x-2)²
    # (1+ε²)x² - (4+3ε²)x + 4 = 0
    a = 1.0 + eps_sq
    b_coef = -(4.0 + 3.0 * eps_sq)
    c = 4.0
    disc = b_coef**2 - 4.0 * a * c
    if disc < 0:
        # Should not happen for valid eps
        return 3.0 * np.sqrt(3.0) * M
    sqrt_disc = np.sqrt(disc)
    # Two roots: one in (2, 3), the other outside
    x1 = (-b_coef - sqrt_disc) / (2.0 * a)
    x2 = (-b_coef + sqrt_disc) / (2.0 * a)
    # Pick the root in (2, 3)
    if 2.0 < x1 < 3.0:
        x = x1
    elif 2.0 < x2 < 3.0:
        x = x2
    else:
        # Fallback: use the smaller root > 2
        x = min(x1, x2) if min(x1, x2) > 2.0 else max(x1, x2)
    # b/M = sqrt(x³/(4-x))
    b_over_M = np.sqrt(x**3 / (4.0 - x))
    return b_over_M * M


def vinf_from_eps(eps):
    """Convert energy parameter to asymptotic speed."""
    return np.sqrt(1.0 + 1.0 / eps**2)


def impact_parameter(eps, ell):
    """Impact parameter b = ell / sqrt(eps^2 + 1)."""
    return ell / np.sqrt(eps**2 + 1.0)


# ── Effective potential analysis ────────────────────────────────────────

def eff_potential_spacelike(r, ell, M=1.0):
    """
    Effective potential for spacelike geodesics.
    V_eff(r) = -f(r)(1 - ell^2/r^2)
    (dr/dlambda)^2 = eps^2 - V_eff(r)

    Turning points where eps^2 = V_eff(r).
    """
    f = f_schw(r, M)
    return -f * (1.0 - ell**2 / r**2)


def find_turning_points(eps, ell, M=1.0, r_min=None, r_max=200.0):
    """
    Find radial turning points by solving eps^2 = V_eff(r).
    Only searches exterior (r > 2M).
    Returns sorted list of r values where dr/dlambda = 0.
    """
    if r_min is None:
        r_min = 2.001 * M
    
    # Quick check: V_eff = -f(1-ℓ²/r²).  V_eff >= 0 requires r <= |ℓ|.
    # If |ℓ| <= r_min (≈ 2M), V_eff is always negative → no turning points.
    if abs(ell) <= r_min:
        return []

    def V_minus_eps2(r):
        f_val = f_schw(r, M)
        if f_val <= 0:
            return -eps**2  # negative, no root
        return eff_potential_spacelike(r, ell, M) - eps**2

    # Sample the function to bracket roots
    r_sample = np.geomspace(r_min, r_max, 5000)
    V_vals = np.array([V_minus_eps2(r) for r in r_sample])

    turning_points = []
    for i in range(len(r_sample) - 1):
        if V_vals[i] * V_vals[i + 1] < 0:
            # Root bracketed
            r_lo, r_hi = r_sample[i], r_sample[i + 1]
            for _ in range(50):
                r_mid = 0.5 * (r_lo + r_hi)
                V_mid = V_minus_eps2(r_mid)
                if V_vals[i] * V_mid <= 0:
                    r_hi = r_mid
                else:
                    r_lo = r_mid
            turning_points.append(0.5 * (r_lo + r_hi))

    return sorted(set(round(tp, 10) for tp in turning_points))


# ── Exterior alpha(r) profile ───────────────────────────────────────────
#
# Key physics: tan(alpha) = |ell| sqrt(f) / (r |dr/dlambda|)
# As r -> 2M, f -> 0, so tan(alpha) -> 0  =>  alpha -> 0
# This is GRAVITATIONAL STRAIGHTENING: all superluminal signals
# become purely radial at the horizon, regardless of initial angle.

def compute_alpha_profile(eps, ell, M=1.0, r_start=50.0, r_min=None, n_pts=500):
    """
    Compute alpha(r) on a radial grid from r_start down to near the horizon
    (or the innermost turning point for bouncing trajectories).

    Returns dict with r, alpha, drdl, f_vals, plus diagnostics.

    This avoids the coordinate singularity at r=2M by stopping at f > 0.
    """
    # Find turning points to determine r_min
    turning_pts = find_turning_points(eps, ell, M)
    tp_outside = [tp for tp in turning_pts if tp > 2.01 * M]
    can_escape = len(tp_outside) > 0

    if can_escape:
        # Innermost turning point where particle bounces
        r_turn = min(tp_outside)
        if r_min is None or r_min < r_turn:
            r_min = r_turn
    elif r_min is None:
        # Stop where f(r) = 0.001 (very close to horizon)
        r_min = 2.0 * M / (1.0 - 0.001)  # r ≈ 2.002 M

    r = np.linspace(r_start, r_min, n_pts)
    f_vals = f_schw(r, M)

    alpha = np.full_like(r, np.nan)
    drdl_vals = np.zeros_like(r)

    for i in range(len(r)):
        ri = r[i]
        fi = f_vals[i]
        drdl_sq = eps**2 + fi * (1.0 - ell**2 / max(ri, 1e-12)**2)
        drdl = np.sqrt(max(0, drdl_sq))
        drdl_vals[i] = drdl
        if fi > 1e-10 and drdl > 1e-12:
            tan_alpha = abs(ell) * np.sqrt(fi) / (ri * drdl)
            alpha[i] = np.arctan(tan_alpha)

    # Diagnostics
    alpha_inf = alpha[0] if np.isfinite(alpha[0]) else np.nan

    # Straightening radius: where alpha drops to half of alpha_inf.
    # Only meaningful for trapped trajectories (alpha → 0 at horizon).
    # For bouncing trajectories, alpha → π/2 at turning point → NaN.
    r_straighten = np.nan
    if np.isfinite(alpha_inf) and alpha_inf > 1e-6 and not can_escape:
        half_alpha = alpha_inf / 2.0
        for i in range(len(r)):
            if np.isfinite(alpha[i]) and alpha[i] < half_alpha:
                r_straighten = r[i]
                break

    # Alpha at innermost valid point
    alpha_inner = np.nan
    for i in range(len(r) - 1, -1, -1):
        if np.isfinite(alpha[i]):
            alpha_inner = alpha[i]
            break

    return {
        'success': True,
        'eps': eps,
        'ell': ell,
        'v_inf': vinf_from_eps(eps),
        'b': impact_parameter(eps, ell),
        'b_over_M': impact_parameter(eps, ell) / M,
        'alpha_inf': float(alpha_inf) if np.isfinite(alpha_inf) else np.nan,
        'alpha_near_horizon': float(alpha_inner) if np.isfinite(alpha_inner) else np.nan,
        'r_straighten': float(r_straighten) if np.isfinite(r_straighten) else np.nan,
        'can_escape': can_escape,
        'r_inner': float(r_min),
        'turning_points': turning_pts,
        'n_turning_points': len(turning_pts),
        'r': r,
        'alpha': alpha,
        'f': f_vals,
        'drdl': drdl_vals,
    }


# ── Single parameter evaluation (worker) ────────────────────────────────

def evaluate_params(args_tuple):
    """Evaluate a single (v_inf, b/M) point."""
    idx, v_inf, b_over_M, M = args_tuple

    try:
        eps = eps_from_vinf(v_inf)
    except ValueError:
        return {'idx': idx, 'v_inf': v_inf, 'b_over_M': b_over_M,
                'success': False, 'error': 'invalid_vinf'}

    ell = b_over_M * M * np.sqrt(eps**2 + 1.0)

    prof = compute_alpha_profile(eps, ell, M=M)

    return {
        'idx': idx,
        'v_inf': v_inf,
        'b_over_M': b_over_M,
        'eps': eps,
        'ell': ell,
        'M': M,
        **{k: v for k, v in prof.items()
           if k not in ('r', 'alpha', 'f', 'drdl')},
    }


# ── Parameter grid ──────────────────────────────────────────────────────

def build_parameter_grid(args):
    """Build list of (v_inf, b/M) tasks with refined grid around b_esc."""
    def parse_list(s):
        return [float(x) for x in s.split(',')]

    if args.v_inf:
        v_inf_vals = parse_list(args.v_inf)
    else:
        v_inf_vals = np.logspace(np.log10(args.v_min), np.log10(args.v_max),
                                 args.n_v)

    M_val = args.M_val

    # If user specified b values explicitly, use those for all v_inf
    if args.b:
        b_vals = parse_list(args.b)
        tasks = []
        idx = 0
        for v_inf in v_inf_vals:
            for b in b_vals:
                tasks.append((idx, v_inf, b, M_val))
                idx += 1
        return tasks

    # Adaptive grid: fine sampling around analytical b_esc for each v_inf
    tasks = []
    idx = 0
    for v_inf in v_inf_vals:
        eps = eps_from_vinf(v_inf)
        b_esc = analytical_b_esc(eps, M_val) / M_val  # dimensionless

        # Build a refined grid around b_esc
        # Outer grid: from 0.1 to ~2*b_esc (or b_max), fewer points
        b_max = max(2.0 * b_esc, args.b_max)
        b_outer = np.logspace(np.log10(0.1), np.log10(b_max),
                             max(6, args.n_b // 2))

        # Inner grid: fine sampling around b_esc
        # Width: ±25% of b_esc, with args.n_b points
        b_lo = max(0.1, b_esc * 0.75)
        b_hi = b_esc * 1.25
        b_inner = np.linspace(b_lo, b_hi, args.n_b)

        # Combine and deduplicate (sort)
        b_all = np.unique(np.concatenate([b_outer, b_inner]))

        for b in b_all:
            tasks.append((idx, v_inf, b, M_val))
            idx += 1

    return tasks


# ── Main sweep ──────────────────────────────────────────────────────────

def run_sweep(tasks, workers, output_csv):
    """Run parameter sweep with multiprocessing."""
    n_total = len(tasks)
    print(f"Running {n_total} superluminal geodesic integrations "
          f"across {workers} workers...")
    v_inf_all = [t[1] for t in tasks]
    b_all = [t[2] for t in tasks]
    print(f"  v_inf range: [{min(v_inf_all):.2f}, {max(v_inf_all):.2f}]")
    print(f"  b/M    range: [{min(b_all):.3f}, {max(b_all):.3f}]")
    print()

    results = []
    n_done = 0
    n_success = 0
    t_start = time.perf_counter()

    fieldnames = [
        'idx', 'v_inf', 'b_over_M', 'eps', 'ell', 'M', 'success', 'error',
        'alpha_inf', 'alpha_near_horizon', 'r_straighten', 'r_inner',
        'can_escape', 'n_turning_points',
    ]

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(evaluate_params, t): t for t in tasks}

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                # Write only summary fields (not arrays)
                writer.writerow({k: result[k] for k in fieldnames if k in result})
                n_done += 1
                if result.get('success'):
                    n_success += 1

                if n_done % max(1, n_total // 20) == 0 or n_done == n_total:
                    elapsed = time.perf_counter() - t_start
                    rate = n_done / elapsed if elapsed > 0 else 0
                    pct = 100 * n_done / n_total
                    print(f"  [{n_done:5d}/{n_total}] {pct:5.1f}%  "
                          f"success={n_success}  rate={rate:.1f}/s  "
                          f"elapsed={elapsed:.0f}s")

    t_total = time.perf_counter() - t_start
    print(f"\nDone. {n_success}/{n_total} successful "
          f"({100*n_success/n_total:.1f}%)")
    print(f"Total time: {t_total:.1f}s  "
          f"Rate: {n_total/t_total:.1f} integrations/s")
    print(f"Results saved to: {output_csv}")

    return results


# ── Plotting ────────────────────────────────────────────────────────────

def plot_results(csv_path):
    """Generate diagnostic plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plots. "
              "Install with: pip install matplotlib")
        return

    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    data = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            data.append(row)

    if not data:
        print("No data in CSV.")
        return

    successful = [d for d in data if d['success'] == 'True']
    if not successful:
        print("No successful integrations in CSV.")
        return

    def sf(val, default=np.nan):
        try:
            return float(val) if val and str(val).strip() else default
        except (ValueError, TypeError):
            return default

    v_inf = np.array([sf(d['v_inf']) for d in successful])
    b_M = np.array([sf(d['b_over_M']) for d in successful])
    alpha_inf = np.array([sf(d['alpha_inf']) for d in successful])
    alpha_horizon = np.array([sf(d['alpha_near_horizon']) for d in successful])
    r_straighten = np.array([sf(d['r_straighten']) for d in successful])
    can_escape = np.array([d['can_escape'] == 'True' for d in successful])

    # Get unique v_inf values for coloring
    unique_v = np.sort(np.unique(v_inf))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(unique_v)))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Superluminal Geodesics in Schwarzschild: Angle Dynamics',
                 fontsize=13)

    # 1. alpha at infinity vs impact parameter
    ax = axes[0, 0]
    for i, uv in enumerate(unique_v):
        mask = np.abs(v_inf - uv) < 1e-10
        order = np.argsort(b_M[mask])
        ax.plot(b_M[mask][order], np.degrees(alpha_inf[mask][order]),
                'o-', markersize=3, color=colors[i],
                label=f'v_inf={uv:.1f}', alpha=0.7)
    ax.set_xlabel('Impact parameter b / M')
    ax.set_ylabel('alpha_inf (degrees)')
    ax.set_title('Asymptotic angle vs. impact parameter')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.axhline(45, color='gray', linestyle=':', alpha=0.3)

    # 2. Straightening: alpha_near_horizon / alpha_inf
    ax = axes[0, 1]
    ratio = np.where(np.isfinite(alpha_horizon) & (alpha_inf > 0.001),
                     alpha_horizon / alpha_inf, np.nan)
    for i, uv in enumerate(unique_v):
        mask = (np.abs(v_inf - uv) < 1e-10) & np.isfinite(ratio)
        if mask.sum() > 0:
            order = np.argsort(b_M[mask])
            ax.plot(b_M[mask][order], ratio[mask][order],
                    'o-', markersize=3, color=colors[i],
                    label=f'v_inf={uv:.1f}', alpha=0.7)
    ax.set_xlabel('Impact parameter b / M')
    ax.set_ylabel('alpha_near_horizon / alpha_inf')
    ax.set_title('Gravitational straightening (all -> 0 at horizon)')
    ax.axhline(0, color='red', linestyle='--', alpha=0.3)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 3. Straightening radius vs impact parameter
    ax = axes[1, 0]
    for i, uv in enumerate(unique_v):
        mask = (np.abs(v_inf - uv) < 1e-10) & np.isfinite(r_straighten)
        if mask.sum() > 0:
            order = np.argsort(b_M[mask])
            ax.plot(b_M[mask][order], r_straighten[mask][order],
                    'o-', markersize=3, color=colors[i],
                    label=f'v_inf={uv:.1f}', alpha=0.7)
    ax.axhline(2.0, color='black', linestyle='--', alpha=0.5,
               label='Horizon r=2M')
    ax.set_xlabel('Impact parameter b / M')
    ax.set_ylabel('r_straighten / M (where alpha = alpha_inf/2)')
    ax.set_title('Straightening radius')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 4. Escape diagram: can_escape vs (b/M, v_inf)
    ax = axes[1, 1]
    for i, uv in enumerate(unique_v):
        mask = np.abs(v_inf - uv) < 1e-10
        # Escaping
        esc = mask & can_escape
        if esc.sum() > 0:
            ax.scatter(b_M[esc], np.full(esc.sum(), np.log10(uv)),
                       c=[colors[i]], s=30, marker='o', alpha=0.7)
        # Falling in
        fall = mask & ~can_escape
        if fall.sum() > 0:
            ax.scatter(b_M[fall], np.full(fall.sum(), np.log10(uv)),
                       c=[colors[i]], s=50, marker='x', alpha=0.7,
                       linewidths=2)
    ax.set_xlabel('Impact parameter b / M')
    ax.set_ylabel('log10(v_inf)')
    ax.set_title('Escape diagram: o=escape, x=fall in')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = csv_path.replace('.csv', '.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.close()


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Superluminal geodesic angle dynamics in Schwarzschild',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--workers', '-w', type=int, default=os.cpu_count(),
                        help=f'Worker processes (default: {os.cpu_count()})')
    parser.add_argument('--output', '-o', default='theta_dynamics_results.csv',
                        help='Output CSV')
    parser.add_argument('--v-inf', help='Comma-separated v_inf values')
    parser.add_argument('--b', help='Comma-separated b/M values')
    parser.add_argument('--v-min', type=float, default=1.01,
                        help='Min v_inf (default: 1.01, slightly superluminal)')
    parser.add_argument('--v-max', type=float, default=100.0,
                        help='Max v_inf (default: 100)')
    parser.add_argument('--n-v', type=int, default=8,
                        help='Number of v_inf points in logspace (default: 8)')
    parser.add_argument('--b-min', type=float, default=0.1,
                        help='Min b/M (default: 0.1)')
    parser.add_argument('--b-max', type=float, default=20.0,
                        help='Max b/M (default: 20)')
    parser.add_argument('--n-b', type=int, default=20,
                        help='Number of b/M points in logspace (default: 20)')
    parser.add_argument('--M-val', type=float, default=1.0,
                        help='Black hole mass in solar masses (default: 1.0)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Plot from existing CSV')
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

    # Summary
    successful = [r for r in results if r.get('success')]
    if successful:
        escaped_count = sum(1 for r in successful if r.get('can_escape'))
        trapped_count = len(successful) - escaped_count
        print(f"\nSummary:")
        print(f"  Escaped (bounced): {escaped_count}/{len(successful)}")
        print(f"  Trapped (fell in): {trapped_count}/{len(successful)}")

        # ── Escape boundary analysis ──
        print(f"\n  Escape boundary b_esc (analytical vs. numerical):")
        print(f"  {'v_inf':>8s}  {'b_esc(analyt)':>14s}  {'b_esc(numer)':>14s}  {'grid span':>12s}")
        print(f"  {'-'*8}  {'-'*14}  {'-'*14}  {'-'*12}")
        v_inf_all = sorted(set(float(r['v_inf']) for r in successful))
        b_esc_numerical = {}
        for v in v_inf_all:
            eps = eps_from_vinf(v)
            b_esc_analyt = analytical_b_esc(eps, M=1.0)
            group = [r for r in successful if abs(float(r['v_inf']) - v) < 1e-10]
            b_vals = sorted([float(r['b_over_M']) for r in group])
            escapes = [float(r['b_over_M']) for r in group if r.get('can_escape')]
            captured = [float(r['b_over_M']) for r in group if not r.get('can_escape')]
            # Numerical boundary: midpoint between max captured and min escaped
            if captured and escapes:
                b_num = 0.5 * (max(captured) + min(escapes))
            elif escapes:
                b_num = min(escapes)
            else:
                b_num = float('nan')
            b_esc_numerical[v] = b_num
            grid_span = f"[{min(b_vals):.2f}, {max(b_vals):.2f}]"
            print(f"  {v:8.2f}  {b_esc_analyt:14.4f}  {b_num:14.4f}  {grid_span:>12s}")
        print()

        alphas = [np.degrees(r['alpha_inf']) for r in successful
                  if np.isfinite(r.get('alpha_inf', np.nan))]
        if alphas:
            print(f"  alpha_inf range: [{min(alphas):.1f}, {max(alphas):.1f}] deg")

        # Straightening
        straight = []
        for r in successful:
            if r.get('can_escape'):
                continue  # bouncing trajectories don't reach the horizon
            ah = r.get('alpha_near_horizon', np.nan)
            ai = r.get('alpha_inf', np.nan)
            if np.isfinite(ah) and np.isfinite(ai) and ai > 0.001:
                straight.append(ah / ai)
        if straight:
            print(f"  Straightening (alpha_inner/alpha_inf): "
                  f"[{min(straight):.3f}, {max(straight):.3f}]")
            print(f"    (Values < 1 = gravitational straightening)")
            print(f"    (Ratio > 1 = anti-straightening: trajectory becomes more tangential)")

    if not args.no_plot:
        plot_results(args.output)


if __name__ == '__main__':
    main()
