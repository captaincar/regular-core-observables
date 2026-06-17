#!/usr/bin/env python3
"""
Tier 1, Calculation 4: ΔN_eff parameter scan for 1+3 observer-sector dark radiation.

Sweeps over (g_*, T_dec) — the effective degrees of freedom and decoupling
temperature of the 1+3 sector — computing ΔN_eff at CMB recombination.

Uses multiprocessing for dense parameter grids. Compares against:
  - Planck 2018:  N_eff = 2.99 ± 0.17  → ΔN_eff ≤ 0.3 at ~2σ
  - CMB-S4 projected: σ(N_eff) ≈ 0.03

Usage:
    python delta_neff_scan.py                      # default sweep
    python delta_neff_scan.py --workers 8          # use 8 CPU cores
    python delta_neff_scan.py --plot-only          # plot from saved CSV
"""

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import csv
import os
import sys
import time
from pathlib import Path

# ── Physical constants ──────────────────────────────────────────────────
# SM relativistic degrees of freedom as function of temperature
# Approximate values from Saikawa & Shirai (2018), Husdal (2016)

# g_*s(T) — entropic d.o.f. — used to track temperature ratio after decoupling
# g_*ρ(T) — energy density d.o.f.

# For our purposes, we need g_*s at decoupling and at neutrino decoupling (~MeV)
# to compute the temperature ratio.

# Standard Model: g_*s(T > 100 GeV) ≈ 106.75
#                 g_*s(T ~ 1 MeV)   ≈ 10.75  (after e± annihilation)
#                 g_*s(T_CMB)       ≈ 3.91   (photons + neutrinos)

G_STAR_S_HIGH = 106.75   # T > electroweak
G_STAR_S_NEUTRINO = 10.75  # T ~ 1 MeV (neutrino decoupling)
G_STAR_S_CMB = 3.91       # today (photons + slightly warmer neutrinos)


def temperature_1p3_sector(T_dec_GeV, T_cmb=2.7255e-9):
    """
    Compute the temperature of the 1+3 sector today given its decoupling
    temperature T_dec (in GeV).

    After decoupling, the 1+3 sector cools as T ∝ 1/a independently.
    The SM entropy is separately conserved, so:

        (T_1p3 / T_SM)³ = g_*s(T_SM) / g_*s(T_dec)

    For the neutrino case (standard ΔN_eff from neutrino heating):
        T_ν = (4/11)^(1/3) T_γ  because g_*s(e±) = 7/2 → ratio = (4/11)^(1/3)

    General formula: if the 1+3 sector decouples at T_dec where g_*s = g_dec,
    and SM photons today have g_*s = 2 (photons only), then:

        T_1p3 / T_γ = (2 / g_dec)^(1/3)

    Wait — this is more subtle. Let me be precise.

    Entropy conservation for the SM:  g_*s(T_SM) a³ T_SM³ = constant.
    For the 1+3 sector post-decoupling: a³ T_1p3³ = constant.

    If decoupling happens at temperature T_dec, then at that moment T_1p3 = T_SM = T_dec.
    After decoupling, the SM entropy is dumped into photons when species annihilate,
    heating photons relative to the decoupled sector.

    The ratio today:
        T_1p3 / T_γ = (g_*s(T_γ) / g_*s(T_dec))^(1/3)

    where T_γ = T_CMB = 2.7255 K = 2.35e-13 GeV today, and g_*s(T_γ) ≈ 2 (photons).

    Actually: g_*s(T_γ) = 2 (for two photon polarizations). So:

        T_1p3 = T_CMB * (2 / g_*s(T_dec))^(1/3)

    This assumes the 1+3 sector was in thermal equilibrium with SM at T_dec
    and decoupled instantly. That's the simplifying assumption.
    """
    g_dec = g_star_s(T_dec_GeV)
    ratio = (2.0 / g_dec) ** (1.0 / 3.0)
    T_1p3_K = 2.7255 * ratio
    return T_1p3_K, ratio


def g_star_s(T_GeV):
    """
    Effective entropic degrees of freedom g_*s as function of temperature.

    Simplified step-function model adequate for order-of-magnitude ΔN_eff.
    """
    T = T_GeV
    if T > 100.0:        # Above electroweak: all SM particles
        return 106.75
    elif T > 1.0:        # Between EW and QCD: quarks + gluons + leptons + photons
        return 86.25
    elif T > 0.1:        # After QCD phase transition: pions + leptons + photons
        return 25.0  # approximate
    elif T > 0.001:      # After muon annihilation: e± + photons + neutrinos
        return 10.75
    elif T > 1e-6:       # After e± annihilation
        return 3.91
    else:                # Below neutrino decoupling
        return 3.36


def compute_delta_neff(g_star_1p3, T_dec_GeV):
    """
    Compute ΔN_eff from a 1+3 sector with g_* effective degrees of freedom
    decoupling at temperature T_dec (in GeV).

    After decoupling, the 1+3 sector's energy density relative to one SM
    neutrino species determines ΔN_eff.

    Energy density of one neutrino species at CMB:
        ρ_ν = (7/8) (π²/30) T_ν⁴

    Energy density of the 1+3 sector:
        ρ_1p3 = g_*_1p3 (π²/30) T_1p3⁴  (if bosonic)
              = g_*_1p3 (7/8) (π²/30) T_1p3⁴  (if fermionic)

    We'll assume bosonic (scalar field) as a simple model.

    ΔN_eff = ρ_1p3 / ρ_ν = g_*_1p3 * (T_1p3 / T_ν)⁴
    """
    T_1p3_K, ratio = temperature_1p3_sector(T_dec_GeV)

    # Neutrino temperature: T_ν = (4/11)^(1/3) T_CMB ≈ 1.945 K
    T_nu_K = 2.7255 * (4.0 / 11.0) ** (1.0 / 3.0)

    # For bosonic 1+3 sector
    delta_neff_bosonic = g_star_1p3 * (T_1p3_K / T_nu_K) ** 4.0

    # For fermionic 1+3 sector
    delta_neff_fermionic = g_star_1p3 * (7.0 / 8.0) / (7.0 / 8.0) * (T_1p3_K / T_nu_K) ** 4.0
    # fermi/bose ratio cancels: (7/8)/(7/8) = 1; result same if both are fermionic
    # If 1+3 is bosonic and neutrinos are fermionic, we need factor (1)/(7/8)
    # Actually neutrinos ARE fermionic: ρ_ν = (7/8)(π²/30)T_ν⁴
    # Bosonic 1+3: ρ_1p3 = g_*(π²/30)T_1p3⁴
    # So ΔN_eff = g_* * (T_1p3/T_ν)⁴ / (7/8)  for bosonic sector
    delta_neff = g_star_1p3 / (7.0 / 8.0) * (T_1p3_K / T_nu_K) ** 4.0

    return delta_neff, T_1p3_K


def evaluate_neff_point(args_tuple):
    """Evaluate single (g_star, T_dec) point for parallel sweep."""
    idx, g_star, T_dec = args_tuple
    try:
        dneff, T1 = compute_delta_neff(g_star, T_dec)
        return {
            'idx': idx,
            'g_star': g_star,
            'T_dec_GeV': T_dec,
            'delta_neff': dneff,
            'T_1p3_K': T1,
            'excluded_planck': dneff > 0.3,
            'detectable_cmbs4': dneff > 0.03,
        }
    except Exception as e:
        return {
            'idx': idx,
            'g_star': g_star,
            'T_dec_GeV': T_dec,
            'delta_neff': np.nan,
            'error': str(e),
        }


def build_neff_grid(args):
    """Build (g_star, T_dec) grid."""
    def parse_list(s):
        return [float(x) for x in s.split(',')]

    g_vals = parse_list(args.g_star) if args.g_star else np.linspace(0.5, 20, args.n_g)
    T_vals = parse_list(args.T_dec) if args.T_dec else np.logspace(-4, 16, args.n_T)

    tasks = []
    idx = 0
    for g in g_vals:
        for T in T_vals:
            tasks.append((idx, g, T))
            idx += 1
    return tasks


def run_neff_sweep(tasks, workers, output_csv):
    """Run ΔN_eff parameter sweep."""
    n_total = len(tasks)
    print(f"Running {n_total} dN_eff evaluations across {workers} workers...")
    print(f"  g_* range: [{min(t[1] for t in tasks):.1f}, {max(t[1] for t in tasks):.1f}]")
    print(f"  T_dec range: [{min(t[2] for t in tasks):.1e}, {max(t[2] for t in tasks):.1e}] GeV")
    print()

    results = []
    n_done = 0
    t_start = time.perf_counter()

    fieldnames = ['idx', 'g_star', 'T_dec_GeV', 'delta_neff', 'T_1p3_K',
                  'excluded_planck', 'detectable_cmbs4']

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(evaluate_neff_point, t): t for t in tasks}

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                writer.writerow(result)
                n_done += 1

                if n_done % max(1, n_total // 10) == 0 or n_done == n_total:
                    elapsed = time.perf_counter() - t_start
                    rate = n_done / elapsed if elapsed > 0 else 0
                    pct = 100 * n_done / n_total
                    print(f"  [{n_done:5d}/{n_total}] {pct:5.1f}%  "
                          f"rate={rate:.1f}/s  elapsed={elapsed:.0f}s")

    t_total = time.perf_counter() - t_start
    print(f"\nDone. Total time: {t_total:.1f}s")
    print(f"Results saved to: {output_csv}")

    # Summary statistics
    valid = [r for r in results if not np.isnan(r.get('delta_neff', np.nan))]
    if valid:
        excluded = sum(1 for r in valid if r['delta_neff'] > 0.3)
        detectable = sum(1 for r in valid if 0.03 < r['delta_neff'] <= 0.3)
        invisible = sum(1 for r in valid if r['delta_neff'] <= 0.03)
        print(f"  Excluded by Planck (dN_eff > 0.3):    {excluded}/{len(valid)}")
        print(f"  Detectable by CMB-S4 (0.03-0.3):      {detectable}/{len(valid)}")
        print(f"  Undetectable (dN_eff <= 0.03):         {invisible}/{len(valid)}")

    return results


def plot_neff_results(csv_path):
    """Contour plot of ΔN_eff in (g_*, T_dec) space."""
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

    g_arr = np.array([float(d['g_star']) for d in data])
    T_arr = np.array([float(d['T_dec_GeV']) for d in data])
    dneff_arr = np.array([float(d['delta_neff']) for d in data])

    # Remove NaN
    valid = ~np.isnan(dneff_arr)
    g_arr = g_arr[valid]
    T_arr = T_arr[valid]
    dneff_arr = dneff_arr[valid]

    # Reshape for contour plot
    g_unique = np.unique(g_arr)
    T_unique = np.unique(T_arr)

    if len(g_unique) < 2 or len(T_unique) < 2:
        print("Need at least 2 g_* and 2 T_dec values for contour plot.")
        # Fallback to scatter
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(T_arr, g_arr, c=np.log10(dneff_arr),
                        s=20, alpha=0.8, cmap='RdBu_r')
        ax.set_xscale('log')
        ax.set_xlabel('Decoupling temperature $T_{dec}$ [GeV]')
        ax.set_ylabel('$g_*$ (1+3 sector d.o.f.)')
        ax.set_title('$\\Delta N_{eff}$ from 1+3 observer sector')
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('$\\log_{10}(\\Delta N_{eff})$')
        ax.grid(True, alpha=0.3)
    else:
        G, T = np.meshgrid(g_unique, T_unique)
        D = np.zeros_like(G)
        for i in range(len(g_unique)):
            for j in range(len(T_unique)):
                mask = (np.abs(g_arr - g_unique[i]) < 1e-10) & \
                       (np.abs(T_arr - T_unique[j]) < 1e-10)
                if mask.any():
                    D[j, i] = dneff_arr[mask][0]

        fig, ax = plt.subplots(figsize=(10, 7))
        levels = np.logspace(-3, 2, 11)
        cf = ax.contourf(T, G, D, levels=levels, norm='log', cmap='RdYlBu_r', extend='both')
        cs = ax.contour(T, G, D, levels=[0.03, 0.3], colors=['orange', 'red'],
                        linewidths=[1.5, 2.0], linestyles=['--', '-'])
        ax.clabel(cs, fmt={0.03: 'CMB-S4', 0.3: 'Planck excl.'})

        ax.set_xscale('log')
        ax.set_xlabel('Decoupling temperature $T_{dec}$ [GeV]', fontsize=12)
        ax.set_ylabel('$g_*$ (1+3 sector degrees of freedom)', fontsize=12)
        ax.set_title('$\\Delta N_{eff}$ from 1+3 Observer Sector', fontsize=14)
        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_label('$\\Delta N_{eff}$', fontsize=11)
        ax.grid(True, alpha=0.2, which='both')

        # Annotate key regimes
        ax.annotate('GUT scale', xy=(1e16, 1), fontsize=8, ha='center', color='gray')
        ax.annotate('EW scale', xy=(100, 1), fontsize=8, ha='center', color='gray')
        ax.annotate('QCD scale', xy=(0.1, 1), fontsize=8, ha='center', color='gray')
        ax.axvline(100, color='gray', linestyle=':', alpha=0.4)
        ax.axvline(0.1, color='gray', linestyle=':', alpha=0.4)

    plt.tight_layout()
    plot_path = csv_path.replace('.csv', '.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.close()


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='ΔN_eff parameter scan for 1+3 observer-sector dark radiation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--workers', '-w', type=int, default=os.cpu_count(),
                        help=f'Number of worker processes (default: CPU count = {os.cpu_count()})')
    parser.add_argument('--output', '-o', default='delta_neff_results.csv',
                        help='Output CSV file')
    parser.add_argument('--g-star', help='Comma-separated g_* values (e.g. 0.5,1,5,10)')
    parser.add_argument('--T-dec', help='Comma-separated T_dec values in GeV')
    parser.add_argument('--n-g', type=int, default=20,
                        help='Number of g_* points (default: 20)')
    parser.add_argument('--n-T', type=int, default=40,
                        help='Number of T_dec points (default: 40)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip computation, only plot from CSV')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting')

    args = parser.parse_args()

    if args.plot_only:
        plot_neff_results(args.output)
        return

    tasks = build_neff_grid(args)
    if not tasks:
        print("No tasks generated.")
        sys.exit(1)

    run_neff_sweep(tasks, args.workers, args.output)

    if not args.no_plot:
        plot_neff_results(args.output)


if __name__ == '__main__':
    main()
