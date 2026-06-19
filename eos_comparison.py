#!/usr/bin/env python3
"""EOS comparison: M_max and rho0 bounds for SLy, APR4, MS1b, H4.

For each EOS:
1. Pure M_max calibration
2. M_max at each rho0 (r_c = 2 km, n = 2)
3. rho0 threshold where M_max drops below NICER 1-sigma (2.01) and 2-sigma (1.94)
"""
import numpy as np
import tov_tidal as tt
import sys

# NICER bounds
NICER_1SIGMA = 2.01  # M_sun
NICER_2SIGMA = 1.94  # M_sun

RHO0_VALS = [0, 1e10, 3e10, 1e11, 3e11, 1e12, 3e12, 1e13, 3e13, 1e14]
R_C_KM = 2.0
N_PTS = 4000


def find_mmax(eos_name, rho0, n_pts=N_PTS, r_c_km=R_C_KM):
    """Find M_max for given EOS and rho0 — grid scan + golden-section refinement."""
    tt.setup_eos(eos_name)
    # Coarse grid scan
    p_c_vals = np.logspace(33.0, 37.5, 80)
    best = 0.0
    best_log10 = 33.0
    for p_c in p_c_vals:
        try:
            sol = tt.solve_tov_tidal(p_c, rho0_cgs=rho0, r_c_km=r_c_km,
                                      n=2, n_points=n_pts)
        except Exception:
            continue
        if sol and sol.get('success') and not sol.get('at_boundary', False):
            if sol['M_Msun'] > best:
                best = sol['M_Msun']
                best_log10 = np.log10(p_c)

    if best <= 0:
        return 0.0, 0.0

    # Golden-section refinement
    M_refined, log10_opt = tt._golden_section_mmax(
        best_log10, rho0_cgs=rho0, r_c_km=r_c_km, n=2, n_points=n_pts)
    if M_refined and M_refined > best:
        return M_refined, 10**log10_opt
    return best, 10**best_log10


def find_rho0_bound(eos_name, m_thresh):
    """Find the highest rho0 where M_max >= m_thresh."""
    tt.setup_eos(eos_name)
    lo_idx = 0
    hi_idx = len(RHO0_VALS) - 1

    # Binary search on rho0 index
    while lo_idx < hi_idx:
        mid = (lo_idx + hi_idx + 1) // 2
        mm, _ = find_mmax(eos_name, RHO0_VALS[mid])
        if mm >= m_thresh:
            lo_idx = mid
        else:
            hi_idx = mid - 1

    rho_bound = RHO0_VALS[lo_idx]
    mm_bound, _ = find_mmax(eos_name, rho_bound)
    # Also check one step above
    if lo_idx + 1 < len(RHO0_VALS):
        mm_next, _ = find_mmax(eos_name, RHO0_VALS[lo_idx + 1])
    else:
        mm_next = 0.0

    return rho_bound, mm_bound, mm_next


def main():
    eos_list = ['sly', 'apr4', 'ms1b', 'h4']
    results = {}

    print("=" * 70)
    print("EOS COMPARISON: M_max and NICER rho0 bounds")
    print("=" * 70)

    for eos_name in eos_list:
        print(f"\n--- {eos_name.upper()} ---")
        sys.stdout.flush()

        # Pure M_max
        mm_pure, pc_pure = find_mmax(eos_name, 0)
        print(f"  Pure M_max = {mm_pure:.3f} Msun  (at p_c = {pc_pure:.2e})")

        # M_max at each rho0
        print(f"  {'rho0 (g/cm3)':>14s}  {'M_max (Msun)':>12s}  {'Status':>12s}")
        mm_vals = {}
        for rho0 in RHO0_VALS:
            mm, _ = find_mmax(eos_name, rho0)
            mm_vals[rho0] = mm
            if rho0 == 0:
                status = "pure"
            elif mm >= NICER_1SIGMA:
                status = "NICER-OK(1s)"
            elif mm >= NICER_2SIGMA:
                status = "NICER-OK(2s)"
            else:
                status = "EXCLUDED"
            print(f"  {rho0:14.1e}  {mm:12.3f}  {status:>12s}")
            sys.stdout.flush()

        results[eos_name] = {
            'M_max_pure': mm_pure,
            'M_max_by_rho0': mm_vals,
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: NICER rho0 bounds by EOS")
    print("=" * 70)
    print(f"  NICER 1-sigma: M_max >= {NICER_1SIGMA} Msun")
    print(f"  NICER 2-sigma: M_max >= {NICER_2SIGMA} Msun")
    print()
    print(f"  {'EOS':>6s}  {'M_max pure':>10s}  {'rho0(1s) bound':>15s}  {'rho0(2s) bound':>15s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*15}  {'-'*15}")

    for eos_name in eos_list:
        r = results[eos_name]
        mm_pure = r['M_max_pure']
        mmv = r['M_max_by_rho0']

        # Find 1-sigma and 2-sigma bounds
        rho1s = "none"
        rho2s = "none"
        for rho0 in sorted(mmv.keys()):
            if mmv[rho0] >= NICER_1SIGMA:
                rho1s = f"{rho0:.1e}"
            if mmv[rho0] >= NICER_2SIGMA:
                rho2s = f"{rho0:.1e}"

        # If pure M_max already below threshold
        if mm_pure < NICER_1SIGMA:
            rho1s = "pure < 2.01"
        if mm_pure < NICER_2SIGMA:
            rho2s = "pure < 1.94"

        print(f"  {eos_name:>6s}  {mm_pure:10.3f}  {rho1s:>15s}  {rho2s:>15s}")


if __name__ == '__main__':
    main()
