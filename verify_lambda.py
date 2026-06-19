#!/usr/bin/env python3
"""Verify Lambda(1.4) values against manuscript after golden-section recalibration.
Runs TOV+tidal for SLy EOS with calibrated K1, computes Lambda at M=1.4 for each rho0.
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from tov_tidal import setup_eos, solve_tov_tidal

setup_eos('sly', tune_K1=False)  # Use already-calibrated K1

rho0_vals = [0.0, 1e10, 3e10, 1e11, 3e11, 1e12, 3e12, 1e13, 3e13]
r_c = 2.0  # km
n = 2

print("=" * 70)
print("LAMBDA(1.4 Msun) VERIFICATION — SLy EOS (K1 = 2.1098e-10)")
print(f"r_c = {r_c} km, n = {n}")
print("=" * 70)
print(f"{'rho0 (g/cm3)':>16s}  {'M(Msun)':>8s}  {'R(km)':>8s}  {'C':>8s}  {'Lambda(1.4)':>12s}  {'Status':>6s}")
print("-" * 70)

# For each rho0, scan p_c to find M near 1.4 and interpolate Lambda
for rho0 in rho0_vals:
    best = None
    # Scan p_c around the right range
    for p_c in np.logspace(34.0, 36.5, 40):
        sol = solve_tov_tidal(p_c, rho0, r_c, n, n_points=4000)
        if sol is None or sol['M_Msun'] <= 0:
            continue
        if sol['M_Msun'] < 0.8:
            continue
        if best is None or abs(sol['M_Msun'] - 1.4) < abs(best['M_Msun'] - 1.4):
            best = sol
    
    if best is not None:
        # Refine with a finer scan near the best p_c
        import math
        p_c_best = float(best['p_c_cgs'])
        for dp in np.linspace(-0.05, 0.05, 11):
            p_c_test = p_c_best * 10**dp
            sol = solve_tov_tidal(p_c_test, rho0, r_c, n, n_points=4000)
            if sol is not None and sol['M_Msun'] > 0.8:
                if abs(sol['M_Msun'] - 1.4) < abs(best['M_Msun'] - 1.4):
                    best = sol
        
        # Linear interpolation to M=1.4 using nearby points
        # Collect points near M=1.4
        pts = []
        for dp in np.linspace(-0.3, 0.3, 60):
            p_c_test = p_c_best * 10**dp
            sol = solve_tov_tidal(p_c_test, rho0, r_c, n, n_points=4000)
            if sol is not None and sol['M_Msun'] > 0:
                pts.append((sol['M_Msun'], sol['Lambda'], sol['R_km'], sol['C']))
        
        pts.sort()
        if len(pts) >= 2 and pts[0][0] <= 1.4 <= pts[-1][0]:
            M_arr = np.array([p[0] for p in pts])
            L_arr = np.array([p[1] for p in pts])
            Lam14 = float(np.interp(1.4, M_arr, L_arr))
            R_arr = np.array([p[2] for p in pts])
            R14 = float(np.interp(1.4, M_arr, R_arr))
            C14 = 1.477 / R14  # G*Msun/c^2 in km ≈ 1.477 km
            
            rho0_str = 'Pure SLy' if rho0 == 0 else f'{rho0:.1e}'
            # Manuscript values
            manuscript = {0: 531, 1e10: 519, 3e10: 506, 1e12: 325, 1e13: 96, 3e13: 52}
            ms_val = manuscript.get(rho0, '—')
            delta = ''
            if rho0 in manuscript:
                pct = abs(Lam14 - ms_val) / ms_val * 100
                if pct < 1.0:
                    delta = ' OK'
                elif pct < 3.0:
                    delta = f' ~{pct:.1f}%'
                else:
                    delta = f' **{pct:.1f}%**'
            
            print(f"{rho0_str:>16s}  {1.4:8.3f}  {R14:8.2f}  {C14:8.4f}  {Lam14:12.1f}  {delta:>6s}")
        else:
            rho0_str = 'Pure SLy' if rho0 == 0 else f'{rho0:.1e}'
            print(f"{rho0_str:>16s}  {'—':>8s}  {'—':>8s}  {'—':>8s}  {'—':>12s}  {'NO1.4':>6s}")
    else:
        rho0_str = 'Pure SLy' if rho0 == 0 else f'{rho0:.1e}'
        print(f"{rho0_str:>16s}  {'—':>8s}  {'—':>8s}  {'—':>8s}  {'—':>12s}  {'FAIL':>6s}")

print()
print("Manuscript values for comparison (line 222-229):")
print("  Pure SLy: 531 | 1e10: 519 | 3e10: 506 | 1e12: 325 | 1e13: 96 | 3e13: 52")
