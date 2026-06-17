"""
Extended TOV sweep v2 — n_points=2000 to match existing CSV,
with fine p_c sampling near each rho0's M_max peak.
"""
import sys
sys.path.insert(0, '.')
from tov_tidal import run_sweep, solve_tov_tidal
import numpy as np
import csv

def main():
    # Use n_points=2000 to match existing CSV
    # Coarse p_c grid + fine grid near the peak
    p_c_coarse = np.logspace(33.0, 35.0, 20)
    p_c_fine = np.logspace(35.0, 35.8, 30)  # fine near M_max
    p_c_vals = np.unique(np.sort(np.concatenate([p_c_coarse, p_c_fine])))
    
    rho0_vals = [0.0, 1e9, 3e9, 1e10, 3e10, 1e11, 3e11, 1e12, 3e12, 1e13, 3e13]
    r_c_vals = [2.0]
    n_vals = [2]

    print(f"Sweep: {len(p_c_vals)} p_c x {len(rho0_vals)} rho0 = {len(p_c_vals)*len(rho0_vals)} tasks")
    print(f"p_c range: {p_c_vals[0]:.1e} to {p_c_vals[-1]:.1e}")

    results = run_sweep(p_c_vals, rho0_vals, r_c_vals, n_vals,
                         w_chi_inf=0.0, n_points=2000, n_workers=16)

    # Find exact Lambda at 1.4 Msun via bisection (matching CSF interpolation)
    print("\n--- Lambda(1.4 Msun) via bisection (n_points=2000) ---")
    for rho0 in rho0_vals:
        lo, hi = 1e33, 5e35
        best = None
        for it in range(30):
            mid = (lo + hi) / 2
            res = solve_tov_tidal(mid, rho0, 2.0, n=2, w_chi_inf=0.0, n_points=2000)
            if not res or not res['success'] or res.get('at_boundary', False):
                hi = mid * 0.95 if res and res.get('at_boundary') else hi
                lo = mid * 1.05
                continue
            best = res
            if res['M_Msun'] < 1.4:
                lo = mid
            else:
                hi = mid
            if abs(res['M_Msun'] - 1.4) < 0.0005:
                break
        if best:
            print(f"  rho0={rho0:>8.0f}: M={best['M_Msun']:.4f}, Lambda={best['Lambda']:.0f}, R={best['R_km']:.2f}")
        else:
            print(f"  rho0={rho0:>8.0f}: NO SOLUTION NEAR 1.4")

    # Save
    fieldnames = ['p_c_cgs', 'rho0_gcm3', 'r_c_km', 'n', 'M_Msun', 'R_km', 'C',
                  'y_R', 'k2', 'Lambda', 'cs2_max', 'success', 'at_boundary']
    path = r'd:\DEV\fizyka\tov_tidal_extended.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                'p_c_cgs': r['p_c_cgs'], 'rho0_gcm3': r['rho0_gcm3'],
                'r_c_km': r['r_c_km'], 'n': r['n'], 'M_Msun': r['M_Msun'],
                'R_km': r['R_km'], 'C': r['C'], 'y_R': r['y_R'],
                'k2': r['k2'], 'Lambda': r['Lambda'], 'cs2_max': r['cs2_max'],
                'success': r['success'], 'at_boundary': r.get('at_boundary', False)
            })
    print(f"\nSaved {len(results)} results to {path}")

    # M_max comparison: article vs new
    print("\n--- M_max: Article (old CSV) vs Fresh Sweep ---")
    article_mmax = {
        0: 2.029, 1e9: 2.029, 3e9: 2.029,
        1e10: 2.029, 3e10: 2.028,
        1e11: 1.963, 3e11: 1.960,
        1e12: 1.952, 3e12: 1.851, 1e13: 1.776, 3e13: 1.702
    }
    for rho0 in rho0_vals:
        pts = [r for r in results if r['rho0_gcm3'] == rho0 and r['success'] and not r.get('at_boundary', False)]
        if pts:
            best = max(pts, key=lambda x: x['M_Msun'])
            old = article_mmax.get(rho0, 0)
            diff = best['M_Msun'] - old
            flag = " <--- FIX" if abs(diff) > 0.005 else ""
            print(f"  rho0={rho0:>8.0f}: old={old:.3f}  new={best['M_Msun']:.4f}  diff={diff:+.4f}{flag}")
        else:
            print(f"  rho0={rho0:>8.0f}: NO VALID SOLUTIONS")

if __name__ == '__main__':
    main()
