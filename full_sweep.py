"""
Full TOV sweep — extended p_c grid, all r_c/n/rho0 combinations.
Saves to tov_tidal_results.csv (replaces truncated original).
"""
import sys
sys.path.insert(0, '.')
from tov_tidal import run_sweep
import numpy as np
import csv
import os
import shutil

def main():
    # Extended p_c: coarse low-end + fine high-end to capture M_max peak
    p_c_coarse = np.logspace(33.0, 35.2, 25)
    p_c_fine = np.logspace(35.2, 35.8, 22)  # dense near M_max
    p_c_vals = np.unique(np.sort(np.concatenate([p_c_coarse, p_c_fine])))
    
    rho0_vals = [0.0, 1e9, 3e9, 1e10, 3e10, 1e11, 3e11, 1e12, 3e12, 1e13, 3e13]
    r_c_vals = [0.5, 1.0, 2.0]
    n_vals = [0, 1, 2]

    n_tasks = len(p_c_vals) * len(rho0_vals) * len(r_c_vals) * len(n_vals)
    print(f"Sweep: {len(p_c_vals)}p_c * {len(rho0_vals)}rho0 * {len(r_c_vals)}r_c * {len(n_vals)}n = {n_tasks} tasks")
    print(f"p_c: {p_c_vals[0]:.1e} to {p_c_vals[-1]:.1e}")

    results = run_sweep(p_c_vals, rho0_vals, r_c_vals, n_vals,
                         w_chi_inf=0.0, n_points=2000, n_workers=16)

    # Backup old CSV
    path = r'd:\DEV\fizyka\tov_tidal_results.csv'
    backup = r'd:\DEV\fizyka\tov_tidal_results_old.csv'
    if os.path.exists(path):
        shutil.copy2(path, backup)
        print(f"Backed up old CSV to {backup}")

    # Save new CSV
    fieldnames = ['p_c_cgs', 'rho0_gcm3', 'r_c_km', 'n', 'M_Msun', 'R_km', 'C',
                  'y_R', 'k2', 'Lambda', 'cs2_max', 'success', 'at_boundary']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                'p_c_cgs': r['p_c_cgs'], 'rho0_gcm3': r['rho0_gcm3'],
                'r_c_km': r['r_c_km'], 'n': r['n'],
                'M_Msun': r['M_Msun'], 'R_km': r['R_km'],
                'C': r['C'], 'y_R': r['y_R'], 'k2': r['k2'],
                'Lambda': r['Lambda'], 'cs2_max': r['cs2_max'],
                'success': r['success'], 'at_boundary': r.get('at_boundary', False)
            })
    print(f"Saved {len(results)} results to {path}")

    # Quick M_max summary for article reference case (r_c=2.0, n=2)
    print("\n--- M_max (r_c=2.0, n=2) ---")
    for rho0 in rho0_vals:
        pts = [r for r in results 
               if r['rho0_gcm3'] == rho0 and r['r_c_km'] == 2.0 and r['n'] == 2
               and r['success'] and not r.get('at_boundary', False)]
        if pts:
            best = max(pts, key=lambda x: x['M_Msun'])
            print(f"  rho0={rho0:>8.0f}: M_max={best['M_Msun']:.4f}")
        else:
            print(f"  rho0={rho0:>8.0f}: NO DATA")

if __name__ == '__main__':
    main()
