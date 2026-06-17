import sys
sys.path.insert(0, '.')
from tov_tidal import solve_tov_tidal
import numpy as np

r_c = 2.0; n = 2

# Targeted: find Lambda at EXACTLY M=1.4 using bisection on p_c
def find_at_mass(target_M, rho0, p_c_lo=1e33, p_c_hi=1e36, tolerance=0.001):
    lo, hi = p_c_lo, p_c_hi
    for _ in range(40):
        p_c_mid = (lo + hi) / 2
        result = solve_tov_tidal(p_c_mid, rho0, r_c, n=n, w_chi_inf=0.0, n_points=2000)
        if not result or not result['success'] or result.get('at_boundary', False):
            lo = p_c_mid
            continue
        M = result['M_Msun']
        if abs(M - target_M) < tolerance:
            return result
        if M < target_M:
            lo = p_c_mid
        else:
            hi = p_c_mid
    return None

# Find M_max precisely
def find_mmax(rho0):
    best = None
    for log_pc in np.linspace(33.0, 37.0, 200):
        p_c = 10**log_pc
        result = solve_tov_tidal(p_c, rho0, r_c, n=n, w_chi_inf=0.0, n_points=2000)
        if result and result['success'] and not result.get('at_boundary', False):
            if best is None or result['M_Msun'] > best['M_Msun']:
                best = result
    return best

print("=== M_max (with cs2_max) ===")
for rho0 in [0.0, 1e10, 3e10, 1e11, 3e11, 1e12, 3e12, 1e13, 3e13]:
    r = find_mmax(rho0)
    if r:
        print("rho0={:.1e}: M_max={:.4f} Msun, cs2_max={:.4f}".format(
            rho0, r["M_Msun"], r["cs2_max"]))

print()
print("=== Lambda at M=1.4 Msun (r_c={} km) ===".format(r_c))
for rho0 in [0.0, 1e10, 1e12, 1e13, 3e13]:
    r = find_at_mass(1.4, rho0)
    if r:
        print("rho0={:.1e}: M={:.4f}, Lambda={:.1f}, R={:.2f} km, cs2_max={:.4f}".format(
            rho0, r["M_Msun"], r["Lambda"], r["R_km"], r["cs2_max"]))
    else:
        print("rho0={:.1e}: could not find M=1.4 solution".format(rho0))

# Also compare: what Lambda does the CSV have at exactly 1.4?
print()
print("=== CSV: Lambda at M~1.4 Msun ===")
import csv
with open('tov_tidal_results.csv') as f:
    reader = list(csv.DictReader(f))
for rho0 in [0.0, 1e10, 1e12, 1e13, 3e13]:
    nearby = [r for r in reader 
              if float(r['rho0_gcm3']) == rho0 
              and r['success'] == 'True' 
              and r['at_boundary'] == 'False'
              and 1.38 <= float(r['M_Msun']) <= 1.42]
    if nearby:
        best = min(nearby, key=lambda x: abs(float(x['M_Msun']) - 1.4))
        print("CSV rho0={:.1e}: M={:.4f}, Lambda={:.1f}, R={:.2f}".format(
            rho0, float(best['M_Msun']), float(best['Lambda']), float(best['R_km'])))
    else:
        print("CSV rho0={:.1e}: no points in [1.38, 1.42]".format(rho0))
