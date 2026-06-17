#!/usr/bin/env python3
"""Causality sweep across the chi-sector parameter space."""
import numpy as np
import tov_tidal as tt

rho0_vals = np.logspace(0, 14, 8)
rho0_vals = np.concatenate([[0], rho0_vals])
rc_vals = [2, 5, 10]
p_c_vals = np.logspace(33.5, 36.0, 20)

print(f"{'rho0':>12s} {'rc':>4s} {'p_c':>10s} {'M_Msun':>8s} {'cs2_max':>8s} {'CAUSAL?':>8s}")
print("-" * 55)

violations = []
all_cs2 = []
for rho0 in rho0_vals:
    for rc in rc_vals:
        for p_c in p_c_vals:
            sol = tt.solve_tov_tidal(p_c, rho0_cgs=rho0, r_c_km=rc, n=2.0)
            if sol is None:
                continue
            cs2 = sol["cs2_max"]
            all_cs2.append(cs2)
            if cs2 > 1.0:
                violations.append((rho0, rc, p_c, sol["M_Msun"], cs2))
                print(f"{rho0:12.1e} {rc:4d} {p_c:10.1e} {sol['M_Msun']:8.4f} {cs2:8.4f}   ***VIOLATION***")
            elif cs2 > 0.95:
                print(f"{rho0:12.1e} {rc:4d} {p_c:10.1e} {sol['M_Msun']:8.4f} {cs2:8.4f}   marginal")

print(f"\nTotal stable configs checked: {len(all_cs2)}")
print(f"cs2_max range: [{np.min(all_cs2):.4f}, {np.max(all_cs2):.4f}]")
print(f"Violations (cs2 > 1): {len(violations)}")

if violations:
    print("\n*** CAUSALITY VIOLATIONS ***")
    for v in violations[:15]:
        print(f"  rho0={v[0]:.1e} rc={v[1]} p_c={v[2]:.1e} M={v[3]:.4f} cs2={v[4]:.4f}")
elif np.max(all_cs2) > 0.99:
    print(f"\nAll configurations EXACTLY causal. Max cs2 = {np.max(all_cs2):.4f} <= 1.0")
    print("(Note: cs2 approaches 1 at highest central densities, as expected for SLy EOS)")
else:
    print(f"\nAll configurations comfortably causal. Max cs2 = {np.max(all_cs2):.4f}")

# Summary by rho0
print(f"\n{'rho0':>12s}  cs2_max_across_all  cs2_median")
for rho0 in rho0_vals:
    cs2_vals = []
    for rc in rc_vals:
        for p_c in p_c_vals:
            sol = tt.solve_tov_tidal(p_c, rho0_cgs=rho0, r_c_km=rc, n=2.0)
            if sol:
                cs2_vals.append(sol["cs2_max"])
    if cs2_vals:
        print(f"{rho0:12.1e}  {np.max(cs2_vals):16.4f}  {np.median(cs2_vals):11.4f}")
