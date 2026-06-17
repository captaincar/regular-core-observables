#!/usr/bin/env python3
"""
Tidal deformability GW170817 constraint plot.
Loads tov_tidal_results.csv and creates publication-quality Lambda(M) plot
with GW170817 excluded region shaded, NICER constraints overlaid.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

# Load results
results = []
with open(r'd:\DEV\fizyka\tov_tidal_results.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        r = {k: float(v) if v != 'True' and v != 'False' and v != '' and v != 'nan' else
             (True if v == 'True' else (False if v == 'False' else np.nan))
             for k, v in row.items()}
        results.append(r)

normal = [r for r in results if r['success'] and not r.get('at_boundary', False)]
rho0_vals = sorted(set(r['rho0_gcm3'] for r in normal))
r_c_vals = sorted(set(r['r_c_km'] for r in normal))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# ═══════════════════════════════════════════════════════════════════════════
# Panel 1: Lambda vs M with GW170817 excluded region
# ═══════════════════════════════════════════════════════════════════════════
ax = axes[0]
colors = plt.cm.plasma(np.linspace(0.05, 0.95, len(rho0_vals)))

# Shade GW170817 excluded region
# Lambda(1.4) < 580 at 90% CI, rising to ~800 for less conservative analyses
# We shade Lambda > 800 as excluded and Lambda > 580 as "disfavored"
M_range = np.logspace(np.log10(0.8), np.log10(2.5), 200)

# Conservative bound: Lambda(1.4) < 580 (Abbott+2018, 90% CI low-spin)
# This translates to a band that widens away from 1.4 Msun
# For simplicity, shade Lambda > 800 at M=1.4 as excluded
lam_bound = 580  # GW170817 90% CI
# Approximate scaling Lambda ~ (R/M)^5 / k2, so as M departs from 1.4, bound relaxes
# Use flat Lambda bound for demonstration (conservative)

# Shade excluded region (Lambda > bound for all M > 1.0)
ax.fill_between([1.0, 2.5], lam_bound, 1e6, color='red', alpha=0.12, label='GW170817 excluded (Λ > 580)')
ax.fill_between([1.0, 2.5], lam_bound, 1e6, color='red', alpha=0.06)

# GW170817 90% CI band at M=1.4
ax.axvspan(1.18, 1.60, ymin=0, ymax=np.log(lam_bound)/np.log(1e6), color='red', alpha=0.08)
ax.axhline(lam_bound, color='darkred', linestyle='--', linewidth=2, alpha=0.8)
ax.annotate('GW170817: Λ(1.4) < 580\n(90% CI, low-spin)', 
            xy=(1.4, 580), xytext=(1.55, 1500),
            fontsize=9, color='darkred', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkred', alpha=0.7))

# NICER M-R constraints
ax.axvline(2.08, color='blue', linestyle=':', alpha=0.5, linewidth=1)
ax.axvspan(1.6, 2.08, color='blue', alpha=0.05)
ax.annotate('PSR J0740+6620\nM = 2.08±0.07 M⊙', xy=(2.08, 5), xytext=(1.65, 3),
            fontsize=8, color='blue', alpha=0.8)

# Plot Lambda(M) curves
for r_c in r_c_vals[:2]:  # r_c = 2, 5 km (most relevant)
    for j, rho0 in enumerate(rho0_vals):
        if rho0 == 0.0:
            continue
        subset = [r for r in normal
                  if abs(r['rho0_gcm3'] - rho0) < 1e-10
                  and abs(r['r_c_km'] - r_c) < 1e-10]
        if len(subset) < 3:
            continue
        subset.sort(key=lambda x: x['M_Msun'])
        M_arr = np.array([s['M_Msun'] for s in subset])
        L_arr = np.array([s['Lambda'] for s in subset])
        
        label = f'ρ₀={rho0:.1e}' if rho0 < 1e12 else f'ρ₀={rho0/1e12:.1f}×10¹²'
        ls = '-' if r_c == 2.0 else '--'
        ax.loglog(M_arr, L_arr, ls, color=colors[j], alpha=0.8, linewidth=1.8,
                  label=f'{label}, r_c={r_c:.0f} km')

# Pure SLy reference
sly = [r for r in normal if r['rho0_gcm3'] == 0.0]
if sly:
    sly.sort(key=lambda x: x['M_Msun'])
    M_sly = np.array([s['M_Msun'] for s in sly])
    L_sly = np.array([s['Lambda'] for s in sly])
    ax.loglog(M_sly, L_sly, 'k-', linewidth=2.5, label='Pure SLy (ρ₀=0)', zorder=10)

ax.set_xlabel('Mass M (M⊙)', fontsize=13)
ax.set_ylabel('Tidal deformability Λ', fontsize=13)
ax.set_title('Tidal deformability vs mass — GW170817 constraint', fontsize=14, fontweight='bold')
ax.legend(fontsize=7, loc='lower left', ncol=2)
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(0.8, 2.5)
ax.set_ylim(1, 50000)

# ═══════════════════════════════════════════════════════════════════════════
# Panel 2: M-R diagram with NICER constraints
# ═══════════════════════════════════════════════════════════════════════════
ax2 = axes[1]

# NICER constraints on M-R
# PSR J0740+6620: M = 2.08±0.07, R = 12.39+1.30/-0.98 (Miller+2021)
# PSR J0030+0451: M = 1.44±0.15, R = 13.02+1.24/-1.06 (Riley+2019)

# J0740+6620
from matplotlib.patches import Ellipse
ell_j0740 = Ellipse((12.39, 2.08), width=2*1.14, height=2*0.07, 
                     edgecolor='blue', facecolor='blue', alpha=0.15, linewidth=2)
ax2.add_patch(ell_j0740)
ax2.annotate('PSR J0740+6620', xy=(12.39, 2.08), xytext=(13.5, 2.15),
             fontsize=9, color='blue', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='blue', alpha=0.6))

# J0030+0451
ell_j0030 = Ellipse((13.02, 1.44), width=2*1.15, height=2*0.15,
                     edgecolor='teal', facecolor='teal', alpha=0.12, linewidth=1.5)
ax2.add_patch(ell_j0030)
ax2.annotate('PSR J0030+0451', xy=(13.02, 1.44), xytext=(10.5, 1.35),
             fontsize=8, color='teal', alpha=0.8)

# GW170817 tidal constraint: requires R(1.4) ~ 11.9±1.4 km for SLy-like EOS
ax2.axhspan(1.17, 1.60, color='red', alpha=0.06)
ax2.annotate('GW170817\nM = 1.186–1.60 M⊙', xy=(9.5, 1.45), fontsize=8, color='darkred', alpha=0.8)

for r_c in r_c_vals[:2]:
    for j, rho0 in enumerate(rho0_vals):
        if len([r for r in normal if abs(r['rho0_gcm3'] - rho0) < 1e-10
                and abs(r['r_c_km'] - r_c) < 1e-10]) < 3:
            continue
        subset = [r for r in normal
                  if abs(r['rho0_gcm3'] - rho0) < 1e-10
                  and abs(r['r_c_km'] - r_c) < 1e-10]
        subset.sort(key=lambda x: x['M_Msun'])
        M_arr = np.array([s['M_Msun'] for s in subset])
        R_arr = np.array([s['R_km'] for s in subset])
        
        ls = '-' if r_c == 2.0 else '--'
        ax2.plot(R_arr, M_arr, ls, color=colors[j], alpha=0.8, linewidth=1.8)

# Pure SLy
if sly:
    sly.sort(key=lambda x: x['M_Msun'])
    R_sly = np.array([s['R_km'] for s in sly])
    ax2.plot(R_sly, M_sly, 'k-', linewidth=2.5, label='Pure SLy', zorder=10)

ax2.set_xlabel('Radius R (km)', fontsize=13)
ax2.set_ylabel('Mass M (M⊙)', fontsize=13)
ax2.set_title('Mass-Radius diagram — NICER constraints', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(8.5, 16)
ax2.set_ylim(0.8, 2.4)

plt.tight_layout()
plt.savefig(r'd:\DEV\fizyka\tidal_gw170817_constraint.png', dpi=200, bbox_inches='tight')
print("Saved: tidal_gw170817_constraint.png")
plt.close()
