#!/usr/bin/env python3
"""
Post-merger GW frequency prediction from TOV M-R curves.
Uses f2 ~ 2*f_max fitting (Bernuzzi+2015, Takami+2015) where
f_max ~ 6.5 kHz * (M_sun / M_tov) for equal-mass systems.
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
colors = plt.cm.plasma(np.linspace(0.05, 0.95, len(rho0_vals)))

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

# ═══════════════════════════════════════════════════════════════════════════
# Panel 1: f2 vs M
# ═══════════════════════════════════════════════════════════════════════════
ax = axes[0]

# Bernuzzi+2015, Takami+2015 fitting formulas:
# f_peak ~ 6.5 kHz * (M_sun/M_tov) for dominant post-merger oscillation
# f2 ~ 2 * f_peak in many numerical simulations
# More precisely: f2 = 2 * f_peak * (1 + correction)
# For TOV mass and compactness, f_peak ∝ sqrt(M/R^3) ∝ sqrt(C^3)/M

for r_c in r_c_vals[:2]:
    for j, rho0 in enumerate(rho0_vals):
        subset = [r for r in normal
                  if abs(r['rho0_gcm3'] - rho0) < 1e-10
                  and abs(r['r_c_km'] - r_c) < 1e-10]
        if len(subset) < 3:
            continue
        subset.sort(key=lambda x: x['M_Msun'])
        M_arr = np.array([s['M_Msun'] for s in subset])
        C_arr = np.array([s['C'] for s in subset])
        
        # Post-merger frequency from universal relations
        # f2 ≈ 2 * f_peak, f_peak from compactness via universal relation
        # Bernuzzi+2015 (PRD 91, 044056): f_peak(kHz) ~ C^{3/2}/M with
        # approximate coefficient ~12.8 (order-of-magnitude; systematic
        # uncertainty ~30% from EOS dependence, see also Bauswein&Janka 2012)
        # NOTE: 'Breschi+2019' cited in earlier version was wrong (that paper
        # covers phase transitions, not f2 universal relations).
        
        f2_kHz = 12.8 * C_arr**1.5 / M_arr
        
        ls = '-' if r_c == 2.0 else '--'
        label = f'ρ₀={rho0:.1e}' if rho0 < 1e12 else f'ρ₀={rho0/1e12:.1f}×10¹²'
        ax.semilogy(M_arr, f2_kHz, ls, color=colors[j], alpha=0.8, linewidth=1.8,
                    label=f'{label}, r_c={r_c:.0f} km')

# Pure SLy
sly = [r for r in normal if r['rho0_gcm3'] == 0.0]
if sly:
    sly.sort(key=lambda x: x['M_Msun'])
    M_sly = np.array([s['M_Msun'] for s in sly])
    C_sly = np.array([s['C'] for s in sly])
    f2_sly = 12.8 * C_sly**1.5 / M_sly
    ax.semilogy(M_sly, f2_sly, 'k-', linewidth=2.5, label='Pure SLy', zorder=10)

# Detector sensitivity bands
ax.axhspan(1000, 8000, color='green', alpha=0.08)
ax.annotate('LIGO A+ (O5)\nsensitivity band', xy=(1.0, 4000), fontsize=9, color='green', alpha=0.8)
ax.axhspan(100, 4000, color='purple', alpha=0.06)
ax.annotate('Einstein Telescope\npost-merger band', xy=(1.0, 200), fontsize=9, color='purple', alpha=0.8)

ax.set_xlabel('Mass M (M⊙)', fontsize=13)
ax.set_ylabel('Post-merger frequency f₂ (kHz)', fontsize=13)
ax.set_title('Post-merger GW frequency prediction', fontsize=14, fontweight='bold')
ax.legend(fontsize=7, loc='upper right', ncol=2)
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(0.8, 2.5)
ax.set_ylim(50, 10000)

# ═══════════════════════════════════════════════════════════════════════════
# Panel 2: f2 shift relative to SLy
# ═══════════════════════════════════════════════════════════════════════════
ax2 = axes[1]

if sly:
    sly_interp_C = np.interp  # will use later
    M_grid = np.linspace(0.9, 2.2, 100)
    C_sly_grid = np.interp(M_grid, M_sly, C_sly)
    f2_sly_grid = 12.8 * C_sly_grid**1.5 / M_grid
    
    for r_c in r_c_vals[:2]:
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
            C_arr = np.array([s['C'] for s in subset])
            f2_arr = 12.8 * C_arr**1.5 / M_arr
            
            # Interpolate f2 comparison at same M
            M_common = np.linspace(max(M_arr[0], 0.9), min(M_arr[-1], 2.2), 50)
            f2_interp = np.interp(M_common, M_arr, f2_arr)
            f2_sly_interp = np.interp(M_common, M_sly, f2_sly)
            
            df2_pct = (f2_interp - f2_sly_interp) / f2_sly_interp * 100
            
            ls = '-' if r_c == 2.0 else '--'
            ax2.plot(M_common, df2_pct, ls, color=colors[j], alpha=0.8, linewidth=1.8)

ax2.axhline(0, color='k', linestyle=':', alpha=0.3)
ax2.axhline(10, color='red', linestyle='--', alpha=0.3, linewidth=1)
ax2.axhline(-10, color='red', linestyle='--', alpha=0.3, linewidth=1)
ax2.annotate('±10% — typical ET\npost-merger precision', xy=(1.0, 12), fontsize=8, color='red', alpha=0.6)

ax2.set_xlabel('Mass M (M⊙)', fontsize=13)
ax2.set_ylabel('Δf₂ / f₂(SLy) (%)', fontsize=13)
ax2.set_title('Post-merger frequency shift from χ sector', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0.9, 2.2)

plt.tight_layout()
plt.savefig(r'd:\DEV\fizyka\post_merger_f2.png', dpi=200, bbox_inches='tight')
print("Saved: post_merger_f2.png")

# Print key numbers
print("\nKEY PREDICTIONS:")
print("=" * 60)
for r_c in r_c_vals[:1]:
    for rho0 in rho0_vals:
        if rho0 == 0.0:
            continue
        subset = [r for r in normal
                  if abs(r['rho0_gcm3'] - rho0) < 1e-10
                  and abs(r['r_c_km'] - r_c) < 1e-10]
        if len(subset) < 3:
            continue
        subset.sort(key=lambda x: x['M_Msun'])
        M_arr = np.array([s['M_Msun'] for s in subset])
        C_arr = np.array([s['C'] for s in subset])
        f2_arr = 12.8 * C_arr**1.5 / M_arr
        
        if M_arr[0] <= 1.4 <= M_arr[-1]:
            f2_14 = np.interp(1.4, M_arr, f2_arr)
            f2_sly_14 = np.interp(1.4, M_sly, f2_sly) if sly else 0
            df2 = (f2_14 - f2_sly_14) / f2_sly_14 * 100
            print(f"r_c={r_c:.0f} km, rho0={rho0:.1e}: f2(1.4)={f2_14:.3f} kHz, Delta = {df2:+.1f}%")

plt.close()
