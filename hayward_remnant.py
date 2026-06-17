#!/usr/bin/env python3
"""Hayward black hole remnant: T_H(M) and extremal mass for chi-sector length scale L."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Constants
G_SI = 6.67430e-11
c_SI = 2.99792458e8
kB_SI = 1.380649e-23
hbar_SI = 1.054571817e-34
Msun_kg = 1.98847e30
Mp_kg = 2.176434e-8  # Planck mass

def hayward_fprime_at_horizon(M, L):
    """
    Compute f'(r_h) for Hayward metric f(r) = 1 - 2Mr^2/(r^3 + 2ML^2).
    M, L in geometric units (meters).
    Returns (r_h, f'(r_h)) in 1/meters.
    """
    # Horizon cubic: r^3 - 2M r^2 + 2M L^2 = 0
    coeffs = [1.0, -2*M, 0.0, 2*M*L**2]
    roots = np.roots(coeffs)
    # Pick the largest real root > 0 (outer horizon)
    real_roots = sorted([r.real for r in roots if abs(r.imag) < 1e-15 and r.real > 0])
    if not real_roots:
        return 0.0, 0.0
    r_h = real_roots[-1]  # outermost
    # f'(r_h) = 2Mr_h(r_h^3 - 4ML^2) / (r_h^3 + 2ML^2)^2
    r3 = r_h**3
    denom = r3 + 2*M*L**2
    fp = 2 * M * r_h * (r3 - 4*M*L**2) / denom**2
    return r_h, fp

def T_Hawking(M_g, L_m):
    """Temperature in Kelvin. M_g = mass in grams, L_m = L in meters."""
    M_kg = M_g * 1e-3
    L_m_eff = L_m
    # Convert to geometric
    M_geom = G_SI * M_kg / c_SI**2  # meters
    _, fp = hayward_fprime_at_horizon(M_geom, L_m_eff)
    if fp <= 0:
        return 0.0
    T = hbar_SI * c_SI * fp / (4 * np.pi * kB_SI)
    return T

# Analytical: M_ext = 3*sqrt(3)/4 * L ~ 1.29904 * L
L_vals_m = np.array([1.616e-35, 1e-15, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2])
L_labels = ["l_Planck", "1 fm", "1 pm", "1 A", "100 A", "1 um", "100 um", "1 cm", "1 m", "100 m"]
M_ext_geom = 3 * np.sqrt(3) / 4 * L_vals_m  # meters
M_ext_kg = M_ext_geom * c_SI**2 / G_SI
M_ext_Msun = M_ext_kg / Msun_kg

print("=" * 70)
print("HAYWARD BLACK HOLE REMNANT ANALYSIS")
print("=" * 70)
print()
print("Hayward metric: f(r) = 1 - 2Mr^2/(r^3 + 2ML^2)")
print("Extremal condition: T_H = 0  =>  r_h^3 = 4ML^2")
print("Extremal mass: M_ext = 3*sqrt(3)/4 * L  ~  1.299 * L")
print()
print(f"{'L scale':>12s}  {'L (m)':>12s}  {'M_ext (kg)':>14s}  {'M_ext (Msun)':>14s}")
print("-" * 56)
for i, L in enumerate(L_vals_m):
    print(f"{L_labels[i]:>12s}  {L:12.4e}  {M_ext_kg[i]:14.4e}  {M_ext_Msun[i]:14.4e}")

# Physical interpretation
print()
print("=" * 70)
print("PHYSICAL INTERPRETATION")
print("=" * 70)
l_planck = 1.616e-35
m_planck_kg = 2.176e-8
print(f"  L = l_Planck ({l_planck:.1e} m): M_rem ~ 1.3 m_Planck ~ 2.8e-8 kg")
print("    -> Planck-scale remnant, inaccessible to all observations")
print(f"  L = 1 fm = 1e-15 m: M_rem = {3*np.sqrt(3)/4*1e-15*c_SI**2/G_SI:.2e} kg = {3*np.sqrt(3)/4*1e-15*c_SI**2/(G_SI*Msun_kg):.2e} Msun")
print("    -> ~10^12 kg (~10^15 g), primordial black hole dark matter range")
print("    -> If L ~ QCD scale, remnants could constitute dark matter!")
print(f"  L = 1 mm = 1e-3 m: M_rem = {3*np.sqrt(3)/4*1e-3*c_SI**2/G_SI:.2e} kg = {3*np.sqrt(3)/4*1e-3*c_SI**2/(G_SI*Msun_kg):.2e} Msun")
print("    -> ~10^23 kg, asteroid-mass remnant")
print(f"  L = 1 m: M_rem = {3*np.sqrt(3)/4*1.0*c_SI**2/G_SI:.2e} kg = {3*np.sqrt(3)/4*1.0*c_SI**2/(G_SI*Msun_kg):.2e} Msun")
print("    -> ~10^26 kg, super-Earth mass remnant -> excluded by dynamics")

# T_H(M) curve for L = 1e-15 m (QCD scale)
print()
print("=" * 70)
print("T_H(M) FOR L = 1 fm (QCD SCALE)")
print("=" * 70)
L_fm = 1e-15
M_ext_fm_g = M_ext_kg[1] * 1000  # grams
M_vals_g = np.logspace(np.log10(M_ext_fm_g * 1.001), np.log10(M_ext_fm_g * 1000), 100)
print(f"M_ext = {M_ext_fm_g:.2e} g = {M_ext_kg[1]:.2e} kg")
print(f"{'M (g)':>14s}  {'M/M_ext':>10s}  {'T_H (K)':>14s}")
print("-" * 42)
for M_g in [M_ext_fm_g * 1.001, M_ext_fm_g * 1.01, M_ext_fm_g * 1.1, M_ext_fm_g * 2, M_ext_fm_g * 10, M_ext_fm_g * 100]:
    T = T_Hawking(M_g, L_fm)
    print(f"{M_g:14.4e}  {M_g/M_ext_fm_g:10.4f}  {T:14.4e}")

# Plot T_H(M) for several L values
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: T_H(M) for various L
L_plot_vals = [1e-15, 1e-12, 1e-10, 1e-8, 1e-6]
L_plot_labels = ["L=1 fm", "L=1 pm", "L=1 A", "L=100 A", "L=1 um"]
colors = plt.cm.viridis(np.linspace(0, 0.9, len(L_plot_vals)))
for i, L in enumerate(L_plot_vals):
    M_ext_g = 3*np.sqrt(3)/4 * L * c_SI**2 / G_SI * 1000
    M_v = np.logspace(np.log10(M_ext_g * 1.0001), np.log10(M_ext_g * 100), 200)
    T_v = np.array([T_Hawking(M, L) for M in M_v])
    ax1.loglog(M_v, T_v, color=colors[i], label=L_plot_labels[i])
    ax1.axvline(M_ext_g, color=colors[i], ls=':', alpha=0.5)
ax1.set_xlabel('M (g)')
ax1.set_ylabel('T_H (K)')
ax1.set_title('Hayward Black Hole Temperature')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Panel 2: M_ext vs L
L_range = np.logspace(-35, 2, 100)
M_ext_range = 3*np.sqrt(3)/4 * L_range * c_SI**2 / G_SI
M_ext_Msun_range = M_ext_range / Msun_kg
ax2.loglog(L_range, M_ext_Msun_range, 'b-', lw=2)
# Annotations
ax2.axhline(1.0, color='gray', ls='--', alpha=0.5, label='1 Msun')
ax2.axhline(1e-15, color='red', ls='--', alpha=0.5, label='PBH DM lower bound')
ax2.axhline(1e-10, color='orange', ls='--', alpha=0.5, label='PBH DM upper bound')
ax2.set_xlabel('L (m)')
ax2.set_ylabel('M_ext (Msun)')
ax2.set_title('Extremal Mass vs. chi-sector Length Scale')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hayward_remnant.png', dpi=150)
print("\nSaved hayward_remnant.png")
print("Done.")
