#!/usr/bin/env python3
"""
Conceptual diagram for the manuscript: black hole with three observational channels.
Simple schematic showing ringdown (QNM shift), tides (Lambda), and echoes.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Arc, Wedge
import matplotlib.patheffects as pe

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.set_aspect('equal')
ax.axis('off')

# Background
fig.patch.set_facecolor('#0a0a2e')

# Central black hole — dark circle with glow
bh_center = (6, 4)
bh_radius = 1.5

# Accretion disk glow
for i, r_mult in enumerate(np.linspace(1.0, 2.5, 15)):
    alpha = 0.03 + 0.02 * (1 - i/15)
    glow = Circle(bh_center, bh_radius * r_mult, 
                  facecolor='none', edgecolor='#ff6600', 
                  linewidth=3*(1-i/15), alpha=alpha)
    ax.add_patch(glow)

# Photon sphere ring
ps_ring = Circle(bh_center, bh_radius * 1.5, 
                 facecolor='none', edgecolor='#ffaa00', 
                 linewidth=2, linestyle='--', alpha=0.6)
ax.add_patch(ps_ring)
ax.annotate('photon sphere', xy=(bh_center[0] + 2.25, bh_center[1] + 0.3),
            fontsize=8, color='#ffaa00', alpha=0.7, rotation=15)

# Black hole silhouette
bh = Circle(bh_center, bh_radius, facecolor='#000011', edgecolor='#330066', linewidth=2)
ax.add_patch(bh)

# Core — fuzzy blue glow
core = Circle(bh_center, 0.4, facecolor='#4488ff', edgecolor='#88bbff', 
              linewidth=1.5, alpha=0.5)
ax.add_patch(core)
ax.annotate('core\n(L, ρ₀)', xy=(bh_center[0] - 0.3, bh_center[1] + 0.05),
            fontsize=8, color='#88bbff', ha='center', va='center', fontweight='bold')

# ── Channel 1: Ringdown (QNM) ──
# Arrow pointing to upper-right
ring_center = (8.5, 6.0)
ring_box = FancyBboxPatch((ring_center[0]-1.8, ring_center[1]-0.8), 3.6, 1.6,
                          boxstyle="round,pad=0.1", facecolor='#1a1a4e', 
                          edgecolor='#4488ff', linewidth=1.5, alpha=0.85)
ax.add_patch(ring_box)

ax.annotate('RINGDOWN', xy=(ring_center[0], ring_center[1]+0.3),
            fontsize=13, color='#4488ff', ha='center', fontweight='bold')
ax.annotate(r'$|\Delta f/f| \approx 0.049\,(L/M)^2$', xy=(ring_center[0], ring_center[1]-0.1),
            fontsize=10, color='white', ha='center')
ax.annotate('Current: no constraint (GW150914)\nET/CE: L/M > 0.045 detectable',
            xy=(ring_center[0], ring_center[1]-0.5),
            fontsize=8, color='#aaaacc', ha='center')

# Connecting wave patterns from BH to ringdown
x_wave = np.linspace(bh_center[0] + bh_radius, ring_center[0] - 1.8, 50)
y_wave = bh_center[1] + bh_radius * 0.5 + 0.3 * np.sin(x_wave * 4)
ax.plot(x_wave, y_wave + 1.2, color='#4488ff', linewidth=1.5, alpha=0.6)

# ── Channel 2: Tides (Lambda) ──
tide_center = (8.5, 2.5)
tide_box = FancyBboxPatch((tide_center[0]-1.8, tide_center[1]-0.8), 3.6, 1.6,
                          boxstyle="round,pad=0.1", facecolor='#1a4e1a',
                          edgecolor='#44ff44', linewidth=1.5, alpha=0.85)
ax.add_patch(tide_box)

ax.annotate('TIDES', xy=(tide_center[0], tide_center[1]+0.3),
            fontsize=13, color='#44ff44', ha='center', fontweight='bold')
ax.annotate(r'$\Lambda(1.4\,M_\odot)$ drops 571 → 50', xy=(tide_center[0], tide_center[1]-0.1),
            fontsize=10, color='white', ha='center')
ax.annotate('Current: GW170817 constrains ρ₀\nNICER constrains M–R',
            xy=(tide_center[0], tide_center[1]-0.5),
            fontsize=8, color='#aaccaa', ha='center')

# Tidal deformation illustration — ellipsoidal companion
comp_center = (9.8, 1.2)
companion = Circle(comp_center, 0.5, facecolor='#224422', edgecolor='#44ff44', 
                   linewidth=1.5, alpha=0.7)
ax.add_patch(companion)
# Tidal bulge
from matplotlib.patches import Ellipse
bulge = Ellipse((comp_center[0] - 0.15, comp_center[1] + 0.05), 0.3, 0.7,
                angle=30, facecolor='none', edgecolor='#44ff44', linewidth=1, alpha=0.5)
ax.add_patch(bulge)

# ── Channel 3: Echoes ──
echo_center = (3.5, 6.0)
echo_box = FancyBboxPatch((echo_center[0]-1.8, echo_center[1]-0.8), 3.6, 1.6,
                          boxstyle="round,pad=0.1", facecolor='#4e1a1a',
                          edgecolor='#ff4444', linewidth=1.5, alpha=0.85)
ax.add_patch(echo_box)

ax.annotate('ECHOES', xy=(echo_center[0], echo_center[1]+0.3),
            fontsize=13, color='#ff4444', ha='center', fontweight='bold')
ax.annotate(r'$L_{\rm echo} = L_{\rm ext} = 0.770\,M$', xy=(echo_center[0], echo_center[1]-0.1),
            fontsize=10, color='white', ha='center')
ax.annotate('Echoes → no horizon at all\nNot a black hole test',
            xy=(echo_center[0], echo_center[1]-0.5),
            fontsize=8, color='#ccaaaa', ha='center')

# Echo pulse pattern
for j in range(3):
    y_pos = bh_center[1] + 0.3 - j * 0.4
    pulse = Circle((bh_center[0] - 1.8, y_pos), 0.08 + j*0.02,
                   facecolor='#ff6666', alpha=0.4 - j*0.1)
    ax.add_patch(pulse)

# ── Connecting lines ──
for target_x, color in [(ring_center[0] - 1.8, '#4488ff'), 
                          (tide_center[0] - 1.8, '#44ff44'),
                          (echo_center[0] + 1.8, '#ff4444')]:
    # Dotted radial lines
    pass

# ── Title ──
ax.annotate('Three Ways to Catch a Fuzzy Core', xy=(6, 7.5),
            fontsize=18, color='white', ha='center', fontweight='bold',
            path_effects=[pe.withStroke(linewidth=3, foreground='#000033')])

# ── Legend / key at bottom ──
key_y = 0.4
key_items = [
    ('#4488ff', 'Ringdown: listen to the vibration (future)'),
    ('#44ff44', 'Tides: measure the squeezing (current)'),
    ('#ff4444', 'Echoes: look for horizon loss (ruled out)'),
]
for i, (color, label) in enumerate(key_items):
    x_start = 1.5 + i * 4
    ax.plot([x_start, x_start + 0.8], [key_y, key_y], color=color, linewidth=3)
    ax.annotate(label, xy=(x_start + 1.0, key_y), fontsize=9, color='#cccccc', va='center')

plt.tight_layout()
plt.savefig(r'd:\DEV\fizyka\conceptual_diagram.png', dpi=200, bbox_inches='tight',
            facecolor='#0a0a2e', edgecolor='none')
print("Saved: conceptual_diagram.png")
plt.close()
