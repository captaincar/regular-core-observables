#!/usr/bin/env python3
"""
TOV chi-sector solver with tidal deformability (Lambda).
Adds Hinderer y-ODE to the two-fluid TOV integration.
Computes Love number k2 and dimensionless tidal deformability Lambda.

Physical units -> geometric conversion for y-ODE.
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

# =========================================================================
# Physical constants (cgs)
# =========================================================================
G = 6.67430e-8       # cm^3/(g s^2)
c = 2.99792458e10    # cm/s
Msun = 1.98847e33    # g
km = 1e5             # cm
Gc2 = G / c**2       # cm/g  — converts mass to length
Gc4 = G / c**4       # cm·s²/g — converts pressure/energy density to cm^{-2}

# =========================================================================
# Piecewise polytropes — Read et al. 2009, PRD 79, 124032, Table II
# =========================================================================
# Segment boundaries (g/cm^3); same for all EOS in the Read parameterisation.
# =========================================================================
RHO_CRUST_CORE = 2.4e14   # crust → core ~ nuclear saturation
RHO_DIV1       = 5.0e14   # core stiff → intermediate
RHO_DIV2       = 1.0e15   # intermediate → soft

# -------------------------------------------------------------------------
# EOS parameter table — K_CORE1 is calibrated for each EOS to reproduce
# the tabulated M_max (Read et al. Table II).
# -------------------------------------------------------------------------
EOS_PARAMS = {
    'sly': {
        'label':        'SLy',
        'gamma_crust':  1.357,
        'gamma_1':      3.005,
        'gamma_2':      2.988,
        'gamma_3':      2.851,
        'K1_target':    2.1098e-10, # calibrated → M_max = 2.050 M⊙ (refined)
        'M_max_lit':    2.05,       # Read et al. Table II
        'ref':          'Read+2009 (Douchin-Haensel 2001 SLy)',
    },
    'apr4': {
        'label':        'APR4',
        'gamma_crust':  1.357,
        'gamma_1':      2.830,
        'gamma_2':      3.445,
        'gamma_3':      3.348,
        'K1_target':    8.3104e-08, # calibrated → M_max = 2.200 M⊙ (refined)
        'M_max_lit':    2.20,       # Read et al. Table II
        'ref':          'Read+2009 (Akmal-Pandharipande-Ravenhall APR4)',
    },
    'ms1b': {
        'label':        'MS1b',
        'gamma_crust':  1.357,
        'gamma_1':      3.456,
        'gamma_2':      3.011,
        'gamma_3':      1.425,
        'K1_target':    1.5115e-16, # calibrated → M_max = 2.801 M⊙ (refined; piecewise approx limit)
        'M_max_lit':    2.76,       # Read et al. Table II
        'ref':          'Read+2009 (Müller-Serot MS1b)',
    },
    'h4': {
        'label':        'H4',
        'gamma_crust':  1.357,
        'gamma_1':      2.909,
        'gamma_2':      2.246,
        'gamma_3':      2.144,
        'K1_target':    8.4944e-09, # calibrated → M_max = 2.029 M⊙ (refined)
        'M_max_lit':    2.03,       # Read et al. Table II
        'ref':          'Read+2009 (Lackey-Nayyar-Owen H4)',
    },
}

# Active EOS — swapped by setup_eos()
EOS_NAME = 'sly'
_EOS = EOS_PARAMS[EOS_NAME]

# Pre-computed globals (populated by setup_eos)
GAMMA_CRUST = GAMMA_CORE1 = GAMMA_CORE2 = GAMMA_CORE3 = 1.0
K_CRUST = K_CORE1 = K_CORE2 = K_CORE3 = 1.0
P_CC = P_DIV1 = P_DIV2 = 0.0


def setup_eos(eos_name='sly', tune_K1=False, n_points_mmax=4000):
    """Select EOS and (optionally) auto-calibrate K_CORE1 to hit M_max_lit.

    Setting tune_K1=True runs a binary search on K_CORE1 so that the
    pure-EOS M_max matches the literature target within 0.005 M⊙.
    Side-effect: updates the global EOS_PARAMS entry.
    """
    global EOS_NAME, _EOS
    global GAMMA_CRUST, GAMMA_CORE1, GAMMA_CORE2, GAMMA_CORE3
    global K_CORE1, K_CRUST, K_CORE2, K_CORE3
    global P_CC, P_DIV1, P_DIV2

    eos_name = eos_name.lower()
    if eos_name not in EOS_PARAMS:
        raise ValueError(f"Unknown EOS: {eos_name}. Choices: {list(EOS_PARAMS.keys())}")
    EOS_NAME = eos_name
    _EOS = EOS_PARAMS[eos_name]

    GAMMA_CRUST = _EOS['gamma_crust']
    GAMMA_CORE1 = _EOS['gamma_1']
    GAMMA_CORE2 = _EOS['gamma_2']
    GAMMA_CORE3 = _EOS['gamma_3']
    K_CORE1 = _EOS['K1_target']

    if tune_K1:
        target = _EOS['M_max_lit']
        print(f"  Auto-calibrating K_CORE1 for {_EOS['label']} -> M_max ~ {target:.2f} Msun ...")
        # Phase 1: Binary search on log10(K1) with coarse grid (fast)
        lo, hi = -18.0, -5.0
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            K_CORE1 = 10.0 ** mid
            _setup_continuity()
            mm = _pure_mmax_quick(n_points_mmax, refine=False)
            if mm is None or mm < target:
                lo = mid
            else:
                hi = mid
        # Phase 2: Refine with golden-section (slower but accurate)
        lo, hi = lo - 0.5, hi + 0.5  # widen range for safety
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            K_CORE1 = 10.0 ** mid
            _setup_continuity()
            mm = _pure_mmax_quick(n_points_mmax, refine=True)
            if mm is None or mm < target:
                lo = mid
            else:
                hi = mid
        K_CORE1 = 10.0 ** (0.5 * (lo + hi))
        EOS_PARAMS[eos_name]['K1_target'] = K_CORE1
        _EOS['K1_target'] = K_CORE1
        # Final verified check
        _setup_continuity()
        mm_final = _pure_mmax_quick(n_points_mmax, refine=True)
        print(f"    K_CORE1 = {K_CORE1:.4e} -> M_max = {mm_final:.3f} Msun")

    _setup_continuity()
    print(f"  EOS: {_EOS['label']}  ({_EOS['ref']})")
    print(f"    K_CORE1 = {K_CORE1:.4e}")


def _setup_continuity():
    """Recompute K_CRUST, K_CORE2, K_CORE3 from K_CORE1 and the Gamma chain."""
    global K_CRUST, K_CORE2, K_CORE3
    global P_CC, P_DIV1, P_DIV2
    P_CC   = K_CORE1 * RHO_CRUST_CORE ** GAMMA_CORE1
    K_CRUST = P_CC / (RHO_CRUST_CORE ** GAMMA_CRUST)
    P_DIV1 = K_CORE1 * RHO_DIV1 ** GAMMA_CORE1
    K_CORE2 = P_DIV1 / (RHO_DIV1 ** GAMMA_CORE2)
    P_DIV2 = K_CORE2 * RHO_DIV2 ** GAMMA_CORE2
    K_CORE3 = P_DIV2 / (RHO_DIV2 ** GAMMA_CORE3)


def _golden_section_mmax(log10_pc_best, rho0_cgs=0.0, r_c_km=2.0, n=2,
                         n_points=4000, tol_log10=1e-3):
    """Refine M_max via golden-section search on log10(p_c).

    Parameters
    ----------
    log10_pc_best : float
        log10 of the best p_c from a coarse grid scan.
    Returns (M_max, log10_pc_opt) or (None, None) on failure.
    """
    phi = (np.sqrt(5) - 1) / 2  # golden ratio conjugate, ~0.618
    a = log10_pc_best - 0.30  # factor ~0.5 in p_c
    c = log10_pc_best + 0.30  # factor ~2.0 in p_c
    b = c - phi * (c - a)
    d = a + phi * (c - a)

    # Evaluate initial test points
    def _eval(lp):
        sol = solve_tov_tidal_nosweep(10**lp, rho0_cgs, r_c_km, n, n_points=n_points)
        return sol['M_Msun'] if sol else 0.0

    Mb = _eval(b)
    Md = _eval(d)

    max_iter = 40
    for _ in range(max_iter):
        if Mb > Md:
            c, d = d, b
            Md = Mb
            b = c - phi * (c - a)
            Mb = _eval(b)
        else:
            a, b = b, d
            Mb = Md
            d = a + phi * (c - a)
            Md = _eval(d)

        if abs(c - a) < tol_log10:
            break

    # Best point is at the bracket midpoint
    log10_mid = 0.5 * (a + c)
    M_best = _eval(log10_mid)
    return (M_best, log10_mid) if M_best > 0 else (None, None)


def _pure_mmax_quick(n_points, refine=True):
    """M_max for pure-EOS calibration — coarse grid + optional golden-section refinement."""
    # Coarse grid scan
    p_c_vals = np.logspace(34.3, 37.8, 30)
    best = 0.0
    best_log10 = 34.3
    for p_c in p_c_vals:
        sol = solve_tov_tidal_nosweep(p_c, rho0_cgs=0.0, r_c_km=2.0, n=2,
                                       n_points=n_points)
        if sol and sol['M_Msun'] > best:
            best = sol['M_Msun']
            best_log10 = np.log10(p_c)

    if best <= 0 or not refine:
        return best if best > 0 else None

    # Golden-section refinement
    M_refined, _ = _golden_section_mmax(best_log10, rho0_cgs=0.0, r_c_km=2.0,
                                         n=2, n_points=n_points)
    return M_refined if M_refined else best


def solve_tov_tidal_nosweep(p_c_cgs, rho0_cgs, r_c_km, n=2,
                             w_chi_inf=0.0, r_max_km=200.0, n_points=4000):
    """Single-integration wrapper — same as solve_tov_tidal but returns None
    on BH gracefully (used for calibration sweeps)."""
    try:
        return solve_tov_tidal(p_c_cgs, rho0_cgs, r_c_km, n,
                                w_chi_inf=w_chi_inf, r_max_km=r_max_km,
                                n_points=n_points)
    except Exception:
        return None


def eos_rho_from_p(p):
    if p <= 0:
        return 0.0
    if p <= P_CC:
        return (p / K_CRUST) ** (1.0 / GAMMA_CRUST)
    elif p <= P_DIV1:
        return (p / K_CORE1) ** (1.0 / GAMMA_CORE1)
    elif p <= P_DIV2:
        return (p / K_CORE2) ** (1.0 / GAMMA_CORE2)
    else:
        return (p / K_CORE3) ** (1.0 / GAMMA_CORE3)


def eos_p_from_rho(rho):
    if rho <= 0:
        return 0.0
    if rho <= RHO_CRUST_CORE:
        return K_CRUST * rho**GAMMA_CRUST
    elif rho <= RHO_DIV1:
        return K_CORE1 * rho**GAMMA_CORE1
    elif rho <= RHO_DIV2:
        return K_CORE2 * rho**GAMMA_CORE2
    else:
        return K_CORE3 * rho**GAMMA_CORE3


def eos_eps_from_rho(rho):
    p = eos_p_from_rho(rho)
    if rho <= RHO_CRUST_CORE:
        gamma = GAMMA_CRUST
    elif rho <= RHO_DIV1:
        gamma = GAMMA_CORE1
    elif rho <= RHO_DIV2:
        gamma = GAMMA_CORE2
    else:
        gamma = GAMMA_CORE3
    return rho * c**2 + p / (gamma - 1.0)


def eos_eps_from_p(p):
    rho = eos_rho_from_p(p)
    if rho <= 0:
        return 0.0
    return eos_eps_from_rho(rho)


def eos_deps_dp(p):
    """d(eps)/dp for the matter component (needed for c_s^2).

    eps = rho*c^2 + p/(gamma-1) where rho = (p/K)^(1/gamma)
    deps/dp = c^2 * drho/dp + 1/(gamma-1)
             = c^2 * rho / (gamma * p) + 1/(gamma-1)
    """
    rho = eos_rho_from_p(p)
    if rho <= 0:
        return 1.0  # avoid div by zero
    if rho <= RHO_CRUST_CORE:
        gamma = GAMMA_CRUST
    elif rho <= RHO_DIV1:
        gamma = GAMMA_CORE1
    elif rho <= RHO_DIV2:
        gamma = GAMMA_CORE2
    else:
        gamma = GAMMA_CORE3
    return c**2 * rho / (gamma * max(p, 1e-30)) + 1.0 / (gamma - 1.0)


# =========================================================================
# Chi-sector profile
# =========================================================================

def chi_profile(r_cm, r_c, n):
    rr = np.maximum(r_cm, 1e-30)
    return 1.0 / (1.0 + (rr / r_c)**n)


def w_chi_profile(r_cm, r_c, n, w_chi_inf=0.0):
    step = chi_profile(r_cm, r_c, n)
    return -1.0 + (1.0 + w_chi_inf) * (1.0 - step)


# =========================================================================
# Two-fluid TOV with tidal y-ODE
# =========================================================================

def solve_tov_tidal(p_c_cgs, rho0_cgs, r_c_km, n=2, w_chi_inf=0.0,
                    r_max_km=200.0, n_points=4000):
    """
    Integrate two-fluid TOV + Hinderer y-ODE in physical units.

    Returns dict with M_R, Lambda, etc. or None if BH.
    """
    r_c_cm = r_c_km * km
    r_max_cm = r_max_km * km

    # Grid
    n_inner = n_points // 2
    r_inner = np.logspace(np.log10(1.0), np.log10(r_c_cm), n_inner)
    r_outer = np.linspace(r_c_cm * 1.01, r_max_cm, n_points - n_inner)
    r = np.concatenate([r_inner, r_outer])
    dr = np.diff(r)
    dr = np.append(dr, dr[-1])

    # Allocate arrays
    m = np.zeros(n_points)
    p_m = np.zeros(n_points)
    p_chi = np.zeros(n_points)
    y = np.zeros(n_points)
    cs2_arr = np.zeros(n_points)  # for causality check

    # Initial conditions
    r0 = r[0]
    eps_m0 = eos_eps_from_p(p_c_cgs)
    m[0] = (4.0 / 3.0) * np.pi * r0**3 * (eps_m0 / c**2 + rho0_cgs / c**2)
    p_m[0] = p_c_cgs
    p_chi[0] = -rho0_cgs * c**2
    y[0] = 2.0  # regular solution at r->0
    cs2_arr[0] = 0.0

    surface_idx = n_points

    for i in range(n_points - 1):
        ri = r[i]
        mi = m[i]
        pm_i = p_m[i]
        pchi_i = p_chi[i]
        yi = y[i]
        h = dr[i]

        # Horizon check
        if mi / max(ri, 1e-30) >= c**2 / (2.0 * G):
            surface_idx = i + 1
            break

        def rhs(_r, _m, _pm, _pchi, _y):
            _r = max(_r, 1.0)

            # --- TOV in physical units ---
            f_val = max(1.0 - 2.0 * G * _m / (c**2 * _r), 1e-15)

            chi_val = chi_profile(np.array([_r]), r_c_cm, n)[0]
            rho_chi_eps = rho0_cgs * chi_val * c**2  # energy density (erg/cm^3)
            w_val = w_chi_profile(np.array([_r]), r_c_cm, n, w_chi_inf)[0]
            p_chi_val = w_val * rho_chi_eps

            eps_m_val = eos_eps_from_p(_pm)
            p_total = _pm + _pchi
            eps_total = eps_m_val + rho_chi_eps

            # dm/dr, dp_m/dr, dp_chi/dr
            m_term = _m + 4.0 * np.pi * _r**3 * p_total / c**2
            common = G * m_term / (c**2 * _r**2 * f_val)

            dm_dr = 4.0 * np.pi * _r**2 * eps_total / c**2
            dpm_dr = -(eps_m_val + _pm) * common
            dpchi_dr = -(rho_chi_eps + _pchi) * common

            # Clamp
            max_deriv = 1e30
            dpm_dr = max(-max_deriv, min(max_deriv, dpm_dr))
            dpchi_dr = max(-max_deriv, min(max_deriv, dpchi_dr))

            # --- Tidal y-ODE (geometric units, G=c=1) ---
            # Convert to geometric
            m_geom = Gc2 * _m
            eps_m_geom = Gc4 * eps_m_val
            eps_chi_geom = Gc4 * rho_chi_eps
            eps_total_geom = eps_m_geom + eps_chi_geom
            p_m_geom = Gc4 * _pm
            p_chi_geom_val = Gc4 * _pchi
            p_total_geom = p_m_geom + p_chi_geom_val

            e_lambda = 1.0 / max(1.0 - 2.0 * m_geom / _r, 1e-15)
            nu_prime = 2.0 * (m_geom + 4.0 * np.pi * _r**3 * p_total_geom) / (_r * (_r - 2.0 * m_geom))
            nu_prime = max(-1e10, min(1e10, nu_prime))

            # Effective speed of sound squared
            # c_s^2 = dp_total/deps_total
            deps_m_dr = eos_deps_dp(_pm) * dpm_dr  # d(eps_m)/dr
            deps_chi_dr = rho0_cgs * c**2 * (-n * (_r / r_c_cm)**n / (_r * (1.0 + (_r / r_c_cm)**n)**2))  # analytical
            dptot_dr = dpm_dr + dpchi_dr
            depstot_dr = deps_m_dr + deps_chi_dr
            if abs(depstot_dr) > 1e-30:
                cs2 = max(abs(dptot_dr / depstot_dr), 1e-10)
            else:
                cs2 = 1e-10  # near-constant density

            # y-ODE: r·dy/dr + y^2 + y·F + r^2·Q = 0
            # → dy/dr = (-y^2 - yF - r^2·Q) / r
            F = e_lambda * (1.0 + 4.0 * np.pi * _r**2 * (p_total_geom - eps_total_geom))
            Q = (4.0 * np.pi * e_lambda * (5.0 * eps_total_geom + 9.0 * p_total_geom
                 + (eps_total_geom + p_total_geom) / cs2)
                 - 6.0 * e_lambda / _r**2 - nu_prime**2)

            dy_dr = (-_y**2 - _y * F - _r**2 * Q) / _r
            dy_dr = max(-1e10, min(1e10, dy_dr))

            return dm_dr, dpm_dr, dpchi_dr, dy_dr

        # RK4 step
        k1m, k1pm, k1pchi, k1y = rhs(ri, mi, pm_i, pchi_i, yi)

        r_half = ri + 0.5 * h
        k2m, k2pm, k2pchi, k2y = rhs(r_half,
                                     mi + 0.5*h*k1m, pm_i + 0.5*h*k1pm,
                                     pchi_i + 0.5*h*k1pchi, yi + 0.5*h*k1y)

        k3m, k3pm, k3pchi, k3y = rhs(r_half,
                                     mi + 0.5*h*k2m, pm_i + 0.5*h*k2pm,
                                     pchi_i + 0.5*h*k2pchi, yi + 0.5*h*k2y)

        r_next = ri + h
        k4m, k4pm, k4pchi, k4y = rhs(r_next,
                                     mi + h*k3m, pm_i + h*k3pm,
                                     pchi_i + h*k3pchi, yi + h*k3y)

        m[i + 1] = mi + (h / 6.0) * (k1m + 2*k2m + 2*k3m + k4m)
        p_m[i + 1] = pm_i + (h / 6.0) * (k1pm + 2*k2pm + 2*k3pm + k4pm)
        p_chi[i + 1] = pchi_i + (h / 6.0) * (k1pchi + 2*k2pchi + 2*k3pchi + k4pchi)
        y[i + 1] = yi + (h / 6.0) * (k1y + 2*k2y + 2*k3y + k4y)

        # --- Store c_s^2 at this step (causality check) ---
        # Compute c_s^2 = dp_tot/dε_tot from the stored profiles
        chi_now = chi_profile(np.array([r[i+1]]), r_c_cm, n)[0]
        rho_chi_now = rho0_cgs * chi_now * c**2
        eps_m_now = eos_eps_from_p(p_m[i+1])
        eps_tot_now = eps_m_now + rho_chi_now
        # Use the rhs function indirectly: re-evaluate derivatives at this point
        # Actually compute directly:
        if i > 0 and abs(p_m[i+1] - p_m[i-1]) > 1e-30:
            dpm_dr_est = (p_m[i+1] - p_m[i-1]) / (r[i+1] - r[i-1])
            # dε/dp from EOS
            deps_dpm = eos_deps_dp(p_m[i+1])
            depsm_dr_est = deps_dpm * dpm_dr_est
            # χ-sector contribution
            depschi_dr_est = rho0_cgs * c**2 * (-n * (r[i+1]/r_c_cm)**n / (r[i+1] * (1.0 + (r[i+1]/r_c_cm)**n)**2))
            dpchi_dr_est = (p_chi[i+1] - p_chi[i-1]) / (r[i+1] - r[i-1])
            dptot_dr_est = dpm_dr_est + dpchi_dr_est
            depstot_dr_est = depsm_dr_est + depschi_dr_est
            if abs(depstot_dr_est) > 1e-30:
                cs2_arr[i+1] = abs(dptot_dr_est / depstot_dr_est)
            else:
                cs2_arr[i+1] = 0.0
        else:
            cs2_arr[i+1] = 0.0

        # Blowup check
        if not (np.isfinite(m[i+1]) and np.isfinite(p_m[i+1]) and np.isfinite(p_chi[i+1])
                and np.isfinite(y[i+1])):
            surface_idx = i + 1
            break

        # Surface: p_total reaches zero
        p_total_nxt = p_m[i + 1] + p_chi[i + 1]
        if p_total_nxt <= 0.0:
            p_total_i = pm_i + pchi_i
            if abs(p_total_nxt - p_total_i) < 1e-15:
                alpha = 0.5
            else:
                alpha = p_total_i / (p_total_i - p_total_nxt)
            alpha = max(0.0, min(1.0, alpha))
            r_surface = ri + alpha * h
            m_surface = mi + alpha * (m[i+1] - mi)
            y_surface = yi + alpha * (y[i+1] - yi)
            r[i+1] = r_surface
            m[i+1] = m_surface
            p_m[i+1] = 0.0
            p_chi[i+1] = 0.0
            y[i+1] = y_surface
            surface_idx = i + 2
            break

    # Trim
    if surface_idx < n_points:
        r = r[:surface_idx]
        m = m[:surface_idx]
        p_m = p_m[:surface_idx]
        p_chi = p_chi[:surface_idx]
        y = y[:surface_idx]

    # Post-integration horizon check (matching tov_physical.py threshold)
    M_g = m[-1]  # grams
    R_cm = r[-1]  # cm
    m_r_ratio = M_g / max(R_cm, 1e-30)
    schwarzschild_crit = c**2 / (2.0 * G)
    if m_r_ratio >= 0.49 * schwarzschild_crit:
        return None

    if M_g < 1e20 or R_cm < 1e3:
        return None

    at_boundary = (surface_idx >= n_points)

    # Maximum c_s^2 anywhere in the star (causality check)
    cs2_max = np.max(cs2_arr[:surface_idx]) if surface_idx > 0 else 0.0

    # Compute Love number k2 and Lambda
    M_geom = Gc2 * M_g
    C = M_geom / R_cm  # geometric compactness GM/(c^2 R)
    y_R = y[-1]

    # k2 formula (Hinderer 2008, Eq. 17)
    C2 = C * C
    C3 = C2 * C
    C4 = C3 * C
    C5 = C4 * C
    om2C = 1.0 - 2.0 * C
    log_om2C = np.log(max(om2C, 1e-30))

    num = (8.0 * C5 / 5.0) * om2C**2 * (2.0 + 2.0 * C * (y_R - 1.0) - y_R)
    den = (2.0 * C * (6.0 - 3.0 * y_R + 3.0 * C * (5.0 * y_R - 8.0))
           + 4.0 * C3 * (13.0 - 11.0 * y_R + C * (3.0 * y_R - 2.0) + 2.0 * C2 * (1.0 + y_R))
           + 3.0 * om2C**2 * (2.0 - y_R + 2.0 * C * (y_R - 1.0)) * log_om2C)

    if abs(den) < 1e-30:
        k2 = 0.0
    else:
        k2 = num / den

    Lambda = (2.0 / 3.0) * k2 / max(C5, 1e-30)

    return {
        'R_km': R_cm / km,
        'M_Msun': M_g / Msun,
        'C': C,
        'y_R': y_R,
        'k2': k2,
        'Lambda': Lambda,
        'cs2_max': cs2_max,
        'p_c_cgs': p_c_cgs,
        'rho0_gcm3': rho0_cgs,
        'r_c_km': r_c_km,
        'n': n,
        'success': True,
        'at_boundary': at_boundary,
    }


# =========================================================================
# Worker
# =========================================================================

def _integrate_one(args):
    p_c, rho0, r_c, n, w_chi_inf, n_points = args
    try:
        sol = solve_tov_tidal(p_c, rho0, r_c, n, w_chi_inf, n_points=n_points)
        if sol is None:
            return {'p_c_cgs': p_c, 'rho0_gcm3': rho0, 'r_c_km': r_c, 'n': n,
                    'M_Msun': np.nan, 'R_km': np.nan, 'Lambda': np.nan,
                    'cs2_max': np.nan,
                    'success': False, 'fail_reason': 'BH', 'at_boundary': False}
        return {
            'p_c_cgs': p_c, 'rho0_gcm3': rho0, 'r_c_km': r_c, 'n': n,
            'M_Msun': sol['M_Msun'], 'R_km': sol['R_km'],
            'C': sol['C'], 'y_R': sol['y_R'], 'k2': sol['k2'],
            'Lambda': sol['Lambda'],
            'cs2_max': sol.get('cs2_max', np.nan),
            'success': True, 'fail_reason': '',
            'at_boundary': sol.get('at_boundary', False),
        }
    except Exception as e:
        return {'p_c_cgs': p_c, 'rho0_gcm3': rho0, 'r_c_km': r_c, 'n': n,
                'M_Msun': np.nan, 'R_km': np.nan, 'Lambda': np.nan,
                'cs2_max': np.nan,
                'success': False, 'fail_reason': str(e)[:80], 'at_boundary': False}


# =========================================================================
# Sweep
# =========================================================================

def run_sweep(p_c_vals, rho0_vals, r_c_vals, n_vals,
              w_chi_inf=0.0, n_points=4000, n_workers=8):
    tasks = []
    for rho0 in rho0_vals:
        for r_c in r_c_vals:
            for n in n_vals:
                for p_c in p_c_vals:
                    tasks.append((p_c, rho0, r_c, n, w_chi_inf, n_points))

    total = len(tasks)
    n_workers = min(n_workers, total)
    print(f"Tasks: {total}, workers: {n_workers}")

    results = []
    bh_count = 0
    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_integrate_one, t): t for t in tasks}
        for fut in as_completed(futures):
            completed += 1
            if completed % max(1, total // 20) == 0 or completed == total:
                print(f"  [{completed:5d}/{total}] {100*completed/total:4.0f}%  "
                      f"stars={len(results)}  BH={bh_count}")
            try:
                r = fut.result()
                if r is not None:
                    if r['success']:
                        results.append(r)
                    else:
                        bh_count += 1
            except Exception:
                pass

    print(f"  Stable stars: {len(results)}  Black holes: {bh_count}")
    return results


# =========================================================================
# Plotting
# =========================================================================

def plot_tidal_results(results, output_path='tov_tidal_plot.png'):
    if not results:
        print("No results to plot!")
        return

    normal = [r for r in results if r['success'] and not r.get('at_boundary', False)]
    if not normal:
        print("No normal stars!")
        return

    r_c_vals = sorted(set(r['r_c_km'] for r in normal))
    rho0_vals = sorted(set(r['rho0_gcm3'] for r in normal))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(rho0_vals)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: Lambda-M diagram
    ax = axes[0, 0]
    for j, r_c in enumerate(r_c_vals[:1]):  # show one r_c for clarity
        for i, rho0 in enumerate(rho0_vals):
            subset = [r for r in normal
                      if abs(r['rho0_gcm3'] - rho0) < 1e-10
                      and abs(r['r_c_km'] - r_c) < 1e-10]
            if len(subset) < 2:
                continue
            subset.sort(key=lambda x: x['M_Msun'])
            M_arr = np.array([s['M_Msun'] for s in subset])
            L_arr = np.array([s['Lambda'] for s in subset])
            label = f'rho0={rho0:.1e}'
            ax.loglog(M_arr, L_arr, '-', color=colors[i], alpha=0.8, linewidth=1.2,
                      label=label)
    # GW170817 bound
    ax.axhline(800, color='red', linestyle='--', alpha=0.7, linewidth=1.5,
               label='GW170817: Lambda(1.4) < 800')
    ax.axvline(1.4, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Mass (Msun)')
    ax.set_ylabel('Lambda')
    ax.set_title(f'Tidal deformability: chi-sector — EOS: {EOS_PARAMS[EOS_NAME]["label"]}')
    ax.legend(fontsize=6, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.8, 2.2)

    # Panel 2: Lambda_1.4 vs rho0
    ax = axes[0, 1]
    for r_c in r_c_vals:
        lam14 = []
        rr = []
        for rho0 in rho0_vals:
            subset = [r for r in normal
                      if abs(r['rho0_gcm3']-rho0) < 1e-10
                      and abs(r['r_c_km']-r_c) < 1e-10]
            if len(subset) >= 2:
                subset.sort(key=lambda x: x['M_Msun'])
                # Interpolate Lambda at M=1.4
                M_arr = np.array([s['M_Msun'] for s in subset])
                L_arr = np.array([s['Lambda'] for s in subset])
                if M_arr[0] <= 1.4 <= M_arr[-1]:
                    lam_at_14 = np.interp(1.4, M_arr, L_arr)
                    lam14.append(lam_at_14)
                    rr.append(rho0)
        if len(lam14) > 1:
            ax.plot(rr, lam14, 'o-', markersize=6, linewidth=1.5,
                    label=f'rc={r_c:.0f} km')
    ax.axhline(800, color='red', linestyle='--', alpha=0.7,
               label='GW170817 bound')
    ax.set_xlabel('rho0 (g/cm^3)')
    ax.set_ylabel('Lambda(1.4 Msun)')
    ax.set_title(f'Lambda(1.4) vs chi-sector density — EOS: {EOS_PARAMS[EOS_NAME]["label"]}')
    ax.set_xscale('log')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 3: Lambda_1.4 constraint contour
    ax = axes[1, 0]
    rho0_pos = [v for v in rho0_vals if v > 0]
    if not rho0_pos:
        rho0_pos = rho0_vals
    rho0_grid = np.logspace(np.log10(min(rho0_pos)*0.5), np.log10(max(rho0_pos)*1.5), 40)
    r_c_grid = np.linspace(min(r_c_vals), max(r_c_vals), 40)

    lam_grid = np.zeros((len(r_c_grid), len(rho0_grid)))
    lam_grid[:] = np.nan
    for i, r_c in enumerate(r_c_grid):
        for j, rho0 in enumerate(rho0_grid):
            subset = [r for r in normal
                      if abs(r['r_c_km'] - r_c) < 0.5
                      and abs(r['rho0_gcm3'] - rho0)/max(rho0, 1e-30) < 0.2]
            if len(subset) >= 2:
                subset.sort(key=lambda x: x['M_Msun'])
                M_arr = np.array([s['M_Msun'] for s in subset])
                L_arr = np.array([s['Lambda'] for s in subset])
                if M_arr[0] <= 1.4 <= M_arr[-1]:
                    lam_grid[i, j] = np.interp(1.4, M_arr, L_arr)

    X, Y = np.meshgrid(rho0_grid, r_c_grid)
    cs = ax.contourf(X, Y, lam_grid, levels=20, cmap='plasma', alpha=0.8)
    ax.contour(X, Y, lam_grid, levels=[800], colors='white', linewidths=2,
               linestyles='-')
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label('Lambda(1.4 Msun)')
    ax.set_xscale('log')
    ax.set_xlabel('rho0 (g/cm^3)')
    ax.set_ylabel('r_c (km)')
    ax.set_title('Lambda_1.4(rho0, r_c) — white line = GW170817 bound (800)')

    # Panel 4: k2 vs compactness
    ax = axes[1, 1]
    for j, r_c in enumerate(r_c_vals[:1]):
        for i, rho0 in enumerate(rho0_vals[:3]):
            subset = [r for r in normal
                      if abs(r['rho0_gcm3'] - rho0) < 1e-10
                      and abs(r['r_c_km'] - r_c) < 1e-10]
            if len(subset) < 2:
                continue
            subset.sort(key=lambda x: x['C'])
            C_arr = np.array([s['C'] for s in subset])
            k2_arr = np.array([s['k2'] for s in subset])
            ax.plot(C_arr, k2_arr, '-', alpha=0.8, linewidth=1.5,
                    label=f'rho0={rho0:.1e}')
    ax.set_xlabel('Compactness C = M/R')
    ax.set_ylabel('Love number k2')
    ax.set_title('k2 vs compactness')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")
    plt.close()


# =========================================================================
# Main
# =========================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--n-pts', type=int, default=4000)
    ap.add_argument('--eos', type=str, default='sly',
                    choices=['sly', 'apr4', 'ms1b', 'h4'],
                    help='Nuclear EOS (default: sly)')
    ap.add_argument('--tune', action='store_true',
                    help='Auto-calibrate K_CORE1 to hit literature M_max')
    args = ap.parse_args()

    setup_eos(args.eos, tune_K1=args.tune)

    print("=" * 60)
    print(f"TOV + tidal deformability: chi-sector — EOS: {_EOS['label']}")
    print("=" * 60)

    # Parameter sweep — focused on rho0 range where Lambda changes
    p_c_vals = np.logspace(33.0, 37.5, 60)
    rho0_vals = [0.0, 1e10, 3e10, 1e11, 3e11, 1e12, 3e12, 1e13, 3e13, 1e14, 3e14]
    r_c_vals = [2.0, 5.0, 10.0]
    n_vals = [2]

    print(f"\nSweep: {len(p_c_vals)} p_c x {len(rho0_vals)} rho0 x {len(r_c_vals)} r_c")
    results = run_sweep(p_c_vals, rho0_vals, r_c_vals, n_vals,
                        n_points=args.n_pts, n_workers=args.workers)

    # Save
    if results:
        fields = ['p_c_cgs', 'rho0_gcm3', 'r_c_km', 'n', 'M_Msun', 'R_km',
                  'C', 'y_R', 'k2', 'Lambda', 'cs2_max', 'success', 'at_boundary']
        with open(f'tov_tidal_results_{args.eos}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows([{k: r[k] for k in fields} for r in results])

    # Summary: Lambda(1.4) for each (rc, rho0)
    print("\n" + "=" * 60)
    print("SUMMARY: Lambda(1.4 Msun) vs chi-sector parameters")
    print("=" * 60)
    normal = [r for r in results if r['success'] and not r.get('at_boundary', False)]
    for r_c in sorted(set(r['r_c_km'] for r in normal)):
        for rho0 in sorted(set(r['rho0_gcm3'] for r in normal)):
            subset = [r for r in normal
                      if abs(r['r_c_km']-r_c) < 1e-10
                      and abs(r['rho0_gcm3']-rho0) < 1e-10]
            if len(subset) >= 2:
                subset.sort(key=lambda x: x['M_Msun'])
                M_arr = np.array([s['M_Msun'] for s in subset])
                L_arr = np.array([s['Lambda'] for s in subset])
                M_max = max(M_arr)
                if M_arr[0] <= 1.4 <= M_arr[-1]:
                    lam14 = np.interp(1.4, M_arr, L_arr)
                    flag = '[TIGHT]' if lam14 < 800 else '[OK]'
                    print(f"  rc={r_c:.0f}km  rho0={rho0:.1e}  M_max={M_max:.3f}  "
                          f"Lambda(1.4)={lam14:.1f}  {flag}")

    # Plot
    plot_tidal_results(results, output_path=f'tov_tidal_plot_{args.eos}.png')


if __name__ == '__main__':
    main()
