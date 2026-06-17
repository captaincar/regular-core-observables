# Regular Core Observables

[![DOI](https://zenodo.org/badge/1272502238.svg)](https://doi.org/10.5281/zenodo.20737351)

This repository contains the manuscript, public summary, code, and generated figures for a computational study of observational constraints on regular black holes and negative-pressure cores in compact objects.

The analysis focuses on three observational channels:

- black-hole ringdown and quasinormal-mode shifts
- neutron-star tidal deformability and mass-radius constraints
- gravitational-wave echoes and the extremality threshold

It also includes a Hayward-remnant calculation relevant to primordial-black-hole dark matter scenarios.

## Main files

- `manuscript.md`: full research manuscript
- `popular-summary.md`: non-technical summary
- `manuscript.html`: HTML rendering target for the manuscript
- `popular-summary.html`: HTML rendering target for the public summary

## Core scripts

- `kerr_qnm.py`: Hayward/Bardeen/Dymnikova QNM analysis and detector thresholds
- `tov_tidal.py`: two-fluid TOV solver with tidal deformability
- `post_merger_f2.py`: post-merger frequency shifts from TOV outputs
- `causality_sweep.py`: sound-speed consistency scan
- `hayward_remnant.py`: Hawking-temperature and remnant-mass calculation
- `tidal_gw170817.py`: GW170817 tidal-constraint plots
- `conceptual_diagram.py`: overview figure used in the manuscript

## Reproducibility

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Generate the HTML outputs:

```bash
python build_manuscript_html.py
python build_popular_summary_html.py
```

On Windows, if `python` is not on PATH, use:

```bash
py build_manuscript_html.py
py build_popular_summary_html.py
```

## Citation

Use the Zenodo version DOI for formal citation of this archived release:
`10.5281/zenodo.20737352`

The badge above points to the Zenodo concept DOI:
`10.5281/zenodo.20737351`

The concept DOI is a stable project-level landing page. The version DOI refers
to this exact archived release.

## Notes

- The manuscript is a phenomenological study. It does **not** derive the regular core from a microphysical theory.
- The slow-rotation Kerr extension is explicitly treated as first-order and is not reliable at astrophysical spins such as GW150914.
- The popular-summary files are intended for general readers; the manuscript files are the canonical research version.