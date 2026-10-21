[![DOI](https://zenodo.org/badge/1428406687.svg)](https://doi.org/10.5281/zenodo.23535545)

# Condition-dependent validation of wearable core temperature monitoring — analysis code

Analysis code for the manuscript *"Condition-dependent validation of wearable core temperature
monitoring across sensing configurations and thermal environments"* (submitted to npj Digital
Medicine).

This repository contains **code only**. No data are included — neither the source datasets nor any
derived result files; these are described in the manuscript's Data availability section. Running the
scripts below regenerates every number, table and numerical figure panel reported in the paper.

---

## Layout

```
analysis/   estimation pipeline, perturbations, decomposition, bootstrap intervals, dynamic metrics
tables/     builds Tables 1–3
figures/    Figure 2 and Figure 3 panel builders + assembly
```

## Which script produces what

| Manuscript item | Script | Output written by the script |
|---|---|---|
| Five sensing configurations, participant-grouped LOSO (nRMSE 0.9973 → 0.6381) | `analysis/_twist.py`, `analysis/_twist_folds.py` | `_twist_results.csv`, `_twist_folds.csv` |
| Signal perturbations (site dropout, gaps, noise, offset, drift) | `analysis/_degrade.py`, `analysis/_twist.py` | rows in `_twist_results.csv` |
| **Table 1** — decomposition of pooled nRMSE² (Eq. 6), within-unit / reweighting / interaction | `analysis/_drift_decomp.py` | `_drift_decomp_summary.csv` |
| **Figure 3c** — per-site nRMSE with 95 % participant-bootstrap CI (2,000 resamples) | `analysis/_panel_b_stats.py` | `_panel_b_sites.csv` |
| **Figure 2a** — ambient-temperature slopes for the eleven skin sites | `analysis/_fig2_v2_numbers.py` | `_panel_numbers.json` |
| Five dynamic-fidelity measures (variance fidelity, derivative error, direction agreement, lag-1 autocorrelation error, peak cross-correlation) | `analysis/_final_dyn.py` | `_final_dyn.csv` |
| Ear-canal dataset: correlations and slopes across 10/20/30 °C (Fig. 2f,g; Table 2) | `analysis/_obs_fujii.py`, `analysis/_obs_lib.py` | printed tables (see script headers) |
| PROSPIE exploratory analyses (condition-level aggregates, paired contrasts) | `analysis/_prospie_v2.py`, `analysis/_flib.py` | printed tables |
| **Tables 1–3** as a Word file | `tables/_build_tables.py` | `tables.docx` |
| **Figures 2–4** | `figures/fig2g_earcanal_build.py`, `figures/fig3{a..f}_*.py`, `figures/fig2_assemble.py`, `figures/fig3_assemble.py` | PDF/PNG panels + assembled sheets |

## Environment

```
pip install -r requirements.txt
```

Python 3.11. The scripts were run on Windows; nothing is platform-specific beyond file paths.

## Before running: two configuration steps

**1. Set the data root.** No absolute paths are hard-coded in this release. Every script opens with
a placeholder base path instead, in one of two forms, whichever the script's own path handling
requires:

* `P = Path("")` — pathlib-based scripts, which join paths with `P / "file"`;
* `P = ""` — scripts that use `os.path.join(P, ...)` or `P + "file"`.

Set it to the directory that holds the datasets obtained from their original sources:

| File | Line |
|---|---|
| `analysis/_twist.py` | 24 |
| `analysis/_twist_folds.py` | 36 |
| `analysis/_degrade.py` | 21 |
| `analysis/_prospie_v2.py` | 24 |
| `analysis/_obs_fujii.py` | 10 |
| `analysis/_flib.py` | 29 |
| `analysis/_panel_b_stats.py` | 23 |
| `figures/fig2g_earcanal_build.py` | 74 |
| `figures/fig3b_site_recovery.py` | 37 |
| `figures/fig3c_cost_profile.py` | 54 |
| `figures/fig3d_build.py` | 57 |
| `figures/fig3e_build.py` | 57 |
| `figures/fig3f_build.py` | 71 |

**2. Figure style module.** The figure scripts import `figure_style`, an internal Matplotlib style
module that is not part of this release. `figures/figure_style.py` is a compatibility shim that
exposes the same public API (`apply_style`, `mm_fig`, `save_figure`, `tidy_axes`, `FONT_PT`) so the
scripts run; it does **not** reproduce the exact typography of the published figures. The published
figures are the authoritative versions, and every numerical panel can be regenerated from the
scripts above.

## Licence and citation

Code is released under the MIT Licence (see `LICENSE`). If you use it, please cite the manuscript
(see `CITATION.cff`).
