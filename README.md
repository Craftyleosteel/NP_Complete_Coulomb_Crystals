# When is finding the ground state of an ion Coulomb crystal NP-hard?

Code, data, and formal proofs accompanying the paper by **Paco Navarro** (Department of Physics,
UC Santa Barbara) and **Michael Emmerich** (Faculty of Information Technology, University of
Jyväskylä), *"When is finding the ground state of an ion Coulomb crystal NP-hard?"*
(arXiv link to be added).

**Short version:** for identical ions in a fixed trap ("free" crystals), the unary-encoded
ground-state decision problem cannot be NP-hard unless the polynomial hierarchy collapses, even
though the problem is hard in practice. This is a statement about one input encoding, not an
algorithm for finding configurations. When the ions are pinned to a prescribed set of sites,
choosing which sites to fill so as to minimize the Coulomb energy is NP-hard in two and three
dimensions, and NP-complete at finite precision.

## New to this? Start with the explainer

[`explainer/explainer.pdf`](explainer/explainer.pdf) is a short, accessible version written for
physicists with no background in computational complexity. It covers why the problem is interesting,
what "NP-hard" means in physics terms, the quadrupole trick behind the construction, and the
simulations. The same piece is in [`explainer/explainer.html`](explainer/explainer.html); download it
together with the `explainer_img/` folder and open it in a browser.

## Contents

| Path | What it contains | Reproduces |
|---|---|---|
| `explainer/` | The accessible explainer (PDF and HTML) | – |
| `checks/multipole_law.py`, `checks/s1_constants.py` | Symbolic and numerical checks of the proof's constants | Lemma 1 (angular law), field coefficients, λ bounds, Lemma 2 (sensitivity) |
| `checks/certify_budget.py` | Interval-arithmetic certification of the lattice sums | Table II and Lemma 3 (adverse load ≤ 42.225 b⁴ < 47.999 b⁴) |
| `figures/make_figures.py` | Schematic and analytic figures | Figs. 3, 4, 5, 12 |
| `figures/free_crystals.py` | Free-crystal census | Fig. 1; the data behind Fig. 2 (`free_summary.json`) |
| `figures/simulations.py` | Pinned cube-graph instance at b = 0.02 | Figs. 6, 7, 8 |
| `figures/single_ion_barriers.py` | Single-ion barrier bounds on the same instance | Sec. IV B, Fig. 9 data |
| `figures/physical_regime.py` | Laboratory-parameter scan | Fig. 10 and Sec. V |
| `figures/*.json` | Saved results quoted in the paper | – |
| `lean/` | Lean 4 + Mathlib formalization, with a build log | The machine-checked components of Appendix B |

## Reproducing the results

Python 3.12 or later (tested on 3.14):

```bash
pip install -r requirements.txt

python checks/multipole_law.py          # angular law at s = 1 and s = 2 (symbolic)
python checks/s1_constants.py           # field coefficient, lambda bounds, sensitivity constants
python checks/certify_budget.py         # interval-arithmetic certificate, under a minute

python figures/make_figures.py          # schematic and analytic figures, seconds
python figures/simulations.py pinned    # pinned cube-graph instance: collapse, annealing, barrier path
python figures/single_ion_barriers.py   # single-ion barrier bounds (Sec. IV B), ~30 s
python figures/free_crystals.py         # free-crystal census and schematic; uses all cores, ~1.5 min on 24
python figures/physical_regime.py full  # laboratory-parameter scan (Fig. 10), a few minutes
```

The `*.json` files in `figures/` hold the results quoted in the paper, so the figures can be
replotted without rerunning the simulations (`python figures/physical_regime.py full replot`).

**Single-ion barriers.** `single_ion_barriers.py` rebuilds the b = 0.02 instance, walks the
680-flip route of Fig. 7, resolves each reversal into its four possible single-ion intermediates,
and checks all 1360 × 1360 single-ion moves at each of the 681 route states. It reproduces the
interval [4.28536, 4.36686] × 10⁶ λ, the highest saddle at reversal 435, and the strict-local-minimum
property of every route state. These are double-precision evaluations, not interval-certified
bounds. The barrier-tree figure itself (Fig. 9) was drawn from the same data in a companion study by
M. Emmerich.

**Laboratory scan.** `physical_regime.py full` rebuilds the cube-graph instance at each selector
half-size b for three levels of control over the well depths and records the three diagnostics of
Sec. V (collapse onto λH_B, cut criterion, single-ion stability of the lowest domain-wall-free
configuration), followed by the finite-stiffness study. The saved `physical_regime.json` reproduces
the windows b ≤ 0.03, 0.12, 0.33 and the stiffness thresholds ω_w ≥ 16 ω_a (b = 0.25) and 63 ω_a
(b = 0.1). The single-ion check establishes local stability only.

### Lean

Install [elan](https://github.com/leanprover/elan), then:

```bash
cd lean
lake exe cache get   # prebuilt Mathlib (several GB)
lake build
```

The toolchain is pinned in `lean/lean-toolchain` (Lean v4.36.0-rc2). `#print axioms` on the main
theorems reports only `propext`, `Classical.choice`, and `Quot.sound`. A log of a successful build of
this commit is in [`lean/BUILD_LOG.txt`](lean/BUILD_LOG.txt).

## What is and is not machine-checked

The Lean development checks the source-problem combinatorics, the gap-transfer logic, the selector
inequalities, the spin-level flip criterion, and the assembly of the main theorem. That assembly is
**conditional** on a `GeometricReduction` object, which Lean does not construct. The geometric estimates
(multipole bounds, lattice sums, field compensation, layout) are proved by hand in the paper; the
tightest of them, the lattice sums, are certified independently by `checks/certify_budget.py`.
Interval arithmetic checks the sums, not that the listed local geometries exhaust every layout; that
coverage is part of the written proof.

## Use of AI tools

This work was prepared with substantial assistance from AI tools: Anthropic Claude (via Claude Code)
for code, the Lean formalization, derivations, and drafting, and OpenAI Codex for adversarial proof
audits and programming assistance. The authors directed the project, checked the results, and take
full responsibility for them. See the acknowledgments of the paper for details.

## License

MIT. See [`LICENSE`](LICENSE).
