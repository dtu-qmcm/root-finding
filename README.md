# root-finding — diffrax metabolic models

Two kinetic metabolic models built as [diffrax](https://docs.kidger.site/diffrax/)
vector fields and simulated to steady state. Parameters are organized as nested
[`equinox.Module`](https://docs.kidger.site/equinox/) structures, accessed as
`p.<enzyme>.<param>`.

## Models

### `model1.py` — reversible mass-action network

```
R1:  A_e          <->  B_c
R2:  B_c + X1_c   <->  C_c + X2_c
R3:  C_c          <->  D_e
R4:  X2_c + Y2_e  <->  X1_c + Y1_e
```

Metabolites tagged `_c` are *balanced* (ODE state variables: `B_c, C_c, X1_c,
X2_c`); those tagged `_e` are *boundary conditions* held fixed. Each reaction
uses reversible mass-action kinetics. `X1_c + X2_c` is a conserved moiety, used
as a correctness check.

### `model2.py` — glycolysis bistability RHS skeleton

A faithful reproduction of the right-hand side of the kinetic model in
Mulukutla et al. (2014), *Bistability in Glycolysis Pathway as a Physiological
Switch in Energy Metabolism*, PLoS ONE,
[DOI 10.1371/journal.pone.0098756](https://doi.org/10.1371/journal.pone.0098756).

All 12 mass-balance ODEs (`GLC, G6P, F6P, F16BP, F26BP, DHAP, GAP, BPG13, PG3,
PG2, PEP, PYR`) are present, with the documented rate-law forms (King–Altman
steady-state laws from Mulquiney & Kuchel 1999; Monod–Wyman–Changeux allostery
for PFK and PK; the bifunctional PFKFB kinase/bisphosphatase cycle; GLUT1 and
mitochondrial pyruvate transport).

The original algebraic rate equations live in the paper's Information S1 as
MathType images. They were extracted by rendering the embedded WMF objects with
LibreOffice and reading them; the full transcription (rate laws + constants) is
in [`EQUATIONS.md`](EQUATIONS.md), and `model2.py` implements them verbatim.

Parameter provenance:
- **PFK, PK, PFK2, F26BPase, ALDO, PGK, ENO** — exact constants listed in the
  supplement (recovered from the equation images).
- **HK, GPI, TPI, GAPDH, PGM, LDH, GLUT, PYRH** — the supplement states these
  were "adopted from Mulquiney & Kuchel 1999" and does not re-list them; the
  values used are representative literature/erythrocyte constants (marked
  `# repr.`).
- Cofactors (ATP, ADP, Mg, NAD, …) are clamped at the Table S4 values, so the
  bi-bi / ter-ter laws reduce to functions of the 12 state metabolites.
- A per-enzyme `level` (relative abundance, a Vmax multiplier) is set to 1.0; the
  paper derives these from transcriptomics. Absolute levels are not given, so the
  defaults are chosen to yield a stable positive steady state rather than to
  match the published bistable region exactly.

## Structure

- `model1.py`, `model2.py` — importable RHS modules: parameters
  (`equinox.Module`), rate laws, and the `vector_field` for each model (plus a
  `simulate` helper). They define the models but do not run anything on import.
- `log_rhs.py` — wraps any `vector_field` into its log-transformed form,
  `d log(x)/dt = f(t, exp(u)) / exp(u)` with `u = log(x)`. Integrating in log
  space keeps every state strictly positive and rescales multiplicative
  dynamics. Provides `log_vector_field`, pre-wrapped versions for both models,
  and `simulate_log` (takes/returns original-space concentrations).
- `main.py` — example script that imports the models and runs all simulations
  to steady state (model 1, model 2, and model 2 in log space), printing states,
  fluxes, and consistency checks.
- `make_figures.py` — simulates both models and writes trajectory plots to
  `figures/`.

## Running

```sh
uv run python main.py          # run both example simulations
uv run python make_figures.py  # regenerate the figures
```
