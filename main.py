"""Run both diffrax metabolic-model demos.

  model1.py — small reversible mass-action network (balanced `_c` species are
              ODE states, boundary `_e` species are clamped).
  model2.py — RHS skeleton of the glycolysis bistability model
              (DOI 10.1371/journal.pone.0098756).

Run an individual model directly (`uv run python model1.py`) for its full
report, or run this file to execute both.
"""

import runpy


def main():
    for mod in ("model1", "model2"):
        print(f"\n{'=' * 70}\nRunning {mod}.py\n{'=' * 70}")
        runpy.run_module(mod, run_name="__main__")


if __name__ == "__main__":
    main()
