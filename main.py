"""Example: simulate both metabolic models to steady state.

The right-hand sides (parameters, rate laws, vector fields) are defined in the
model modules; this script imports them and runs both simulations as a demo.

  model1.py — small reversible mass-action network (balanced `_c` species are
              ODE states, boundary `_e` species are clamped).
  model2.py — glycolysis bistability model, exact published rate laws
              (DOI 10.1371/journal.pone.0098756; see EQUATIONS.md).

Run: uv run python main.py
"""

from __future__ import annotations

import jax.numpy as jnp

import model1
import model2


def run_model1() -> None:
    p = model1.default_params()
    bc = model1.default_boundary()
    y0 = jnp.array([0.1, 0.1, 0.5, 0.5])  # B_c, C_c, X1_c, X2_c
    sol = model1.simulate(y0=y0, p=p, bc=bc)
    yf = sol.ys[-1]

    print("=== Model 1: reversible mass-action network ===")
    print(f"integrated to t = {float(sol.ts[-1]):.1f}, steps = {sol.stats['num_steps']}")
    print("\nsteady state:")
    for name, val in zip(["B_c", "C_c", "X1_c", "X2_c"], yf):
        print(f"  {name:5s} = {float(val): .6f}")

    v = model1.reaction_rates(yf, p, bc)
    print("\nfluxes at steady state (all ~equal at steady state):")
    for i, val in enumerate(v, 1):
        print(f"  v{i} = {float(val): .6e}")

    # Moiety conservation check: X1_c + X2_c is constant.
    pool0, poolf = float(y0[2] + y0[3]), float(yf[2] + yf[3])
    print(f"\nX1_c + X2_c : t0 = {pool0:.6f}, tf = {poolf:.6f}, "
          f"drift = {abs(poolf - pool0):.2e}")
    assert abs(poolf - pool0) < 1e-6, "moiety X1_c+X2_c not conserved!"
    print("moiety conservation OK")


def run_model2() -> None:
    p = model2.default_params()
    sol = model2.simulate(p)
    yf = sol.ys[-1]

    print("=== Model 2: glycolysis bistability (exact published rate laws) ===")
    print("(DOI 10.1371/journal.pone.0098756; equations transcribed in EQUATIONS.md)")
    print(f"integrated to t = {float(sol.ts[-1]):.0f} h, steps = {sol.stats['num_steps']}, "
          f"result = {sol.result}")

    print("\nsteady-state metabolites (mM):")
    for s, val in zip(model2.SPECIES, yf):
        print(f"  {s:6s} = {float(val): .6e}")

    assert bool(jnp.all(jnp.isfinite(yf))), "non-finite state!"
    assert bool(jnp.all(yf >= -1e-9)), "negative concentration!"

    dydt = model2.vector_field(sol.ts[-1], yf, p)
    max_rate = float(jnp.max(jnp.abs(dydt)))
    glyc = float(model2.r_HK(yf, p))
    print(f"\nsteady-state check: max|dy/dt| = {max_rate:.3e} mM/h "
          f"({abs(max_rate / glyc) * 100:.3f}% of glycolytic flux)")
    assert abs(max_rate / glyc) < 1e-3, "did not reach steady state (increase t1)"

    fl = model2.fluxes(yf, p)
    print("\nsteady-state fluxes (mM/h):")
    for name, val in fl.items():
        print(f"  {name:8s} = {float(val): .6e}")
    print(f"\nflux balance: HK={float(fl['HK']):.4e}  PFK={float(fl['PFK']):.4e}  "
          f"PK={float(fl['PK']):.4e}  (lower chain = 2x upper)")


def main() -> None:
    run_model1()
    print()
    run_model2()


if __name__ == "__main__":
    main()
