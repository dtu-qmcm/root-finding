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

import diffrax
import jax.numpy as jnp

import log_rhs
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


def run_model2_logspace() -> None:
    """Integrate model 2 in log space and compare to direct integration."""
    p = model2.default_params()
    y0 = jnp.array([1.0, 0.1, 0.1, 0.02, 0.005, 0.04, 0.02,
                    0.01, 0.05, 0.01, 0.02, 0.05])

    # log-space RHS: d log(x)/dt = f(t, x) / x  (states stay strictly positive)
    ts, xs = log_rhs.simulate_log(
        model2.vector_field, y0, p, solver=diffrax.Kvaerno5(), t1=5000.0, dt0=1e-4)
    x_log = xs[-1]
    x_direct = model2.simulate(p).ys[-1]

    print("=== Model 2 in log space (d log(x)/dt = f(t,x)/x) ===")
    print("steady-state comparison (mM):")
    print(f"  {'species':6s} {'log-space':>14s} {'direct':>14s} {'rel.diff':>10s}")
    for s, xl, xd in zip(model2.SPECIES, x_log, x_direct):
        rel = abs(float(xl) - float(xd)) / float(xd)
        print(f"  {s:6s} {float(xl):14.6e} {float(xd):14.6e} {rel:10.2e}")
    max_rel = float(jnp.max(jnp.abs(x_log - x_direct) / x_direct))
    print(f"\nmax relative difference vs direct integration: {max_rel:.2e}")
    assert max_rel < 1e-4, "log-space steady state disagrees with direct!"


def main() -> None:
    run_model1()
    print()
    run_model2()
    print()
    run_model2_logspace()


if __name__ == "__main__":
    main()
