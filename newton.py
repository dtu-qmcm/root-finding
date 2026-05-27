"""Newton's method for algebraic systems, with the Jacobian from JAX autodiff.

Given a residual ``F(y) = 0`` we iterate

    y_{k+1} = y_k - alpha * J(y_k)^{-1} F(y_k),   J = dF/dy,

where the Jacobian ``J`` is obtained by automatic differentiation
(``jax.jacobian``) rather than by hand, and ``alpha`` is an optional relaxation
(damping) factor in (0, 1].

The examples find the steady states (roots of ``dy/dt = 0``) of two models:

  * ``model1`` — small mass-action network.  Its vector field is rank-deficient
    (the conserved moiety X1_c + X2_c makes one ODE redundant), so its Jacobian
    is singular and plain ``solve`` would fail.  We restore a full-rank square
    system by replacing the redundant row with ``X1_c + X2_c - pool = 0``.

  * ``model2`` — glycolysis (12 states).  Its Jacobian is full rank, so the
    plain ``dy/dt`` residual works directly.  The model is stiff, so we hot-start
    Newton from a short time integration (a cold start diverges).

Run: uv run python newton.py
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

import model1
import model2

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------- #
# Generic Newton solver                                                       #
# --------------------------------------------------------------------------- #
@eqx.filter_jit
def newton_solve(residual, y0, args=None, *, alpha=1.0, atol=1e-12, max_steps=50):
    """Solve ``residual(y, args) = 0`` by Newton's method with autodiff Jacobian.

    Returns ``(y, residual_norm, n_iters)``.
    """
    jac = jax.jacobian(residual)

    def cond(state):
        y, k = state
        converged = jnp.max(jnp.abs(residual(y, args))) <= atol
        return (k < max_steps) & jnp.logical_not(converged)

    def body(state):
        y, k = state
        F = residual(y, args)
        J = jac(y, args)
        dy = jnp.linalg.solve(J, F)
        return y - alpha * dy, k + 1

    y, k = jax.lax.while_loop(cond, body, (y0, 0))
    return y, jnp.max(jnp.abs(residual(y, args))), k


# --------------------------------------------------------------------------- #
# model1 steady-state residual (augmented to full rank)                       #
# --------------------------------------------------------------------------- #
def steady_residual(y, args):
    """F(y) = dy/dt, with the redundant X2 row swapped for moiety conservation."""
    p, bc, pool = args
    f = model1.vector_field(0.0, y, (p, bc))      # rank-3: X1/X2 rows redundant
    return f.at[3].set(y[2] + y[3] - pool)        # X1_c + X2_c - pool = 0


def run_model1() -> None:
    y0 = jnp.array([0.1, 0.1, 0.5, 0.5])          # B_c, C_c, X1_c, X2_c
    pool = float(y0[2] + y0[3])                    # conserved X1_c + X2_c = 1.0
    args = (model1.default_params(), model1.default_boundary(), pool)

    y, resid, n_iters = newton_solve(steady_residual, y0, args)

    print("=== Newton's method (autodiff Jacobian) on model1 ===")
    print(f"converged in {int(n_iters)} iterations, residual max|F| = {float(resid):.2e}")

    print("\nsteady state:")
    for name, val in zip(["B_c", "C_c", "X1_c", "X2_c"], y):
        print(f"  {name:5s} = {float(val): .6f}")

    p, bc, _ = args
    dydt = model1.vector_field(0.0, y, (p, bc))
    print(f"\nsteady-state check: max|dy/dt| = {float(jnp.max(jnp.abs(dydt))):.2e}")
    print(f"moiety X1_c + X2_c = {float(y[2] + y[3]):.6f} (target {pool:.6f})")

    # Validate against the time-integration steady state in model1.simulate.
    y_sim = model1.simulate(y0=y0, p=p, bc=bc).ys[-1]
    rel = float(jnp.max(jnp.abs(y - y_sim) / jnp.abs(y_sim)))
    print(f"\nmax relative difference vs diffrax integration: {rel:.2e}")
    assert rel < 1e-6, "Newton steady state disagrees with time integration!"
    print("agreement with diffrax integration OK")


def run_model2() -> None:
    p = model2.default_params()
    y0 = jnp.array([1.0, 0.1, 0.1, 0.02, 0.005, 0.04, 0.02,
                    0.01, 0.05, 0.01, 0.02, 0.05])

    # model2 is stiff; hot-start Newton from a short integration (0.5 h).
    y_hot = model2.simulate(p, y0=y0, t1=0.5).ys[-1]
    residual = lambda y, args: model2.vector_field(0.0, y, args)
    y, resid, n_iters = newton_solve(residual, y_hot, p, max_steps=200)

    print("=== Newton's method (autodiff Jacobian) on model2 ===")
    print(f"hot-started from 0.5 h of integration; converged in {int(n_iters)} "
          f"iterations, residual max|F| = {float(resid):.2e}")

    print("\nsteady-state metabolites (mM):")
    for s, val in zip(model2.SPECIES, y):
        print(f"  {s:6s} = {float(val): .6e}")

    assert bool(jnp.all(y > 0)), "negative concentration!"
    dydt = model2.vector_field(0.0, y, p)
    print(f"\nsteady-state check: max|dy/dt| = {float(jnp.max(jnp.abs(dydt))):.2e}")

    # Validate against the full time integration in model2.simulate.
    y_sim = model2.simulate(p, y0=y0).ys[-1]
    rel = float(jnp.max(jnp.abs(y - y_sim) / jnp.abs(y_sim)))
    print(f"\nmax relative difference vs diffrax integration: {rel:.2e}")
    assert rel < 1e-6, "Newton steady state disagrees with time integration!"
    print("agreement with diffrax integration OK")


def main() -> None:
    run_model1()
    print()
    run_model2()


if __name__ == "__main__":
    main()
