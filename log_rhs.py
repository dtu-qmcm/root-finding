"""Log-transformed right-hand side.

For a system  dx/dt = f(t, x, args)  with strictly positive state x > 0, we can
integrate in log space. Let u = log(x); then x = exp(u) and

    d log(x)/dt = (dx/dt) · (1/x) = f(t, exp(u), args) / exp(u).

Integrating `u` instead of `x` keeps every state strictly positive by
construction and rescales multiplicative dynamics, which is often better
conditioned for stiff kinetic models whose concentrations span many orders of
magnitude (e.g. `model2`, where states range over ~10^-4 to 10^0 mM).

`log_vector_field` wraps any `vector_field(t, x, args)` into its log-space form.
Pre-wrapped versions for both models are provided, along with `simulate_log`,
which takes the initial condition in ORIGINAL space and returns the trajectory
back-transformed to original space.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import diffrax

import model1
import model2


def log_vector_field(vector_field):
    """Return d log(x)/dt = f(t, exp(u), args) / exp(u) for state u = log(x)."""

    @jax.jit
    def f_log(t, u, args):
        x = jnp.exp(u)
        return vector_field(t, x, args) / x

    return f_log


# Pre-wrapped log-space RHS for each model.
model1_log_vector_field = log_vector_field(model1.vector_field)
model2_log_vector_field = log_vector_field(model2.vector_field)


def simulate_log(
    vector_field,
    y0,
    args,
    *,
    solver,
    t1: float,
    dt0: float,
    t0: float = 0.0,
    n_save: int = 200,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_steps: int = 2_000_000,
):
    """Integrate `vector_field` in log space.

    Parameters
    ----------
    y0 : initial state in ORIGINAL space (every entry must be > 0).
    solver, t1, dt0, ... : passed to `diffrax.diffeqsolve`.

    Returns
    -------
    (ts, xs) : times and the trajectory back-transformed to original space
               (xs[i] = exp(u(ts[i]))).
    """
    term = diffrax.ODETerm(log_vector_field(vector_field))
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, n_save))
    controller = diffrax.PIDController(rtol=rtol, atol=atol)
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=jnp.log(y0),
        args=args,
        saveat=saveat,
        stepsize_controller=controller,
        max_steps=max_steps,
    )
    return sol.ts, jnp.exp(sol.ys)
