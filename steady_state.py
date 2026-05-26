"""Estimate steady state via root finding (optimistix); benchmark log vs non-log."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import optimistix as optx

import log_rhs
import model1
import model2


def make_solver(vector_field, log):
    f = log_rhs.log_vector_field(vector_field) if log else vector_field
    g = lambda y, args: f(0.0, y, args)
    solver = optx.LevenbergMarquardt(rtol=1e-8, atol=1e-10)

    @jax.jit
    def solve(y0, args):
        sol = optx.root_find(g, solver, jnp.log(y0) if log else y0,
                             args=args, throw=False)
        y = jnp.exp(sol.value) if log else sol.value
        return y, jnp.max(jnp.abs(vector_field(0.0, y, args)))
    return solve


def bench(solve, y0, args, n=100):
    y, resid = jax.block_until_ready(solve(y0, args))
    t = time.perf_counter()
    for _ in range(n):
        jax.block_until_ready(solve(y0, args))
    return (time.perf_counter() - t) / n, y, resid


def main():
    p1, bc1 = model1.default_params(), model1.default_boundary()
    y1 = jnp.array([0.1, 0.1, 0.5, 0.5])
    p2 = model2.default_params()
    y2 = jnp.array([1.0, 0.1, 0.1, 0.02, 0.005, 0.04, 0.02,
                    0.01, 0.05, 0.01, 0.02, 0.05])
    y2 = model2.simulate(p2, y0=y2, t1=0.5).ys[-1]  # hot start: 0.5 h of integration
    cases = [
        ("model1", model1.vector_field, y1, (p1, bc1)),
        ("model2", model2.vector_field, y2, p2),
    ]
    for name, vf, y0, args in cases:
        for log in (False, True):
            dt, y, resid = bench(make_solver(vf, log), y0, args)
            tag = name + ("-log" if log else "")
            print(f"{tag:12s} {dt * 1e3:8.3f} ms/solve  max|f|={float(resid):.2e}")


if __name__ == "__main__":
    main()
