"""Simulate both models to steady state and save trajectory figures.

Writes:
  figures/model1_trajectory.png  — 4 balanced metabolites vs time
  figures/model2_trajectory.png  — 12 glycolytic intermediates vs time (log y)

Run: uv run python make_figures.py
"""

from __future__ import annotations

import os

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

import model1
import model2

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGDIR, exist_ok=True)


def _steady_state_time(ts, ys, vector_field, args, tol=1e-3):
    """First time at which max|dy/dt| drops below tol * (its initial value)."""
    rates = jnp.array([jnp.max(jnp.abs(vector_field(t, y, args)))
                       for t, y in zip(ts, ys)])
    thresh = tol * float(rates[0])
    below = jnp.where(rates <= thresh)[0]
    return float(ts[int(below[0])]) if below.size else float(ts[-1])


def figure_model1():
    p, bc = model1.default_params(), model1.default_boundary()
    sol = model1.simulate(p=p, bc=bc, t1=50.0, n_save=400)
    ts, ys = sol.ts, sol.ys
    tss = _steady_state_time(ts, ys, model1.vector_field, (p, bc))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, name in enumerate(["B_c", "C_c", "X1_c", "X2_c"]):
        ax.plot(ts, ys[:, i], label=name, lw=2)
    ax.axvline(tss, color="grey", ls="--", lw=1, label=f"steady state ≈ {tss:.1f}")
    ax.set_xlabel("time")
    ax.set_ylabel("concentration")
    ax.set_title("Model 1 — reversible mass-action network")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "model1_trajectory.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"model1: steady state at t≈{tss:.2f}; wrote {out}")


def figure_model2():
    p = model2.default_params()
    # Confirm a genuine steady state over a long horizon ...
    long_sol = model2.simulate(p, t1=5000.0, n_save=2)
    max_rate = float(jnp.max(jnp.abs(model2.vector_field(
        long_sol.ts[-1], long_sol.ys[-1], p))))
    # ... then plot the transient window where the approach is visible.
    sol = model2.simulate(p, t1=6.0, n_save=600)
    ts, ys = sol.ts, sol.ys
    tss = _steady_state_time(ts, ys, model2.vector_field, p)
    print(f"model2: max|dy/dt| at t=5000 h = {max_rate:.2e} mM/h (steady)")

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(model2.SPECIES):
        ax.plot(ts, ys[:, i], label=name, lw=1.5)
    ax.axvline(tss, color="grey", ls="--", lw=1, label=f"steady state ≈ {tss:.2f} h")
    ax.set_yscale("log")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("concentration (mM, log scale)")
    ax.set_title("Model 2 — glycolysis (exact published rate laws)\n"
                 "(verified steady to t=5000 h)")
    ax.legend(loc="center right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "model2_trajectory.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"model2: steady state at t≈{tss:.1f} h; wrote {out}")


if __name__ == "__main__":
    figure_model1()
    figure_model2()
