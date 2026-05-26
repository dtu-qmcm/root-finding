"""Model 1 — a small textbook reaction network solved with diffrax.

Reactions (all reversible, reversible mass-action kinetics):

    R1:  A_e            <->  B_c
    R2:  B_c  + X1_c    <->  C_c + X2_c
    R3:  C_c            <->  D_e
    R4:  X2_c + Y2_e    <->  X1_c + Y1_e

Metabolites tagged ``_c`` are *balanced* (ODE state variables); those tagged
``_e`` are *boundary conditions* held at a fixed concentration.

    state    : B_c, C_c, X1_c, X2_c
    boundary : A_e, D_e, Y1_e, Y2_e

The X1/X2 pool is a conserved moiety (X1_c + X2_c = const), which we use as a
correctness check on the integration.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import diffrax

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------- #
# Parameter structure: access as p.R1.kf, p.R2.kr, ...                        #
# --------------------------------------------------------------------------- #
class Reaction(eqx.Module):
    """Forward / reverse mass-action rate constants for one reversible step."""

    kf: float
    kr: float


class Params(eqx.Module):
    R1: Reaction
    R2: Reaction
    R3: Reaction
    R4: Reaction


class Boundary(eqx.Module):
    """Fixed (clamped) boundary metabolite concentrations."""

    A_e: float
    D_e: float
    Y1_e: float
    Y2_e: float


def default_params() -> Params:
    return Params(
        R1=Reaction(kf=1.0, kr=0.5),
        R2=Reaction(kf=2.0, kr=0.5),
        R3=Reaction(kf=1.5, kr=0.3),
        R4=Reaction(kf=1.0, kr=0.4),
    )


def default_boundary() -> Boundary:
    return Boundary(A_e=2.0, D_e=0.1, Y1_e=0.5, Y2_e=1.0)


# --------------------------------------------------------------------------- #
# Reaction rates and vector field                                             #
# --------------------------------------------------------------------------- #
def reaction_rates(y, p: Params, bc: Boundary):
    """Reversible mass-action rate of each of the four reactions."""
    B, C, X1, X2 = y
    v1 = p.R1.kf * bc.A_e - p.R1.kr * B
    v2 = p.R2.kf * B * X1 - p.R2.kr * C * X2
    v3 = p.R3.kf * C - p.R3.kr * bc.D_e
    v4 = p.R4.kf * X2 * bc.Y2_e - p.R4.kr * X1 * bc.Y1_e
    return jnp.array([v1, v2, v3, v4])


# Stoichiometric matrix S: rows = (B, C, X1, X2), cols = (v1, v2, v3, v4)
#         v1  v2  v3  v4
#  B   [   1  -1   0   0 ]
#  C   [   0   1  -1   0 ]
#  X1  [   0  -1   0   1 ]
#  X2  [   0   1   0  -1 ]
STOICH = jnp.array(
    [
        [1.0, -1.0, 0.0, 0.0],
        [0.0, 1.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, -1.0],
    ]
)


@jax.jit
def vector_field(t, y, args):
    """dy/dt = S @ v(y).  y = (B_c, C_c, X1_c, X2_c)."""
    p, bc = args
    v = reaction_rates(y, p, bc)
    return STOICH @ v


def simulate(
    y0=None,
    p: Params | None = None,
    bc: Boundary | None = None,
    t1: float = 50.0,
    n_save: int = 200,
):
    """Integrate the model from y0 to t1 with an adaptive explicit solver."""
    if y0 is None:
        y0 = jnp.array([0.1, 0.1, 0.5, 0.5])  # B, C, X1, X2
    if p is None:
        p = default_params()
    if bc is None:
        bc = default_boundary()

    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, t1, n_save))
    controller = diffrax.PIDController(rtol=1e-8, atol=1e-10)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=t1,
        dt0=0.01,
        y0=y0,
        args=(p, bc),
        saveat=saveat,
        stepsize_controller=controller,
        max_steps=100_000,
    )
    return sol
