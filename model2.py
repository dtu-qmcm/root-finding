"""Model 2 — glycolysis bistability model (exact published rate laws).

Reproduces the right-hand side of the kinetic model in:

    Mulukutla, Yongky, Daoutidis, Hu (2014)
    "Bistability in Glycolysis Pathway as a Physiological Switch in Energy
    Metabolism", PLoS ONE, DOI 10.1371/journal.pone.0098756 (PMC4049617).

The rate equations were extracted from the supplement's MathType images and are
transcribed in `EQUATIONS.md`. This module implements them verbatim. Forms
follow Mulquiney & Kuchel (1999) Biochem J 342:581-596 (King-Altman steady-state
laws), with Monod-Wyman-Changeux allostery for PFK and PK, the bifunctional
PFKFB kinase/bisphosphatase cycle, GLUT1, and mitochondrial pyruvate transport.

Parameter provenance (every constant is `p.<enz>.<param>`):
  * PFK, PK, PFK2, F26BPase, ALDO, PGK, ENO — EXACT values listed in the
    supplement (Information S1) and recovered from the equation images.
  * HK, GPI, TPI, GAPDH, PGM, LDH, GLUT, PYRH — the supplement states these were
    "adopted from Mulquiney & Kuchel 1999" and does not re-list them; the values
    here are representative literature/erythrocyte constants (marked `# repr.`).
  * Cofactors (ATP, ADP, Mg, NAD, ... ) are clamped at the Table S4 values, so
    the bi-bi / ter-ter laws reduce to functions of the 12 state metabolites.
  * `level` on each enzyme is the relative abundance (Vmax multiplier). The paper
    sets these from transcriptomics; absolute values are not given, so they are
    chosen here to yield a stable positive steady state.

State vector (mM), fixed order:
    GLC, G6P, F6P, F16BP, F26BP, DHAP, GAP, BPG13, PG3, PG2, PEP, PYR
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import diffrax

jax.config.update("jax_enable_x64", True)

SPECIES = ["GLC", "G6P", "F6P", "F16BP", "F26BP", "DHAP",
           "GAP", "BPG13", "PG3", "PG2", "PEP", "PYR"]
IDX = {s: i for i, s in enumerate(SPECIES)}


# ======================================================================= #
# Clamped cofactor / boundary concentrations (Table S4 + boundary).       #
# ======================================================================= #
class Fixed(eqx.Module):
    GLCe: float = 5.0        # extracellular glucose (boundary input)
    ATP: float = 0.31
    ADP: float = 0.54
    AMP: float = 0.03
    MgATP: float = 2.69
    MgADP: float = 0.46
    Mg: float = 0.7
    Pi: float = 2.5
    NAD: float = 0.299
    NADH: float = 0.001
    BPG23: float = 3.0       # 2,3-bisphosphoglycerate
    GSH: float = 0.57
    G16BP: float = 0.024     # glucose-1,6-bisphosphate (repr.)
    LAC: float = 2.5         # cytosolic lactate
    ALA: float = 0.2
    PYRmito: float = 0.1
    Hc: float = 10 ** (-7.3) * 1e3   # cytosolic [H+] in mM (pH 7.3)
    Hm: float = 10 ** (-8.0) * 1e3   # mitochondrial [H+] in mM (pH 8.0)


# ----------------------- enzyme parameter modules ----------------------- #
class HK(eqx.Module):                       # constants repr. (M&K 1999, HK2)
    level: float = 1.0
    Vmf: float = 5.0
    Vmr: float = 1.0
    K_GLC: float = 0.1
    K_G6P: float = 0.047
    K_MgATP: float = 1.0
    Ki_MgATP: float = 1.0
    Ki_MgADP: float = 1.0
    Ki_G6P: float = 0.074
    Ki_G16BP: float = 0.03
    Ki_23BPG: float = 4.0
    Ki_GSH: float = 3.0


class GPI(eqx.Module):                       # constants repr. (M&K 1999)
    level: float = 1.0
    Vmf: float = 1500.0
    Vmr: float = 1500.0
    Kf: float = 0.4          # K_G6P
    Kr: float = 0.12         # K_F6P


class PFK(eqx.Module):                        # EXACT (Information S1)
    level: float = 1.0
    Vf: float = 1550.0
    Vr: float = 67.8
    K_F6P: float = 0.06
    K_MgATP: float = 0.068
    K_MgADP: float = 0.54
    K_F16BP: float = 0.65    # PFKL isozyme (Table 1)
    K_F26BP: float = 0.0055
    K_G16BP: float = 0.1
    K_ATP: float = 0.1
    K_AMP: float = 0.3
    K_Mg: float = 0.2
    K_Pi: float = 30.0
    K_23BPG: float = 0.5
    L: float = 2.0e-3


class PFK2(eqx.Module):                       # EXACT (Information S1)
    level: float = 1.0
    Vf: float = 41.6
    Km_ATP: float = 0.15
    Km_F6P: float = 0.032
    Km_F26BP: float = 0.008
    Km_ADP: float = 0.062
    Ki_ATP: float = 0.15
    Ki_F6P: float = 0.001
    Ki_F26BP: float = 0.02
    Ki_ADP: float = 0.23
    Ki_PEP: float = 0.013
    Keq: float = 16.0


class F26BPase(eqx.Module):                   # EXACT (Information S1)
    level: float = 1.0
    V: float = 11.78
    Km_F26BP: float = 1.0e-3
    Ki_F6P: float = 25.0e-3


class ALDO(eqx.Module):                       # EXACT (Information S1)
    level: float = 1.0
    Vmf: float = 675.0
    Vmr: float = 2320.0
    K_F16BP: float = 0.05
    Ki_F16BP: float = 0.0198
    K_DHAP: float = 0.035
    Ki_DHAP: float = 0.011
    K_GAP: float = 0.189
    Ki_23BPG: float = 1.5


class TPI(eqx.Module):                        # constants repr. (M&K 1999)
    level: float = 1.0
    Vmf: float = 10000.0
    Vmr: float = 10000.0
    Kf: float = 0.16         # K_DHAP
    Kr: float = 0.43         # K_GAP


class GAPDH(eqx.Module):
    """Ter-ter ping-pong; with NAD/NADH/Pi/H+ clamped it reduces to a reversible
    GAP<->13BPG law. Effective constants repr. (M&K 1999)."""
    level: float = 1.0
    Vmf: float = 3000.0
    Vmr: float = 3000.0
    Kf: float = 0.1          # effective K_GAP (lumps NAD/Pi)
    Kr: float = 0.1          # effective K_13BPG (lumps NADH/H+)
    Keq_eff: float = 1.0     # effective GAP->13BPG equilibrium (cofactors folded)


class PGK(eqx.Module):                        # EXACT (Information S1)
    level: float = 1.0
    Vmf: float = 5.96e4
    Vmr: float = 2.39e4
    K_MgADP: float = 0.1
    Ki_MgADP: float = 0.08
    K_13BPG: float = 0.002
    Ki_13BPG: float = 1.6
    K_MgATP: float = 1.0
    Ki_MgATP: float = 0.186
    K_3PG: float = 1.1
    Ki_3PG: float = 0.205


class PGM(eqx.Module):                        # constants repr. (M&K 1999)
    level: float = 1.0
    Vmf: float = 5000.0
    Vmr: float = 5000.0
    K_3PG: float = 0.2
    K_2PG: float = 0.014


class ENO(eqx.Module):                        # EXACT (Information S1)
    level: float = 1.0
    Vmf: float = 2.106e4
    Vmr: float = 5.542e3
    K_Mg: float = 0.14       # = Ki_Mg
    K_PEP: float = 0.11      # = Ki_PEP
    K_2PG: float = 0.046     # = Ki_2PG


class PK(eqx.Module):                         # EXACT (Information S1)
    level: float = 1.0
    Vf: float = 2.02e4
    Vr: float = 47.5
    K_PEP: float = 0.225
    K_MgADP: float = 0.474
    K_MgATP: float = 3.0
    K_ATP: float = 3.39
    K_PYR: float = 4.0
    K_F16BP: float = 0.04    # PKM2 isozyme (Table 1)
    K_G16BP: float = 0.1
    K_ALA: float = 0.02
    L: float = 0.398


class LDH(eqx.Module):                        # form exact; constants repr.
    level: float = 1.0
    Vmf: float = 2000.0
    Vmr: float = 200.0
    K_PYR: float = 0.3
    Ki_PYR: float = 10.0     # substrate inhibition
    K_LAC: float = 5.0
    Ki_NADH: float = 0.002
    Ki_NAD: float = 0.5


class GLUT(eqx.Module):                       # constants repr. (GLUT1)
    level: float = 1.0
    Vmf: float = 50.0
    Vmr: float = 50.0
    K_GLC: float = 1.5


class PYRH(eqx.Module):                       # mito pyruvate transport
    level: float = 1.0
    Vmf: float = 5.0e5       # absorbs the tiny [H+] scale


class Params(eqx.Module):
    fixed: Fixed
    HK: HK
    GPI: GPI
    PFK: PFK
    PFK2: PFK2
    F26BPase: F26BPase
    ALDO: ALDO
    TPI: TPI
    GAPDH: GAPDH
    PGK: PGK
    PGM: PGM
    ENO: ENO
    PK: PK
    LDH: LDH
    GLUT: GLUT
    PYRH: PYRH


def default_params(**fixed_overrides) -> Params:
    return Params(
        fixed=Fixed(**fixed_overrides),
        HK=HK(), GPI=GPI(), PFK=PFK(), PFK2=PFK2(), F26BPase=F26BPase(),
        ALDO=ALDO(), TPI=TPI(), GAPDH=GAPDH(), PGK=PGK(), PGM=PGM(),
        ENO=ENO(), PK=PK(), LDH=LDH(), GLUT=GLUT(), PYRH=PYRH(),
    )


# ======================================================================= #
# Rate laws — exact forms from EQUATIONS.md (cofactors taken from p.fixed) #
# ======================================================================= #
def r_GLUT(y, p):
    f, e = p.fixed, p.GLUT
    glc = y[IDX["GLC"]]
    num = e.Vmf * f.GLCe / e.K_GLC - e.Vmr * glc / e.K_GLC
    return e.level * num / (1.0 + f.GLCe / e.K_GLC + glc / e.K_GLC)


def r_HK(y, p):
    f, e = p.fixed, p.HK
    glc, g6p = y[IDX["GLC"]], y[IDX["G6P"]]
    num = (e.Vmf * f.MgATP * glc / (e.K_MgATP * e.K_GLC)
           - e.Vmr * f.MgADP * g6p / (e.Ki_MgADP * e.K_G6P))
    N = (1.0 + f.MgATP / e.Ki_MgATP + g6p / e.Ki_G6P + glc / e.K_GLC
         + f.MgATP * glc / (e.K_MgATP * e.K_GLC) + f.MgADP / e.Ki_MgADP
         + f.MgADP * g6p / (e.Ki_MgADP * e.K_G6P) + glc * g6p / (e.K_GLC * e.K_G6P)
         + glc * f.G16BP / (e.K_GLC * e.Ki_G16BP)
         + glc * f.BPG23 / (e.K_GLC * e.Ki_23BPG)
         + glc * f.GSH / (e.K_GLC * e.Ki_GSH))
    return e.level * num / N


def r_GPI(y, p):
    e = p.GPI
    g6p, f6p = y[IDX["G6P"]], y[IDX["F6P"]]
    num = e.Vmf * g6p / e.Kf - e.Vmr * f6p / e.Kr
    return e.level * num / (1.0 + g6p / e.Kf + f6p / e.Kr)


def _mwc_bibi(s1, s2, p1, p2, Ks1, Ks2, Kp1, Kp2):
    """Two-substrate / two-product ordered bi-bi base denominator for the MWC
    R-state velocity: (1+s1/Ks1)(1+s2/Ks2) + (1+p1/Kp1)(1+p2/Kp2) - 1."""
    return (1.0 + s1 / Ks1) * (1.0 + s2 / Ks2) + (1.0 + p1 / Kp1) * (1.0 + p2 / Kp2) - 1.0


def r_PFK(y, p):
    f, e = p.fixed, p.PFK
    f6p, f16bp, f26bp = y[IDX["F6P"]], y[IDX["F16BP"]], y[IDX["F26BP"]]
    num = (e.Vf * f6p * f.MgATP / (e.K_F6P * e.K_MgATP)
           - e.Vr * f16bp * f.MgADP / (e.K_F16BP * e.K_MgADP))
    base = _mwc_bibi(f6p, f.MgATP, f16bp, f.MgADP,
                     e.K_F6P, e.K_MgATP, e.K_F16BP, e.K_MgADP)
    t_state = (e.L * (1.0 + f.ATP / e.K_ATP) ** 4 * (1.0 + f.Mg / e.K_Mg) ** 4
               * (1.0 + f.BPG23 / e.K_23BPG) ** 4)
    r_state = ((1.0 + f6p / e.K_F6P + f16bp / e.K_F16BP) ** 4
               * (1.0 + f.AMP / e.K_AMP) ** 4 * (1.0 + f.G16BP / e.K_G16BP) ** 4
               * (1.0 + f.Pi / e.K_Pi) ** 4 * (1.0 + f26bp / e.K_F26BP) ** 4)
    N_PFK = 1.0 + t_state / r_state
    return e.level * num / base / N_PFK


def r_PFK2(y, p):
    f, e = p.fixed, p.PFK2
    f6p, f26bp, pep = y[IDX["F6P"]], y[IDX["F26BP"]], y[IDX["PEP"]]
    num = e.Vf * (f.ATP * f6p - f.ADP * f26bp / e.Keq)
    D = (e.Ki_ATP * e.Km_F6P + e.Km_F6P * f.ATP + e.Km_ATP * f6p
         + e.Km_ADP * f26bp / e.Keq + e.Km_F26BP * f.ADP / e.Keq + f.ATP * f6p
         + e.Km_ADP * f.ATP * f26bp / (e.Keq * e.Ki_ATP) + f.ADP * f26bp / e.Keq
         + e.Km_ATP * f.ADP * f6p / e.Ki_ADP + f.ATP * f6p * f26bp / e.Ki_F26BP
         + f.ADP * f6p * f26bp / (e.Keq * e.Ki_F6P))
    return e.level * num / (D * (1.0 + pep / e.Ki_PEP))


def r_F26BPase(y, p):
    e = p.F26BPase
    f6p, f26bp = y[IDX["F6P"]], y[IDX["F26BP"]]
    return e.level * e.V * f26bp / ((1.0 + f6p / e.Ki_F6P) * (e.Km_F26BP + f26bp))


def r_ALDO(y, p):
    f, e = p.fixed, p.ALDO
    f16bp, dhap, gap = y[IDX["F16BP"]], y[IDX["DHAP"]], y[IDX["GAP"]]
    num = e.Vmf * f16bp / e.K_F16BP - e.Vmr * gap * dhap / (e.K_GAP * e.Ki_DHAP)
    D = (1.0 + f.BPG23 / e.Ki_23BPG + f16bp / e.K_F16BP
         + (e.K_DHAP * gap / (e.K_GAP * e.Ki_DHAP)) * (1.0 + f.BPG23 / e.Ki_23BPG)
         + dhap / e.Ki_DHAP
         + e.K_DHAP * f16bp * gap / (e.Ki_F16BP * e.K_GAP * e.Ki_DHAP)
         + dhap * gap / (e.K_GAP * e.Ki_DHAP))
    return e.level * num / D


def r_TPI(y, p):
    e = p.TPI
    dhap, gap = y[IDX["DHAP"]], y[IDX["GAP"]]
    num = e.Vmf * dhap / e.Kf - e.Vmr * gap / e.Kr
    return e.level * num / (1.0 + dhap / e.Kf + gap / e.Kr)


def r_GAPDH(y, p):
    e = p.GAPDH
    gap, bpg = y[IDX["GAP"]], y[IDX["BPG13"]]
    num = e.Vmf * gap / e.Kf - e.Vmr * bpg / (e.Kr * e.Keq_eff)
    return e.level * num / (1.0 + gap / e.Kf + bpg / e.Kr)


def r_PGK(y, p):
    f, e = p.fixed, p.PGK
    bpg, pg3 = y[IDX["BPG13"]], y[IDX["PG3"]]
    num = (e.Vmf * bpg * f.MgADP / (e.Ki_MgADP * e.K_13BPG)
           - e.Vmr * pg3 * f.MgATP / (e.Ki_MgATP * e.K_3PG))
    D = (1.0 + bpg / e.Ki_13BPG + f.MgADP / e.Ki_MgADP
         + bpg * f.MgADP / (e.Ki_MgADP * e.K_13BPG) + pg3 / e.Ki_3PG
         + f.MgATP / e.Ki_MgATP + pg3 * f.MgATP / (e.Ki_MgATP * e.K_3PG))
    return e.level * num / D


def r_PGM(y, p):
    e = p.PGM
    pg3, pg2 = y[IDX["PG3"]], y[IDX["PG2"]]
    num = e.Vmf * pg3 / e.K_3PG - e.Vmr * pg2 / e.K_2PG
    return e.level * num / (1.0 + pg3 / e.K_3PG + pg2 / e.K_2PG)


def r_ENO(y, p):
    f, e = p.fixed, p.ENO
    pg2, pep = y[IDX["PG2"]], y[IDX["PEP"]]
    num = (e.Vmf * pg2 * f.Mg / (e.K_Mg * e.K_2PG)
           - e.Vmr * pep * f.Mg / (e.K_Mg * e.K_PEP))
    D = (1.0 + pg2 / e.K_2PG + f.Mg / e.K_Mg + pg2 * f.Mg / (e.K_Mg * e.K_2PG)
         + pep / e.K_PEP + f.Mg / e.K_Mg + pep * f.Mg / (e.K_Mg * e.K_PEP))
    return e.level * num / D


def r_PK(y, p):
    f, e = p.fixed, p.PK
    pep, pyr, f16bp = y[IDX["PEP"]], y[IDX["PYR"]], y[IDX["F16BP"]]
    num = (e.Vf * pep * f.MgADP / (e.K_PEP * e.K_MgADP)
           - e.Vr * pyr * f.MgATP / (e.K_PYR * e.K_MgATP))
    base = _mwc_bibi(pep, f.MgADP, pyr, f.MgATP,
                     e.K_PEP, e.K_MgADP, e.K_PYR, e.K_MgATP)
    t_state = e.L * (1.0 + f.ATP / e.K_ATP) ** 4 * (1.0 + f.ALA / e.K_ALA) ** 4
    r_state = ((1.0 + pep / e.K_PEP + pyr / e.K_PYR) ** 4
               * (1.0 + f16bp / e.K_F16BP + f.G16BP / e.K_G16BP) ** 4)
    N_PK = 1.0 + t_state / r_state
    return e.level * num / base / N_PK


def r_LDH(y, p):
    f, e = p.fixed, p.LDH
    pyr = y[IDX["PYR"]]
    num = (e.Vmf * f.NADH * pyr / (e.Ki_NADH * e.K_PYR)
           - e.Vmr * f.NAD * f.LAC / (e.Ki_NAD * e.K_LAC))
    # ordered bi-bi denominator (NAD/NADH/LAC clamped) with substrate inhibition
    D = (1.0 + pyr / e.Ki_PYR) * (
        1.0 + f.NADH / e.Ki_NADH + f.NAD / e.Ki_NAD + pyr / e.K_PYR
        + f.LAC / e.K_LAC + f.NADH * pyr / (e.Ki_NADH * e.K_PYR)
        + f.NAD * f.LAC / (e.Ki_NAD * e.K_LAC))
    return e.level * num / D


def r_PYRH(y, p):
    f, e = p.fixed, p.PYRH
    pyr = y[IDX["PYR"]]
    return e.level * e.Vmf * (pyr * f.Hc - f.PYRmito * f.Hm)


def fluxes(y, p):
    return {
        "GLUT": r_GLUT(y, p), "HK": r_HK(y, p), "GPI": r_GPI(y, p),
        "PFK": r_PFK(y, p), "PFK2": r_PFK2(y, p), "F26BPase": r_F26BPase(y, p),
        "ALDO": r_ALDO(y, p), "TPI": r_TPI(y, p), "GAPDH": r_GAPDH(y, p),
        "PGK": r_PGK(y, p), "PGM": r_PGM(y, p), "ENO": r_ENO(y, p),
        "PK": r_PK(y, p), "LDH": r_LDH(y, p), "PYRH": r_PYRH(y, p),
    }


# ======================================================================= #
# Vector field: 12 mass balances (confirmed against Information S1)        #
# ======================================================================= #
def vector_field(t, y, args):
    p = args
    vGLUT = r_GLUT(y, p); vHK = r_HK(y, p); vGPI = r_GPI(y, p)
    vPFK = r_PFK(y, p); vPFK2 = r_PFK2(y, p); vF26 = r_F26BPase(y, p)
    vALDO = r_ALDO(y, p); vTPI = r_TPI(y, p); vGAPDH = r_GAPDH(y, p)
    vPGK = r_PGK(y, p); vPGM = r_PGM(y, p); vENO = r_ENO(y, p)
    vPK = r_PK(y, p); vLDH = r_LDH(y, p); vPYRH = r_PYRH(y, p)

    return jnp.array([
        vGLUT - vHK,                 # GLC
        vHK - vGPI,                  # G6P
        vGPI - vPFK - vPFK2 + vF26,  # F6P
        vPFK - vALDO,                # F16BP
        vPFK2 - vF26,                # F26BP
        vALDO - vTPI,                # DHAP
        vALDO + vTPI - vGAPDH,       # GAP
        vGAPDH - vPGK,               # BPG13
        vPGK - vPGM,                 # PG3
        vPGM - vENO,                 # PG2
        vENO - vPK,                  # PEP
        vPK - vLDH - vPYRH,          # PYR
    ])


def simulate(p: Params | None = None, y0=None, t1: float = 5000.0, n_save: int = 200):
    if p is None:
        p = default_params()
    if y0 is None:
        y0 = jnp.array([1.0, 0.1, 0.1, 0.02, 0.005, 0.04, 0.02,
                        0.01, 0.05, 0.01, 0.02, 0.05])
    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Kvaerno5()
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, t1, n_save))
    controller = diffrax.PIDController(rtol=1e-6, atol=1e-9)
    return diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=t1, dt0=1e-4, y0=y0, args=p,
        saveat=saveat, stepsize_controller=controller, max_steps=2_000_000,
    )
