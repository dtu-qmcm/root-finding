# Glycolysis model — rate equations extracted from Information S1

Source: Mulukutla et al. (2014), *Bistability in Glycolysis Pathway as a
Physiological Switch in Energy Metabolism*, PLoS ONE,
[DOI 10.1371/journal.pone.0098756](https://doi.org/10.1371/journal.pone.0098756),
Information S1 (`*.s012.docx`).

The rate equations are MathType images in the supplement. They were extracted
by rendering the embedded WMF objects to PNG with LibreOffice and reading them.
Forms follow Mulquiney & Kuchel (1999) Biochem J 342:581-596 (King–Altman
steady-state laws), with Monod–Wyman–Changeux allostery for PFK and PK and the
bifunctional PFKFB cycle from Kitajima (1984) / Kretschmer (1985).

Notation: `[X]` is the cytosolic concentration of X (the paper writes `C^c_X`);
`[GLC_e]` is extracellular glucose; `[PYR_m]`, `[H_m]` are mitochondrial.
Cofactors ATP/ADP/AMP/MgATP/MgADP/Mg/Pi/NAD/NADH/23BPG/GSH/ALA/LAC are clamped.
Parameter blocks were present in the supplement only for the modified/novel
enzymes (PFK, PFK2, F2,6BPase, ALDO, PGK, ENO, PK); for the others the constants
are "adopted from Mulquiney & Kuchel 1999" and are not re-listed there.

---

## Hexokinase (HK) — rapid-equilibrium random bi-bi, mixed inhibition

```
              Vf·[MgATP][GLC]/(K_MgATP·K_GLC) − Vr·[MgADP][G6P]/(Ki_MgADP·K_G6P)
r_HK  =  ───────────────────────────────────────────────────────────────────────
                                       N_HK

N_HK = 1 + [MgATP]/Ki_MgATP + [G6P]/Ki_G6P + [GLC]/K_GLC
         + [MgATP][GLC]/(K_MgATP·K_GLC) + [MgADP]/Ki_MgADP
         + [MgADP][G6P]/(Ki_MgADP·K_G6P) + [GLC][G6P]/(K_GLC·K_G6P)
         + [GLC][G16BP]/(K_GLC·Ki_G16BP) + [GLC][23BPG]/(K_GLC·Ki_23BPG)
         + [GLC][GSH]/(K_GLC·Ki_GSH)
```
Constants: from Mulquiney & Kuchel 1999 (HK2 isozyme; not re-listed in S1).

## Glucose phosphate isomerase (GPI) — reversible uni-uni

```
          Vmf·[G6P]/Kf − Vmr·[F6P]/Kr
r_GPI = ─────────────────────────────────
          1 + [G6P]/Kf + [F6P]/Kr
```
Constants: from Mulquiney & Kuchel 1999.

## Phosphofructokinase (PFK) — MWC two-state, ordered bi-bi

```
          Vf·[F6P][MgATP]/(K_F6P·K_MgATP) − Vr·[F16BP][MgADP]/(K_F16BP·K_MgADP)     1
r_PFK = ─────────────────────────────────────────────────────────────────────── · ─────
        (1+[F6P]/K_F6P)(1+[MgATP]/K_MgATP) + (1+[F16BP]/K_F16BP)(1+[MgADP]/K_MgADP) − 1   N_PFK

                 L_PFK·(1+[ATP]/K_ATP)^4·(1+[Mg]/K_Mg)^4·(1+[23BPG]/K_23BPG)^4
N_PFK = 1 + ──────────────────────────────────────────────────────────────────────────────────────────────
            (1+[F6P]/K_F6P+[F16BP]/K_F16BP)^4·(1+[AMP]/K_AMP)^4·(1+[G16BP]/K_G16BP)^4·(1+[Pi]/K_Pi)^4·(1+[F26BP]/K_F26BP)^4
```
R-state (active) is stabilised by F6P, F16BP, AMP, G16BP, Pi, F26BP;
T-state (inactive) by ATP, Mg, 23BPG. Tetramer → exponent 4.

Constants (mM, rates mM·h⁻¹):
Vf = 1550, Vr = 67.8, K_F6P = 0.06, K_MgATP = 0.068, K_MgADP = 0.54,
K_F16BP = 0.65, K_F26BP = 0.0055, K_G16BP = 0.1, K_ATP = 0.1, K_AMP = 0.3,
K_Mg = 0.2, K_Pi = 30, K_23BPG = 0.5, L_PFK = 2×10⁻³.
(K_F16BP = 0.65 → PFKL isozyme, per Table 1.)

## PFKFB kinase domain (PFK2) — ordered bi-bi, non-competitive PEP inhibition

```
            Vf·([ATP][F6P] − [ADP][F26BP]/Keq)
r_PFK2 = ──────────────────────────────────────────
                    D · (1 + [PEP]/Ki_PEP)

D = Ki_ATP·Km_F6P + Km_F6P·[ATP] + Km_ATP·[F6P]
  + Km_ADP·[F26BP]/Keq + Km_F26BP·[ADP]/Keq + [ATP][F6P]
  + Km_ADP·[ATP][F26BP]/(Keq·Ki_ATP) + [ADP][F26BP]/Keq
  + Km_ATP·[ADP][F6P]/Ki_ADP + [ATP][F6P][F26BP]/Ki_F26BP
  + [ADP][F6P][F26BP]/(Keq·Ki_F6P)
```
Constants: Vf = 41.6, Km_ATP = 0.15, Km_F6P = 0.032, Km_F26BP = 0.008,
Km_ADP = 0.062, Ki_ATP = 0.15, Ki_F6P = 0.001, Ki_F26BP = 0.02,
Ki_ADP = 0.23, Ki_PEP = 0.013, Keq = 16.
(The K/P-ratio isozyme effect is modelled by scaling this Vf.)

## PFKFB bisphosphatase domain (F2,6BPase) — MM, non-competitive F6P inhibition

```
                  V·[F26BP]
r_F26BPase = ──────────────────────────────────────
             (1 + [F6P]/Ki_F6P)·(Km_F26BP + [F26BP])
```
Constants: V = 11.78, Km_F26BP = 1×10⁻³, Ki_F6P = 25×10⁻³.

## Aldolase (ALDO) — ordered uni-bi, reversible

```
          Vmf·[F16BP]/K_F16BP − Vmr·[GAP][DHAP]/(K_GAP·Ki_DHAP)
r_ALDO = ────────────────────────────────────────────────────────
                                  D

D = 1 + [23BPG]/Ki_23BPG + [F16BP]/K_F16BP
  + (K_DHAP·[GAP]/(K_GAP·Ki_DHAP))·(1 + [23BPG]/Ki_23BPG)
  + [DHAP]/Ki_DHAP + K_DHAP·[F16BP][GAP]/(Ki_F16BP·K_GAP·Ki_DHAP)
  + [DHAP][GAP]/(K_GAP·Ki_DHAP)
```
Constants: Vmf = 675, Vmr = 2320, K_F16BP = 0.05, Ki_F16BP = 0.0198,
K_DHAP = 0.035, Ki_DHAP = 0.011, K_GAP = 0.189, Ki_23BPG = 1.5.

## Triose phosphate isomerase (TPI) — reversible uni-uni

```
          Vmf·[DHAP]/Kf − Vmr·[GAP]/Kr
r_TPI = ─────────────────────────────────
          1 + [DHAP]/Kf + [GAP]/Kr
```
Constants: from Mulquiney & Kuchel 1999.

## Glyceraldehyde-3-P dehydrogenase (GAPDH) — ter-ter (bi-uni-uni-bi ping-pong)

Forward: NAD⁺ + Pi + GAP → 1,3BPG + NADH + H⁺. Large reversible King–Altman law:

```
            Vmf·[NAD][Pi][GAP]/(K_NAD·Ki_Pi·Ki_GAP) − Vmr·[13BPG][NADH][H]/(Ki_13BPG·K_NADH)
r_GAPDH = ─────────────────────────────────────────────────────────────────────────────────────
                                                  D
```
with `D` a sum of ~13 King–Altman terms in [GAP], [13BPG], [NAD], [NADH], [Pi],
[H⁺] (the full denominator is reproduced verbatim in Mulquiney & Kuchel 1999,
Table/eqns for GAPDH). Constants: from Mulquiney & Kuchel 1999.

## Phosphoglycerate kinase (PGK) — random bi-bi, reversible

```
          Vmf·[13BPG][MgADP]/(Ki_MgADP·K_13BPG) − Vmr·[3PG][MgATP]/(Ki_MgATP·K_3PG)
r_PGK = ──────────────────────────────────────────────────────────────────────────────
          1 + [13BPG]/Ki_13BPG + [MgADP]/Ki_MgADP + [13BPG][MgADP]/(Ki_MgADP·K_13BPG)
            + [3PG]/Ki_3PG + [MgATP]/Ki_MgATP + [3PG][MgATP]/(Ki_MgATP·K_3PG)
```
Constants: Vmf = 5.96×10⁴, Vmr = 2.39×10⁴, K_MgADP = 0.1, Ki_MgADP = 0.08,
K_13BPG = 0.002, Ki_13BPG = 1.6, K_MgATP = 1, Ki_MgATP = 0.186, K_3PG = 1.1,
Ki_3PG = 0.205, Keq = 3.2×10³.

## Phosphoglycerate mutase (PGM/PGAM) — reversible uni-uni

```
           Vmf·[3PG]/K_3PG − Vmr·[2PG]/K_2PG
r_PGAM = ─────────────────────────────────────
           1 + [3PG]/K_3PG + [2PG]/K_2PG
```
Constants: from Mulquiney & Kuchel 1999.

## Enolase (ENO) — random bi-bi (with Mg), reversible

```
          Vmf·[2PG][Mg]/(Ki_Mg·K_2PG) − Vmr·[PEP][Mg]/(Ki_Mg·K_PEP)
r_ENO = ───────────────────────────────────────────────────────────────
          1 + [2PG]/Ki_2PG + [Mg]/Ki_Mg + [2PG][Mg]/(Ki_Mg·K_2PG)
            + [PEP]/Ki_PEP + [Mg]/Ki_Mg + [PEP][Mg]/(Ki_Mg·K_PEP)
```
Constants: Vmf = 2.106×10⁴, Vmr = 5.542×10³, K_Mg = Ki_Mg = 0.14,
K_PEP = Ki_PEP = 0.11, K_2PG = Ki_2PG = 0.046, Keq = 3.0.

## Pyruvate kinase (PK) — MWC two-state, ordered bi-bi

```
          Vf·[PEP][MgADP]/(K_PEP·K_MgADP) − Vr·[PYR][MgATP]/(K_PYR·K_MgATP)        1
r_PK = ──────────────────────────────────────────────────────────────────────── · ────
       (1+[PEP]/K_PEP)(1+[MgADP]/K_MgADP) + (1+[PYR]/K_PYR)(1+[MgATP]/K_MgATP) − 1  N_PK

                  L_PK·(1+[ATP]/K_ATP)^4·(1+[ALA]/K_ALA)^4
N_PK = 1 + ──────────────────────────────────────────────────────────────────────
           (1+[PEP]/K_PEP+[PYR]/K_PYR)^4·(1+[F16BP]/K_F16BP+[G16BP]/K_G16BP)^4
```
R-state (active) stabilised by PEP, PYR, F16BP, G16BP; T-state by ATP, ALA.

Constants: Vf = 2.02×10⁴, Vr = 47.5, K_PEP = 0.225, K_MgADP = 0.474,
K_MgATP = 3, K_ATP = 3.39, K_PYR = 4, K_F16BP = 0.04, K_G16BP = 0.1,
L_PK = 0.398, K_ALA = 0.02.
(K_F16BP = 0.04 → PKM2 isozyme, per Table 1.)

## Lactate dehydrogenase (LDH) — ordered bi-bi, pyruvate substrate inhibition

```
          Vmf·[NADH][PYR]/(Ki_NADH·K_PYR) − Vmr·[NAD][LAC]/(Ki_NAD·K_LAC)
r_LDH = ─────────────────────────────────────────────────────────────────────
          (1 + [PYR]/Ki_PYR)·(...)  +  (bi-bi King–Altman terms in NAD/NADH/PYR/LAC)
```
The denominator is the standard ordered bi-bi form with the extra substrate-
inhibition factor `(1 + [PYR]/Ki_PYR)`. Constants: from Mulquiney & Kuchel 1999.

## Glucose transporter (GLUT1) — reversible symmetric carrier (uni-uni)

```
           Vmf·[GLC_e]/K_GLC − Vmr·[GLC_c]/K_GLC
r_GLUT = ────────────────────────────────────────────
           1 + [GLC_e]/K_GLC + [GLC_c]/K_GLC
```

## Mitochondrial pyruvate transporter (PYRH) — reversible mass action (H⁺-coupled)

```
r_PYRH = Vmf·( [PYR_c][H⁺_c] − [PYR_m][H⁺_m] )
```

---

## Mass-balance ODEs (12 balanced intermediates) — confirmed from S1

```
d[GLC]/dt   = r_GLUT − r_HK
d[G6P]/dt   = r_HK   − r_GPI
d[F6P]/dt   = r_GPI  − r_PFK − r_PFK2 + r_F26BPase
d[F16BP]/dt = r_PFK  − r_ALDO
d[F26BP]/dt = r_PFK2 − r_F26BPase
d[DHAP]/dt  = r_ALDO − r_TPI
d[GAP]/dt   = r_ALDO + r_TPI − r_GAPDH
d[13BPG]/dt = r_GAPDH − r_PGK
d[3PG]/dt   = r_PGK  − r_PGAM
d[2PG]/dt   = r_PGAM − r_ENO
d[PEP]/dt   = r_ENO  − r_PK
d[PYR]/dt   = r_PK   − r_LDH − r_PYRH
```
These match the vector field implemented in `model2.py`.
