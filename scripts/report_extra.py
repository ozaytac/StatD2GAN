#!/usr/bin/env python3
"""Post-freeze reporting layer for StatD2GAN v3 -> generated/numbers_extra.tex.

No new inference. Every macro is either
  (a) a re-expression (ratio, range, count, percentage) of a frozen result file, or
  (b) a design constant read from the frozen protocol.yaml (`proto`), or from the frozen,
      sha-verified source under src/ (`code`; each pattern must match exactly once), or
  (c) prefixed `ph`: a post-freeze descriptive observation that is NOT a pre-specified
      decision rule (comparisons, counts or ranges first computed after the freeze).
Before writing, every frozen input and every frozen script present locally is checked against
its sha256 in MANIFEST.json, and the values that duplicate a frozen decision
(results/decisions.json) are asserted equal to it. Outputs are registered in
MANIFEST_post_freeze.json; MANIFEST.json is never modified.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from post_freeze import register, verify_frozen  # noqa: E402

INPUTS = ["abl_perseed", "baselines", "seq_level", "projection", "wilcoxon", "bootstrap",
          "baseline_tests", "pct_change", "run_noise", "ks_floor", "ks_ratio"]
WORDS = "zero one two three four five six seven eight nine ten eleven twelve".split()
# Analysis choice (declared, not a protocol constant): a critic weight counts as "at its lower
# bound" if within this relative tolerance, because renormalisation after clipping moves it
# slightly off the bound; the ratio cap counts as binding within the same tolerance.
REL_TOL = 0.02


def cap(s: str) -> str:
    return "".join(p[:1].upper() + p[1:] for p in s.replace("-", "_").split("_"))


def fmt(x: float, d: int) -> str:
    return f"{x:.{d}f}"


def fp(p: float) -> str:
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def rel(v: str) -> str:
    """Attach the relation so that $p_{\\mathrm{Holm}}\\macro$ reads p=0.420 or p<0.001."""
    return v if v[:1] in "<>=" else f"={v}"


def and_list(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1] if xs else "none"


def rule_const(text: str, pattern: str) -> str:
    hit = re.search(pattern, text)
    if not hit:
        raise SystemExit(f"decision-rule constant not found in protocol: {pattern}")
    return hit.group(1)


def main(root: Path, paper: Path | None = None) -> None:
    paths = [root / "results" / f"{k}.csv" for k in INPUTS] + [root / "results" / "decisions.json", root / "generated" / "numbers.tex",
                                                              root / "protocol.yaml"]
    n_checked = verify_frozen(root, paths)
    print(f"verified {n_checked} frozen files against MANIFEST.json")

    R = {k: pd.read_csv(root / "results" / f"{k}.csv", comment="#") for k in INPUTS}
    DEC = json.loads((root / "results" / "decisions.json").read_text())
    P = yaml.safe_load((root / "protocol.yaml").read_text())
    LOCS = list(P["data"]["locations"])
    m: dict[str, str] = {}

    # -- decision-rule constants, parsed from the frozen rule text -------------------
    dr = P["decision_rules"]
    m["protoDOneRatio"] = rule_const(dr["D1_calibration_floor"], r"is <= ([\d.]+)")
    m["protoDTwoMinLocs"] = rule_const(dr["D2_component"], r">= (\d+) of \d+ location")
    m["protoDFiveFrac"] = rule_const(dr["D5_projection"], r"<= ([\d.]+) \* mean floor")
    m["protoDSixPct"] = rule_const(dr["D6_variance_collapse"], r"< (\d+)% of the real")
    m["protoDSixMinLocs"] = rule_const(dr["D6_variance_collapse"], r">= (\d+) of \d+ locations")

    floor = R["ks_floor"].groupby("loc").ks_floor.mean()          # as in scripts/analyze.py

    # -- D1 statistic (frozen) and calibrated KS / reference ratios ------------------
    kr = R["ks_ratio"]
    d1_med = float(kr[kr.config == "full_model"].ks_ratio.median())
    d1_rec = float(re.search(r"median ks/floor=([\d.]+)", DEC["D1"]["detail"]).group(1))
    assert round(d1_med, 3) == d1_rec, (d1_med, d1_rec)
    m["ksRatioFullMedian"] = fmt(d1_med, 3)
    kc = kr[kr.config != "no_calibration"]
    top = kc.loc[kc.ks_ratio.idxmax()]
    m["phKsRatioCellMin"], m["phKsRatioCellMax"] = fmt(kc.ks_ratio.min(), 2), fmt(kc.ks_ratio.max(), 2)
    m["phKsRatioCellMaxLoc"], m["phKsRatioCellMaxSeed"] = cap(top["loc"]), str(int(top.seed))
    m["phKsRatioCellMaxArm"] = top.config.replace("_", r"\_")
    m["phKsRatioCellsBelowOne"], m["phKsRatioCells"] = str(int((kc.ks_ratio < 1).sum())), str(len(kc))

    # -- KS / reference before and after calibration, all generators (descriptive) --
    abl, base = R["abl_perseed"], R["baselines"]
    gens = {"Full": abl[abl.config == "full_model"], "Rcgan": base[base.config == "rcgan"],
            "Timegan": base[base.config == "timegan"]}
    for g, df in gens.items():
        for stage, col in [("Cal", "ks_stat"), ("Raw", "ks_stat_raw")]:
            r = df[col] / df["loc"].map(floor)
            m[f"phKsRatio{stage}{g}Med"] = fmt(r.median(), 2)
            m[f"phKsRatio{stage}{g}Min"], m[f"phKsRatio{stage}{g}Max"] = fmt(r.min(), 2), fmt(r.max(), 2)

    # -- baselines vs full_model (frozen tests, family of two) -----------------------
    met = {"ks_stat": "Ks", "kendall_mae": "Tau", "acf_mae": "Acf", "phys_viol_rate": "Phys"}
    for _, r in R["baseline_tests"].iterrows():
        k = f"Base{cap(r.contrast)}{met[r.metric]}"
        m[f"med{k}"] = fmt(r.median_diff, 4 if r.metric == "ks_stat" else 3)
        m[f"pHolm{k}"] = rel(fp(r.p_holm))
        m[f"nLocPos{k}"] = str(int(r.n_loc_median_pos))

    # -- seed-level signs for D_sort removal (frozen pos_by_loc) ----------------------
    w = R["wilcoxon"].set_index(["metric", "contrast"])
    pos = dict(re.findall(r"(\w\w):(\d)/\d", w.loc[("kendall_mae", "no_D_sorted"), "pos_by_loc"]))
    for loc in LOCS:
        m[f"nPosSeedsNoDSortedTau{cap(loc)}"] = pos[loc[:2].upper()]
    full_mean = abl[abl.config == "full_model"].groupby("loc").kendall_mae.mean()
    a = abl[abl.config == "no_D_sorted"].set_index(["loc", "seed"]).kendall_mae
    f = abl[abl.config == "full_model"].set_index(["loc", "seed"]).kendall_mae
    pc_seed = 100 * (a - f.loc[a.index]) / a.index.get_level_values(0).map(full_mean)
    lo = pc_seed.idxmin()
    m["phNoDSortedSeedMinPct"], m["phNoDSortedSeedMinLoc"] = fmt(pc_seed.min(), 0), cap(lo[0])

    # -- sequence level (temperature; seed medians vs real held-out) ------------------
    s = R["seq_level"]
    med = s.groupby(["loc", "model"])[["seq_mean_sd", "within_sd"]].median().unstack()
    for stat, tag in [("seq_mean_sd", "SeqPct"), ("within_sd", "WithinPct")]:
        t = med[stat]
        pct = t.div(t["real_val"], axis=0) * 100
        for mod, mt in [("statd2gan", "Stat"), ("timegan", "Timegan"), ("rcgan", "Rcgan")]:
            for loc in LOCS:
                m[f"{tag}{mt}{cap(loc)}"] = fmt(pct.loc[loc, mod], 0)
            m[f"{tag}{mt}Min"], m[f"{tag}{mt}Max"] = fmt(pct[mod].min(), 0), fmt(pct[mod].max(), 0)
        closer = ((t.timegan - t.real_val).abs() < (t.statd2gan - t.real_val).abs()).sum()
        m[f"phTimeganCloser{'Between' if stat == 'seq_mean_sd' else 'Within'}"] = str(int(closer))
    # D6 (frozen) cross-check
    real = s[s.model == "real_val"].set_index("loc").seq_mean_sd
    for mod in ("rcgan", "timegan"):
        n = int((med["seq_mean_sd"][mod] < float(m["protoDSixPct"]) / 100 * real).sum())
        assert DEC["D6"][mod]["detail"].startswith(f"{n}/"), (mod, n)
    g = s[s.model != "real_val"].copy()
    g["pct"] = g.seq_mean_sd / g["loc"].map(real) * 100
    for mod, mt in [("rcgan", "Rcgan"), ("timegan", "Timegan"), ("statd2gan", "Stat")]:
        gm = g[g.model == mod]
        low = gm.loc[gm.pct.idxmin()]
        m[f"phSeedsBelowTen{mt}"] = str(int((gm.pct < float(m["protoDSixPct"])).sum()))
        m[f"phSeedMinPct{mt}"], m[f"phSeedMinLoc{mt}"] = fmt(low.pct, 1), cap(low["loc"])
        m[f"phSeedMinSeed{mt}"] = str(int(low.seed))
    m["phSeedsPerModel"] = str(int((g.model == "rcgan").sum()))

    # -- run-to-run noise -------------------------------------------------------------
    rn = R["run_noise"]
    noise = float(rn.median_pairwise_abs_dtau.iloc[0])
    m["phEffOverNoiseNoDSorted"] = fmt(float(w.loc[("kendall_mae", "no_D_sorted"), "median_diff"]) / noise, 1)

    # -- bootstrap (descriptive) vs Wilcoxon + Holm -----------------------------------
    b = R["bootstrap"].set_index(["metric", "contrast"])
    j = w.join(b, how="inner")
    excl, sig = (j.lo > 0) | (j.hi < 0), j.p_holm < P["inference"]["alpha"]
    m["phBootDiscord"], m["phBootDiscordReverse"] = str(int((excl & ~sig).sum())), str(int((sig & ~excl).sum()))
    m["nHolmSig"] = str(int(sig.sum()))
    m["nBootDistinct"] = str(comb(2 * len(LOCS) - 1, len(LOCS)))
    n_locs = int(re.search(r"\\newcommand\{\\nLocs\}\{(\d+)\}", (root / "generated" / "numbers.tex").read_text()).group(1))
    assert n_locs == len(LOCS)
    m["protoSignTestPMin"] = f"{2.0 ** (1 - n_locs):g}"          # exact two-sided sign test, all n agree

    # -- projection (D5), asserted against the frozen decision -------------------------
    p = R["projection"].copy()
    p["dks"] = (p.proj_ks_stat - p.cal_ks_stat).abs()
    thr = float(m["protoDFiveFrac"]) * float(floor.mean())          # as in scripts/analyze.py
    zero = bool((p.proj_phys_viol_rate == 0).all())
    rec = re.search(r"all zero=(\w+), max\|dKS\|=([\d.]+)", DEC["D5"]["detail"])
    assert f"{p.dks.max():.5f}" == rec.group(2) and str(zero) == rec.group(1), (p.dks.max(), rec.groups())
    assert DEC["D5"]["holds"] == (zero and p.dks.max() <= thr)
    for loc in LOCS:
        m[f"dKsMax{cap(loc)}"] = fmt(p[p["loc"] == loc].dks.max(), 4)
    worst = p.loc[p.dks.idxmax()]
    m["dKsMaxLoc"], m["dKsMaxSeed"] = cap(worst["loc"]), str(int(worst.seed))
    m["dKsMedianAll"], m["dKsThreshold"] = fmt(p.dks.median(), 4), fmt(thr, 4)
    m["nProjRunsZero"], m["nProjRuns"] = str(int((p.proj_phys_viol_rate == 0).sum())), str(len(p))
    m["dKsMaxRunCalPhys"] = fmt(float(worst.cal_phys_viol_rate), 3)
    # post-freeze descriptive: locations whose largest |dKS| exceeds the D5 threshold
    over = [cap(l) for l in LOCS if p[p["loc"] == l].dks.max() > thr]
    m["phDKsLocsOverThr"], m["phDKsLocsOverThrList"] = str(len(over)), and_list(over)
    m["phDKsRunsOverThr"] = str(int((p.dks > thr).sum()))

    # -- D_sort removal: location means against the run-noise band (post-freeze) ----------
    band = float(R["run_noise"].band_pct.iloc[0])
    pv = R["pct_change"].set_index(["loc", "config"]).kendall_mae.xs("no_D_sorted", level="config")
    m["pctNoDSortedTauMin"], m["pctNoDSortedTauMinLoc"] = fmt(pv.min(), 0), cap(pv.idxmin())
    m["pctNoDSortedTauMax"], m["pctNoDSortedTauMaxLoc"] = fmt(pv.max(), 0), cap(pv.idxmax())
    inb = [cap(l) for l in LOCS if abs(pv[l]) < band]
    m["phNoDSortedInBandN"], m["phNoDSortedInBandList"] = str(len(inb)), and_list(inb)

    # -- percentage changes vs full_model (frozen pct_change) ---------------------------
    pcd = R["pct_change"].set_index(["loc", "config"])
    for cfg, col, tag in [("rank_D_sort", "phys_viol_rate", "PctRankDSortPhys"),
                          ("rank_D_sort", "kendall_mae", "PctRankDSortTau"),
                          ("no_calibration", "ks_stat", "PctNoCalKs")]:
        v = pcd.xs(cfg, level="config")[col]
        for loc in LOCS:
            m[f"{tag}{cap(loc)}"] = fmt(v[loc], 0)
        m[f"{tag}Min"], m[f"{tag}Max"] = fmt(v.min(), 0), fmt(v.max(), 0)

    # -- relation-carrying copies of the frozen Holm p-values (numbers.tex, sha-verified) --
    frozen_tex = (root / "generated" / "numbers.tex").read_text()
    for name, val in re.findall(r"\\newcommand\{\\pHolm(\w+)\}\{([^}]*)\}", frozen_tex):
        m[f"pHolmRel{name}"] = rel(val)

    # -- design constants (frozen protocol) ----------------------------------------------
    dat, ev, mod = P["data"], P["evaluation"], P["model"]
    y0, y1 = dat["years"]
    thou = lambda x: f"{x:,}".replace(",", "{,}")  # noqa: E731
    m.update(protoVersion=P["version"], protoYearFirst=str(y0), protoYearLast=str(y1),
             protoHeldOutFirst=str(y1 - dat["val_years"] + 1), protoValYears=str(dat["val_years"]),
             protoEmbargoHours=str(dat["embargo_hours"]), protoSeqLen=str(dat["seq_len"]),
             protoStride=str(dat["step"]), protoNFeatures=str(len(dat["features"])),
             protoNScaled=str(len(dat["scalable_idx"])), protoNEval=thou(ev["n_eval"]),
             protoNQuantiles=thou(int(re.search(r"(\d+) quantiles", ev["calibration"]).group(1))),
             protoTauSubsample=thou(ev["tau_subsample"]), protoNTauPairs=str(len(ev["tau_pairs"])),
             protoAcfLags=str(ev["acf_nlags"]), protoAcfNSeq=str(ev["acf_n_seq"]),
             protoNAcfFeatures=str(len(ev["acf_features"])), protoCalibDrawSeed=str(ev["calib_draw_seed"]),
             protoEvalDrawSeed=str(ev["eval_draw_seed"]), protoMetricSeed=str(ev["metric_seed"]),
             protoAlpha=str(P["inference"]["alpha"]), protoNArms=str(len(P["arms"])),
             protoNSeeds=str(len(P["seeds"])), protoNContrasts=str(len(P["arms"]) - 1),
             protoNBaselines=str(len(P["baselines"]["models"])), protoNBoot=thou(P["inference"]["bootstrap"]["n_boot"]),
             protoEpochs=str(mod["total_epochs"]), protoWarmup=str(mod["warmup_epochs"]),
             protoPilotMinPct=str(P["pilot"]["gate_min_pct"]),
             protoPilotMinSeeds=str(P["pilot"]["gate_min_positive_seeds"]),
             protoBudgetHours=str(P["budget"]["gpu_hours"]), nLocsProto=str(len(LOCS)),
             nRunsTotal=str((len(P["arms"]) + len(P["baselines"]["models"])) * len(P["seeds"]) * len(LOCS)))

    # -- model, training and baseline constants (frozen protocol) -----------------------
    phy, rc, tg = P["physics"], P["baselines"]["rcgan"], P["baselines"]["timegan"]
    sci = lambda x: (lambda b, e: f"{b}\\times 10^{{{int(e)}}}" if b != "1" else f"10^{{{int(e)}}}")(*f"{x:.0e}".split("e"))  # noqa: E731
    m.update(protoLatent=str(mod["latent_dim"]), protoHidden=str(mod["hidden_dim"]),
             protoBatch=str(mod["batch_size"]), protoLrG=sci(mod["lr_g"]), protoLrD=sci(mod["lr_d"]),
             protoGpFreq=str(mod["gp_freq"]), protoGpLambda=f"{mod['gp_lambda']:g}",
             protoGpLambdaLazy=f"{mod['gp_lambda'] * mod['gp_freq']:g}",
             protoEvoEta=f"{mod['evo']['eta']:g}", protoEvoLo=f"{mod['evo']['bounds'][0]:g}",
             protoEvoHi=f"{mod['evo']['bounds'][1]:g}", protoEvoRatio=f"{mod['evo']['max_ratio']:g}",
             protoDewTol=f"{phy['dew_tol']:g}", protoPresLo=f"{phy['pres_lo']:g}",
             protoPresHi=f"{phy['pres_hi']:g}", protoHumLo=f"{phy['hum_lo']:g}", protoHumHi=f"{phy['hum_hi']:g}",
             protoRcLatent=str(rc["latent"]), protoRcHidden=str(rc["hidden"]), protoRcLayers=str(rc["layers"]),
             protoRcEpochs=str(rc["epochs"]), protoRcBatch=str(rc["batch_size"]), protoRcLr=sci(rc["lr"]),
             protoTgLatent=str(tg["latent"]), protoTgHidden=str(tg["hidden"]), protoTgLayers=str(tg["layers"]),
             protoTgItersAe=thou(tg["iters_ae"]), protoTgItersSup=thou(tg["iters_sup"]),
             protoTgItersJoint=thou(tg["iters_joint"]), protoTgBatch=str(tg["batch_size"]),
             protoTgLr=sci(tg["lr"]), protoTgGamma=f"{tg['gamma']:g}")

    # -- data layout implied by the protocol and src/s2g/data.py (complete hourly record,
    #    asserted at preprocessing; held-out = last val_years calendar years; embargo cut from
    #    the end of train; windows of seq_len with stride `step`) ------------------------
    t0 = pd.Timestamp(f"{y0}-01-01")
    v0 = pd.Timestamp(f"{y1 - dat['val_years'] + 1}-01-01")
    h_tr = int((v0 - t0) / pd.Timedelta(hours=1)) - dat["embargo_hours"]
    h_va = int((pd.Timestamp(f"{y1 + 1}-01-01") - v0) / pd.Timedelta(hours=1))
    nwin = lambda h: (h - dat["seq_len"]) // dat["step"] + 1  # noqa: E731
    h_all = int((pd.Timestamp(f"{y1 + 1}-01-01") - t0) / pd.Timedelta(hours=1))
    m.update(protoNTrainSeq=thou(nwin(h_tr)), protoNValSeq=thou(nwin(h_va)), protoNValHours=thou(h_va),
             protoNRecordHours=thou(h_all),
             protoSeedList=and_list([str(s) for s in P["seeds"]]), protoSeedFirst=str(P["seeds"][0]),
             protoPilotLocs=and_list([cap(l) for l in P["pilot"]["locations"]]),
             protoBootSeed=str(P["inference"]["bootstrap"]["seed"]))
    lo_q, hi_q = map(float, re.findall(r"[\d.]+", P["inference"]["bootstrap"]["interval"]))
    m["protoBootCI"] = f"{hi_q - lo_q:g}"
    man = json.loads((root / "MANIFEST.json").read_text())
    import hashlib
    assert man["protocol_sha256"] == hashlib.sha256((root / "protocol.yaml").read_bytes()).hexdigest()
    m["protoShaShort"] = man["protocol_sha256"][:16]
    for loc, (la, lo) in dat["locations"].items():
        m[f"protoLat{cap(loc)}"], m[f"protoLon{cap(loc)}"] = f"{la:.2f}", f"{lo:.2f}"
        assert la > 0 and lo > 0                     # all points N/E; text writes N and E

    # -- constants that live only in the frozen source (sha-verified above) ------------
    # Each pattern must match exactly once in its file; otherwise the script aborts.
    def code(rel: str, pattern: str, flags: int = 0) -> tuple[str, ...]:
        hits = re.findall(pattern, (root / rel).read_text(), flags)
        if len(hits) != 1:
            raise SystemExit(f"code constant {pattern!r} matched {len(hits)}x in {rel}")
        return hits[0] if isinstance(hits[0], tuple) else (hits[0],)

    mo, lo_, tr, bl = "src/s2g/models.py", "src/s2g/losses.py", "src/s2g/train.py", "src/s2g/baselines.py"
    m["codeTendLag"], = code("src/s2g/data.py", r"f\.msl\.diff\((\d+)\)")
    m["codeGLayers"], = code(mo, r'nn\.LSTM\(noise_dim, M\["hidden_dim"\], (\d+),')
    m["codeDTempHidden"], m["codeDTempLayers"] = code(mo, r"class TemporalDiscriminator\(LSTMCritic\):\n.*?super\(\)\.__init__\((\d+), (\d+)\)")
    m["codeDStatHidden"], m["codeDStatLayers"] = code(mo, r"class StatDiscriminator\(LSTMCritic\):\n.*?super\(\)\.__init__\((\d+), (\d+)\)")
    (m["codeConvOneOut"], m["codeConvOneK"], _c, m["codeConvTwoOut"], m["codeConvTwoK"]) = code(
        mo, r"CircularConv1d\(NF, (\d+), (\d+)\), CircularConv1d\((\d+), (\d+), (\d+)\)")
    m["codeLeaky"], = code(mo, r"F\.leaky_relu\(self\.c1\(s\), ([\d.]+)\)")
    m["codeRankAnchors"], m["codeRankTemp"] = code(mo, r"def soft_rank\(x, anchors=(\d+), tau=([\d.]+)\)")
    m["codeMomentClamp"], = code(mo, r"\.clamp\(-(\d+), \d+\)")
    m["codeMomentHOne"], m["codeMomentHTwo"] = code(mo, r"SN\(nn\.Linear\(4 \* NF, (\d+)\)\).*?SN\(nn\.Linear\(\d+, (\d+)\)\)", re.S)
    m["codePhysDewW"], m["codePhysBeta"] = code(lo_, r"(\d+) \* \(F\.softplus\(dew - t \+ PHY\[\"dew_tol\"\], beta=(\d+)\)")
    m["codePhysPresW"], = code(lo_, r"(\d+) \* \(F\.relu\(pres")
    m["codePhysHumW"], = code(lo_, r"(\d+) \* \(F\.relu\(hum")
    m["codeFitLo"], m["codeFitHi"] = code(lo_, r"np\.clip\(fit / \(fit\.sum\(\) \+ 1e-8\), ([\d.]+), ([\d.]+)\)")
    m["codeCdfW"], = code(lo_, r"def cdf_loss\(real, fake, w=([\d.]+)\)")
    m["codeQuantW"], = code(lo_, r"def quantile_loss\(real, fake, w=([\d.]+)\)")
    m["codeQuantLevels"], = code(lo_, r"torch\.tensor\(\[([\d., ]+)\]")
    (m["codeTailFracMax"], m["codeTailFracBase"], m["codeTailFracSlope"], m["codeTailStart"],
     m["codeTailRamp"]) = code(lo_, r"min\(([\d.]+), ([\d.]+) \+ ([\d.]+) \* \(epoch - (\d+)\) / (\d+)\)")
    assert code(lo_, r"if epoch >= (\d+):") == (m["codeTailStart"],)
    m["codeTailScale"], = code(lo_, r"\*= ([\d.]+)")
    m["codeLrDecayEp"], m["codeLrDecayDiv"] = code(tr, r'if ep == (\d+):\n.*?M\["lr_g"\] / (\d+)')
    m["codeNCriticLate"], m["codeNCriticEp"], m["codeNCritic"] = code(tr, r"\((\d+) if ep > (\d+) else (\d+)\)")
    m["codePhysW"], = code(tr, r"\+ ([\d.]+) \* physics_loss\(fake, sc, mn\)")
    m["codeAuxStart"], m["codeAuxW"], m["codeAuxRamp"] = code(tr, r"if ep >= (\d+) and cfg\.use_aux_losses:\n.*?min\(([\d.]+), [\d.]+ \* \(ep - \d+\) / (\d+)\)")
    clip = set(re.findall(r"clip_grad_norm_\([^,]+, ([\d.]+)\)", (root / tr).read_text()))
    assert len(clip) == 1, clip
    m["codeClipNorm"] = clip.pop()
    b1, b2 = code(tr, r"opt_g = Adam\(G\.parameters\(\), lr=M\[\"lr_g\"\], betas=\(([\d.]+), ([\d.]+)\)\)")
    m["codeAdamBetaOne"], m["codeAdamBetaTwo"] = f"{float(b1):g}", f"{float(b2):g}"
    rb1, rb2 = code(bl, r"og, od = torch\.optim\.Adam\(G\.parameters\(\), c\[\"lr\"\], betas=\(([\d.]+), ([\d.]+)\)\)")
    m["codeRcBetaOne"], m["codeRcBetaTwo"] = f"{float(rb1):g}", f"{float(rb2):g}"
    m["codeTgAeW"], = code(bl, r"\((\d+) \* mse\(m\.rec\(m\.emb\(x\)\), x\)\.sqrt\(\)\)")
    m["codeTgGUpdates"], = code(bl, r"for _ in range\((\d+)\):\n\s+x = _batch")
    m["codeTgSupW"], m["codeTgMomW"] = code(bl, r"\(gu \+ (\d+) \* gs\.sqrt\(\) \+ (\d+) \* gv\)")
    m["codeTgEmbRecW"], m["codeTgEmbSupW"] = code(bl, r"e_loss = (\d+) \* mse\(m\.rec\(h\), x\)\.sqrt\(\) \+ ([\d.]+) \* mse")
    m["codeTgDThresh"], = code(bl, r"if dl\.item\(\) > ([\d.]+):")
    m["codeTgSupLayersOffset"], = code(bl, r"gru\(h, h, n - (\d+)\)")
    m["codeTgSupLayers"] = str(int(m["protoTgLayers"]) - int(m.pop("codeTgSupLayersOffset")))
    del _c
    for k in [k for k in m if k.startswith("code") and re.fullmatch(r"\d+\.\d+", m[k])]:
        m[k] = f"{float(m[k]):g}"                          # 30.0 -> 30, 0.15 unchanged

    # -- critic weights (optional: needs runs/**/*full_model_seed*.json) ----------------
    FIT_LAST = 100                                         # reporting window, final epochs
    lo_b, ratio = float(mod["evo"]["bounds"][0]), float(mod["evo"]["max_ratio"])
    runs = sorted(glob.glob(str(root / "runs" / "**" / "*full_model_seed*.json"), recursive=True))
    n_expected = len(LOCS) * len(P["seeds"])
    if runs and len(runs) != n_expected:
        raise SystemExit(f"found {len(runs)} of {n_expected} full_model run JSONs: supply all or none")
    if runs:
        verify_frozen(root, [Path(x) for x in runs])
        fr, cb, fits = [], [], []
        for fn in runs:
            d = json.loads(Path(fn).read_text())
            critics = list(d["params"])[1:]                    # order of train.py's critic list
            i_sort = critics.index("SortedDiscriminator")
            L, F = np.array(d["lambda_per_epoch"]), np.array(d["fit_per_epoch"])
            fr.append((L[:, i_sort] <= lo_b * (1 + REL_TOL)).mean())
            cb.append((L.max(1) / L.min(1) >= ratio * (1 - REL_TOL)).mean())
            fits.append(np.median(F[-FIT_LAST:], 0))
        fits = np.median(np.array(fits), 0)
        m["phFitLastEpochs"] = str(FIT_LAST)
        m["phLambdaSortAtFloorPct"] = fmt(100 * np.median(fr), 0)
        m["phRatioCapBindsPct"] = fmt(100 * np.median(cb), 0)
        for i, nm in enumerate(c.replace("Discriminator", "") for c in critics):
            m[f"phFit{nm}"] = fmt(fits[i], 1)
        m["phNRunsLambda"] = str(len(runs))
    else:
        print("note: no runs/**/*full_model_seed*.json found; critic-weight macros skipped")

    # -- distance of the calibrated sample to the training block (post-freeze re-inference,
    #    results_extra/distance_cal_train.csv; optional, sha-checked against
    #    MANIFEST_post_freeze.json; descriptive, no tests) ----------------------------------
    dist_csv = root / "results_extra" / "distance_cal_train.csv"
    dist_tab = root / "generated" / "table_dist_pf.tex"
    if dist_csv.exists():
        pf_man = json.loads((root / "MANIFEST_post_freeze.json").read_text())["files"]
        assert pf_man["results_extra/distance_cal_train.csv"] == hashlib.sha256(dist_csv.read_bytes()).hexdigest()
        dz = pd.read_csv(dist_csv, comment="#")
        head_line = dist_csv.read_text().splitlines()[0]
        base_cfg = set(P["baselines"]["models"])
        arm = ~dz.config.isin(base_cfg)
        assert len(dz) == (len(P["arms"]) - 1 + len(base_cfg)) * len(P["seeds"]) * len(LOCS)   # no_calibration out
        assert dz.gate_ok.all() and dz.triangle_ok.all() and (dz.min_slack_feature >= -1e-12).all()
        assert (dz.device == dz.json_gpu).all() and dz.device.nunique() == dz.torch.nunique() == 1
        gap = dz.ks_cal_val - dz.ks_floor
        assert str(int((gap[arm] < 0).sum())) == m["phKsRatioCellsBelowOne"] and str(int(arm.sum())) == m["phKsRatioCells"]
        assert np.allclose(dz.ks_floor, dz["loc"].map(floor), atol=1e-12)
        ratio, below = dz.d_cal_train / dz.ks_floor, gap < 0
        use = (-gap[below] / dz.d_cal_train[below]).max()
        assert use < 1                                       # Proposition, lower side
        top = dz.assign(r=ratio).nlargest(3, "r")
        names = {"rcgan": "RCGAN", "timegan": "TimeGAN"}
        disp = lambda c: names.get(c, r"\texttt{" + c.replace("_", r"\_") + "}")  # noqa: E731
        tol = re.search(r"\|KS - JSON\| <= ([\d.e-]+)", head_line).group(1)
        m.update(phDistNRuns=str(len(dz)),
                 phDistRatioMed=fmt(ratio.median(), 3), phDistRatioPninety=fmt(float(np.quantile(ratio, 0.9)), 3),
                 phDistGapMed=fmt((gap.abs() / dz.ks_floor).median(), 3),
                 phDistGapOverDMed=fmt((gap.abs() / dz.d_cal_train).median(), 3),
                 phDistBelowMaxRatio=fmt(use, 3), phDistBelowAll=str(int(below.sum())),
                 phDistRawOverCalMed=fmt((dz.d_raw_train / dz.d_cal_train).median(), 0),
                 phDistGateTol=sci(float(tol)),
                 phDistGateMaxDiff=f"{max(dz.gate_raw_diff.max(), dz.gate_cal_diff.max()):g}",
                 phDistDevice=dz.device.iloc[0], phDistTorch=dz.torch.iloc[0])
        for k, (_, t) in zip(("One", "Two", "Three"), top.iterrows()):
            m[f"phDistTop{k}Ratio"], m[f"phDistTop{k}Loc"] = fmt(t.r, 3), cap(t["loc"])
            m[f"phDistTop{k}Model"], m[f"phDistTop{k}Seed"] = disp(t.config), str(int(t.seed))
        # descriptive table: arms by location, baselines pooled, all runs
        dz = dz.assign(r=ratio, g=(gap.abs() / dz.ks_floor))
        groups = [(cap(l), dz[arm & (dz["loc"] == l)]) for l in LOCS] + \
                 [(names[b], dz[dz.config == b]) for b in P["baselines"]["models"]] + [("All runs", dz)]
        rows = []
        for i, (lab, x) in enumerate(groups):
            fl = fmt(x.ks_floor.iloc[0], 4) if x["loc"].nunique() == 1 else "--"
            rows.append(" & ".join([lab, str(len(x)), fl, fmt(x.d_cal_train.median(), 4), fmt(x.r.median(), 3),
                                    fmt(x.r.max(), 3), fmt(x.g.median(), 3), fmt(x.d_raw_train.median(), 3)]) + r" \\")
            if i in (len(LOCS) - 1, len(LOCS) + len(base_cfg) - 1):
                rows.append(r"\midrule")
        dist_tab.write_text("\n".join([
            "% produced by scripts/report_extra.py from results_extra/distance_cal_train.csv (post-freeze; descriptive, no tests)",
            r"\begin{table}[!htb]", r"\centering\small",
            r"\caption{Distance of the calibrated synthetic sample to the training block (post-freeze; descriptive, no tests; "
            r"$\bar d^{\,\mathrm{raw,train}}$ is not a quality ranking). "
            r"Medians over runs unless marked max. Location rows pool the calibrated ablation arms; baseline rows pool locations. "
            r"$\bar d^{\,\mathrm{cal,train}}$, $\bar d^{\,\mathrm{raw,train}}$: feature-mean sup-distance of the calibrated "
            r"and uncalibrated sample to the training block; floor: $\mathrm{KS}^{\mathrm{floor}}$; "
            r"$\Delta=\lvert\mathrm{KS}^{\mathrm{cal}}-\mathrm{KS}^{\mathrm{floor}}\rvert$.}",
            r"\label{tab:dist}", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
            r" & Runs & Floor & $\bar d^{\,\mathrm{cal,train}}$ & \multicolumn{2}{c}{$\bar d^{\,\mathrm{cal,train}}/$floor} & "
            r"$\Delta/$floor & $\bar d^{\,\mathrm{raw,train}}$ \\", r"\cmidrule(lr){5-6}",
            r" & & & & median & max & & \\",
            r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]))
    else:
        print("note: results_extra/distance_cal_train.csv not found; phDist* macros and tab:dist skipped")

    # -- training environment of the frozen campaign (runs/_env/*.json, frozen) ----------
    envs = sorted((root / "runs" / "_env").glob("*.json"))
    if envs:
        verify_frozen(root, envs)
        tv = {json.loads(e.read_text()).get("torch") for e in envs}
        assert len(tv) == 1, tv
        m["phTrainTorch"] = tv.pop()
    else:
        print("note: runs/_env/*.json not found; phTrainTorch skipped")

    # -- small integers spelled out for running prose ------------------------------------
    for k in ["protoValYears", "nLocsProto", "protoNSeeds", "protoNBaselines", "protoNArms",
              "protoNContrasts", "protoNTauPairs", "protoNAcfFeatures", "protoNScaled",
              "phTimeganCloserBetween", "phTimeganCloserWithin", "phBootDiscord", "nHolmSig",
              "phDKsLocsOverThr", "phNoDSortedInBandN", "phDKsRunsOverThr"]:
        word = WORDS[int(m[k])]
        m[f"{k}Word"], m[f"{k}WordCap"] = word, word.capitalize()

    n_all = len(m)
    if paper is not None:                                   # Rule 4: keep only macros the text calls
        called = set(re.findall(r"\\([A-Za-z]+)", paper.read_text()))
        m = {k: v for k, v in m.items() if k in called}
    out = root / "generated" / "numbers_extra.tex"
    out.parent.mkdir(exist_ok=True)
    head = ("% produced by scripts/report_extra.py (post-freeze reporting layer; no new inference)\n"
            f"% inputs: protocol.yaml, results/decisions.json, {', '.join(f'results/{x}.csv' for x in INPUTS)}"
            f"{', runs/**/*full_model_seed*.json' if runs else ''}\n"
            "% ph* macros: post-freeze descriptive observations, not pre-specified decision rules\n")
    out.write_text(head + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(m.items())))
    print(f"wrote {out} ({len(m)} of {n_all} macros{' called by ' + paper.name if paper else ''})")
    register(root, [root / "scripts" / "report_extra.py", root / "scripts" / "post_freeze.py", out]
             + ([dist_tab] if dist_csv.exists() else []))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--paper", type=Path, default=None, help="prune to macros this .tex calls")
    a = ap.parse_args()
    main(a.root, a.paper)
