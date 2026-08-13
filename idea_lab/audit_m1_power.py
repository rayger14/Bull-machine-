"""
M1 -- POSITIVE CONTROLS / POWER CURVES  (wyckoff_audit add.60)
=============================================================
The test never run: prove the verdict machinery CAN detect a real edge when one is
injected. Three controls mirror the three real verdict machineries used in add.48-59:

  (a) ENTRY/EXIT paired-ΔR machinery  (G-EXIT, exit-toolkit add.56/58/59):
      overlay flags a random 50% of the door's own trades and improves their realized
      R by a KNOWN delta; run the exact paired mean-ΔR bootstrap (CI-excludes-0).
  (b) SIZING/ENTRY in-out split machinery  (G-ENTRY, conviction sizing add.58/59):
      overlay flags a random 50%, boosts their R by delta; run the exact unpaired
      difference-of-means bootstrap (in vs out, CI-excludes-0).
  (c) CADENCE machinery  (campaign-v2 pass rules add.58/59):
      inject +1 synthetic profitable trade per campaign at the door's own win profile;
      check whether the pre-registered C1-C4 rules + paired-ΔPnL bootstrap detect it.

DELIVERABLE: empirical power curve over 200 seeds per delta -> the Minimum Detectable
Effect (MDE @ 80% power) of each machinery at our actual n. STUDY ONLY.
"""
from __future__ import annotations
import os, sys, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from audit_common import (load_basket, basket_R, basket_stats_from_R,
                          FIXED_RISK, INITIAL_CASH)

CACHE = ("/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/"
         "da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad/audit_basket.pkl")
NSEED = 200
BBOOT = 5000          # bootstrap resamples per test (studies used 10k; 5k for 200x speed, checked stable)
DELTAS = [0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]


def get_basket():
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    assets = load_basket(with_campaigns=True)
    R = np.array([x[2] for x in basket_R(assets)])
    # per-campaign v1 R sums for the cadence machinery + baseline verdict inputs
    camp_rows = []   # (name, cid, v1R_in_window, v2R, n_v2_entries)
    for name, a in assets.items():
        if "camps" not in a:
            continue
        v1_by_time = sorted(a["v1"]["trades"], key=lambda t: t["entry_time"])
        for cid, ts in a["camps"].items():
            c_start = min(t["entry_time"] for t in ts)
            c_end = max(t["exit_time"] for t in ts)
            v2R = sum(t["R"] for t in ts)
            v1R = sum(t["R"] for t in v1_by_time if c_start <= t["entry_time"] <= c_end)
            camp_rows.append((name, cid, v1R, v2R, len(ts)))
    v1_all_R = [(t["exit_time"], t["R"]) for a in assets.values() for t in a["v1"]["trades"]]
    v2_all_R = [(t["exit_time"], t["R"]) for a in assets.values() for t in a["v2"]["trades"]]
    data = dict(R=R, camp_rows=camp_rows, v1_all_R=v1_all_R, v2_all_R=v2_all_R)
    with open(CACHE, "wb") as f:
        pickle.dump(data, f)
    return None if False else _reload(assets)  # keep assets out of cache size; reload path below


def _reload(assets):
    R = np.array([x[2] for x in basket_R(assets)])
    return dict(R=R)


def load_cached():
    with open(CACHE, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# (a) PAIRED ΔR machinery  (G-EXIT / exit-toolkit)
# ---------------------------------------------------------------------------
def power_paired(R, delta, noisy_sigma, rng, nseed=NSEED, B=BBOOT):
    """Overlay flags 50% of trades; flagged trades get +delta (+ optional N(0,sigma)).
    Test = paired mean-ΔR bootstrap CI-lower>0. Return detection fraction over nseed."""
    n = len(R); hits = 0
    for _ in range(nseed):
        flag = rng.random(n) < 0.5
        dR = np.where(flag, delta, 0.0).astype(float)
        if noisy_sigma > 0:
            dR = dR + np.where(flag, rng.normal(0, noisy_sigma, n), 0.0)
        # bootstrap mean ΔR
        bidx = rng.integers(0, n, size=(B, n))
        means = dR[bidx].mean(1)
        lo = np.percentile(means, 2.5)
        if lo > 0:
            hits += 1
    return hits / nseed


# ---------------------------------------------------------------------------
# (b) IN/OUT SPLIT machinery  (G-ENTRY / conviction sizing)
# ---------------------------------------------------------------------------
def power_split(R, delta, rng, nseed=NSEED, B=BBOOT):
    """Overlay flags 50%, adds +delta to the flagged cohort's R. Test = unpaired
    difference-of-means bootstrap (in vs out) CI-lower>0. Detection fraction."""
    n = len(R); hits = 0
    for _ in range(nseed):
        flag = rng.random(n) < 0.5
        Rin = R[flag] + delta
        Rout = R[~flag]
        if len(Rin) < 2 or len(Rout) < 2:
            continue
        bi = rng.integers(0, len(Rin), size=(B, len(Rin)))
        bo = rng.integers(0, len(Rout), size=(B, len(Rout)))
        dstat = Rin[bi].mean(1) - Rout[bo].mean(1)
        lo = np.percentile(dstat, 2.5)
        if lo > 0:
            hits += 1
    return hits / nseed


def mde_at(powers, deltas, target=0.80):
    """Linear-interpolate the delta achieving `target` power."""
    for i in range(1, len(deltas)):
        if powers[i - 1] < target <= powers[i]:
            x0, x1 = deltas[i - 1], deltas[i]
            y0, y1 = powers[i - 1], powers[i]
            return x0 + (target - y0) * (x1 - x0) / (y1 - y0 + 1e-12)
    if powers[-1] >= target:
        return deltas[0] if powers[0] >= target else deltas[-1]
    return float("inf")   # not reachable within grid


def run_ab(data):
    R = np.asarray(data["R"], float)
    n = len(R)
    print("=" * 92)
    print(f"M1 POWER CURVES  |  basket n={n}  meanR={R.mean():+.3f}  sdR={R.std(ddof=1):.3f}  "
          f"WR={(R>0).mean():.1%}  seeds={NSEED}")
    print("=" * 92)

    rng = np.random.default_rng(20260813)
    print("\n(a) PAIRED ΔR machinery (G-EXIT / exit-toolkit) -- overlay improves flagged 50% by δ")
    print("    two variants: DETERMINISTIC δ (pure statistics layer) and NOISY δ+N(0,σ), σ=sdR")
    sd = R.std(ddof=1)
    print(f"    {'δ(R)':>7} {'pow_det':>9} {'pow_noisy':>10}")
    pdet, pnoisy = [], []
    for d in DELTAS:
        a1 = power_paired(R, d, 0.0, rng)
        a2 = power_paired(R, d, sd, rng)
        pdet.append(a1); pnoisy.append(a2)
        print(f"    {d:>7.3f} {a1:>9.2f} {a2:>10.2f}")
    print(f"    MDE@80%: deterministic={mde_at(pdet,DELTAS):.3f}R   noisy={mde_at(pnoisy,DELTAS):.3f}R")

    print("\n(b) IN/OUT SPLIT machinery (G-ENTRY / conviction sizing) -- flagged 50% mean lifted by δ")
    print(f"    {'δ(R)':>7} {'power':>9}")
    psplit = []
    for d in DELTAS:
        pw = power_split(R, d, rng)
        psplit.append(pw)
        print(f"    {d:>7.3f} {pw:>9.2f}")
    print(f"    MDE@80%: split={mde_at(psplit,DELTAS):.3f}R   MDE@50%: {mde_at(psplit,DELTAS,0.50):.3f}R")
    return dict(pdet=pdet, pnoisy=pnoisy, psplit=psplit, sd=sd, n=n)


# ---------------------------------------------------------------------------
# (c) CADENCE machinery  (campaign-v2 pass rules)
# ---------------------------------------------------------------------------
def run_c(data):
    print("\n" + "=" * 92)
    print("(c) CADENCE machinery (campaign-v2 C1-C4 + paired-ΔPnL bootstrap)")
    print("=" * 92)
    camp_rows = data["camp_rows"]          # (name, cid, v1R_window, v2R, n_v2)
    v1_all = data["v1_all_R"]; v2_all = data["v2_all_R"]   # lists of (exit_time_str, R)
    b1 = basket_stats_from_R([r for (_, r) in v1_all])
    b2 = basket_stats_from_R([r for (_, r) in v2_all])
    n_camp = len(camp_rows)
    # win profile = positive-R v1 trades
    winR = np.array([r for (_, r) in v1_all if r > 0])
    rng = np.random.default_rng(4242)

    print(f"  baseline: v1 PF={b1['PF']:.2f} PnL=${b1['PnL']:,.0f} n={b1['n']}  |  "
          f"v2 PF={b2['PF']:.2f} PnL=${b2['PnL']:,.0f} n={b2['n']}  campaigns={n_camp}")
    print(f"  win profile: {len(winR)} positive v1 trades, meanR={winR.mean():+.3f}")

    # Actual add.58/59 paired-ΔPnL (v2 - v1 per campaign)
    base_delta = np.array([(v2R - v1R) * FIXED_RISK for (_, _, v1R, v2R, _) in camp_rows])
    print(f"\n  ACTUAL v2-vs-v1 paired: mean Δ$/camp = ${base_delta.mean():,.0f}  "
          f"CI[{np.percentile(_boot(base_delta,rng),2.5):,.0f},"
          f"{np.percentile(_boot(base_delta,rng),97.5):,.0f}]  (straddles 0 => the real add.58/59 null)")

    # INJECTION: add k synthetic winners per campaign (at door win profile) to the v2 side.
    print(f"\n  INJECT +k profitable trades/campaign at the door's own win profile:")
    print(f"    {'k':>4} {'v2*_n':>7} {'cad_mult':>9} {'v2*_PnL':>11} {'meanΔ$/camp':>12} "
          f"{'CI_lo$':>9} {'paired_det':>11} {'C2_pass':>8} {'C3_pass':>8}")
    for k in [0, 1, 2, 3]:
        # synthetic v2* basket = v2 + k*n_camp injected winners
        inj_R = []
        inj_delta = base_delta.copy()
        for ci, (_, _, v1R, v2R, _) in enumerate(camp_rows):
            add = 0.0
            for _ in range(k):
                r = rng.choice(winR)
                inj_R.append(r); add += r
            inj_delta[ci] = (v2R + add - v1R) * FIXED_RISK
        v2star = v2_all + [(None, r) for r in inj_R]
        # basket_stats ignores exit_time ordering effect on PF/PnL (PnL is order-invariant); use R list
        bs = basket_stats_from_R([r for (_, r) in v2star])
        cad_mult = (b2["n"] + k * n_camp) / max(b1["n"], 1)
        # paired detection: CI lower of mean Δ$/camp > 0 ?
        bootd = _boot(inj_delta, rng)
        ci_lo = np.percentile(bootd, 2.5)
        paired_det = ci_lo > 0
        c2 = bs["PnL"] >= b1["PnL"]
        c3 = cad_mult >= 1.5
        print(f"    {k:>4} {b2['n']+k*n_camp:>7} {cad_mult:>9.2f} {bs['PnL']:>11,.0f} "
              f"{inj_delta.mean():>12,.0f} {ci_lo:>9,.0f} {str(paired_det):>11} "
              f"{str(c2):>8} {str(c3):>8}")
    print("\n  READS: k=0 is the actual study (paired CI straddles 0 = null). If +1 real winner/campaign")
    print("  makes paired-det True AND C2/C3 pass, the cadence machinery CAN detect real added cadence")
    print("  => the campaign-topology rejection is DECISIVE (topology added ~0 trades), not blind.")


def _boot(vals, rng, B=BBOOT):
    v = np.asarray(vals, float)
    idx = rng.integers(0, len(v), size=(B, len(v)))
    return v[idx].mean(1)


if __name__ == "__main__":
    if not os.path.exists(CACHE):
        print("Building basket cache (first run)...")
        assets = load_basket(with_campaigns=True)
        R = np.array([x[2] for x in basket_R(assets)])
        camp_rows = []
        v1_all_R, v2_all_R = [], []
        for name, a in assets.items():
            for t in a["v1"]["trades"]:
                v1_all_R.append((str(t["exit_time"]), t["R"]))
            if "camps" not in a:
                continue
            for t in a["v2"]["trades"]:
                v2_all_R.append((str(t["exit_time"]), t["R"]))
            v1_by_time = sorted(a["v1"]["trades"], key=lambda t: t["entry_time"])
            for cid, ts in a["camps"].items():
                c_start = min(t["entry_time"] for t in ts)
                c_end = max(t["exit_time"] for t in ts)
                v2R = sum(t["R"] for t in ts)
                v1R = sum(t["R"] for t in v1_by_time if c_start <= t["entry_time"] <= c_end)
                camp_rows.append((name, cid, v1R, v2R, len(ts)))
        data = dict(R=R, camp_rows=camp_rows, v1_all_R=v1_all_R, v2_all_R=v2_all_R)
        with open(CACHE, "wb") as f:
            pickle.dump(data, f)
    data = load_cached()
    run_ab(data)
    run_c(data)
