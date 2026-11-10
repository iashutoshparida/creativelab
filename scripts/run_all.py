"""Train the CTR model, run the creative agent for each game, and compare
uniform vs bandit testing on the selected creatives. Writes results/.

Usage:  python -m scripts.run_all [--impressions 2000000] [--tests 100]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from creativelab.bandit_test import compare
from creativelab.ctr_model import CTRModel, baseline_logreg, evaluate, simulate_log
from creativelab.pipeline import CreativeAgent
from creativelab.world import CREATIVE_NOISE_SD, blended_ctr

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impressions", type=int, default=2_000_000)
    ap.add_argument("--creatives", type=int, default=600)
    ap.add_argument("--tests", type=int, default=100)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()

    log, _ = simulate_log(args.creatives, args.impressions, seed=11)
    holdout, _ = simulate_log(args.creatives, 400_000, seed=12)
    model = CTRModel.fit(log)
    metrics = evaluate(model, baseline_logreg(log.sample(400_000, random_state=0)), holdout)
    metrics.update(train_impressions=args.impressions, train_creatives=args.creatives, holdout_impressions=400_000)

    games = json.loads((ROOT / "data" / "games.json").read_text())
    rng = np.random.default_rng(5)
    selections, rank_corrs, lifts, ctr_sets = {}, [], [], []
    for g_i, game in enumerate(games):
        agent = CreativeAgent(model, n_variants=40, top_k=8, seed=g_i)
        out = agent.run(game)
        for c in out["creatives"]:
            c.quality = float(rng.normal(0, CREATIVE_NOISE_SD))  # hidden from the model
        true = np.array([blended_ctr(c) for c in out["creatives"]])
        pred = np.array(out["scores"])
        rank_corrs.append(float(spearmanr(true, pred).statistic))
        top = np.array([blended_ctr(c) for c in out["selected"]])
        lifts.append(float(top.mean() / true.mean() - 1))
        ctr_sets.append(top)
        selections[game["id"]] = {
            "log": out["log"],
            "selected": [
                {"hook": c.hook, "visual": c.visual, "cta": c.cta, "length": c.length, "headline": c.headline,
                 "body": c.body, "pred_ctr": round(c.meta["pred_ctr"], 5), "true_ctr": round(blended_ctr(c), 5)}
                for c in out["selected"]
            ],
        }
    metrics["variant_rank_spearman_mean"] = float(np.mean(rank_corrs))
    metrics["top8_vs_all40_true_ctr_lift_mean"] = float(np.mean(lifts))

    # Bandit vs uniform: each test uses one game's top 8 with fresh hidden quality draws.
    sets = []
    for t in range(args.tests):
        base = ctr_sets[t % len(ctr_sets)]
        logit = np.log(base / (1 - base)) + rng.normal(0, CREATIVE_NOISE_SD, len(base))
        sets.append(1 / (1 + np.exp(-logit)))
    testing = compare(sets, seed=100)

    (RESULTS / "ctr_metrics.json").write_text(json.dumps(metrics, indent=2))
    (RESULTS / "selections.json").write_text(json.dumps(selections, indent=2))
    (RESULTS / "bandit_vs_uniform.json").write_text(json.dumps(testing, indent=2))
    print("CTR model:", json.dumps(metrics, indent=2))
    print("Bandit vs uniform:", json.dumps(testing, indent=2))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
