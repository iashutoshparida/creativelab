"""Uniform A/B split vs Thompson sampling for finding the best creative.

Both arms serve impressions in batches of 1,000 and stop when the Beta
posterior gives one creative at least a 95% chance of being best, or when
the impression cap is hit. We record how many impressions each needed, how
many clicks it collected on the way, and whether it picked the true best.
"""

from __future__ import annotations

import numpy as np

BATCH = 1000
CAP = 600_000
STOP_P = 0.95


def p_best(clicks: np.ndarray, imps: np.ndarray, rng: np.random.Generator, draws: int = 2000) -> np.ndarray:
    a = 1 + clicks
    b = 1 + imps - clicks
    samples = rng.beta(a[:, None], b[:, None], (len(a), draws))
    return np.bincount(samples.argmax(0), minlength=len(a)) / draws


def run_test(true_ctr: np.ndarray, policy: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    k = len(true_ctr)
    clicks = np.zeros(k)
    imps = np.zeros(k)
    while imps.sum() < CAP:
        if policy == "uniform":
            alloc = np.full(k, BATCH // k)
        else:
            alloc = rng.multinomial(BATCH, p_best(clicks, imps, rng, 500))
        clicks += rng.binomial(alloc, true_ctr)
        imps += alloc
        pb = p_best(clicks, imps, rng)
        if pb.max() >= STOP_P:
            break
    winner = int(np.argmax(pb))
    return {
        "impressions": int(imps.sum()),
        "clicks": int(clicks.sum()),
        "correct": winner == int(np.argmax(true_ctr)),
        "hit_cap": bool(imps.sum() >= CAP),
    }


def compare(ctr_sets: list[np.ndarray], seed: int = 0) -> dict:
    rows = []
    for t, ctrs in enumerate(ctr_sets):
        u = run_test(ctrs, "uniform", seed + t)
        b = run_test(ctrs, "thompson", seed + t)
        rows.append({"uniform": u, "thompson": b})

    def stat(arm, key, fn=np.median):
        return float(fn([r[arm][key] for r in rows]))

    u_imp, b_imp = stat("uniform", "impressions"), stat("thompson", "impressions")
    return {
        "tests": len(rows),
        "median_impressions_uniform": u_imp,
        "median_impressions_thompson": b_imp,
        "impression_reduction": 1 - b_imp / u_imp,
        "correct_rate_uniform": stat("uniform", "correct", np.mean),
        "correct_rate_thompson": stat("thompson", "correct", np.mean),
        "ctr_during_test_uniform": float(sum(r["uniform"]["clicks"] for r in rows) / sum(r["uniform"]["impressions"] for r in rows)),
        "ctr_during_test_thompson": float(sum(r["thompson"]["clicks"] for r in rows) / sum(r["thompson"]["impressions"] for r in rows)),
        "cap_hits_uniform": int(sum(r["uniform"]["hit_cap"] for r in rows)),
        "cap_hits_thompson": int(sum(r["thompson"]["hit_cap"] for r in rows)),
    }
