import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from creativelab import policy
from creativelab.bandit_test import run_test
from creativelab.ctr_model import CTRModel, evaluate, baseline_logreg, simulate_log
from creativelab.pipeline import CreativeAgent, plan_combinations
from creativelab.world import HOOKS

GAMES = json.loads((Path(__file__).resolve().parent.parent / "data" / "games.json").read_text())


@pytest.fixture(scope="module")
def model():
    log, _ = simulate_log(200, 200_000, seed=1)
    return CTRModel.fit(log)


def test_policy_blocks_and_allows():
    assert not policy.check("Win real money", "Play now.").ok
    assert not policy.check("Earn ca$h fast", "Play now.").ok
    assert not policy.check("Only 1% beat this", "Play now.").ok
    assert policy.check("Join 1v1 duels", "60 cars to unlock. Download free.").ok
    assert policy.check("Win the weekly cup", "Race friends. Play now.").ok


def test_plan_is_balanced():
    combos = plan_combinations(40, np.random.default_rng(0))
    assert len(set(combos)) == 40
    hooks = Counter(c[0] for c in combos)
    assert max(hooks.values()) - min(hooks.values()) <= 1
    assert set(hooks) == set(HOOKS)


def test_agent_selects_top_k_without_duplicate_pairs(model):
    out = CreativeAgent(model, n_variants=40, top_k=8, seed=0, use_llm=False).run(GAMES[0])
    sel = out["selected"]
    assert len(sel) == 8
    assert len({(c.hook, c.visual) for c in sel}) == 8
    preds = [c.meta["pred_ctr"] for c in sel]
    assert preds == sorted(preds, reverse=True)
    assert all(policy.check(c.headline, c.body).ok for c in sel)


def test_failed_copy_is_rewritten_once(model):
    calls = Counter()

    def writer(c, game, feedback):
        calls[c.key()] += 1
        if feedback is None and c.hook == "reward":
            return "Win real money", "Play now."  # breaks policy on the first pass
        return "Daily rewards", "New levels every day. Play now."

    out = CreativeAgent(model, n_variants=24, top_k=8, seed=1, writer=writer).run(GAMES[1])
    assert len(out["creatives"]) == 24  # all fixed on rewrite, none dropped
    assert any(n == 2 for n in calls.values())


def test_copy_that_keeps_failing_is_dropped(model):
    def writer(c, game, feedback):
        return ("Guaranteed win", "Play now.") if c.hook == "fail_bait" else ("New season", "Play now.")

    out = CreativeAgent(model, n_variants=24, top_k=8, seed=2, writer=writer).run(GAMES[2])
    assert all(c.hook != "fail_bait" for c in out["creatives"])


def test_bandit_finds_clear_winner():
    ctrs = np.array([0.03, 0.03, 0.03, 0.06])
    for policy_name in ("uniform", "thompson"):
        r = run_test(ctrs, policy_name, seed=3)
        assert r["correct"]


def test_ctr_model_beats_chance(model):
    holdout, _ = simulate_log(200, 100_000, seed=2)
    log, _ = simulate_log(200, 50_000, seed=1)
    m = evaluate(model, baseline_logreg(log), holdout)
    assert m["auc"] > 0.55
    assert m["auc"] <= m["auc_oracle"] + 0.01
