"""Simulated impression logs and an XGBoost click-through-rate model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from .world import (AUDIENCE_MIX, AUDIENCES, CTAS, GENRES, HOOKS, LENGTHS, PLACEMENT_MIX, PLACEMENTS, VISUALS,
                    Creative, logit_ctr, random_creative)

CATS = {
    "genre": GENRES, "hook": HOOKS, "visual": VISUALS, "cta": CTAS,
    "length": tuple(str(x) for x in LENGTHS), "placement": PLACEMENTS, "audience": AUDIENCES,
}
FEATURE_NAMES = [f"{k}={v}" for k, vals in CATS.items() for v in vals]


def encode(df: pd.DataFrame) -> np.ndarray:
    cols = []
    for k, vals in CATS.items():
        s = df[k].astype(str).to_numpy()
        cols.extend((s == v).astype(np.float32) for v in vals)
    return np.column_stack(cols)


def simulate_log(n_creatives: int, n_impressions: int, seed: int) -> tuple[pd.DataFrame, list[Creative]]:
    """Impression-level log: one row per impression with a click label."""
    rng = np.random.default_rng(seed)
    creatives = [random_creative(rng, f"g{i % 30}", GENRES[i % 3]) for i in range(n_creatives)]
    ci = rng.integers(0, n_creatives, n_impressions)
    pl = rng.choice(PLACEMENTS, n_impressions, p=list(PLACEMENT_MIX.values()))
    au = rng.choice(AUDIENCES, n_impressions, p=list(AUDIENCE_MIX.values()))
    logits = np.array([logit_ctr(creatives[c], p, a) for c, p, a in zip(ci, pl, au)])
    clicks = (rng.random(n_impressions) < 1 / (1 + np.exp(-logits))).astype(np.int8)
    df = pd.DataFrame({
        "creative": ci,
        "genre": [creatives[c].genre for c in ci],
        "hook": [creatives[c].hook for c in ci],
        "visual": [creatives[c].visual for c in ci],
        "cta": [creatives[c].cta for c in ci],
        "length": [creatives[c].length for c in ci],
        "placement": pl,
        "audience": au,
        "click": clicks,
        "true_p": 1 / (1 + np.exp(-logits)),
    })
    return df, creatives


@dataclass
class CTRModel:
    booster: xgb.Booster

    @classmethod
    def fit(cls, df: pd.DataFrame, seed: int = 0) -> "CTRModel":
        d = xgb.DMatrix(encode(df), label=df["click"].to_numpy(), feature_names=FEATURE_NAMES)
        params = dict(objective="binary:logistic", eval_metric="logloss", max_depth=4, eta=0.1,
                      subsample=0.8, colsample_bytree=0.9, min_child_weight=50, tree_method="hist", seed=seed)
        return cls(xgb.train(params, d, num_boost_round=300))

    def predict_rows(self, df: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(xgb.DMatrix(encode(df), feature_names=FEATURE_NAMES))

    def predict_blended(self, creatives: list[Creative]) -> np.ndarray:
        """Expected CTR of each creative under the standard traffic mix."""
        rows = [
            dict(genre=c.genre, hook=c.hook, visual=c.visual, cta=c.cta, length=c.length, placement=p, audience=a,
                 w=pw * aw, idx=i)
            for i, c in enumerate(creatives)
            for p, pw in PLACEMENT_MIX.items()
            for a, aw in AUDIENCE_MIX.items()
        ]
        df = pd.DataFrame(rows)
        df["p"] = self.predict_rows(df) * df["w"]
        return df.groupby("idx")["p"].sum().reindex(range(len(creatives))).to_numpy()


def baseline_logreg(train: pd.DataFrame) -> LogisticRegression:
    """Main effects only: no interactions between genre, hook, audience and visual."""
    m = LogisticRegression(max_iter=200, C=1.0)
    m.fit(encode(train), train["click"].to_numpy())
    return m


def evaluate(model: CTRModel, base: LogisticRegression, test: pd.DataFrame) -> dict:
    y = test["click"].to_numpy()
    p = model.predict_rows(test)
    pb = base.predict_proba(encode(test))[:, 1]
    return {
        "auc": float(roc_auc_score(y, p)),
        "auc_logreg": float(roc_auc_score(y, pb)),
        "auc_oracle": float(roc_auc_score(y, test["true_p"])),  # ceiling: the true click probabilities
        "logloss": float(log_loss(y, p)),
        "logloss_logreg": float(log_loss(y, pb)),
        "ctr_observed": float(y.mean()),
        "ctr_predicted": float(p.mean()),
    }
