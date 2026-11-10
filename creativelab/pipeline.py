"""Creative agent as a LangGraph pipeline.

plan   -> choose N attribute combinations that cover every hook, visual,
          CTA and length as evenly as possible (a good test design)
write  -> fill headline and body (Claude or templates)
check  -> policy guardrails; failures go back to `write` once with the
          reasons, and anything still failing is dropped
score  -> XGBoost CTR model predicts each creative's blended CTR
select -> top K by predicted CTR, at most one per hook+visual pair
"""

from __future__ import annotations

import itertools
from typing import TypedDict

import numpy as np
from langgraph.graph import END, StateGraph

from . import policy
from .ctr_model import CTRModel
from .world import CTAS, HOOKS, LENGTHS, VISUALS, Creative
from .writer import llm_available, llm_copy, template_copy


class CreativeState(TypedDict, total=False):
    game: dict
    creatives: list[Creative]
    failures: dict[int, list[str]]
    rewrites: int
    scores: list[float]
    selected: list[Creative]
    log: list[str]
    tokens: int
    cost_usd: float


def plan_combinations(n: int, rng: np.random.Generator) -> list[tuple[str, str, str, int]]:
    """Greedy balanced design: each pick favours attribute levels used least so far."""
    combos = list(itertools.product(HOOKS, VISUALS, CTAS, LENGTHS))
    rng.shuffle(combos)
    counts = {a: 0 for a in (*HOOKS, *VISUALS, *CTAS, *LENGTHS)}
    chosen = []
    for _ in range(n):
        best = min(combos, key=lambda t: sum(counts[x] for x in t))
        combos.remove(best)
        chosen.append(best)
        for x in best:
            counts[x] += 1
    return chosen


class CreativeAgent:
    def __init__(self, model: CTRModel, n_variants: int = 40, top_k: int = 8, seed: int = 0, use_llm: bool | None = None,
                 writer=None):
        self.model, self.n, self.k = model, n_variants, top_k
        self.rng = np.random.default_rng(seed)
        self.use_llm = llm_available() if use_llm is None else use_llm
        self.writer = writer  # optional override, used in tests and evals
        self.graph = self._build()

    def plan(self, s: CreativeState) -> CreativeState:
        g = s["game"]
        creatives = [Creative(g["id"], g["genre"], h, v, c, L) for h, v, c, L in plan_combinations(self.n, self.rng)]
        return {"creatives": creatives, "log": [f"planned {len(creatives)} variants"], "rewrites": 0, "tokens": 0, "cost_usd": 0.0}

    def write(self, s: CreativeState) -> CreativeState:
        creatives, game, failures = s["creatives"], s["game"], s.get("failures") or {}
        targets = list(failures) if failures else list(range(len(creatives)))
        batch = [creatives[i] for i in targets]
        tokens = cost = 0
        if self.writer is not None:
            copies = [self.writer(c, game, failures.get(i)) for c, i in zip(batch, targets)]
        elif self.use_llm:
            copies, tokens, cost = llm_copy(batch, game, {j: failures[i] for j, i in enumerate(targets) if i in failures})
        else:
            copies = [template_copy(c, game) for c in batch]
        for i, (h, b) in zip(targets, copies):
            creatives[i].headline, creatives[i].body = h, b
        return {"creatives": creatives, "rewrites": s["rewrites"] + (1 if failures else 0),
                "tokens": s["tokens"] + tokens, "cost_usd": s["cost_usd"] + cost,
                "log": s["log"] + [f"wrote {len(targets)} ({'llm' if self.use_llm else 'template'})"]}

    def check(self, s: CreativeState) -> CreativeState:
        failures = {}
        for i, c in enumerate(s["creatives"]):
            r = policy.check(c.headline, c.body)
            if not r.ok:
                failures[i] = r.reasons
        return {"failures": failures, "log": s["log"] + [f"policy failures: {len(failures)}"]}

    def route_after_check(self, s: CreativeState) -> str:
        if s["failures"] and s["rewrites"] == 0:
            return "write"
        return "score"

    def score(self, s: CreativeState) -> CreativeState:
        keep = [c for i, c in enumerate(s["creatives"]) if i not in s.get("failures", {})]
        dropped = len(s["creatives"]) - len(keep)
        scores = self.model.predict_blended(keep).tolist() if keep else []
        return {"creatives": keep, "scores": scores, "log": s["log"] + [f"dropped {dropped} after rewrite, scored {len(keep)}"]}

    def select(self, s: CreativeState) -> CreativeState:
        order = np.argsort(-np.asarray(s["scores"]))
        seen, picked = set(), []
        for i in order:
            c = s["creatives"][i]
            if (c.hook, c.visual) in seen:
                continue
            seen.add((c.hook, c.visual))
            c.meta["pred_ctr"] = float(s["scores"][i])
            picked.append(c)
            if len(picked) == self.k:
                break
        return {"selected": picked, "log": s["log"] + [f"selected {len(picked)}"]}

    def _build(self):
        g = StateGraph(CreativeState)
        for name in ("plan", "write", "check", "score", "select"):
            g.add_node(name, getattr(self, name))
        g.set_entry_point("plan")
        g.add_edge("plan", "write")
        g.add_edge("write", "check")
        g.add_conditional_edges("check", self.route_after_check, {"write": "write", "score": "score"})
        g.add_edge("score", "select")
        g.add_edge("select", END)
        return g.compile()

    def run(self, game: dict) -> CreativeState:
        return self.graph.invoke({"game": game})
