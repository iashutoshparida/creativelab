"""Creative attributes and a simulated ad market with known click-through rates.

A creative is a set of attributes (hook, visual, call to action, length) plus
copy text. Its true CTR depends on those attributes, on the game's genre, on
the placement it runs in and on the audience, through main effects and a few
interactions (for example, "challenge" hooks work for racing games, not for
relaxing puzzle games). Because the true CTR is known, we can check how well
a model trained on impression logs recovers it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HOOKS = ("question", "challenge", "reward", "curiosity", "social_proof", "fail_bait")
VISUALS = ("gameplay", "character", "ugc_style", "satisfying_loop", "cinematic")
CTAS = ("play_now", "download_free", "try_level_1", "join_friends")
LENGTHS = (6, 15, 30)
GENRES = ("puzzle", "racing", "rpg")
PLACEMENTS = ("rewarded_video", "interstitial", "feed", "banner")
AUDIENCES = ("casual", "core", "lapsed")

BASE = -4.0
HOOK_FX = dict(question=0.05, challenge=0.15, reward=0.20, curiosity=0.25, social_proof=0.0, fail_bait=0.30)
VISUAL_FX = dict(gameplay=0.20, character=0.0, ugc_style=0.25, satisfying_loop=0.15, cinematic=-0.10)
CTA_FX = dict(play_now=0.10, download_free=0.05, try_level_1=0.15, join_friends=0.0)
LENGTH_FX = {6: 0.10, 15: 0.15, 30: -0.15}
PLACEMENT_FX = dict(rewarded_video=0.50, interstitial=0.10, feed=0.0, banner=-0.60)
AUDIENCE_FX = dict(casual=0.0, core=0.10, lapsed=-0.20)
# (genre, hook) and (audience, visual) interactions
GENRE_HOOK_FX = {
    ("racing", "challenge"): 0.35, ("puzzle", "challenge"): -0.25,
    ("rpg", "curiosity"): 0.20, ("puzzle", "fail_bait"): 0.25, ("rpg", "fail_bait"): -0.30,
    ("racing", "social_proof"): 0.15, ("rpg", "reward"): 0.15,
}
AUDIENCE_VISUAL_FX = {("core", "cinematic"): 0.35, ("casual", "satisfying_loop"): 0.25, ("lapsed", "ugc_style"): 0.20}
CREATIVE_NOISE_SD = 0.12  # unexplained creative quality


@dataclass
class Creative:
    game_id: str
    genre: str
    hook: str
    visual: str
    cta: str
    length: int
    headline: str = ""
    body: str = ""
    quality: float = 0.0  # hidden noise term, set by the market
    meta: dict = field(default_factory=dict)

    def key(self) -> tuple:
        return (self.game_id, self.hook, self.visual, self.cta, self.length)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit_ctr(c: Creative, placement: str, audience: str) -> float:
    return (
        BASE + HOOK_FX[c.hook] + VISUAL_FX[c.visual] + CTA_FX[c.cta] + LENGTH_FX[c.length]
        + PLACEMENT_FX[placement] + AUDIENCE_FX[audience]
        + GENRE_HOOK_FX.get((c.genre, c.hook), 0.0) + AUDIENCE_VISUAL_FX.get((audience, c.visual), 0.0)
        + c.quality
    )


def true_ctr(c: Creative, placement: str, audience: str) -> float:
    return float(_sigmoid(logit_ctr(c, placement, audience)))


# Traffic mix used when we talk about "a creative's CTR" without a placement.
PLACEMENT_MIX = dict(rewarded_video=0.35, interstitial=0.30, feed=0.25, banner=0.10)
AUDIENCE_MIX = dict(casual=0.55, core=0.30, lapsed=0.15)


def blended_ctr(c: Creative) -> float:
    return float(sum(pw * aw * true_ctr(c, p, a) for p, pw in PLACEMENT_MIX.items() for a, aw in AUDIENCE_MIX.items()))


def random_creative(rng: np.random.Generator, game_id: str, genre: str) -> Creative:
    c = Creative(game_id, genre, str(rng.choice(HOOKS)), str(rng.choice(VISUALS)), str(rng.choice(CTAS)), int(rng.choice(LENGTHS)))
    c.quality = float(rng.normal(0, CREATIVE_NOISE_SD))
    return c
