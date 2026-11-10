"""Copy writer: Claude when ANTHROPIC_API_KEY is set, templates otherwise.

The writer only fills in headline and body for a creative whose hook, visual,
CTA and length the planner already chose, so the LLM changes wording, never
the test design. Every output still goes through the policy check.
"""

from __future__ import annotations

import json
import os

from .world import Creative

MODEL = os.environ.get("CREATIVELAB_MODEL", "claude-haiku-4-5")
PRICE_PER_MTOK = (1.0, 5.0)

CTA_TEXT = dict(play_now="Play now", download_free="Download free", try_level_1="Try level 1", join_friends="Join your friends")

HOOK_HEADLINES = {
    "question": "Ready for {usp0}?",
    "challenge": "Think you can beat level 10?",
    "reward": "Daily rewards in {name}",
    "curiosity": "Nobody expected this in {name}",
    "social_proof": "Your next favourite {genre} game",
    "fail_bait": "Can you do better than this?",
}


def llm_available() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def template_copy(c: Creative, game: dict) -> tuple[str, str]:
    usp = game["usp"]
    headline = HOOK_HEADLINES[c.hook].format(name=game["name"], usp0=usp[0], genre=game["genre"].replace("rpg", "RPG"))
    body = f"{usp[0].capitalize()}, {usp[1]} and {usp[2]} in {game['name']}. {CTA_TEXT[c.cta]}."
    return headline[:40], body[:125]


def llm_copy(batch: list[Creative], game: dict, feedback: dict[int, list[str]] | None = None) -> tuple[list[tuple[str, str]], int, float]:
    import anthropic

    specs = [
        {"i": i, "hook": c.hook, "visual": c.visual, "cta": CTA_TEXT[c.cta], "seconds": c.length,
         **({"fix": feedback[i]} if feedback and i in feedback else {})}
        for i, c in enumerate(batch)
    ]
    prompt = (
        "Write mobile game ad copy. Game store page:\n" + game["store_page"] + "\n\n"
        "For each spec, write a headline (max 40 chars) and body (max 125 chars) in the given hook style, "
        "ending the body with the CTA. No money or prize promises, no guarantees, no competitor names, "
        "no fake alerts. If a spec has 'fix', it failed policy for those reasons; rewrite it.\n"
        "Return only a JSON list of {\"i\", \"headline\", \"body\"}.\n" + json.dumps(specs)
    )
    msg = anthropic.Anthropic().messages.create(model=MODEL, max_tokens=4000, messages=[{"role": "user", "content": prompt}])
    text = msg.content[0].text
    data = json.loads(text[text.find("[") : text.rfind("]") + 1])
    by_i = {d["i"]: (d["headline"], d["body"]) for d in data}
    tokens = msg.usage.input_tokens + msg.usage.output_tokens
    cost = (msg.usage.input_tokens * PRICE_PER_MTOK[0] + msg.usage.output_tokens * PRICE_PER_MTOK[1]) / 1e6
    return [by_i.get(i, template_copy(c, game)) for i, c in enumerate(batch)], tokens, cost
