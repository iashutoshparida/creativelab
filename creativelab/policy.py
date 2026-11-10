"""Ad copy guardrails, checked before any creative reaches testing.

Rules follow common app-store and ad-network ad policies: no promises of
money or guaranteed wins, no fake system alerts, no competitor or platform
names, no health claims, honest "free" claims, no made-up statistics, and
length limits per field. Text is also checked after undoing common character
swaps ("ca$h", "w1n", "m0ney") that are used to dodge filters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_HEADLINE = 40
MAX_BODY = 125

_SWAPS = str.maketrans({"$": "s", "0": "o", "1": "i", "3": "e", "4": "a", "@": "a", "5": "s", "7": "t"})


def _normalise(text: str) -> str:
    # Only swap characters inside words, so "1v1" and "60 cars" stay as they are.
    return re.sub(r"(?<=[a-z])[$01345@7]|[$01345@7](?=[a-z]{2,})", lambda m: m.group().translate(_SWAPS), text.lower())


RULES: list[tuple[str, re.Pattern]] = [
    ("money_promise", re.compile(
        r"\b(win|earn|make|get)\s+(real\s+)?(money|cash|rupees|dollars|\$|₹|rs\.?)|\bcash\s*(prize|out)|\bpaypal\b|"
        r"\breal[\s-]*money\b|\bpays?\s+(you|out)\b|\bpayouts?\b|\bget\s+paid\b|\bwithdraw|\bwinnings\b|\bincome\b|"
        r"\bplay\s+to\s+earn\b|\bupi\b", re.I)),
    ("guarantee", re.compile(
        r"\bguar[ae]nt[e]{1,2}[ds]?\b|\bguar[ae]nt[e]{1,2}d\b|\b100%\s*(win|free)\b|\bnever\s+lose\b|\bcan'?t\s+lose\b|"
        r"\bno\s+way\s+to\s+lose\b|\bcertain\s+victory\b|\bvictory\s+assured\b|\blosing\s+(is\s+)?impossible\b|"
        r"\balways\s+win\b|\bevery\s+(round|match|game)\s+is\s+a\s+win\b", re.I)),
    ("fake_alert", re.compile(
        r"\b(virus|warning|alert|urgent\s+system|system\s+message)\b|\byour\s+(phone|device|battery)\s+(is|needs|has)\b|"
        r"\bupdate\s+required\b|\b(device|system)\s+error\b|\berror\b.{0,3}\b(memory|storage|0x)|\bmemory\s+low\b", re.I)),
    ("competitor", re.compile(
        r"\b(candy[\s-]*crush|pubg|free\s*fire|clash\s+of\s+clans|roblox|genshin|subway\s+surfers|ludo\s+king|fortnite|"
        r"minecraft|among\s+us)\b", re.I)),
    ("platform_claim", re.compile(
        r"\b(apple|google|facebook|meta)('s)?[\s-]+(approved|official|recommended|certified|favourite|favorite|verified)\b|"
        r"\b(approved|endorsed|verified|picked|recommended)\s+by\s+(apple|google|facebook|meta)\b", re.I)),
    ("health_claim", re.compile(
        r"\bcures?\b|\btreats?\s+(insomnia|adhd|anxiety|depression|stress|dementia)|\bboosts?\s+your\s+iq\b|"
        r"\breduces?\s+anxiety\b|\bclinically\b|\bblood\s+pressure\b|\bheal(s)?\s+your\b|\bdepression\b|\badhd\b|"
        r"\bdementia\b|\bdoctors?\s+(recommend|love)|\bimproves?\s+your\s+memory\b|\bstress\s+hormones?\b|\binsomnia\b", re.I)),
    ("misleading_free", re.compile(
        r"\bfree\s+(gems|coins|diamonds|lives|spins)\s+(forever|with\s+no\s+limit)|\b(unlimited|infinite|endless|limitless)"
        r"\s+free\b|\bnever\s+run\s+out\b|\bfully\s+free\b|\ball\s+items\s+free\b|\bnever\s+pay\b|\bno\s+purchases\s+ever\b", re.I)),
    ("unverified_stat", re.compile(
        r"\bonly\s+\d+(\.\d+)?\s*%|\b\d+(\.\d+)?\s*%\s+(of\s+(players|people|gamers)|can'?t)|#1\s+(game|app)|"
        r"\bmillions?\s+of\s+players\b|\b\d+\s*(m|million|k)\+?\s+(users|players|downloads)\b|\bmost\s+downloaded\b|"
        r"\btop[\s-]rated\s+by\b|\b\d+\s+in\s+\d+\s+(win|players)|\brated\s+best\b", re.I)),
    ("all_caps_shouting", re.compile(r"\b[A-Z]{6,}\b(?:\s+\b[A-Z]{4,}\b){2,}")),
]


@dataclass
class PolicyResult:
    ok: bool
    reasons: list[str]


def check(headline: str, body: str) -> PolicyResult:
    texts = [headline, body, _normalise(headline), _normalise(body)]
    reasons = [name for name, rx in RULES if any(rx.search(t) for t in texts)]
    if len(headline) > MAX_HEADLINE:
        reasons.append("headline_too_long")
    if len(body) > MAX_BODY:
        reasons.append("body_too_long")
    if not headline.strip() or not body.strip():
        reasons.append("empty_field")
    return PolicyResult(not reasons, reasons)
