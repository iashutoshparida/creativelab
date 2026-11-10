"""Policy-check eval: 120 ads that each break one rule, and 120 clean ads.

Violating ads are written in varied phrasings, including some that try to
slip past simple filters ("ca$h", "w1n"). The clean set mixes template copy
with hand-written copy that uses risky-looking but allowed words ("win the
cup", "download free", "earn stars"). CI fails if recall drops below 90% or
the false-positive rate goes above 5%.

    python -m evals.policy_eval
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from creativelab import policy
from creativelab.world import Creative
from creativelab.writer import HOOK_HEADLINES, CTA_TEXT, template_copy
from evals.heldout import HELDOUT_CLEAN, HELDOUT_VIOLATIONS

ROOT = Path(__file__).resolve().parent.parent
MIN_RECALL, MAX_FPR = 0.90, 0.05

# 10 categories x 12 = 120 violating (headline, body) pairs. {g} = game name.
VIOLATIONS = {
    "money_promise": [
        ("Win real money in {g}", "Play and cash out daily."), ("Earn cash every level", "{g} pays you to play."),
        ("Make money playing {g}", "Start today."), ("Cash prize every week", "Top players in {g} get paid."),
        ("Get paid via PayPal", "Play {g} and withdraw."), ("Win ₹500 today", "Only in {g}."),
        ("Earn $50 a day", "Just play {g}."), ("Make cash fast", "{g} rewards real money."),
        ("Win real cash", "Join {g} tournaments."), ("Earn ca$h playing", "{g} is the easiest way."),
        ("W1n money now", "{g} pays out daily."), ("Real-money rewards", "Play {g} and earn rupees."),
    ],
    "guarantee": [
        ("Guaranteed win in {g}", "Every match."), ("100% win rate", "Play {g} now."), ("You never lose in {g}", "Try it."),
        ("Guaranteed rewards", "{g} gives everyone gems."), ("Wins guaranteed", "Download {g}."),
        ("100% free forever", "{g} has no limits."), ("Never lose again", "{g} makes you a pro."),
        ("Guarantee your top rank", "Play {g}."), ("Our players never lose", "{g}."),
        ("Win every time, guaranteed", "{g}."), ("Certain victory", "{g} makes losing impossible."),
        ("Can't lose in {g}", "Every round is a win."),
    ],
    "fake_alert": [
        ("WARNING: phone slow", "Install {g} to fix it."), ("Virus detected!", "Tap to clean with {g}."),
        ("Alert: storage full", "{g} frees space."), ("Your phone is at risk", "Get {g} now."),
        ("Security alert", "Download {g} immediately."), ("Your device is infected", "{g} can help."),
        ("System warning", "Update {g}."), ("Battery alert!", "Play {g} to cool down."),
        ("Your phone is overheating", "Open {g}."), ("Critical alert", "Tap here for {g}."),
        ("Phone update required", "Install {g} to continue."), ("Device error 0x80", "Fix it by installing {g}."),
    ],
    "competitor": [
        ("Better than Candy Crush", "Try {g}."), ("Like PUBG but faster", "{g}."), ("Free Fire fans love {g}", "Play now."),
        ("Bored of Clash of Clans?", "Switch to {g}."), ("The Roblox alternative", "{g} is here."),
        ("Genshin players, try {g}", "Free to play."), ("CandyCrush who?", "{g} is better."),
        ("Quit PUBG today", "{g} is the new king."), ("Roblox kids love {g}", "Download now."),
        ("Better than Genshin", "{g} has more heroes."), ("Not another Candy Crush", "{g} is new."),
        ("Clash of Clans is old", "Play {g}."),
    ],
    "platform_claim": [
        ("Apple approved game", "{g}."), ("Google recommended", "Play {g}."), ("Facebook official game", "{g}."),
        ("Meta approved", "Try {g}."), ("Apple recommended puzzle", "{g}."), ("Google official pick", "{g}."),
        ("Facebook approved fun", "{g}."), ("Meta recommended RPG", "{g}."), ("Google approved racing", "{g}."),
        ("Apple official app", "{g}."), ("Endorsed by Google", "{g} is safe."), ("Picked by Apple editors", "{g}."),
    ],
    "health_claim": [
        ("{g} cures boredom stress", "Science says so."), ("Boosts your IQ", "Play {g} daily."),
        ("Reduces anxiety fast", "{g} calms you."), ("Treats insomnia", "Play {g} at night."),
        ("{g} cures memory loss", "Play daily."), ("Boost your IQ in a week", "{g}."),
        ("Reduce anxiety with {g}", "Try now."), ("Cure your stress", "Play {g}."),
        ("{g} treats ADHD", "Doctors love it."), ("Boosts your IQ by 20 points", "{g}."),
        ("Clinically calming", "{g} lowers blood pressure."), ("Heal your mind", "{g} fixes depression."),
    ],
    "misleading_free": [
        ("Free gems forever", "{g}."), ("Unlimited free coins", "In {g}."), ("Free diamonds forever", "{g}."),
        ("Unlimited free gems", "Play {g}."), ("Free coins forever in {g}", "No catch."),
        ("Unlimited free spins", "{g}."), ("Free gems forever, no ads", "{g}."), ("Get unlimited free lives", "{g}."),
        ("Unlimited free diamonds", "Only in {g}."), ("Free diamonds forever", "Download {g}."),
        ("Infinite free gems", "{g} gives them away."), ("Gems never run out", "{g} is fully free."),
    ],
    "unverified_stat": [
        ("Only 1% beat level 50", "{g}."), ("Only 2% finish {g}", "Are you one?"), ("90% of players fail", "{g}."),
        ("#1 game in India", "{g}."), ("Millions of players", "Join {g}."), ("Only 0.5% reach the end", "{g}."),
        ("99% of people quit", "Not you? Try {g}."), ("#1 app this week", "{g}."), ("Only 3% solve this", "{g}."),
        ("80% of gamers can't", "Can you? {g}."), ("Top-rated by 10M users", "{g}."), ("Most downloaded game ever", "{g}."),
    ],
    "length": [
        ("This headline is far too long to fit in the ad slot for {g}", "Play now."),
        ("An extremely long headline that keeps on going for {g}", "Play."),
        ("Ready for {g}?", "x" * 130), ("Play {g}", "A body that rambles on " * 7),
        ("A headline that is definitely over forty characters", "{g}."), ("Short", "y" * 126),
        ("Racing, puzzles, heroes and more all in one {g}", "Play."), ("Try {g}", "Long body " * 14),
        ("The most relaxing, beautiful and fun game in the world", "{g}."), ("{g} is here", "z" * 140),
        ("You will not believe how much there is to do in {g}", "Go."), ("Play", "Tap to play {g} now. " * 7),
    ],
    "empty": [
        ("", "Play {g} now."), ("Play {g}", ""), ("   ", "{g}."), ("{g}", "  "), ("", ""), ("\t", "Try {g}."),
        ("Try {g}", "\n"), ("", "Download {g}."), (" ", " "), ("Join {g}", ""), ("", "{g} is here."), ("{g}!", "   "),
    ],
}

CLEAN_HANDWRITTEN = [
    ("Win the weekly cup", "Race friends in {g}. Play now."), ("Download free today", "{g} is free to play with in-app purchases."),
    ("Earn stars, unlock cars", "Progress in {g}. Try level 1."), ("Relax with {g}", "Calm music and easy controls."),
    ("New season is live", "Fresh levels in {g}. Play now."), ("Build your squad", "Explore and fight bosses in {g}."),
    ("Play offline anytime", "{g} works without Wi-Fi."), ("Beat your best time", "{g} tracks every lap."),
    ("Your daily puzzle", "A new challenge in {g} every day."), ("Meet the heroes", "Twelve chapters of story in {g}."),
    ("Join a guild", "Raid with friends in {g}."), ("Free to start", "{g} has optional purchases."),
    ("Tune your ride", "60 cars to unlock in {g}."), ("Short sessions, big fun", "{g} fits your break."),
    ("Climb the leaderboard", "Weekly ranks in {g}."), ("Match, build, repeat", "Rebuild a seaside town in {g}."),
    ("Treat yourself to a break", "{g} is a calm puzzle game."), ("Alerting all racers", "Season 3 starts in {g}."),
    ("Win races, earn trophies", "{g}. Download free."), ("Google Play top new game", "{g} launched this week."),
]

GAMES = ["Tile Harbor", "Drift Kings", "Realm of Ash"]


def clean_set() -> list[tuple[str, str]]:
    """60 template ads (3 games x 20) plus 60 hand-written ads (3 games x 20)."""
    out = []
    games = json.loads((ROOT / "data" / "games.json").read_text())
    for g in games:
        variants = [(h, c) for h in HOOK_HEADLINES for c in CTA_TEXT][:20]
        out += [template_copy(Creative(g["id"], g["genre"], h, "gameplay", c, 15), g) for h, c in variants]
    for g in GAMES:
        out += [(h.format(g=g), b.format(g=g)) for h, b in CLEAN_HANDWRITTEN]
    return out


def violation_set() -> list[tuple[str, str, str]]:
    out = []
    for cat, items in VIOLATIONS.items():
        for j, (h, b) in enumerate(items):
            g = GAMES[j % 3]
            out.append((cat, h.format(g=g), b.format(g=g)))
    return out


def main() -> int:
    bad = violation_set()
    good = clean_set()
    missed = [(c, h, b) for c, h, b in bad if policy.check(h, b).ok]
    false_pos = [(h, b, policy.check(h, b).reasons) for h, b in good if not policy.check(h, b).ok]
    recall = 1 - len(missed) / len(bad)
    fpr = len(false_pos) / len(good)
    by_cat = {c: sum(1 for m in missed if m[0] == c) for c in VIOLATIONS}
    h_missed = [(c, h, b) for c, h, b in HELDOUT_VIOLATIONS if policy.check(h, b).ok]
    h_fp = [(h, b, policy.check(h, b).reasons) for h, b in HELDOUT_CLEAN if not policy.check(h, b).ok]
    heldout = {"violating_ads": len(HELDOUT_VIOLATIONS), "clean_ads": len(HELDOUT_CLEAN),
               "recall": 1 - len(h_missed) / len(HELDOUT_VIOLATIONS), "false_positive_rate": len(h_fp) / len(HELDOUT_CLEAN),
               "missed": h_missed, "false_positives": h_fp}
    report = {"violating_ads": len(bad), "clean_ads": len(good), "recall": recall, "false_positive_rate": fpr,
              "missed_by_category": by_cat, "missed": missed, "false_positives": false_pos,
              "heldout": heldout, "gate": {"min_recall": MIN_RECALL, "max_fpr": MAX_FPR}}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "policy_eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"recall {recall:.1%} ({len(bad) - len(missed)}/{len(bad)}), false positives {fpr:.1%} ({len(false_pos)}/{len(good)})")
    print(f"held-out: recall {heldout['recall']:.1%} ({len(HELDOUT_VIOLATIONS) - len(h_missed)}/{len(HELDOUT_VIOLATIONS)}), "
          f"false positives {heldout['false_positive_rate']:.1%} ({len(h_fp)}/{len(HELDOUT_CLEAN)})")
    for m in h_missed:
        print("  held-out miss:", m)
    for fp in h_fp:
        print("  held-out false positive:", fp)
    for c, n in by_cat.items():
        if n:
            print(f"  missed {n} {c}")
    for fp in false_pos:
        print("  false positive:", fp)
    return 0 if recall >= MIN_RECALL and fpr <= MAX_FPR else 1


if __name__ == "__main__":
    sys.exit(main())
