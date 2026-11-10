# CreativeLab

An agent that designs, writes, checks and ranks mobile game ad creatives, and
a test harness showing that a bandit finds the winning creative with fewer
impressions than a uniform A/B split.

Given a game's store page, a LangGraph pipeline plans 40 creative variants
(hook, visual, call to action, length), writes the copy, rejects anything
that breaks ad policy, scores every variant with an XGBoost click-through-rate
(CTR) model, and picks the top 8 to test.

Everything runs on a simulated ad market with known true CTRs (see "Limits").
That is what makes the model checkable: we know the right answer.

## Results

From `python -m scripts.run_all` (2M training impressions, 3 games, 100 bandit tests):

**Choosing creatives**

| | |
|---|---|
| True CTR of the agent's top 8 vs the average of all 40 variants | **+33%** |
| Rank correlation, predicted vs true CTR, across the 40 variants | 0.88 |

**CTR model** (400K held-out impressions)

| | XGBoost | Logistic regression | Ceiling (true click probabilities) |
|---|---|---|---|
| AUC | **0.615** | 0.607 | 0.621 |
| Log loss | 0.1487 | 0.1490 | |

Clicks are noisy, so even perfect knowledge only reaches 0.621 AUC. The
XGBoost model recovers about 95% of the achievable gain over chance, and
beats the baseline because it learns interactions (a "challenge" hook works
for racing games and hurts puzzle games).

**Testing the top 8: Thompson sampling vs uniform split**

Both stop when one creative has a 95% posterior chance of being best.

| | Uniform A/B | Thompson sampling |
|---|---|---|
| Median impressions to a decision | 88,500 | **59,000 (-33%)** |
| Picked the true best creative | 95% | 95% |
| CTR while the test ran | 4.61% | **5.43%** |
| Tests that hit the 600K cap | 11 | 6 |

**Policy guardrail**

| | Recall (violations caught) | False positives |
|---|---|---|
| Main set: 120 violating + 120 clean ads | 100% | 0% |
| Held-out set: 40 violating + 40 clean ads | **97.5%** | **0%** |

The rules were tuned on the main set (first version: 87.5% recall, 2.6%
false positives), so the held-out score is the honest one. Its single
"miss" is a labelling mistake: that body is 119 characters, under the
125-character limit. Both sets were written by the author of the rules, so
expect lower recall on ads written by someone trying to get around them.

## How it works

```
store page ──► plan ──► write ──► check ──► score ──► select ──► top 8 to test
                         ▲          │
                         └──────────┘  failed policy: one rewrite with reasons,
                                       then dropped
```

- **plan** (`pipeline.py`): picks 40 of the 360 possible attribute combinations
  so that every hook, visual, CTA and length appears about equally often.
  A balanced design is what lets a test separate their effects.
- **write** (`writer.py`): with `ANTHROPIC_API_KEY` set, Claude writes headline
  and body for each planned variant and is told why any earlier attempt failed.
  Without a key, templates are used. The writer cannot change the test design.
- **check** (`policy.py`): blocks money promises, guarantees, fake system
  alerts, competitor and platform names, health claims, misleading "free"
  claims, made-up statistics and over-length copy, including character swaps
  such as "ca$h" and "w1n".
- **score** (`ctr_model.py`): XGBoost on one-hot creative attributes, genre,
  placement and audience, trained on simulated impression logs. A creative's
  score is its expected CTR over the usual placement and audience mix.
- **select**: top 8 by predicted CTR, at most one per hook-and-visual pair so
  the test still covers different ideas.
- **bandit test** (`bandit_test.py`): Beta-Bernoulli Thompson sampling against
  an equal split, with the same stopping rule for both.

## Run it

```bash
pip install -r requirements.txt
pytest -q                          # 7 unit tests
python -m evals.policy_eval        # policy gate: fails CI below 90% recall or above 5% false positives
python -m scripts.run_all          # trains the model, runs the agent, runs 100 bandit tests
```

Example output for each game is in `results/selections.json`.

## Limits

- The ad market is simulated. The effect sizes are invented, so the 33% and
  -33% figures show the method under known conditions, not live results.
- The CTR model scores attributes, not the copy text. Two creatives with the
  same attributes get the same score whatever the wording.
- The policy check is rule-based. It is fast and explainable but misses
  phrasing nobody wrote a rule for. An LLM reviewer would be the next layer.
- Creative fatigue and changing traffic during a test are not modelled.
