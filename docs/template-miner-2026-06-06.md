# Analyzer-conditioned template miner (2026-06-06)

## Purpose

Shift away from hindsight best-action labels and test whether any fixed action template has train-stable positive expectancy under simple analyzer text conditions.

## Method

- Rule form: `fixed action | analyzer_fact [& analyzer_fact]`
- Facts are extracted from the prompt summary only, e.g. `regime`, `trend_alignment`, `location`, `risk_state`, macro dollar state, kimchi premium state, and context tags.
- Candidate rules are mined on train only.
- Val/OOS are reporting-only in the initial strict scan.

## Strict scan result

Command used:

```bash
python -m training.economic_template_miner \
  --max-terms 2 \
  --min-train-n 80 \
  --min-train-ci-pct 0.0 \
  --min-val-n 20 \
  --min-val-mean-pct 0.0
```

Result:

- train candidate rules: `29,120`
- train survivors with `n>=80` and train 95% CI lower bound `>=0`: `0`
- reported rules: `0`
- val-stable rules: `0`

## Interpretation

At the coarse analyzer text level, there is no statistically defensible fixed action template. This reinforces the previous drift diagnostic: the current dataset mostly encodes a hindsight action lottery rather than a reusable tradable action prior.

## Runtime note

A relaxed `max_terms=3` scan caused combinatorial blow-up even after fact-frequency pruning. The script now has `--min-fact-count`, but deeper mining needs additional bounding (top-N facts by train information gain, streaming heaps, or precomputed row fact indices) before being useful.

## Next implication

The next structural change should alter the label/action representation, not keep searching this label space. Candidate directions:

1. Generate path-shape analyzer labels: excursion, adverse excursion, invalidation, time-to-target, time-to-stop.
2. Define fewer action templates with explicit stop/target/invalidation instead of many hold-only labels.
3. Train analyzer to describe path/risk state, then train trader/RL on those descriptions with a reward based on realized strict path PnL.
