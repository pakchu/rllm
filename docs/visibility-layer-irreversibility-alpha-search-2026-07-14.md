# Visibility-layer irreversibility alpha — one-shot result

## Verdict

**Rejected as an executable alpha.** The frozen rule choosing the direction of
whichever price/flow HVG layer was more irreversible lost in fit and in the
one-shot 2023 inspection. No 2024+ outcome was opened.

The protocol was committed before outcome access in `311a46a`. Support-only
preflight and an independent critic cleared one fixed pre-2024 run. No HVG
window, degree metric, layer mapping, threshold, entry, hold, cost or admission
rule changed after 2023 became visible.

## Fixed primary result

All returns include the complete calendar window, including idle time. Replay
uses 0.5x leverage, 6 bp per side, minute-05 next-open entry, fixed 12-hour hold,
and favorable-first/adverse-second strict OHLC MDD.

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| Fit 2020-06..2022 | -15.56% | -6.33% | 21.21% | -0.30 | 339 | 168 / 171 |
| 2023 | -15.08% | -15.09% | 19.00% | -0.79 | 139 | 77 / 62 |
| 2023 H1 | -7.57% | -14.68% | 10.65% | -1.38 | 76 | 38 / 38 |
| 2023 H2 | -8.13% | -15.49% | 11.26% | -1.38 | 63 | 39 / 24 |

The mean-trade test also rejects usable edge: fit mean `-0.0409%`, approximate
`p=0.574`, effect size `d=-0.031`; 2023 mean `-0.1144%`, `p=0.090`,
`d=-0.144`.

Zero implementation cost did not rescue 2023:

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| Fit, 0 bp/side | +3.49% | +1.34% | 18.23% | 0.07 | 339 |
| 2023, 0 bp/side | -7.70% | -7.70% | 13.38% | -0.58 | 139 |
| 2023 H1, 0 bp/side | -3.25% | -6.46% | 8.38% | -0.77 | 76 |
| 2023 H2, 0 bp/side | -4.59% | -8.91% | 8.44% | -1.05 | 63 |

## Frozen controls

| Policy | Fit abs. return | Fit ratio | 2023 abs. return | 2023 ratio |
|---|---:|---:|---:|---:|
| Primary layer arbitration | -15.56% | -0.30 | -15.08% | -0.79 |
| Same events, flow follow | -2.89% | -0.07 | -11.33% | -0.77 |
| Same events, price follow | -6.72% | -0.16 | -18.43% | -0.84 |
| Price-HVG-only, price follow | +12.09% | 0.24 | +4.54% | 0.46 |
| Flow-HVG-only, flow follow | +41.97% | 0.78 | +2.12% | 0.21 |
| Exact direction flip | -25.81% | -0.29 | -1.20% | -0.17 |
| Signal delayed 6 hours | -33.14% | -0.37 | -21.86% | -0.77 |
| Signal delayed 7 days | -39.56% | -0.37 | +5.83% | 0.81 |

The flow-HVG component is the only representation-level clue: it was positive
in fit and both 2023 halves, but fit/2023 ratios were only `0.78/0.21`, and it
lost in 2022 H1. It is a weak beta token, not a promoted alpha. The price-only
component lost in 2023 H2. The seven-day placebo loses fit and therefore does
not identify a local mechanism.

Minute-05/10/15 entries all lost fit and 2023. Fixed 6/12/24-hour holds produced
fit absolute returns of `-34.93%`, `-15.56%`, and `-32.90%`; 2023 returns were
`-0.04%`, `-15.08%`, and `-20.76%`. These report-only diagnostics cannot replace
the primary.

## Representation audit

The descriptor passed every preregistered novelty check:

- Spearman(layer ratio, cross-map dominance): `-0.022`.
- Spearman(layer score, price volatility): `-0.104`.
- Spearman(layer score, mean absolute flow): `+0.107`.
- Spearman(layer ratio, 28-block price trend): `+0.132`.
- Spearman(layer score, order-3 price permutation entropy): `+0.188`.
- Primary-versus-cross-map event Jaccard: `0.157`.

Thus the HVG representation is not a simple cross-map, volatility, flow-scale,
trend or ordinal-entropy clone. Distinctness does not imply predictability: the
fixed dominant-layer arbitration is decisively wrong, while each raw layer
alone remains far below the risk-efficiency target.

## Leakage and source limits

- The returned analysis frame is strictly before `2024-01-01`; a gzip
  cutoff-crossing chunk may be decoded then immediately discarded. Discarded
  rows enter no returned frame, feature, hash, support count or outcome.
- Each HVG contains exactly 168 completed six-hour blocks ending with current
  `[T-6h,T)`. Tests lock this intentional difference from cross-map's
  current-excluded library.
- Both degree distributions share support
  `0..max(max_in_degree,max_out_degree)` before Jeffreys smoothing.
- The q80 threshold is shifted one state and never sees its current score.
- Signal at minute 00 enters minute-05 open; trades are split-contained and
  non-overlapping.
- The artifact hashes only the returned pre-2024 frame. 2024+ remains sealed.
- HVG irreversibility is a time-asymmetry magnitude, not proof of causal
  direction or future predictability.

## Frozen conclusion

Do not tune the 168-block window, HVG edge/tie rule, Jeffreys prior, symmetric
KL, layer ratio, q80 gate, arbitration map, delays or holds on this inspected
sample. Retain separate price/flow HVG irreversibility and their ratio only as
weak beta representations for a materially different preregistered learner or
a genuinely fresh forward shadow period.

Reproduce:

```bash
PYTHONPATH=. .venv/bin/python -m training.search_visibility_layer_irreversibility_alpha
```
