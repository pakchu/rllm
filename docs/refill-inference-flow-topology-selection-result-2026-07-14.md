# RIFT-96 frozen pre-2024 result — 2026-07-14

## Verdict

**Rejected without repair.** The hash-frozen evaluator opened only 2020–2023.
Calendar 2024 test, calendar 2025 eval, and 2026 YTD remain sealed.

| split | absolute return | CAGR | strict MDD | CAGR/MDD | gross underlying mean | trades |
|---|---:|---:|---:|---:|---:|---:|
| 2020–2022 train | -30.07% | -11.24% | 36.92% | -0.30 | -6.13 bp | 374 |
| 2023 selection | -11.30% | -11.31% | 12.09% | -0.94 | -16.02 bp | 85 |
| 2023 H1 | -9.44% | -18.13% | 10.12% | -1.79 | -34.94 bp | 42 |
| 2023 H2 | -2.06% | -4.04% | 3.94% | -1.03 | +2.46 bp | 43 |

Train and selection weekly-cluster one-sided p-values were `0.96250` and
`0.99780`, respectively. Both halves were net negative, and both train and
selection failed the frozen `>12 bp` gross hurdle.

## Falsification controls

The exact short flip was not a promotable reverse alpha:

- train: `-12.05%`, gross `+6.13 bp`, ratio `-0.18`, p `0.71689`;
- 2023: `+1.64%`, gross `+16.02 bp`, ratio `0.29`, p `0.33314`.

Its gross sign and magnitude did not generalize. The 1-hour stale-setup control
was also diagnostic rather than profitable: train gross `13.38 bp` but net
`-0.09%` with p `0.44845`; 2023 gross fell to `11.35 bp`, below break-even,
with net `-0.86%` and p `0.52280`.

Matched same-bar, simple two-bar momentum, Spot-only, centroid-free,
no-path-quality, and no-derivatives-crowd controls all failed. The primary did
not beat any control except its one-bar delayed copy on the minimum
train/selection ratio.

## Root cause and stop decision

A rare, reproducible two-bar pressure sequence existed, but it did not identify
persistent offer-side scarcity. The long action was adverse on average, the
short action was regime-dependent, and the stale setup performed similarly or
better. This falsifies the claimed setup-to-confirmation causal timing rather
than exposing a threshold shortage.

No percentile, confirmation rule, direction, hold, cost, or gross hurdle will
be repaired on RIFT-96. The next candidate must use a different state
transition and cannot promote the 2023-only short flip or the near-break-even
stale control.

## Artifact

- result SHA-256:
  `2e80a924f540d06e751e1e3d57d4b918bc27f78964df8a692fd5c1aaebe49525`
- evaluator-freeze manifest SHA-256:
  `830a1ad7ee726097980ca4e3b57a3c88f3e414f22fb7f1b672b6738d8d2fa75f`
