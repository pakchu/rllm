# AFGI-12 preregistration — six-alt flow geometry Gross9 marginal

## Decision

Freeze one terminal supervised attempt on the existing six-alt, price-free
hourly source before fitting the exact policy or reading its BTC outcomes.

`AFGI-12` asks whether the **cross-sectional geometry** of completed-hour
aggressive flow and activity in ETH, SOL, BNB, XRP, DOGE, and ADA can identify
a sparse 12-hour BTC long/short sleeve that improves frozen Gross9 at identical
configured gross.

This is not a repair of the retired same-source rules:

- FCIR-12 centrality/quiet-crowd rule: rejected at 2023 train;
- DTAC-8 flow/premium tail consensus: rejected at 2023 train;
- TGR-12 mean-ticket leader gap: rejected at its source-support gate.

AFGI uses no directed centrality, premium, tail-vote, mean-ticket feature,
symbolic event threshold, or repaired side. If this battery fails, the
six-alt hourly flow/activity supervised-model family closes.

## Frozen source and causal clock

- Source:
  `data/binance_six_alt_price_free_flow_2023_2026/six_alt_price_free_flow_1h_2023-01-01_2026-06-01.csv.gz`
- Source SHA-256:
  `bf4d67ee02948444712a6ff7862a0d4f4ae4ae2a704c9d0586538043c169f6b9`
- Fields read: feature-availability time, symbol, quote volume, trade count,
  taker-flow fraction, and validity.
- Each source interval `[H-1h,H)` is available at `H`.
- Signal is attached to the BTC five-minute bar at `H`; one complete latency
  bucket is left empty; entry is the BTC open at `H+5m`.
- Every symbol must be valid. Missing values are never carried, interpolated,
  zero-filled, or replaced with stale data.

The feature builder cannot read BTC price, return, funding, PnL, Gross9 state,
or any 2025/2026 value.

## Frozen 52-feature representation

For each symbol, retain six values:

1. current completed-hour flow fraction;
2. current-inclusive six-hour flow mean;
3. current-inclusive 24-hour flow mean;
4. six-hour mean minus 24-hour mean;
5. quote-volume robust z-score;
6. trade-count robust z-score.

The robust reference is the preceding 720 positional hours excluding current,
with at least 672 observations. It uses linear q25/q50/q75 and scale
`(q75-q25)/1.349`; non-positive scale invalidates the row.

Sixteen cross-sectional features add:

- current and six-hour population mean, standard deviation, and signed breadth;
- current absolute-flow HHI;
- current flow/activity population correlations;
- six-hour versus 24-hour cosine;
- prior-only 168-hour flow-correlation PC1 variance share, effective rank,
  normalized loading entropy, current/six-hour projections, and one-hour
  subspace drift.

PCA excludes current. `np.linalg.eigh` eigenvalues are sorted descending and
tiny negatives are clipped to zero. PC1 is oriented to positive loading sum;
when that sum is within `1e-12` of zero, the first frozen-order maximum-absolute
loading is made positive. Subspace drift is `1-abs(v_t dot v_(t-1))`.
Zero HHI/cosine/correlation denominators invalidate the row. All 52 values must
be finite.

## Frozen learner and folds

- ExtraTreesRegressor, scikit-learn `1.7.2`;
- 256 trees, depth 4, minimum leaf 96, max-features 0.75, no bootstrap;
- seeds `7 / 71 / 715`;
- fixed 12-hour / 144-bar no-stop path;
- 0.5x unit leverage and 6 bp/notional/side;
- exact realized funding under the existing Gross9 multiplicative engine;
- four targets: long/short exact net return and strict adverse drawdown.

For each side and seed, utility is predicted return minus `0.5 * adverse`.
Side score is seed mean utility minus `0.5 * population seed std`. Choose the
larger side, with long winning exact ties. Entry requires
`score >= max(0, prior-OOS q95)`.

Every fit purges labels unless entry, all held bars, funding events, and exit
are strictly before its fit boundary:

| Prediction fold | Fit ends before | Threshold source | Candidate trades |
|---|---|---|:---:|
| 2023 Q2 | 2023-04-01 | none | no, calibration only |
| 2023 Q3 | 2023-07-01 | 2023 Q2 OOS scores | yes |
| 2023 Q4 | 2023-10-01 | 2023 Q3 OOS scores | yes |
| 2024 | 2024-01-01 | 2023 Q4 OOS scores | yes |

The 10 bp stress replay changes only entry/exit cost. It cannot refit, rescore,
change thresholds, sides, coordination, or schedules.

## Candidate and portfolio cells

Coordination modes are fixed:

1. unrestricted;
2. Gross9 flat at the completed signal bar;
3. continuous Gross9 drawdown at least 5% at the signal bar.

The 2024 boundary does not reset Gross9 state. Candidate weights are
`0.25 / 0.50 / 0.75 / 1.00`, producing exactly 12 portfolio cells.

For candidate weight `c`, the combined portfolio keeps frozen Gross9 weights
and adds `c`. The same-gross control contains Gross9 only, with every baseline
weight multiplied by `(9+c)/9`. Both have static configured gross `9+c`.

Selection uses only common-coverage calendars:

- `2023-07-01 <= time < 2024-01-01`;
- `2024-01-01 <= time < 2025-01-01`.

Standalone must be positive in both windows, have CAGR/strict-MDD at least 1.5,
strict MDD at most 15%, at least 25/50 trades, at least 20% on each side, and
positive 10 bp stress return.

The combined portfolio must retain at least 97% of unscaled Gross9 absolute
return, keep MDD at most 20%, improve CAGR/MDD by at least 0.05 versus the
same-gross control in **both** windows, reduce unscaled Gross9 MDD in at least
one window, keep exact-entry Jaccard at most 0.25 versus every Gross9 sleeve,
and stay profitable with candidate-only 10 bp stress.

Overlap against FCIR, DTAC, and TGR is reported diagnostically and cannot repair
or select a cell. Exactly one pre-2025 top row may survive.

## Outcome boundary

No 2025 or 2026 candidate result may open unless one pre-2025 cell passes.
Later historical windows can veto only the frozen top row; they cannot rerank,
repair, or certify it. Even a historical survivor remains shadow-only until an
immutable 90-day prospective gate with at least 30 trades passes.

Machine contract:
`results/alt_flow_geometry_inference_marginal_preregistration_2026-07-28.json`

Machine contract SHA-256:
`a27e2ed6afe9dcbf158b9b6fd7091c2d2beb617034e825f7c7444c1fa54c7fa4`
