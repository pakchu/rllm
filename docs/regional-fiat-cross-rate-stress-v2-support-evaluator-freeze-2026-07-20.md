# RFXS2-576 source-support evaluator freeze — 2026-07-20

## Freeze decision

Freeze the source-only RFXS2-576 support evaluator before any real residual,
z-score, state, event count, comparator statistic, or support pass/fail result
is computed.

The evaluator may decide only whether the already-preregistered candidate has
enough source incidence, balance, calendar breadth, contributor breadth, and
novelty to justify writing a later strict outcome evaluator. It cannot read or
compute a trade return.

## Frozen implementation

```text
training/build_regional_fiat_cross_rate_stress_support.py
SHA-256 dc9f2237160de4db843a32b053d0ebdd46dc44aa2074e07d1579088199ea41a4

tests/test_build_regional_fiat_cross_rate_stress_support.py
SHA-256 e2991a7a60e1e7ced784b7e95627a00fbc3c80ee7a64ff7c6beb7fbcdb098737
```

The production entry point rejects an uncommitted evaluator by comparing its
worktree bytes with `git show HEAD:<path>`. It also requires frozen source commit
`e576d22c6f2d567d4b40358f755bef4b27c188d4` to be an ancestor of that HEAD.

## Complete input whitelist

Only these seven files may be opened:

1. `data/binance_regional_fiat_cross_rate_btc_2020-11_2023/`
   `BTC_regional_fiat_cross_rate_1d_2020-11-01_2023-12-31.csv.gz`
2. `data/binance_regional_fiat_cross_rate_btc_2020-11_2023/build_manifest.json`
3. `docs/regional-fiat-cross-rate-stress-mechanism-decision-2026-07-20.md`
4. `docs/regional-fiat-cross-rate-stress-rfxs576-source-rejection-2026-07-20.md`
5. `docs/regional-fiat-cross-rate-stress-v2-mechanism-decision-2026-07-20.md`
6. `results/fiat_quote_participation_rotation_clocks_2026-07-17.csv`
7. `data/stablecoin_denominator_dislocation_clocks_2023.csv.gz`

Every byte hash is hard-coded in `STATIC_INPUT_SHA256`. The test suite walks all
production path literals, asserts the exact whitelist, and verifies that the
only `pd.read_csv` targets are the source panel, FQPR clock, and SDDR clock.

There is no USD-M execution, funding, 2024+ source, future return, PnL, equity,
CAGR, MDD, or portfolio input path.

## Frozen causal implementation

- Validate the exact 1,156-day UTC grid `[2020-11-01, 2024-01-01)` and the
  following-day availability timestamp before deriving a feature.
- Compute each regional return residual against BTCUSDT.
- For each residual, compute median and MAD from exactly 180 strictly prior
  complete residuals; current-day inclusion, expanding fallback, standard
  deviation fallback, clipping, and zero-MAD repair are prohibited.
- Require all three regional z-scores, take their median, and assign the frozen
  `+1 / 0 / -1` state at `+/-1`.
- A candidate requires a nonzero state different from an immediately preceding
  **valid** state. The first finite state after warmup has no preceding valid
  state and therefore cannot itself emit an event. This is the literal frozen
  onset rule, not a result-selected suppression.
- Decision is source day plus one UTC day, entry is five minutes later, and exit
  is exactly 576 five-minute bars later.
- Reserve each independent clock globally before split slicing. A suppressed
  transition still consumes its source-state transition. A split-crossing
  reserved event can suppress a later candidate but is scored in neither split.
- Accept only clocks whose source, decision, entry, exact 576-bar interval, and
  exit are contained in the declared split.

## Frozen support and novelty interpretation

All count, side, contributor, month, quarter, and half gates are exactly those
in the RFXS2 mechanism. Calendar buckets use UTC entry time. A contributor is
an event-day region at or beyond the threshold in the event's state direction;
it does not need a new individual crossing.

Spearman uses average ranks on all finite paired source days and fails closed
for fewer than three observations, a constant ranked vector, or a non-finite
result.

For FQPR and SDDR, exact-entry Jaccard uses only their frozen primary clocks.
Signed exposure is side on every occupied half-open five-minute interval and
zero otherwise across the complete frozen comparison horizon. The novelty gate
uses the **absolute magnitude** of Pearson correlation between those signed
series. This resolves “signed occupied-exposure correlation at most 0.40” as an
orthogonality gate; a perfectly inverse clock cannot pass as low correlation.

## Frozen controls

The evaluator builds independent EUR-only, TRY-only, BRL-only,
three-book-sign-only, BTC-return-shadow, and one-day-stale clocks. Direction
flip and deterministic random side clone the exact primary reservation clock.
The random seed string is:

```text
RFXS2-576-random-side-20260720|<UTC ISO entry_time>
```

Controls can diagnose or later falsify the mechanism but cannot replace a
failed primary.

## Output and stop rule

The committed evaluator will write only:

```text
results/regional_fiat_cross_rate_stress_v2_support_2026-07-20.json
results/regional_fiat_cross_rate_stress_v2_clocks_2026-07-20.csv
```

The JSON records evaluator commit/hash, every static input hash, all source-only
gates, and explicit false flags for execution, funding, 2024+, future-return,
PnL/CAGR/MDD, and outcomes. `allow_nan=False` prevents an undefined statistic
from being serialized as a passing value.

Any failed train, 2023-selection, or novelty gate retires RFXS2-576 and forbids
an outcome evaluator. Only a complete pass authorizes a separately written,
tested, committed, and hash-frozen strict evaluator.

## Pre-result verification

- Synthetic source-only tests: `18 passed`
- Python syntax compilation: passed
- Independent code review: PASS, zero remaining findings
- Ruff/mypy/pyright: unavailable in the environment
- Real source-derived event/support/novelty values opened: zero
- Execution/funding/outcome values opened: zero
