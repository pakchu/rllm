# LCLR-24 preregistration — London Cash-Lead Release

Status: **frozen before real LCLR incidence or any post-window outcome is
calculated**.

## Singleton hypothesis

On London weekdays, observe only the 12 completed five-minute candles from
15:00 through 16:00 `Europe/London` for Coinbase BTC-USD and Binance BTCUSDT
perpetual. If both markets moved in the Coinbase direction but Coinbase moved
farther, treat Coinbase as the mandatory cash leader. Follow the completed
Coinbase direction only when at least two of four separately weak source votes
also agree:

1. absolute Coinbase–perpetual displacement is at or above its strictly prior
   median;
2. Coinbase path efficiency is at or above its strictly prior median;
3. Coinbase quote-notional participation is at or above its strictly prior
   median; and
4. the final 15 minutes continue in the full-window Coinbase direction.

There is one policy, one vote count, one window, one latency, and one hold. No
threshold or holding-period grid is allowed.

## Strictly prior features

For each complete London window:

```text
cash_return      = log(Coinbase last close / first open)
perp_return      = log(Binance last close / first open)
relative_return  = cash_return - perp_return
cash_efficiency  = abs(cash_return) / sum(abs(12 Coinbase partition returns))
final_cash_return = sum(final three Coinbase partition returns)
cash_quote_share = Coinbase(volume * close)
                   / [Coinbase(volume * close) + Binance quote volume]
```

Every median is computed over at most the previous 126 weekday source-window
rows, with the current row shifted out before rolling. Incomplete rows remain
NaN and are ignored, and at least 63 finite prior observations are required.
Missing Coinbase or Binance candles invalidate the entire day; they are never
filled. Source values at 16:00 or later are not parsed by the support run.

## Causal execution contract

- The 15:55 candle closes at 16:00 London, which is the decision time.
- The 16:00–16:05 bar is a mandatory complete latency bar.
- Entry is the 16:05 London Binance perpetual open.
- Exit is exactly 24 five-minute bars later: 18:05 London.
- Side is the sign of the completed Coinbase window return.
- Leverage for later evaluation is 0.5x.
- Base cost is 6 bp/notional/side; stress cost is 10 bp/notional/side.
- Funding uses the frozen strict boundary: interior exact-time funding is
  symmetric; exact-entry/exact-exit credits are discarded while debits are
  retained.
- Strict MDD includes the global/pre-entry high-water mark, entry cost, every
  held five-minute path, exact funding, virtual adverse exit fee, and actual
  exit.
- CAGR uses the full calendar split, including warm-up and idle cash.

The support stage loads none of those execution bars, funding marks, or return
labels. It freezes only event incidence and side.

## Outcome-blind support gate

The frozen 2020–2022 source clock must contain:

- at least 180 total events;
- at least 110 across 2020–2021;
- at least 45 in each of 2020 and 2021;
- at least 55 in 2022;
- at least 22 in each 2022 half;
- at least 8 in each of the 12 calendar quarters;
- at least 30% long and 30% short events in the full, train, and test clocks;
  and
- no single calendar quarter above 18% of all events.

Failure rejects LCLR-24 before outcomes. Reducing votes, moving the London
window, lowering a support minimum, changing latency, or changing the hold is
forbidden after incidence is observed.

## Later performance sequence, only if support passes

1. Commit and hash-freeze the strict evaluator.
2. Open 2020–2021 train and calendar-2022 test only.
3. Require in both train and test: positive absolute return, CAGR/strict MDD at
   least 3, strict MDD at most 15%, positive 10 bp/side stress return, positive
   one-bar-delayed return, at least 20 bp mean gross underlying move, and
   weekly-cluster sign-flip `p <= 0.10`.
4. Reject if cash-only, a separately normalized 12:00–13:00 London control, or
   the weekend clock independently explains the full result. Exact side flip
   is diagnostic only.
5. Only one unchanged pre-2023 pass can open a separately frozen 2023+ source.
6. Evaluate 2023, 2024, 2025, and 2026 YTD sequentially, stopping at the first
   failed year.

## Source and novelty boundary

The exact source hashes and official interfaces are recorded in
[`london-cash-lead-release-mechanism-decision-2026-07-20.md`](london-cash-lead-release-mechanism-decision-2026-07-20.md).
The experiment uses no CME benchmark value. It differs from the rejected
all-hours Coinbase policies by its fixed DST-aware 12-partition event primitive
and from the coarse calendar × OI/funding scan by using a completed Coinbase
cash path with explicit cross-venue lead.

This repository has broad prior exposure to BTC history. The strongest honest
claim available is a candidate-level frozen sequence, not a globally pristine
human holdout.
