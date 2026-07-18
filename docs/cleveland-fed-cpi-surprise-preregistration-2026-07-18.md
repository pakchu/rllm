# CFCS-1 Cleveland Fed CPI-surprise preregistration

## Mechanism

CFCS-1 compares first-release headline and core CPI month-over-month values
with the last Cleveland Fed nowcasts strictly before the official release.
When both forecast errors share a sign and their equal mean is at least
0.05 percentage points in magnitude, a downside surprise goes long BTC and an
upside surprise goes short. The signal contains no crypto or market-price
feature.

## Frozen execution

- signal: official 08:30 America/New_York CPI release;
- entry: 08:35 America/New_York on the release day;
- exit: 16:00 America/New_York on the release day;
- exposure: 0.5x BTCUSDT perpetual;
- costs: 6 bp/notional/side, 10 bp stress;
- exact realized funding on `[entry, exit)`;
- full-calendar CAGR and strict intratrade MDD;
- singleton policy with no mutable parameter.

## Source-only density

| Window | Events | Long | Short |
|---|---:|---:|---:|
| 2019_source_history | 6 | 3 | 3 |
| 2020 | 9 | 5 | 4 |
| 2021 | 8 | 2 | 6 |
| 2022 | 9 | 3 | 6 |
| stage1 | 26 | 10 | 16 |
| 2023_h1 | 4 | 4 | 0 |
| 2023_h2 | 4 | 4 | 0 |
| 2023 | 8 | 8 | 0 |

The 0.05pp threshold is the highest source-only grid value retaining at least
24 Stage1 events, eight in each Stage1 year, eight in sealed 2023, and four in
each sealed half. No post-release BTC bar or funding row was loaded to choose
it. Sealed 2023 has eight long and zero short signals; that known source-side
regime imbalance cannot be repaired after outcomes.

## Controls

- headline-only, core-only, and no-concordance mechanism controls;
- exact direction flip;
- one-calendar-day delay and seven-calendar-day placebo;
- every control receives the same costs, funding, strict-MDD, subperiod, and
  significance battery.

## Sequential boundary

Stage1 may physically parse only `[2020-01-01, 2023-01-01)`. Calendar 2023 can
open only after exact replay of a passing Stage1 artifact under an immutable
evaluator. Any Stage1 failure rejects CFCS-1 unchanged. Calendar 2024+ remains
sealed. Portfolio overlap is inspected only after standalone Stage2 passes.

## Vintage limitation

The Cleveland Fed chart is a frozen current historical vintage, not a proven
immutable point-in-time archive. Forward timestamped capture of the final
pre-release nowcast is mandatory before live promotion.

## Frozen identity

- source commit: `a6f824f`
- source panel SHA-256: `e8755bfd15ec135b2a85cedada8880bf5d4518ed07f4eef43b4b3820211d508e`
- clock SHA-256: `cff8d0f8d7810400bc78f833cc91996a7b2cd0e9d5903fe0ef154f0e38a71739`
- preregistration manifest: `61604984515942428977d24afa39299a5083e8e2b36fef3ac7cf95b4eddf6b60`
