# CME weekend-anchor BTC alpha source feasibility — 2026-07-19

## Verdict

**Blocked on primary historical data; no backtest was run.**

A causal CME-close/reopen anchor could be materially different from the already
rejected Binance-only weekend continuation/reversal controls. However, no free,
automatable official CME source was found that supplies the required 2020–2026
outright-contract futures settlements or intraday trades. Substituting a
third-party continuous future would introduce contract-roll and session-clock
ambiguity, so it is not acceptable evidence for alpha promotion.

This is a data-source stop, not a negative result for the economic hypothesis.

## Required experiment, once licensed data exists

- Use outright CME Bitcoin futures contracts, not an undocumented continuous
  series.
- Anchor on the official Friday settlement/close known at the decision time.
- Enter only after the actual CME reopen and one complete 5-minute latency bar.
- Treat holidays, daylight-saving transitions and contract rolls explicitly.
- Split the pre-2026 schedule from the 2026 24/7 schedule change.
- Compare against the already rejected BTC weekend continuation/reversal and
  weekend FX reconciliation controls.
- Freeze train/selection/eval clocks before opening outcomes; apply costs,
  funding, virtual adverse-mark exit cost and full-calendar CAGR.

## Official-source findings

- [CME DataMine](https://www.cmegroup.com/datamine.html) is the official
  historical-data product and describes historical files as purchased data.
- [CME DataMine API](https://www.cmegroup.com/datamine/datamine-api.html)
  provides programmatic access to purchased historical data.
- [CME historical market data on Google Analytics Hub](https://www.cmegroup.com/market-data/connect-data/cme-group-market-data-on-google-analytics-hub.html)
  includes market depth, settlements and time-and-sales history, but requires
  licensing and onboarding.
- [CME settlement-data access FAQ](https://www.cmegroup.com/articles/faqs/access-to-cme-group-settlement-data-faq.html)
  routes archival access through licensed CME delivery products.
- [CME continuous price series](https://www.cmegroup.com/market-data/cme-group-continuous-price-series.html)
  lists Bitcoin history from 2018-01-02 and exposes contract-switch metadata,
  but it is a constructed roll series and licensed market data.
- [CME daily settlement timing](https://www.cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/457085528/Daily+Settlement+Time+Details)
  places the Bitcoin settlement window at 14:59–15:00 Central Time.
- [CME filing 20-306](https://www.cmegroup.com/market-regulation/rule-filings/2020/7/20-306.pdf)
  defines the tiered BTC daily-settlement procedure using that one-minute
  Globex window.
- [CME cryptocurrency futures FAQ](https://www.cmegroup.com/articles/faqs/frequently-asked-questions-cryptocurrency-futures.html)
  and [CME trading hours](https://www.cmegroup.com/trading-hours.html) describe
  the current crypto schedule and maintenance windows.

## Version boundary

CME's current crypto schedule differs from the historical Sunday–Friday
schedule and moved to 24/7 trading in 2026. A weekend-gap definition therefore
has a structural break in 2026 and must not pool both eras under one clock.

## Existing control evidence

The repository's frozen Binance-only weekend control was weak rather than a
promotable alpha:

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD |
| --- | ---: | ---: | ---: | ---: |
| Fit 2020-06..2022 | +19.99% | +7.31% | 23.97% | 0.30 |
| 2023 | +10.75% | +10.75% | 13.60% | 0.79 |

Fixed 12/24/48-hour holds and the weekend FX reconciliation hypothesis also
failed. The CME branch is worth reopening only with the missing primary data;
retesting a Binance-only clock would duplicate rejected work.

## Current BTC alpha decision

Until that data exists, the strongest grounded standalone BTC candidate remains
the frozen annual expanding ExtraTrees rank-7 long policy. Its unchanged trade
schedule survived the hardened strict audit over 2023–2026H1 with 64.04%
absolute return, 15.59% full-calendar CAGR, 5.01% strict MDD, a 3.11 CAGR/MDD
ratio and 74 trades. This remains retrospective validation rather than pristine
future discovery; shadow/live-forward evidence is still required.
