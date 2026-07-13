# Alt derivatives crowding alpha search — 2026-07-13

## Result

Rejected as a standalone alpha. The frozen pre-2024 Top-10 produced **zero
alpha-pool qualifiers** and **zero live-grade qualifiers**. The apparently
strong 2023 relationship inverted in 2025 and remained negative in 2026 YTD.

This rejects the exact static-tail usage tested here. It does not claim that
alt funding/premium crowding is useless as a regime descriptor or as an input
to a separately validated policy.

## Independent data surface

- Binance USD-M funding and premium-index candles for ETH, SOL, BNB, XRP, ADA,
  and DOGE only.
- Funding was joined backward with a 12-hour staleness ceiling.
- Premium was exposed only after hourly `close_time`, joined backward with a
  65-minute staleness ceiling.
- All six symbols had to be available; no backward fill from future rows and no
  indefinite forward fill was allowed.
- Candidate features with fit Spearman `max |rho| >= 0.30` against BTC funding,
  BTC premium change, BTC 8-hour trend, completed daily momentum, or
  `lr_impact_72` were rejected before policy search.

Ten of 16 external features passed admission. The largest admitted correlation
was `0.2817`; the rejected funding-median features reached roughly `0.50`.

## Frozen protocol

- Feature thresholds: 2023-02-15 through 2023-06-30.
- Policy selection: 2023-07-01 through 2023-12-31.
- The Top-10 manifest was written after truncating market and source frames at
  2024-01-01.
- Replay-only windows: 2024, 2025, and 2026 through 2026-06-02.
- 138 predeclared single/pair masks, 237 eligible execution policies.
- Entry at next 5-minute open, fixed non-overlapping hold, no TP/SL.
- 0.5x leverage and 6 bp per side.
- Strict MDD uses the conservative favorable-high-water then adverse-OHLC
  ordering over the entire holding path.
- Manifest SHA-256:
  `0e07d5afba762c116fb9dbd1acf52ef69cc0abbf94beee4cac00252ebbb3d0cf`.
- Replay verifies hashes for the execution cache and all 14 auxiliary files,
  validates both external and BTC admission-feature prefixes, and reuses an
  existing manifest without mutation unless `--refresh-manifest` is explicit.

## Top-10 summary

Cells are `absolute return / strict MDD / CAGR-to-MDD / trades`.

|Rank|Policy|hold/stride|fit 2023H1|select 2023H2|test 2024|eval 2025|2026 to Jun02|combined 2024-2026|
|---:|---|---:|---:|---:|---:|---:|---:|---:|
|1|funding dispersion z2016 low 10%, long|576/24|20.39/10.12/6.38/21|11.26/4.30/5.49/30|-3.35/15.60/-0.21/67|-14.00/15.11/-0.93/54|-3.88/14.34/-0.63/27|-20.10/31.05/-0.29/148|
|2|funding dispersion z2016 low 10%, long|576/12|17.73/10.13/5.43/21|13.45/4.30/6.61/31|-8.73/17.21/-0.51/67|-15.26/16.23/-0.94/54|-0.58/12.07/-0.11/28|-23.10/33.57/-0.31/149|
|3|funding dispersion z2016 low 30%, long|576/24|20.05/10.12/6.26/41|16.14/6.82/5.07/52|-5.43/17.18/-0.32/90|-13.66/25.02/-0.55/97|-7.75/14.84/-1.19/48|-23.98/33.18/-0.32/236|
|4|funding dispersion z2016 low 30%, long|576/12|17.05/10.32/5.10/42|17.24/6.82/5.44/52|2.53/16.83/0.15/90|-15.21/25.87/-0.59/98|-5.55/14.14/-0.91/48|-16.92/31.56/-0.23/237|
|5|premium dispersion z2016 high 20%, long|576/24|15.19/11.33/4.08/45|20.19/9.36/4.71/66|31.12/20.11/1.54/124|-17.66/25.79/-0.68/129|-5.85/13.15/-1.03/52|4.31/29.23/0.06/306|
|6|premium dispersion z8640 high 10%, long|576/24|14.88/8.51/5.31/26|14.99/8.38/3.81/49|33.29/22.09/1.50/88|-13.07/19.79/-0.66/101|-11.48/17.67/-1.44/42|0.47/28.97/0.01/232|
|7|premium median 1-day change low 10%, long|576/12|12.02/9.43/3.78/50|18.41/11.15/3.57/67|52.25/12.44/4.19/143|-15.13/28.43/-0.53/143|-12.81/22.74/-1.23/58|11.61/37.20/0.12/345|
|8|premium dispersion z8640 high 10%, long|576/12|18.89/9.18/6.45/31|13.68/9.27/3.13/59|30.51/22.74/1.34/107|-16.39/22.01/-0.75/114|-9.25/18.12/-1.15/47|-0.87/32.89/-0.01/269|
|9|premium dispersion z2016 low 30%, long|576/24|12.39/10.40/3.54/53|18.29/11.70/3.38/77|25.96/21.72/1.19/144|-8.91/20.79/-0.43/142|-8.72/18.57/-1.06/62|6.38/26.49/0.10/349|
|10|funding dispersion z8640 low 10%, long|576/12|9.45/10.13/2.71/19|23.90/6.82/7.77/42|4.93/14.04/0.35/84|-2.16/17.75/-0.12/85|-6.31/11.72/-1.24/45|-2.68/17.75/-0.06/215|

## Interpretation

The feature family is genuinely low-correlation at the input level, but the
static 2023-selected directional mapping is not stable. All selected policies
were long because 2023 rewarded that mapping; 2025 and 2026 show that this was
direction/regime selection rather than durable edge. The next experiment must
therefore avoid another standalone static-tail search. A better bounded use is
an external veto/state gate on a separately fixed alpha, with thresholds and
Top-10 selection completed before replay.

## Artifacts

- `training/search_alt_derivatives_crowding_alpha.py`
- `tests/test_search_alt_derivatives_crowding_alpha.py`
- `results/alt_derivatives_crowding_top10_manifest_2026-07-13.json`
- `results/alt_derivatives_crowding_alpha_scan_2026-07-13.json`
