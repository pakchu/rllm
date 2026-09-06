# CLWMSR-6 source-support pass — 2026-08-13

The frozen witness-migration sponsorship relay passed every outcome-blind
source gate unchanged. Train/test/eval/final contained 24/50/32/19 accepted
events. The minimum side share was 0.3684 and the maximum calendar-month share
was 0.3333, versus required limits of 0.20 and 0.45.

The source evaluator reconstructed each six-block BIP141 witness share from
the immutable serialized-size and block-weight fields, enforced canonical
chain continuity and exact completed-minute price intervals, and applied the
frozen two-hour confirmation delay. The normalized block source contained
166,682 rows and had SHA-256
`bdacc354120c526e1672df52f67912b527f6e03bfc3bc2191f3c4ba7ba47e3aa`.

Two consecutive evaluator runs reproduced the feature panel, primary clock,
five diagnostic-control clocks, manifest, and report byte-for-byte. No
execution price, post-entry return, funding value, PnL, RV20 value, or Gross9
row was opened. CLWMSR-6 advances only to Gross9 structural novelty.
