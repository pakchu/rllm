# HVCASFR-8 Gross9 rejection — 2026-08-12

HVCASFR-8 passed its preregistered source-support gate but failed the next frozen
Gross9 structural-novelty gate before any execution price, funding value,
post-entry return, or PnL was opened.

The exact-entry, occupied-bar, and absolute signed-exposure checks passed against
all five Gross9 sleeves. The one-to-one six-hour proximity check failed against:

- `fresh_kimchi_fx`: `0.3783783784 > 0.35`
- `markov_transition_long`: `0.3529411765 > 0.35`

The tight passing comparison was `frozen_annual_rank7` at `0.3448275862`.
Rerunning the frozen evaluator reproduced the result JSON byte-for-byte with
SHA-256 `c0e1ea68add81467d7450a38aab14de214244cf24e3b67ea7943fb1c1e8d0895`.

Per the preregistered stopping rule, HVCASFR-8 is rejected unchanged. Its fixed
clock, half-block definition, minimum four symbol flips, side, and eight-hour
hold cannot be retimed, filtered, or repaired, and economic evaluation is not
authorized.
