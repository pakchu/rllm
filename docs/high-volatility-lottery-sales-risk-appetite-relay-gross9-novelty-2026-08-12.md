# HVLSRA-24 Gross9 novelty

Date: 2026-08-12

## Decision

The unchanged HVLSRA-24 Powerball sales clock passes all four preregistered
structural novelty limits against every authoritative Gross9 sleeve. Economic
outcomes remained closed during this comparison.

| Gross9 sleeve | exact Jaccard | matched within 6h | occupied 5m Jaccard | abs exposure correlation |
|---|---:|---:|---:|---:|
| `cand_rex_veto_7` | 0.0000 | 0.0851 | 0.1011 | 0.0619 |
| `fresh_kimchi_fx` | 0.0000 | 0.1622 | 0.0830 | 0.0587 |
| `frozen_annual_rank7` | 0.0000 | 0.0690 | 0.0487 | 0.0367 |
| `markov_transition_long` | 0.0000 | 0.1176 | 0.1080 | 0.0048 |
| `rex_taker_low_range_position` | 0.0000 | 0.0851 | 0.0766 | 0.0213 |

Limits are respectively 0.10, 0.35, 0.25, and 0.35. Every sleeve passes.

The result artifact SHA-256 is
`2d251f67e5b623cc76f4b9ef61e3133f46a57374c8556ba0a36ecf3190b29bb2`.
Two consecutive runs were byte-identical.

The next authorized step is to freeze and publish the strict economic
evaluator before opening train outcomes. Test/eval/final outcomes remain
sealed until each preceding stage passes.
