# LEMRR-16 train economic rejection — 2026-08-09

LEMRR-16 is terminally rejected unchanged at the first sequential economic
stage. Test, eval, final, and the post-stage RV20 q90 decomposition were not
opened.

The 9-trade 2023H2 train clock returned `-0.920513%` at 6 bp per notional
side. Full-calendar CAGR was `-1.818983%`, strict held-bar MDD was `3.048189%`,
and CAGR/MDD was `-0.596742`. Mean gross underlying movement was
`-7.547094 bp`; the 10 bp stress return was `-1.277968%`. Weekly cluster
sign-flip p-value was `0.696873`. The first calendar half lost `-2.110111%`
and the second gained `1.215241%`.

- Preregistration SHA-256: `7153527a55247b42eae6214f1505acc629eff3569df70949b3a94df475c0ba7f`
- Source-support SHA-256: `aab0cb26c40edbb15ed6296c2a349f60327ee5a396d7c4fa6d36d7126bae5ef6`
- Gross9 novelty SHA-256: `13823e398bf69767f0155cc543ae11ff8e778a4bc7a4a4119d8648b0adadb49c`
- Economic evaluator SHA-256: `7b970f1a9f0a7528f806ef8b4b89dd8e622d29ce408363545e347860df382101`
- Train result SHA-256: `662dc6d34f2376390f21825009de386db2be78b30f8a59309f44c14761d5ac4c`

An immediate rerun reproduced the train result byte-for-byte. No weekday,
window, extreme occurrence rule, midpoint condition, side, hold, volatility,
subset, or control repair was attempted, and no diagnostic control may be
promoted.
