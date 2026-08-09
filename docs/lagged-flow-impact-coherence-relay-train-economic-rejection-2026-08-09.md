# LFIC-12 train economic rejection — 2026-08-09

LFIC-12 is terminally rejected unchanged at the first sequential economic
stage. Test, eval, final, and the post-stage RV20 q90 decomposition were not
opened.

The 80-trade 2023H2 train clock returned `-1.590034%` at 6 bp per notional
side. Full-calendar CAGR was `-3.131584%`, strict held-bar MDD was `6.370913%`,
and CAGR/MDD was `-0.491544`. Mean gross underlying movement was only
`8.016857 bp`; the 10 bp stress return was `-4.688063%`. Weekly cluster
sign-flip p-value was `0.593694`. Calendar halves were `1.137198%` and
`-2.696567%`.

- Preregistration SHA-256: `6f407a92d68f8c771ee8cf90bfd5b9df4711f020b33f55f273ef716cf2886485`
- Source-support SHA-256: `d026f8eee0cb9945e297ea7fd304ae868b6471dbeb07a60905643986b9d5e4b5`
- Gross9 novelty SHA-256: `159e6203191a1ce7cc95b6e1077c289727c8c3f63c67fc2cefb0353fdc00722b`
- Economic evaluator SHA-256: `a77ff6bab5c6284a59f0fe3c556507217e32e8cf27f6ab7f2804de9c3164242c`
- Train result SHA-256: `4e9b11e5ced2432ab14780cb97eb4cfb3f3175dfefae242d8e914b66dd4d56e8`

An immediate rerun reproduced the train result byte-for-byte. No lag, rank,
history, onset, block, direction, hold, volatility, subset, or control repair
was attempted, and no diagnostic control may be promoted.
