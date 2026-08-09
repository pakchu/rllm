# AVFCR-12 source rejection — 2026-08-09

AVFCR-12 is terminally rejected unchanged at source support. Event counts were
`104/217/184/94` and maximum monthly shares were all below `0.22`, but the
primary clock was persistently long-biased. Minority-side shares were `0.1442`
in train, `0.1429` in test, `0.2228` in eval, and `0.1702` in final, failing the
frozen `0.20` minimum in three stages.

- Preregistration SHA-256: `4f9d8035c2582ec444aea52dfe188bae80169155aaa5278cca90c337c1f84b33`
- Source evaluator commit: `3f0546f1`
- Source result SHA-256: `ac3d9df9973d90a85d52f584acb857968a25d220256f5be4811b5b11f76a1d31`
- Primary clock SHA-256: `ee0e0b5ee80dd4aedb16396f44cdf9f774a4dff948cbd723cd3fd126dde3e2b9`

The complete DB evaluation reproduced every artifact byte-for-byte. Gross9,
execution prices, post-entry returns, funding PnL, and RV20 were not opened.
No side balancing, asymmetric threshold, direction flip, clock subset, or
diagnostic-control promotion is permitted.
