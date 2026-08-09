# HVSAR-12 economic evaluator freeze — 2026-08-09

The strict sequential evaluator was code-hash bound after source support and
Gross9 novelty passed, but before any post-entry BTC outcome or economic metric
was opened.

It fixes 0.5x quantity through exit, exact settlement-mark funding cash, 6bp
base and 10bp stress costs per notional side, full-calendar CAGR, strict global
HWM with favorable-then-adverse held 5m envelopes, weekly cluster sign flips,
and positive calendar halves. Stages open only in train → test → eval → final
order and stop at the first failure. Controls remain diagnostic-only.

RV20 q90 remains unavailable until all four stages pass unchanged.
