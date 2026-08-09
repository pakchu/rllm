# HVPASR-12 economic evaluator freeze — 2026-08-09

The sequential evaluator was code-hash bound after source support and Gross9
novelty passed, but before any post-entry BTC outcome or economic metric was
opened. It fixes 0.5x quantity, exact funding, 6bp/10bp costs, full-calendar
CAGR, strict favorable-then-adverse held 5m MDD, weekly cluster sign flips, and
positive calendar halves. Stages open train → test → eval → final and stop on
the first failure. Controls are diagnostic-only; RV20 q90 remains unavailable
until all four stages pass unchanged.
