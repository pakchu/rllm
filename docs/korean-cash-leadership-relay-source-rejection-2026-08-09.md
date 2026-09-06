# KCLR-12 terminal source-support rejection

KCLR-12 was rejected before Gross9 rows, execution prices, or post-entry
returns were opened.

The frozen exact-path intersection produced only 277 complete daily sessions,
spanning 2025-10-20 through 2026-07-31.  The strict-prior 270-session rank with
180 observations therefore became available only 97 times.  Consequently the
primary clock had 0/0/0/8 train/test/eval/final events.  The final split also
failed month concentration at 0.75 despite passing its event count and side
balance gates.

Relaxing the exact 480-minute intersection, shortening rank history, changing
the source window, or altering the volatility or leadership rule would modify
the frozen candidate.  KCLR-12 is terminally rejected unchanged; Gross9
novelty and economics remain unopened, and diagnostic controls remain
non-promotable.
