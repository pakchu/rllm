# TBASR-24 — private semantic support pass

After the v3 synthetic artifact was committed and pinned, the private semantic
stage revalidated all 13,610 source pages and 6,791,328 rows against the frozen
raw-stream and page-container hashes. It classified 67,497 selected messages
with Gemma2-2B. No BTC market, funding, execution-path, outcome, return, or PnL
row was loaded.

## Result

All 17 preregistered support checks passed:

- clear consensus events: 2,718 of 5,417 attention events;
- train (2020H2–2021): 1,728, including 527 in 2020H2 and 1,201 in 2021;
- calendar-2022 test: 990, split 498 in H1 and 492 in H2;
- every quarter from 2020Q3 through 2022Q4: at least 207 clear events;
- active weeks: 131 overall, 79 train, and 53 test;
- clear label shares overall: 40.69% bullish and 59.31% bearish;
- maximum quarter concentration: 12.91%;
- exact model-output parse success: 99.905%; and
- meta-instruction guard: zero of 67,497 messages, below the frozen 1% cap.

The public semantic clock contains only event times, aggregate participant and
message counts, crowd label, and the deterministic contrarian side. It contains
no message, rendered prompt, username, participant hash, job ID, or
participant-level label. The private 22 MB resumable label journal remains
ignored and uncommitted.

Hashes:

- support result file:
  `2b89f710d59a5c0708d400541defb43d5e292f6d9bdedbe66d6bdcf614d09e94`;
- support result:
  `5996b7d7497d6bf5e96343f7ceca766363d58aa34280aea0fdb7b8653a8b1725`;
- semantic clock file:
  `af8687564614ec5a1cbd7a1438c908f687af7bd99ceede9539016e5c1b111bd4`;
- semantic clock manifest:
  `fdcd9c7c376b18df2799acf24af04a421ca679e27009e6a539888defc7438aa8`.

This is a source-incidence and directional-breadth pass, not evidence of alpha
or profitability. It permits only the next preregistration: freeze completed
pre-entry price displacement/alignment, entry/exit rules, costs, funding, and
strict held-path MDD before opening any BTC market row.
