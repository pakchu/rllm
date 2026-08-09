# EMDFR-12 terminal source-support rejection

EMDFR-12 was preregistered and its source evaluator was committed and pushed
before exact incidence was opened. The frozen evaluator produced `0/0/0/0`
train/test/eval/final events. USDMXN and USDINR supplied 919 and 793 valid
13:00-16:00 UTC sessions, respectively, but USDCNY supplied no session meeting
the immutable 165-minute and endpoint requirements. Therefore no three-pair
factor row was source-valid and every period failed its minimum-event and side-
balance gates.

Two executions reproduced identical terminal artifacts:

- source-session SHA-256: `9a0bbad9235fc3936c192b7755430dafd03f287047c10fc0c6ce37a5dd44afa4`
- clock SHA-256: `746ddfe50c71e9ba9619697f94fd8352c41907e3fb1af780812edd2c0e0b0d3a`
- result SHA-256: `82822ab27d9dc43a30a91738a1b425f8e73b1e64891209105d2226dfaf989f50`

EMDFR-12 is rejected unchanged at source support. Gross9 rows, execution
prices, funding, post-entry outcomes, economics, and RV20 remain unopened. The
source set, completeness rule, session, factor, thresholds, side, clock, and
hold cannot be repaired.
