# HVSTER-8 terminal source-support rejection

HVSTER-8 was rejected before Gross9 rows, execution prices, or post-entry
returns were opened.

The frozen 2023H1 source-only calibration produced 18/2/22/97
train/test/eval/final events. Test failed its 12-event minimum and had no long
events. Train and eval also failed the maximum-month-share gate at 0.556 and
0.864 respectively. The strong calendar instability is consistent with a
nonstationary execution-count/ticket-size scale, which the singleton deliberately
did not normalize or recalibrate after incidence became visible.

Changing to rolling ranks, recalibrating by year, relaxing count or ticket
tails, changing onset logic, or promoting a component control would alter the
frozen policy. HVSTER-8 is terminally rejected unchanged; novelty and economics
remain unopened.
