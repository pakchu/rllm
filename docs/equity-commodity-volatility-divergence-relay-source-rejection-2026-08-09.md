# ECVDR-12 terminal source-support rejection

ECVDR-12 was rejected before Gross9 rows, execution prices, funding, returns,
or PnL were opened.

The frozen two-pair opposite-sign rule produced 2/12/9/9
train/test/eval/final events.  Train, eval, and final failed their minimum
event gates; train also had a 0.50 maximum month share above the 0.45 ceiling.
Side balance passed wherever events existed.

Weakening either pair confirmation, dropping the volatility gate, or changing
the Cboe clock would repair the frozen rule.  ECVDR-12 is rejected unchanged
and controls remain non-promotable.
