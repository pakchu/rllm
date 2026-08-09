# CLBHAR-6 terminal source-support rejection

CLBHAR-6 produced `16/33/33/21` train/test/eval/final events. Event counts and
month concentration passed every stage, and train/test/eval side balance was
healthy. Final contained `4` long and `17` short events, giving minority share
`0.190476 < 0.20`; therefore the immutable source gate fails.

Two executions reproduced the same artifacts:

- clock SHA-256: `33a82212f7a3c6ead64f48415ba32a06157a714f59f12f816938b8d295b1aa6b`
- result SHA-256: `6129f9c4ede6293c08c45d71301ae529a51e3dff8341ba3a50322283c321f721`

CLBHAR-6 is rejected unchanged. Altering the q80 rank, early/late partition,
same-sign premise, side, height anchor, confirmation, embargo, or hold to add a
single final long is forbidden. Gross9, execution prices, funding, outcomes,
economics, controls, and RV20 remain unopened.
