# CRES-1 2026 outcome-blind support

## Decision

- Support gate: **PASS**.
- 2026 post-entry strategy returns calculated: **no**.
- Historical 2023-2025 training seed is physically separate and ends before 2026.

## Base-event clock

- events: 68 (Q1 36, Q2 32);
- unique ordered continuation-reference pairs: 26;
- maximum ordered-pair share: 0.088;
- event features use only completed hourly bars;
- range risk uses only the 864 completed 5m bars strictly before each signal;
- the clock contains no return, PnL, target, label, edge, prediction, or choice column.

The long/short columns are only the canonical continuation expert reference.
The frozen evaluator must choose continuation, reversion, or flat online using
only the historical seed and earlier 2026 events whose exits are already
observable.

Clock SHA256: `62b40c2474399595acd5c48f2fecb0b8f6b0f96cfb3fce1ec63da3a1c7522088`
Source manifest hash: `d1fc86b0084727c09451b89b412c685822611a5a567b889a88f9d2e82b0a8bfb`
Historical seed SHA256: `cdcd7719b0f3c1e40bcd4610c836fa7ca3f8dd83223e36c4b8a5840db202dec9`
