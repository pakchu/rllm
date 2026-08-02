# Gross9 structural-clock bundle G9CB15 E14 terminal pytest recovery authority — 2026-08-02

Status: **A15 construction authority only.** This docs-only authority supersedes the G9CB14 successor topology only where the consumed, terminal E14 failure requires a new recovery lane. It neither reopens E14 nor grants execution authority for T14, Q15, E15, any downstream official stage, Git publication, runtime mutation, or economics work.

## 1. Immutable lineage and A15 tracked boundary

The immutable repository baseline is:

- worktree: `/tmp/rllm-alpha-orthogonal-20260718`;
- branch: `codex/gross9-structural-clock-bundle-20260731`;
- A14C commit, `HEAD`, and upstream: `bf0a0c9e0339d7da5e2fa3d167ea055828693e15`;
- intended A15 tracked path: `docs/gross9-structural-clock-bundle-g9cb15-e14-terminal-pytest-recovery-authority-2026-08-02.md`;
- required worktree mode: `0644`;
- required Git mode: `100644`.

A15 MUST be a direct child of A14C: its future commit must have exactly one parent equal to `bf0a0c9e0339d7da5e2fa3d167ea055828693e15`. The A15 commit diff against A14C MUST contain exactly one tracked change: addition of the intended A15 path above. No tracked file may be removed, renamed, or otherwise changed. In particular, A15 MUST NOT change `pyproject.toml`, `uv.lock`, source, tests, other documentation, E14 evidence, runtime state, results, or Git history.

A15 construction may add exactly one tracked path, `docs/gross9-structural-clock-bundle-g9cb15-e14-terminal-pytest-recovery-authority-2026-08-02.md`; no other tracked mutation is authorized. The intended tracked document has mode 0644 in the worktree and Git mode `100644`.

The completed A15 file SHA-256, byte size, Git blob, tree, commit, pushed identity, and every descendant identifier are learned only after the corresponding bytes or object exist. This document embeds none of those self-identities and predicts none of them. An amend, rebase, merge parent, extra tracked path, mode mismatch, force-push, force-with-lease, or parent mismatch invalidates A15.

This document intentionally embeds no whole-file hash or byte count for itself and no A15 or descendant commit, tree, or blob object identifier.

## 2. Bound source authorities and present grant

The following exact inputs are jointly authoritative for this document:

| Artifact | Bytes | Mode | SHA-256 |
| --- | ---: | ---: | --- |
| `.omx/context/g9cb15-e14-terminal-pytest-recovery-20260802.md` | 7019 | `0644` | `62755d033dd3ed1dd5d0b5dc65017762103b076b6eb59b1a908766b13ca1560c` |
| `.omx/plans/prd-g9cb15-e14-terminal-pytest-recovery.md` | 13978 | `0644` | `18fb2de0585f6b36903e95301933a390907b6a437814008e5aed1047d2cfab16` |
| `.omx/plans/test-spec-g9cb15-e14-terminal-pytest-recovery.md` | 10280 | `0644` | `a3aa711123d3b1557381a022a3b1cea1ab7e88a7f4401257f6d99a0c0aa42abb` |
| `.omx/plans/g9cb15-e14-terminal-pytest-recovery-planner-round3.md` | 5612 | `0644` | `7b5b131b47a8d443b624d3e19b3ed51f188b7ee50ac067c3a60d6a1646c66c68` |
| `.omx/plans/g9cb15-e14-terminal-pytest-recovery-architect-round3.md` | 11413 | `0644` | `7b129f3f85bfcaeed816993db5d6bc386a6b70ce7d9072f24894f0eff9b4f603` |
| `.omx/plans/g9cb15-e14-terminal-pytest-recovery-critic-round3.md` | 11294 | `0644` | `80fde3ac8a6461c14c35b5797b27be1258bc11f6ef38619f0fdb6672910a658b` |
| `.omx/plans/g9cb15-e14-terminal-pytest-recovery-consensus-handoff.json` | 3375 | `0444` | `1c855411c3fc5b8a771f86819f959f2ad2c1a277eb35f9a0fa5870e5b0bb1ad1` |
| `.omx/state/g9cb15-explicit-user-dependency-authority.json` | 1028 | `0444` | `8baa83ae2fbd2cb7a5c5e55af36a27df8e24e945cd1b5678bc3069d303c7bd5d` |
| `.omx/state/g9cb15-a15-post-review-execution-handoff.json` | 1227 | `0444` | `6d7b72bc102be86d246dc1e0739063ba46a85faddd098e19c16ede85c340bcd1` |
| `docs/gross9-structural-clock-bundle-g9cb14-successor-authority-correction-2026-08-02.md` | 147756 | `0644` | `57954d0af18d479002d6107c4657dd1a208e5c5bb5917a5c623e0e99f6d11906` |

Architect round 3 is `CLEAR`; Critic round 3 is `APPROVE / OKAY`. Those reviews validate the conditional design but do not grant dependency or execution authority. The separate same-thread user authority and reviewed execution handoff authorize construction of A15 only.

The exact dependency permission carried forward is solely:

> permit `pytest==9.1.1` in `dependency-groups.dev` and generated lock closure

The exact approved decision text, reproduced literally, is:

```text
permit pytest==9.1.1 in dependency-groups.dev and generated lock closure
```

That permission allows only the future root addition `[dependency-groups]` with `dev = ["pytest==9.1.1"]` and the resolver-required lock closure generated under Q15's gates. It grants no unrelated dependency, runtime, environment, economics, or Git-history change. It does not itself authorize T14, Q15, E15, or any later stage.

The immutable pre-Q15 tracked dependency baseline is:

| File | Bytes | Mode | SHA-256 |
| --- | ---: | ---: | --- |
| `pyproject.toml` | 900 | `0644` | `972713ffd03a621c8e3a5acf61b8aa5f7aa68d573d68415bfab34a5b68304e90` |
| `uv.lock` | 792262 | `0644` | `ff965ca88c9eb9f17efe74a6d550ab99d093b44cda2467cee6f5738fb60f770a` |

At this baseline, the project and lock contain no dependency group and no `pytest`, `iniconfig`, `pluggy`, or `tomli` package record. A15 and T14 MUST preserve these exact tracked bytes. Any drift before authorized Q15 stops for replanning.

## 3. E14 is consumed, terminal, and immutable

E14 completed one frozen offline sync child successfully and then ran one readiness child. The readiness child exited `1`; stdout was empty; stderr was exactly 120 bytes, ended in one LF, and reported:

```text
Traceback (most recent call last):
  File "<string>", line 9, in <module>
ModuleNotFoundError: No module named 'pytest'
```

The immutable manifest records `state=terminal_failure`, `authority_consumed=true`, `retry_allowed=false`, and `resume_allowed=false`. It records a `ClosureError` with message `readiness did not complete successfully`, ten read-only Git attempts, two mutable payload attempts, no post-success Git entries, no replicas, and no ledger candidate. E14 produced no tracked success ledger and no E14 commit.

### Immutable E14 capture set

Every file below is mode `0444` and MUST remain byte-identical:

| Capture | Bytes | SHA-256 |
| --- | ---: | --- |
| `.omx/captures/g9cb14/E14/bootstrap/manifest.json` | 27795 | `f39475e8a1f8bd72bdbe28b1a0dffaf5d3fbb051e6d13ab6b31fd29142190cc6` |
| `.omx/captures/g9cb14/E14/bootstrap/readiness-attempt.json` | 4033 | `d81874b14e688397638ad2cfe691c6317a1e41fc9341dea19985bb9c9c45440d` |
| `.omx/captures/g9cb14/E14/bootstrap/readiness-outcome.json` | 2023 | `7335723eab31c6cb46d362252d65c29f310cd8489533016a08c4abe9c745d8d3` |
| `.omx/captures/g9cb14/E14/bootstrap/readiness-stderr.bin` | 120 | `b54dde77ccb8e20d54aaad854c498211a2eae1746c35885279c89d21350b6b0f` |
| `.omx/captures/g9cb14/E14/bootstrap/readiness-stdout.bin` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `.omx/captures/g9cb14/E14/bootstrap/reservation.json` | 11907 | `621e42757dfbfdd1f33760268a09051402e69810b05a19f9f3cbd5d91747ea8a` |
| `.omx/captures/g9cb14/E14/bootstrap/spawn-attempt.json` | 1062 | `baa0ba18e4cfccd3dd0a03b8179e4b53571ddddb075547e332284b511c9f18a3` |
| `.omx/captures/g9cb14/E14/bootstrap/sync-outcome.json` | 973 | `cc1e90828930bbedfc584ffd5b29de43eec931993d33ac4eb0cc21b393f879c9` |
| `.omx/captures/g9cb14/E14/bootstrap/sync-stderr.bin` | 2671 | `3923538f52c9e1c50fbc4540e023ade1802af4b0d853ee7f3e0318e064b5cb7e` |
| `.omx/captures/g9cb14/E14/bootstrap/sync-stdout.bin` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

There is no `.omx/captures/g9cb14/E14/bootstrap/verification.json`. Any reference that treats that nonexistent path as capture authority is superseded and invalid.

### Preserved E14 runtime provenance

Every runtime file below is mode `0444` and MUST remain byte-identical:

| Runtime evidence | Bytes | SHA-256 |
| --- | ---: | --- |
| `.omx/runtime/g9cb14-e14/bootstrap_harness.py` | 71668 | `26d25466b963865cf2f81107d4a8f23899608c6b2c4e78896659006b1d347311` |
| `.omx/runtime/g9cb14-e14/materialization.json` | 1604 | `719504b7d236c6b1c162d2e9bbe49c7758eed109e4311623e7cda1635c5e90bd` |
| `.omx/runtime/g9cb14-e14/pre_e14_verifier.py` | 35274 | `5056e392091f3f069f015aa9bc3bdc128afc64447f1f1282fc61ac13d7ea6287` |
| `.omx/runtime/g9cb14-e14/verification.json` | 3804 | `c7b3368af9ef2bb776a8acc569c1b5d501e825fca1f0655fab0c0f0e213f5b53` |

The runtime `verification.json` is provenance only at its actual path; it is not an E14 capture substitute.

E14 MUST NOT be retried, resumed, reconstructed, reclassified as successful, or cleaned up. No E14 sync, readiness child, preflight call, post-success call, replica creation, ledger publication, or alternate environment operation may run. Preserve the existing `.venv`, the complete capture directory, the runtime directory, all raw outputs, modes, and partial terminal state exactly as found. Do not delete, overwrite, repair, install into, reuse for recovery execution, or normalize them. Any E14 byte, path, mode, or existence drift stops the recovery lane for fresh review.

E14 is terminal and immutable: no retry, no resume, no cleanup, no reuse, and no mutation of any E14 path or evidence is authorized.

## 4. Closed successor topology and external-pin gates

The only permitted successor topology is:

```text
A15 → T14 → Q15 → E15 → T13 → Q14 → P14 → C14 → D14 → H14
```

No stage may be skipped, reordered, combined, substituted, retried, resumed, or entered from a stale predecessor. Every transition is fail-closed. Each predecessor must first complete under its own authority, pass independent verification, be committed where that stage has a tracked output, be normally pushed where publication is required, have `HEAD` and upstream proven equal to the fresh pushed identity, and receive an external pin created only after that identity and its exact bytes are learned. A tracked stage cannot self-pin. A missing, stale, mismatched, self-referential, or unverified pin stops before the next stage.

| Stage | Mandatory entry gate | Required gate before successor |
| --- | --- | --- |
| A15 | Exact authorities in section 2 and exact A14C baseline | Verify the sole tracked addition and direct-parent relation; then separately authorize any commit/push, normally push, prove `HEAD`/upstream equality, independently verify, and externally pin A15 before T14 |
| T14 | Exact external A15 pin plus separate T14 authority | Complete the read-only ledger and command freeze, independently verify it, publish only under separate authority, and externally pin its exact learned identity before Q15 |
| Q15 | Exact external T14 pin plus separate Q15 authority | Independently approve the two-file semantic result and closure manifest before commit; then normally push, prove equality, and externally pin the exact Q15 identity and manifest before E15 |
| E15 | Exact external Q15 identity and manifest pin plus separate E15 authority | Independently verify the isolated one-shot evidence, publish only under separate authority, and externally pin E15 before T13 |
| T13 | Exact external E15 pin and the existing T13 authority | Complete once, independently verify, normally publish as required, and externally pin fresh T13 before Q14 |
| Q14 | Exact external T13 pin and the existing Q14 authority | Complete once, independently verify, normally publish as required, and externally pin fresh Q14 before P14 |
| P14 | Exact external Q14 pin and the existing P14 authority | Complete once, independently verify, normally publish as required, and externally pin fresh P14 before C14 |
| C14 | Exact external P14 pin and the existing C14 authority | Complete once, independently verify, normally publish as required, and externally pin fresh C14 before D14 |
| D14 | Exact external C14 pin and the existing D14 authority | Complete once, independently verify, normally publish as required, and externally pin fresh D14 before H14 |
| H14 | Exact external D14 pin and the existing H14 authority | Complete once, normally push, prove `HEAD`/upstream equality, independently verify against that same fresh identity, externally pin H14, and STOP |

This topology records prerequisites; it is not downstream execution authority. Each stage still requires its separate stage-specific handoff or authority.

## 5. T14 read-only authority freeze

T14 is a read-only authority-building stage. Its learning and verification MUST NOT mutate tracked files, Git state, dependencies, either environment, E14 captures/runtime, caches, results, or any other protected input. Publication of a T14 ledger is permitted only by a later separate authority that names its exact output boundary. Before Q15, the immutable T14 ledger MUST freeze every item below with concrete, independently verified values learned through read-only inspection.

The externally pinned immutable ledger MUST bind the ordered (Name, version) distribution list, canonical bytes, count, SHA-256, command, interpreter, environment, stdout, stderr, non-mutation proof, resolver argv, uv executable, system Python, offline rule, process budget, `pyproject.toml` and `uv.lock` boundary, and terminal no retry and no resume semantics.

**5.1 Authoritative baseline inventory**

T14 MUST independently recompute the current `.venv` distribution inventory and record:

1. the exact ordered `(distribution Name, version)` list;
2. canonical compact UTF-8 JSON bytes using sorted keys, `ensure_ascii=false`, `allow_nan=false`, comma/colon separators, and no final LF;
3. canonical byte size, distribution count, and SHA-256;
4. exact recomputation argv, canonical argv bytes and SHA-256, interpreter path and realpath, interpreter version-output bytes and hash, cwd, clean environment map and hash, accepted exit, process count, and stdout/stderr bytes, sizes, hashes, and LF facts; and
5. before/after no-follow facts proving that `.venv`, every E14 capture, and every E14 runtime file were not mutated.

The required recomputation result is exactly `104` distributions and SHA-256 `d0871fdf385acb5263e4219551c64519f6b51d38dad0d84e0a82ba5808833a36`. This value is presently a reproducible contextual observation only: E14 readiness failed before emitting an inventory, and its `observed` value is null. It becomes Q15 baseline authority only through a complete, independently verified, externally pinned T14 ledger. Any other count or hash, missing list, provenance gap, or non-mutation gap stops before Q15 for replanning.

**5.2 Complete pre-Q15 command contract**

T14 MUST also freeze:

- the exact resolver argv array, its canonical bytes, byte size, SHA-256, accepted exit, and one-attempt rule;
- the uv executable path and realpath, exact version-output bytes/size/hash, and executable bytes/size/mode/SHA-256;
- the system Python path and realpath, exact version-output bytes/size/hash, and executable bytes/size/mode/SHA-256;
- exact cwd and the complete clean environment map with canonical bytes/size/SHA-256;
- an explicit network-denied and offline-only rule;
- exact cache roots and the complete read-only resolver input inventory, including each required path, type, mode, size, and SHA-256;
- exact ordered process budgets separated into read-only and mutable classes, with every call site, integer maximum, ordering constraint, and accepted exit; no unspecified child or process-creation surface is allowed;
- T14 freezes Q15's dependency-edit and resolver writable paths as exactly `{pyproject.toml, uv.lock}`; those two paths remain the only writable dependency surfaces, and every other path remains read-only during Q15 dependency-edit and resolver execution. This phase-scoped boundary does not prohibit the separately authorized post-approval sole tracked manifest publication path `docs/gross9-structural-clock-bundle-g9cb15-q15-closure-manifest.json`; and
- terminal semantics for spawn ambiguity, nonzero or unaccepted exit, timeout, interruption, budget excess, unexpected child, path drift, byte drift, cache/input drift, network attempt, and tool, Python, cwd, or environment mismatch: stop with no retry, resume, commit, push, E15, or inferred repair.

This A15 document intentionally supplies none of those future learned values. T14 is incomplete if any field is absent or ambiguous. Q15 is forbidden until an independent verifier approves the complete T14 ledger and a separate external pin binds its exact identity.

## 6. Q15 exact two-file dependency mutation and terminal semantic verification

During Q15 dependency-edit and resolver execution, exactly two tracked dependency files are writable: `pyproject.toml` and `uv.lock`. Those two paths remain the only writable dependency surfaces during that phase; every other path is read-only. The separately authorized post-approval sole tracked publication artifact is `docs/gross9-structural-clock-bundle-g9cb15-q15-closure-manifest.json`; it is not a dependency-edit path or a resolver writable path.

**6.1 Exact root addition**

The only permitted project semantic change is this exact new root table and sole requirement:

```toml
[dependency-groups]
dev = ["pytest==9.1.1"]
```

The equivalent exact root-edit semantic is `dependency-groups.dev = ["pytest==9.1.1"]`.

Existing project metadata, runtime dependencies, source configuration, tool configuration, and every unrelated TOML semantic MUST remain unchanged. No second dependency group or dev requirement is permitted.

**6.2 One exact offline resolver**

After the exact project edit, Q15 MUST invoke the one exact T14-pinned resolver argv once, with the exact pinned uv executable, system Python, cwd, clean environment, offline/network-denied policy, cache inputs, accepted exit, process ordering, and budgets. Only `pyproject.toml` and `uv.lock` are writable dependency files during dependency-edit and resolver execution. No dry run, wrapper reconstruction, command suffix, config/index lookup, network fallback, alternative executable, ambient install, or caller-selected variation is allowed. No second resolver exists. No alternate path exists.

**6.3 Independent semantic lock verifier**

Before any Q15 commit or push, an independent verifier MUST record exact pre/post bytes, sizes, modes, and SHA-256 for both files and decide acceptance from parsed canonical semantic projections, not a line diff or whole-file equality test. It MUST:

1. parse TOML; normalize maps by key and package collections by stable package identity; and serialize compact sorted-key UTF-8 JSON with no final LF;
2. permit the existing virtual root package `rllm` only the resolver-equivalent dev-group metadata/edge `dev → pytest==9.1.1`, while requiring every preexisting root field and runtime edge to remain equal;
3. require every preexisting third-party package to remain canonically equal in normalized name, version, source, dependency edges, markers, sdist and wheel URLs, artifact hashes, and every other resolution semantic;
4. permit new third-party records only when the resolver proves they are required for the missing bounded family `{pytest==9.1.1, iniconfig, pluggy, tomli}`; require `pytest` exactly `9.1.1`; learn rather than guess any required `iniconfig`, `pluggy`, or `tomli` versions and artifact identities; and leave already-locked shared requirements unchanged rather than duplicating them;
5. require global lock schema, version, options, and resolution markers to remain semantically unchanged except where the exact root dev edge necessarily projects;
6. record formatting- or ordering-only byte changes separately without treating them as semantic proof or failure; and
7. reject any removal, unrelated addition or upgrade, changed third-party source/edge/marker/URL/hash, out-of-family record, ambiguous normalization, unexplained resolver output, or unrelated root change.

On acceptance:

```text
The independent verifier MUST create the canonical Q15 closure manifest at `docs/gross9-structural-clock-bundle-g9cb15-q15-closure-manifest.json`, binding the exact pre/post project and lock facts, canonical projections and semantic delta, exact T14 identity, resolver/uv/Python/cwd/environment/cache/process identities, verifier identity, and the exact expected E15 distribution list, count, and SHA-256.
```

The creation and publication step runs under separately granted stage-specific Q15 publication/evidence authority. A15 freezes the manifest path only and grants no creation or publication authority. The manifest is publication/evidence output, not a dependency-edit path or a resolver writable path. Its path is separate from, unequal to, and non-substitutable with `pyproject.toml` and `uv.lock`.

The tracked A15 authority path `docs/gross9-structural-clock-bundle-g9cb15-e14-terminal-pytest-recovery-authority-2026-08-02.md` and the tracked Q15 manifest path `docs/gross9-structural-clock-bundle-g9cb15-q15-closure-manifest.json` are separate, unequal, and non-substitutable. A15 grants no T14/Q15/E15 execution, manifest creation/publication, commit, push, external-pin, or Git-history authority.

The exact Q15 commit records exactly three tracked entries:

1. `M pyproject.toml`
2. `M uv.lock`
3. `A docs/gross9-structural-clock-bundle-g9cb15-q15-closure-manifest.json` mode `100644`

Nothing else is part of the Q15 commit diff. Rename, delete, type change, mode-only substitution, and any fourth entry are rejected. The same Q15 commit is normally pushed, and `HEAD` and upstream equality are proven. Before E15, an external pin binds the exact Q15 commit plus the manifest path, bytes, byte size, and SHA-256.

No retry exists. No alternate path exists. No second resolver exists. The static verifier checks candidate text only and remains static-only and read-only; it creates or modifies no runtime verifier file. Future Q15 runtime behavior remains normative text and static checks only in this round.

Any Q15 mismatch, ambiguity, spawn failure, unaccepted exit, timeout, interruption, budget excess, path drift, semantic drift, verifier rejection, or publication mismatch is terminal for that attempt: preserve the evidence, stop before commit when uncommitted, do not retry or resume, do not run a second resolver, and do not enter E15. A new attempt requires replanning and fresh authority.

Q15 has no retry.

## 7. E15 isolated one-shot bootstrap

E15 may begin only from the exact externally pinned Q15 commit and closure manifest. It MUST:

- create and use only the distinct environment `.venv-g9cb15`;
- consume only the exact Q15-pinned `pyproject.toml`, `uv.lock`, resolver closure, toolchain, cache inputs, and expected inventory;
- preregister and obey an exact one-shot process budget and command contract before any mutable action;
- exclude ambient, user-site, system-site, temporary-environment, and E14-path reuse;
- never execute from, install into, repair, delete, rename, or otherwise touch the preserved E14 `.venv`, captures, or runtime evidence;
- validate import origins and exact equality to the Q15-manifest distribution list, count, and SHA-256; and
- prove E14 non-mutation across the complete E15 attempt.

E15 is one-shot. Any spawn ambiguity, nonzero or unaccepted exit, timeout, interruption, budget excess, origin mismatch, inventory mismatch, unexpected write, ambient/E14 leakage, or evidence mismatch is terminal: preserve evidence, do not retry or resume, do not run a second sync or readiness child, do not infer success, and do not enter T13. A successful E15 still requires independent verification and an external pin before T13.

The E15 boundary is no E14 touch and no retry: `.venv-g9cb15` alone may consume the externally pinned Q15 distribution list, count, and hash.

## 8. Downstream official-stage prohibitions

`T13→Q14→P14→C14→D14→H14` remains the existing official lane. This document changes none of those stages' command arrays, code, tests, write boundaries, role boundaries, process budgets, or acceptance rules. It only replaces their predecessor gate with the fresh, externally pinned E15 chain defined above.

- T13, Q14, P14, C14, D14, and H14 are each one-shot and no-rerun.
- Each stage accepts only the fresh externally pinned immediate predecessor listed in section 4; no ancestor, contextual observation, local-only object, stale upstream, or unpinned artifact may substitute.
- No failure may be retried, resumed, reconstructed as a new invocation, reclassified as success, or bypassed. A terminal failure closes every downstream gate pending fresh planning and authority.
- No official G12 or G13 stage, the consumed C13 invocation, E14 operation, or goal-steering operation may be rerun.
- The existing Q14 team boundary remains unchanged: its team must reach terminal shutdown before leader-only Git work, official commands, steering, push, or stage transition.
- No ambient install, temporary environment, E14 reuse, ad hoc dependency edit, guessed command, guessed closure, network fallback, or unpinned repair may satisfy an official-stage gate.
- No candidate-specific source-support, novelty, alpha discovery, evaluation, economics, or post-H14 continuation may begin during or between these stages.

Any absent or mismatched independent verification, normal push, `HEAD`/upstream equality proof, or external pin at T13, Q14, P14, C14, D14, or H14 stops before its successor. No stage or tracked artifact may contain or predict its own future blob, tree, commit, pushed identity, or external pin.

## 9. Mandatory post-H14 stop and volatility-alpha gate

After H14, STOP. Candidate-specific alpha discovery and all economics remain forbidden until a fresh H14 commit has been normally pushed, `HEAD` and upstream are proven equal to that fresh pushed identity, a fresh independent H14 verification succeeds against the same identity, and an external pin binds that verified identity. Existing H14 command arrays or prior H14 evidence cannot satisfy this gate.

Only after that gate may a separate candidate-specific authority resume work. It MUST be preregistered before candidate inspection, enforce causal availability, and target a high-volatility regime analogous to July 2026 high-volatility conditions without using July 2026 outcomes, candidate outputs, or future observations to select or tune regime membership.

For every decision timestamp `t`, under the BTC calendar-day convention, define:

```text
RV20(t)=sqrt(365*mean(r_d^2))
```

The exact preregistered BTC reference return series uses the 20 calendar-day returns dated `t-20` through `t-1`. The regime is active only when `RV20(t)` is at or above a rolling 90th-percentile threshold computed from the preceding 756 available `RV20` feature observations whose timestamps are strictly before `t`. Before candidate inspection, freeze the BTC reference series, return convention, UTC day boundary, calendar-day alignment, annualization, 20-return window, 756-observation window, quantile interpolation, warm-up, tie, and missing-value rules.

The preregistered evaluation MUST report this regime as a dedicated volatility stress slice while separately passing strict full-calendar CAGR, strict full-calendar MDD, an identical-gross comparison at identical gross exposure, and low Gross9 overlap. It MUST preregister a trivial persistent-long-vol comparator at identical gross and explicitly decompose candidate returns, on both the strict full calendar and the stress slice, into the component attributable to that comparator and residual candidate-specific alpha. Persistent long-vol beta cannot satisfy the candidate-specific claim. These aggregate constraints remain unchanged.

In verifier terminology, the post-H14 gate requires a fresh pushed and independently verified H14 identity; causal RV20 annualized by 365; Q90 from 756 strictly preceding available feature observations; strict full-calendar CAGR/MDD; same gross; low Gross9 overlap; and persistent-long-vol decomposition into comparator-attributable return and residual candidate-specific alpha.

## 10. Non-self-reference, fail-closed stop, and no broader authority

This document is complete only as the single A15 tracked addition described in section 1. It does not authorize its own commit, push, or external pin; those actions require separate Git authority. It does not authorize T14 publication or execution, either Q15 mutation, E15 creation, an official-stage invocation, candidate inspection, or economics work. The user dependency permission remains bounded to the exact future Q15 root addition and resolver-required lock closure and cannot be expanded by interpretation.

No future resolver argv, tool version, Python version, package version other than `pytest==9.1.1`, closure identity, inventory, file hash, byte size, mode, command budget, object identifier, or external pin may be guessed. Every such value must be learned at its assigned stage, independently verified, and externally pinned before use. Any missing authority, source-pin mismatch, lineage mismatch, unexpected mutation, evidence drift, incomplete ledger, semantic ambiguity, failed verification, publication mismatch, or absent external pin stops fail-closed before the next stage.

This authority is complete at the final line; no wrapper prose, unreviewed suffix, or trailing blank line is permitted.
