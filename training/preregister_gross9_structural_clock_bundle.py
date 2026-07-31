"""Create the metadata-only G9CB-7 structural-clock preregistration.

This module deliberately has a stdlib-only import surface.  It authenticates
opaque bytes, Git metadata, permitted JSON metadata, static Python imports,
and the installed-distribution inventory.  It never imports a repository
runtime and never decodes an anchor, source, model, or history value.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence
import zlib


PROTOCOL_VERSION = (
    "gross9_structural_clock_bundle_g9cb7_preregistration_v1"
)
HISTORICAL_PROTOCOL_VERSION = (
    "gross9_structural_clock_bundle_preregistration_v1"
)
FAILED_V2_PROTOCOL_VERSION = (
    "gross9_structural_clock_bundle_preregistration_v2"
)
IDENTITY = "G9CB-7"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPOSITORY_ROOT / "results"
PREREGISTRATION_SOURCE = Path(
    "training/preregister_gross9_structural_clock_bundle.py"
)
PREREGISTRATION_TEST = Path(
    "tests/test_preregister_gross9_structural_clock_bundle.py"
)
ARTIFACT_TEST = Path(
    "tests/test_gross9_structural_clock_bundle_preregistration_artifact.py"
)
BUILDER_SOURCE = Path("training/build_gross9_structural_clock_bundle.py")
BUILDER_TEST = Path("tests/test_build_gross9_structural_clock_bundle.py")
AUTHORITY_DECISION_PATH = Path(
    "docs/"
    "gross9-structural-clock-bundle-g9cb7-successor-authority-decision-"
    "2026-07-31.md"
)
ACTIVE_AUTHORITY_DECISION_PATH = AUTHORITY_DECISION_PATH
G9CB6_AUTHORITY_DECISION_PATH = Path(
    "docs/"
    "gross9-structural-clock-bundle-g9cb6-successor-authority-decision-"
    "2026-07-31.md"
)
G9CB5_AUTHORITY_DECISION_PATH = Path(
    "docs/"
    "gross9-structural-clock-bundle-g9cb5-successor-authority-decision-"
    "2026-07-31.md"
)
G9CB4_AUTHORITY_DECISION_PATH = Path(
    "docs/"
    "gross9-structural-clock-bundle-g9cb4-successor-authority-decision-"
    "2026-07-31.md"
)
G9CB3_AUTHORITY_DECISION_PATH = Path(
    "docs/"
    "gross9-structural-clock-bundle-g9cb3-successor-authority-decision-"
    "2026-07-31.md"
)
G9CB2_AUTHORITY_DECISION_PATH = Path(
    "docs/"
    "gross9-structural-clock-bundle-successor-authority-decision-"
    "2026-07-31.md"
)
PREDECESSOR_AUTHORITY_DECISION_PATH = Path(
    "docs/gross9-structural-clock-bundle-authority-decision-2026-07-31.md"
)
RANK7_AUTHORITY_AMENDMENT_PATH = Path(
    "docs/gross9-structural-clock-bundle-rank7-authority-amendment-2026-07-31.md"
)
RUNTIME_ISOLATION_AMENDMENT_PATH = Path(
    "docs/gross9-structural-clock-bundle-runtime-isolation-amendment-2026-07-31.md"
)
PREREGISTRATION_CORRECTION_AMENDMENT_PATH = Path(
    "docs/"
    "gross9-structural-clock-bundle-preregistration-correction-amendment-"
    "2026-07-31.md"
)
PRIMITIVES_SOURCE = Path("training/gross9_structural_clock_primitives.py")
PRIMITIVES_TEST = Path("tests/test_gross9_structural_clock_primitives.py")
RANK7_FACADE_SOURCE = Path("execution/gross9_rank7_clock_runtime.py")
RANK7_FACADE_TEST = Path("tests/test_gross9_rank7_clock_runtime.py")
HISTORICAL_PREREGISTRATION_PATH = Path(
    "results/gross9_structural_clock_bundle_preregistration_2026-07-31.json"
)
FAILED_V2_PREREGISTRATION_PATH = Path(
    "results/"
    "gross9_structural_clock_bundle_preregistration_v2_2026-07-31.json"
)
PREREGISTRATION_PATH = Path(
    "results/"
    "gross9_structural_clock_bundle_g9cb7_preregistration_2026-07-31.json"
)
ACCESS_CLAIM_PATH = Path(
    "results/"
    "gross9_structural_clock_bundle_g9cb7_access_claim_2026-07-31.json"
)
ATTEMPT_SENTINEL_PATH = Path(
    "results/"
    "gross9_structural_clock_bundle_g9cb7_attempt_consumed_2026-07-31.json"
)
BUNDLE_PATH = Path(
    "results/gross9_structural_clock_bundle_g9cb7_2026-07-31.csv.gz"
)
FINAL_MANIFEST_PATH = Path(
    "results/"
    "gross9_structural_clock_bundle_g9cb7_manifest_2026-07-31.json"
)
WORKER_CAPABILITY_CONSUMPTION_LEDGER_PATHS = (
    Path(
        "results/"
        "gross9_structural_clock_bundle_g9cb7_worker_capability_consumed_pass1_"
        "2026-07-31.json"
    ),
    Path(
        "results/"
        "gross9_structural_clock_bundle_g9cb7_worker_capability_consumed_pass2_"
        "2026-07-31.json"
    ),
)

AUTHORITY_DECISION_COMMIT = "ad5a7e5f6d3edeac0928c1ef93fd0fd2209a9279"
AUTHORITY_DECISION_SHA256 = (
    "faf5b5f427882c97e2437fe32bb1a0b280f87fe780393e4016911a02ce6c2624"
)
AUTHORITY_DECISION_GIT_BLOB = "53860caaefbeb964a46d5668660793f98a929ed4"
G9CB6_AUTHORITY_DECISION_COMMIT = "2695ee61fbb9b5e053dbb9da597ebe2729aad361"
G9CB6_AUTHORITY_DECISION_SHA256 = (
    "b64f9480741eeb4f69ac86736589fbcf8fb75565c436d76316b73f5e076acfca"
)
G9CB6_AUTHORITY_DECISION_GIT_BLOB = (
    "eb743d9f8ecd878b83f8f8873697c58cccef9f1b"
)
G9CB6_PROTOCOL_IMPLEMENTATION_COMMIT = (
    "86c7076e415ed667560bfe41c942ab4a00c75a4d"
)
G9CB6_PREREGISTRATION_SHA256 = (
    "5a04a8616a7c8416e67f349f8fd4a846fda87786c0f54fb1415dcf924bb17374"
)
G9CB6_PREREGISTRATION_GIT_BLOB = (
    "af809793347a647632f07ab1d74f5fbeabaac122"
)
G9CB6_BUILDER_SHA256 = (
    "4fe465368fa074536e85e2e0b54e4ff4800b4cd8a034510015bef78a66d9db93"
)
G9CB6_BUILDER_GIT_BLOB = "09cb9757a230c349cd7b7df9f7ce4a20cfa9b30c"
G9CB5_AUTHORITY_DECISION_COMMIT = "1ca718d9dab1077b041e753f3b011fbf5b23f047"
G9CB5_AUTHORITY_DECISION_SHA256 = (
    "d0b2e14417b4cd46213708597220067c2195d22308da9eb95921bcb59da27385"
)
G9CB5_AUTHORITY_DECISION_GIT_BLOB = (
    "e0bb4b1d26a67c4baf681d8a48e988307c92f9f5"
)
G9CB5_PROTOCOL_IMPLEMENTATION_COMMIT = (
    "02c3c83a5253684057f44f51ee96bcb089b40b2f"
)
G9CB5_PREREGISTRATION_SHA256 = (
    "2c989f97f8046154d8a479d541c1d4b3cb8f70ab1394d2e610fc203207854e1f"
)
G9CB5_PREREGISTRATION_GIT_BLOB = (
    "1f74ddbb8fa019884f674466a29cf0bfb5ec9af1"
)
G9CB5_BUILDER_SHA256 = (
    "d7edaa3277b581c675f81b2364421d862c1897e89cc149335d912753bb182802"
)
G9CB5_BUILDER_GIT_BLOB = "8af92fbdf7200b2e67275d9b41d3e40ebc1449a8"
G9CB4_AUTHORITY_DECISION_COMMIT = "1156e2fd80957d5ef0a6027a09e08ff59349a80d"
G9CB4_AUTHORITY_DECISION_SHA256 = (
    "9199955f62abbb99c8665a5eeee6a32cf9605ba637e2b034d929b1ac91ace626"
)
G9CB4_AUTHORITY_DECISION_GIT_BLOB = (
    "2610246e4d9fb89d775fe7d8d1998282d23e5961"
)
G9CB4_PROTOCOL_IMPLEMENTATION_COMMIT = (
    "750c837a10c4d4ac39fbc8f6097465c82b6dc3ec"
)
G9CB4_PREREGISTRATION_SEAL_COMMIT = (
    "01de73258902d754905319b906345c865a016558"
)
G9CB3_AUTHORITY_DECISION_COMMIT = "a97576c050cf7cdf08738ddb755e63cc92484428"
G9CB3_AUTHORITY_DECISION_SHA256 = (
    "1df555c5149bfe269d2cc2c87375d54032809f13ac36f4e92b5ba00dd6e87cc7"
)
G9CB3_AUTHORITY_DECISION_GIT_BLOB = (
    "43d68f6b7407c19b3b52ef8b7bb7010797dbf3b3"
)
G9CB3_PROTOCOL_IMPLEMENTATION_COMMIT = (
    "211752bf2529b37376b5fecdc190e0c22a1195ff"
)
G9CB3_PREREGISTRATION_SEAL_COMMIT = (
    "7f4985b0685fdb6cbbde891485f08305f076e205"
)
G9CB3_CLAIM_COMMIT = "4365522eba5b01129c6aa12a5e06d314b611c840"
G9CB3_TERMINAL_EVIDENCE_COMMIT = (
    "04b9e53272ab58537235ad290551607dd071ee17"
)
G9CB2_AUTHORITY_DECISION_COMMIT = "0a2847c8589908def4243890727c3640f806e109"
G9CB2_AUTHORITY_DECISION_SHA256 = (
    "b80ad199c803623c4de289d32beea913b3e1c38541e1c4f035ce3d33fa049410"
)
G9CB2_AUTHORITY_DECISION_GIT_BLOB = (
    "4904a47fa75cb455cd3c5007373e149267b7f198"
)
G9CB2_PROTOCOL_IMPLEMENTATION_COMMIT = (
    "f48634af22dcad84ffde885fa970635d133cc126"
)
G9CB2_PREREGISTRATION_SEAL_COMMIT = (
    "04550a47686ee039f82dfdb412d3c3eec4b5d6a1"
)
G9CB2_CLAIM_COMMIT = "731f093eb963b9e7213778ed4f259ee5466cd893"
G9CB2_TERMINAL_EVIDENCE_COMMIT = (
    "edad4de5cf5524c4646c64b0581e47c914e31425"
)
RANK7_AUTHORITY_AMENDMENT_COMMIT = (
    "f1ae4e68bfb0d0b861cd9979762f87e51a55f69d"
)
RANK7_AUTHORITY_AMENDMENT_SHA256 = (
    "a99b1a2b3d738ecc1cea8595eed2d88759c9b5fa7faf751a53b643fcc1a808cb"
)
RANK7_AUTHORITY_AMENDMENT_GIT_BLOB = (
    "0c7781ebe25178c592bb526ac51ee00c5ba840e2"
)
RUNTIME_ISOLATION_AMENDMENT_COMMIT = (
    "2550e0b8ee348b4217744a73d9781dba1e1e91a3"
)
RUNTIME_ISOLATION_AMENDMENT_SHA256 = (
    "354ae3870dd6dedf738b38bdd266d85b24389fe5de10d1fa0b3dbdde18d1c2de"
)
RUNTIME_ISOLATION_AMENDMENT_GIT_BLOB = (
    "c2da15ff249e46a8fac2040d67f531a683b7fd7e"
)
PREREGISTRATION_CORRECTION_AMENDMENT_COMMIT = (
    "eee3383c9b2f88f4ea28f5bfe3a5ff6a650cec0f"
)
PREREGISTRATION_CORRECTION_AMENDMENT_SHA256 = (
    "b79151c3378960017ddb30b7c1040f3027be538acad00776315380c267c6acaf"
)
PREREGISTRATION_CORRECTION_AMENDMENT_GIT_BLOB = (
    "94c0f3e13680f9e0ebbdb07ae7646b9505891e46"
)
HISTORICAL_PREREGISTRATION_SHA256 = (
    "3580a3663b54509d004dc2edac0f18ff9c79cb80b199e8de5e9b1a9feb98d472"
)
HISTORICAL_PREREGISTRATION_GIT_BLOB = (
    "61992d68beff0da255b002776d0efdb4ef96ab93"
)
HISTORICAL_PREREGISTRATION_SEAL_COMMIT = (
    "3810a3b7e24b83591866f2ccf9b63167795718c5"
)
HISTORICAL_PROTOCOL_PARENT_COMMIT = (
    "05437c3d8f2a9c556fde4e950a815b9901f7fc98"
)
HISTORICAL_PREREGISTRATION_MANIFEST_HASH = (
    "5ddf4c5c0aef42e1fb24defa78fccbd4142c8274bc22fd0a7d7e97fa9e8bb9bb"
)
FAILED_V2_PREREGISTRATION_SHA256 = (
    "5e6fe5e23f78103e5e4c6a288bb12df5f6aaa4e00028a211a175221a58b48e84"
)
FAILED_V2_PREREGISTRATION_GIT_BLOB = (
    "6bf7c4fd62818c639b11da943f25353946d141b6"
)
FAILED_V2_PREREGISTRATION_SEAL_COMMIT = (
    "c5c5120cb5af931294524d4833f44440f8949327"
)
FAILED_V2_PROTOCOL_IMPLEMENTATION_COMMIT = (
    "d4ebec8f151fc5db6d318734ca0b6a79afaad1e1"
)
FAILED_V2_PREREGISTRATION_MANIFEST_HASH = (
    "e83d2bec1300c34401931c2b45c6c0b8715f4237eba0ae01811c665718b11a54"
)
DIRECT_AUTHORITY_VERIFICATION_COMMIT = "91b41254319686f8b64bba797708f8e637aeddd3"
EXPECTED_BRANCH = "codex/gross9-structural-clock-bundle-20260731"
UNSEALED_PROTOCOL_IMPLEMENTATION_COMMIT = "0" * 40
_ABSOLUTE_BINDING_ALLOWLIST = frozenset(
    {"/tmp/btcusdt_open_interest_5m_2020_2026.csv"}
)
Q7_PREREGISTRATION_PUBLICATION = "Q7_PREREGISTRATION_PUBLICATION"

PROTOCOL_PATHS = (
    ACTIVE_AUTHORITY_DECISION_PATH,
    G9CB6_AUTHORITY_DECISION_PATH,
    G9CB5_AUTHORITY_DECISION_PATH,
    G9CB4_AUTHORITY_DECISION_PATH,
    G9CB3_AUTHORITY_DECISION_PATH,
    G9CB2_AUTHORITY_DECISION_PATH,
    PREDECESSOR_AUTHORITY_DECISION_PATH,
    RANK7_AUTHORITY_AMENDMENT_PATH,
    RUNTIME_ISOLATION_AMENDMENT_PATH,
    PREREGISTRATION_CORRECTION_AMENDMENT_PATH,
    PREREGISTRATION_SOURCE,
    PREREGISTRATION_TEST,
    ARTIFACT_TEST,
    BUILDER_SOURCE,
    BUILDER_TEST,
    PRIMITIVES_SOURCE,
    PRIMITIVES_TEST,
    RANK7_FACADE_SOURCE,
    RANK7_FACADE_TEST,
)
RUNTIME_IMPORT_ROOTS = (
    RANK7_FACADE_SOURCE,
    PRIMITIVES_SOURCE,
)
G9CB1_CORRECTION_AUTHORITY_DIFF = (
    "A\tdocs/"
    "gross9-structural-clock-bundle-preregistration-correction-amendment-"
    "2026-07-31.md",
)
G9CB1_CORRECTION_PROTOCOL_DIFF = (
    "M\ttests/test_build_gross9_structural_clock_bundle.py",
    "M\ttests/test_gross9_structural_clock_bundle_preregistration_artifact.py",
    "M\ttests/test_preregister_gross9_structural_clock_bundle.py",
    "M\ttraining/build_gross9_structural_clock_bundle.py",
    "M\ttraining/preregister_gross9_structural_clock_bundle.py",
)
FAILED_V2_PREREGISTRATION_DIFF = (
    "A\tresults/"
    "gross9_structural_clock_bundle_preregistration_v2_2026-07-31.json",
)
G9CB2_SUCCESSOR_AUTHORITY_DIFF = (
    "A\tdocs/"
    "gross9-structural-clock-bundle-successor-authority-decision-"
    "2026-07-31.md",
)
G9CB2_SUCCESSOR_PROTOCOL_DIFF = (
    "M\ttests/test_build_gross9_structural_clock_bundle.py",
    "M\ttests/test_gross9_structural_clock_bundle_preregistration_artifact.py",
    "M\ttests/test_preregister_gross9_structural_clock_bundle.py",
    "M\ttraining/build_gross9_structural_clock_bundle.py",
    "M\ttraining/preregister_gross9_structural_clock_bundle.py",
)
G9CB2_ACTIVE_PREREGISTRATION_DIFF = (
    "A\tresults/"
    "gross9_structural_clock_bundle_g9cb2_preregistration_2026-07-31.json",
)
G9CB2_CLAIM_DIFF = (
    "A\tresults/"
    "gross9_structural_clock_bundle_g9cb2_access_claim_2026-07-31.json",
)
G9CB3_SUCCESSOR_AUTHORITY_DIFF = (
    "A\tdocs/"
    "gross9-structural-clock-bundle-g9cb3-successor-authority-decision-"
    "2026-07-31.md",
)
G9CB2_TERMINAL_EVIDENCE_DIFF = (
    "A\tresults/"
    "gross9_structural_clock_bundle_g9cb2_attempt_consumed_2026-07-31.json",
)
G9CB3_PROTOCOL_DIFF = G9CB2_SUCCESSOR_PROTOCOL_DIFF
G9CB3_ACTIVE_PREREGISTRATION_DIFF = (
    "A\tresults/"
    "gross9_structural_clock_bundle_g9cb3_preregistration_2026-07-31.json",
)
G9CB3_CLAIM_DIFF = (
    "A\tresults/"
    "gross9_structural_clock_bundle_g9cb3_access_claim_2026-07-31.json",
)
G9CB4_SUCCESSOR_AUTHORITY_DIFF = (
    "A\tdocs/"
    "gross9-structural-clock-bundle-g9cb4-successor-authority-decision-"
    "2026-07-31.md",
)
G9CB5_SUCCESSOR_AUTHORITY_DIFF = (
    "A\tdocs/"
    "gross9-structural-clock-bundle-g9cb5-successor-authority-decision-"
    "2026-07-31.md",
)
G9CB6_SUCCESSOR_AUTHORITY_DIFF = (
    "A\tdocs/"
    "gross9-structural-clock-bundle-g9cb6-successor-authority-decision-"
    "2026-07-31.md",
)
SUCCESSOR_AUTHORITY_DIFF = (
    (
        "A\tdocs/"
        "gross9-structural-clock-bundle-g9cb7-successor-authority-decision-"
        "2026-07-31.md"
    ),
)
TERMINAL_EVIDENCE_DIFF = (
    "A\tresults/"
    "gross9_structural_clock_bundle_g9cb3_attempt_consumed_2026-07-31.json",
    "A\tresults/"
    "gross9_structural_clock_bundle_g9cb3_worker_capability_consumed_pass1_"
    "2026-07-31.json",
)
SUCCESSOR_PROTOCOL_DIFF = G9CB2_SUCCESSOR_PROTOCOL_DIFF
G9CB4_ACTIVE_PREREGISTRATION_DIFF = (
    "A\tresults/"
    "gross9_structural_clock_bundle_g9cb4_preregistration_2026-07-31.json",
)
ACTIVE_PREREGISTRATION_DIFF = (
    "A\tresults/"
    "gross9_structural_clock_bundle_g9cb7_preregistration_2026-07-31.json",
)

DIRECT_AUTHORITY_BINDINGS = (
    (
        "gross9_portfolio",
        "configs/shadow/portfolio_rank7_capacity_candidate_2026-07-28.json",
        "006f82e1f0affad9f96a08a6c600542feec4a0e1198ed99b8630627de4913450",
        "a78173a3bd43a0c072e5e157d19579391bc10e29",
    ),
    (
        "base_portfolio",
        "configs/live/portfolio_added_alpha_mainnet_live_2026-07-18.json",
        "3f6c929f6b03797093b8b81f50ede533176aa169f5f81a4bb5f616d31afd24ff",
        "d8a2403f7e22dbe2c440c7ca031bc42e8557a86f",
    ),
    (
        "portfolio_runtime",
        "execution/portfolio_live.py",
        "5edd4e9aa749e538d7de6a9990e31b94fbcb444b7e1498714cea82036962863d",
        "801fae922f196c3d819b207045ba3f8d8c9f85d5",
    ),
    (
        "rank7_runtime",
        "execution/rank7_runtime.py",
        "1ba1ab8f0af7cee0bac4885836776d50f2aff9dd30319d47e9a322f82f36c0dc",
        "10294fe2b763de22c8928d061374600a2c90a1f8",
    ),
    (
        "rex_runtime",
        "execution/rex_llm_live.py",
        "2e0de376e967b237afb711dd44503ec45dbb9b6548f575219c1cf93cc2de9c48",
        "a4ab48081786f979ad20da03db39410e8545aaac",
    ),
    (
        "transitive_source_manifest",
        "configs/shadow/portfolio_added_alpha_signal_parity_sources_2026-07-16.json",
        "27a5095b18acaf10c9f5aa68c2ddac1ab1ebe4f506828e1fcfec34c414eb3ba6",
        "9ff9d3efb3fcd0688fbce1a1694417089edc63df",
    ),
    (
        "pre2025_anchor_hash_only",
        "results/gross9_pre2025_authoritative_anchor_2026-07-28.json",
        "329878d90b6cd9c731eb4871ac041256f95f03c14dd261ada681d3a370709875",
        "f0f73d05b666ebc86adb2b068e0d6369c57c8da2",
    ),
    (
        "rex_veto_config",
        "configs/live/rex_veto_7_candidate.json",
        "36df47c4737eb99f4ca5e2b257d9bd2fbf130df9d731b9ac02fcfe5192acd4db",
        "067a43c69b5433185c8c4a79e16e5d59597c9c0e",
    ),
    (
        "fresh_kimchi_config",
        "configs/shadow/fresh_kimchi_fx_2026-07-16.json",
        "f3e764d5d065643905105ae1c46668a22684569289c3781b79fc6b2efcc5154f",
        "310e65980b9e3987054fa6bc04e5abbab36d8cda",
    ),
    (
        "rank7_config",
        "configs/shadow/frozen_annual_rank7_2026-07-16.json",
        "b75621bb604266d1cd2529a29f8bdb6aec3b1f2c14ff00d88673ef007362526d",
        "29ec02f4dab2f49fc09360f65ff1c510347a7847",
    ),
    (
        "rank7_bundle_manifest",
        "artifacts/rank7/frozen_annual_rank7_2026/manifest.json",
        "2c45484dce48658ef7d342df7a3bb8e83cd0f31d4728bbb72fd38e612ec3b7a9",
        "bd375546e6a273e59f14a58aea19f725a5aeb0ad",
    ),
    (
        "markov_config",
        "configs/shadow/markov_transition_long_2026-07-16.json",
        "ebfec66715428b2fffead13e17229fb4369816daeeeab2c02cf0115e7110b755",
        "5f92d86cee2c617c590656c10ff05530196fc150",
    ),
    (
        "rex_taker_config",
        "configs/shadow/rex_taker_low_range_position_2026-07-16.json",
        "d4c56a6f1659189876c1d3f2e519a3dbc2608c754720c5cd1f65a02adb5589e4",
        "ede2d9d632f57eda2a4369a05d12916ef1f5ac5c",
    ),
    (
        "project_lock",
        "pyproject.toml",
        "972713ffd03a621c8e3a5acf61b8aa5f7aa68d573d68415bfab34a5b68304e90",
        "fa8a6907c7e965f588216f23a4e6e51e270bbea0",
    ),
    (
        "resolution_lock",
        "uv.lock",
        "ff965ca88c9eb9f17efe74a6d550ab99d093b44cda2467cee6f5738fb60f770a",
        "e4d529eca8110a530c362eb7883430bb81893140",
    ),
)

RANK7_BUNDLE_MANIFEST_PATH = Path(
    "artifacts/rank7/frozen_annual_rank7_2026/manifest.json"
)
RANK7_BUNDLE_ROOT = RANK7_BUNDLE_MANIFEST_PATH.parent
RANK7_BUNDLE_MANIFEST_HASH = (
    "06211697e4717f15db2c796da606c3785bfc25cac8ffa417fb3274063cb6ac8d"
)
RANK7_FILE_BINDINGS = (
    (
        "state/completed_hourly_history.csv.gz",
        "8d3ef5bae39c36e9955caf8c30bc20deedf375aa2876da9070a32a3fbd0f2f08",
        "e767b3edf7b9186c4d73566216c013573e30fb44",
    ),
    (
        "models/seed_7.npz",
        "b1f1c529cccabdd24465be995f9156fe211e5ad07792b0298ad42c2f1d4ddfb4",
        "ae2bab7b43254b75d26778e6c0189c1d9a9f9d8b",
    ),
    (
        "models/seed_71.npz",
        "df53e7b99090171b87c7e9fe4ef14b3f2a318e371df7d9735bd4d16b89eac5f9",
        "80a392b3f51122c22c385a71ea07265869a13db7",
    ),
    (
        "models/seed_715.npz",
        "ab9dff0aea41e4d55cd5c1a709c7ce061140891845e4596db30caf1505aaacf2",
        "20de028a609a3a766c6720cdaab0f45d40166058",
    ),
    (
        "models/seed_2026.npz",
        "5938b411a04b8b34b2cbed97778da8be33a1dcf574b6ff63480b594ab94fd51a",
        "7da6edb5b7d474699b5bd603ba64668248ea60c1",
    ),
    (
        "models/seed_71515.npz",
        "de955a31433722a61f18038195bdaad39efdb0a2cbfed3f6fe10dcd4a1ed63a5",
        "3caf8ee1c83223c3a96d84733b8517b6a595c701",
    ),
)

SOURCE_MANIFEST_PATH = Path(
    "configs/shadow/portfolio_added_alpha_signal_parity_sources_2026-07-16.json"
)
SOURCE_BINDINGS = (
    (
        "market_5m",
        "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz",
        "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c",
    ),
    (
        "funding",
        "data/binance_um_aux_btc_2020_2026/"
        "BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz",
        "4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7",
    ),
    (
        "premium",
        "data/binance_um_aux_btc_2020_2026/"
        "BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz",
        "b45fcc5a3cf75c8e594effe61a698c4652f841b1d304107e9669524e0fc9d0d7",
    ),
    (
        "open_interest",
        "/tmp/btcusdt_open_interest_5m_2020_2026.csv",
        "e08f93033e56959e8e7a9c1e21f27c5f01efc8d06fa6b4fbbfe7354697122b31",
    ),
    (
        "rex_taker_train",
        "data/rex_pullback_reclaim_q075_h144_ranker_train_2021_2023.jsonl",
        "07f6c4bb43ac92b341ce1a1b54ea6a429983611000148ad6966b81ea4a086df0",
    ),
    (
        "rex_taker_test",
        "data/rex_pullback_reclaim_q075_h144_ranker_test_2024.jsonl",
        "b1f5abf59c901ac109823a50063665ef455e75e70e90135acda77755ab8e5371",
    ),
    (
        "rex_taker_eval",
        "data/rex_pullback_reclaim_q075_h144_ranker_eval_2025_2026h1.jsonl",
        "bbe13d845d8dffcbb3e6c9b0f348390bd9d089c2d7b7bd6bccbafb91e75d9ce7",
    ),
    (
        "rex_veto_source",
        "data/rex_event_reasoning_policy_sft_20260712.jsonl",
        "2f5f477ed7ffd6063bd25b1fdbcb6cbaa804685be43b4522b7105dfba1b75d48",
    ),
)

FROZEN_OPEN_INTEREST_GZIP_PATH = Path(
    "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"
)
FROZEN_OPEN_INTEREST_GZIP_RESOLVED_PATH = FROZEN_OPEN_INTEREST_GZIP_PATH
FROZEN_OPEN_INTEREST_GZIP_SIZE = 72_898_508
FROZEN_OPEN_INTEREST_GZIP_SHA256 = (
    "dbc9e53b09551b469168fe19cc750c5c3ea86278db3055d079103f7654050192"
)
OPEN_INTEREST_PATH = Path("/tmp/btcusdt_open_interest_5m_2020_2026.csv")
OPEN_INTEREST_SIZE = 19_657_777
OPEN_INTEREST_SHA256 = (
    "e08f93033e56959e8e7a9c1e21f27c5f01efc8d06fa6b4fbbfe7354697122b31"
)

FROZEN_ENVIRONMENT = {
    "python_implementation": "CPython",
    "python_version": "3.10.10",
    "platform": "Linux",
    "machine": "x86_64",
    "libc": "glibc 2.39",
    "zlib_compile": "1.3",
    "zlib_runtime": "1.3",
    "selected_distributions": {
        "datasets": "4.6.1",
        "numpy": "2.2.6",
        "pandas": "2.3.3",
        "peft": "0.18.1",
        "scikit-learn": "1.7.2",
        "scipy": "1.15.3",
        "sqlalchemy": "absent",
        "torch": "2.9.0",
        "transformers": "5.7.0.dev0",
        "trl": "0.29.0",
        "websockets": "15.0.1",
    },
    "distribution_count": 108,
    "distribution_inventory_sha256": (
        "a5b435e485426d7254ed222692bf3b9c6444ae992e582084398dc57b960549dc"
    ),
}
WORKER_PROCESS_ENVIRONMENT = {
    "BLIS_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONPATH": REPOSITORY_ROOT.as_posix(),
    "PYTHONPYCACHEPREFIX": (
        REPOSITORY_ROOT / "results/.g9cb7-bytecode-cache-disabled"
    ).as_posix(),
    "PYTHONUNBUFFERED": "1",
    "PYTHONUTF8": "1",
    "TZ": "UTC",
    "VECLIB_MAXIMUM_THREADS": "1",
}

SLEEVES = (
    {
        "order": 0,
        "name": "cand_rex_veto_7",
        "configured_weight": 1.6,
        "side_rule": "exact_rex_decision_integer_1_or_minus_1",
        "hold_bars": 144,
        "entry_delay_bars": 1,
    },
    {
        "order": 1,
        "name": "fresh_kimchi_fx",
        "configured_weight": 2.0,
        "side_rule": "exclusive_long_short_gate_integer_1_or_minus_1",
        "maximum_hold_bars": 288,
        "take_bps": 400,
        "stop_bps": 250,
        "same_bar_policy": "stop_before_take",
        "entry_delay_bars": 1,
    },
    {
        "order": 2,
        "name": "frozen_annual_rank7",
        "configured_weight": 3.0,
        "side_rule": "long_only_integer_1",
        "funding_exit": {"maximum_hold_bars": 576, "take_bps": 400, "stop": None},
        "premium_exit": {"maximum_hold_bars": 144, "take": None, "stop_bps": 300},
        "entry_delay_bars": 1,
    },
    {
        "order": 3,
        "name": "markov_transition_long",
        "configured_weight": 2.0,
        "side_rule": "long_only_integer_1",
        "hold_bars": 576,
        "entry_delay_bars": 1,
    },
    {
        "order": 4,
        "name": "rex_taker_low_range_position",
        "configured_weight": 0.4,
        "side_rule": "exact_rex_decision_integer_1_or_minus_1",
        "hold_bars": 144,
        "entry_delay_bars": 1,
    },
)

CREATION_EVIDENCE_BOUNDARY = {
    "source_bytes_hashed": True,
    "source_value_rows_opened": 0,
    "pre2025_anchor_value_rows_opened": 0,
    "runtime_modules_imported": 0,
    "esdi_runtime_or_private_invocations": 0,
    "model_files_loaded": 0,
    "model_or_history_rows_opened": 0,
    "market_rows_opened": 0,
    "open_interest_rows_opened": 0,
    "funding_rows_opened": 0,
    "premium_rows_opened": 0,
    "outcome_dependent_ohlc_rows_opened": 0,
    "gross9_clock_rows_opened": 0,
    "candidate_rows_opened": 0,
    "comparator_clock_rows_opened": 0,
    "portfolio_return_or_pnl_computed": False,
    "funding_cash_computed": False,
    "economic_metric_computed": False,
    "candidate_or_overlap_metric_computed": False,
}

PERMANENT_PROHIBITED_COUNTERS = {
    "pre2025_anchor_value_rows_opened": 0,
    "candidate_rows_opened": 0,
    "comparator_clock_rows_opened": 0,
    "portfolio_return_values_computed": 0,
    "portfolio_pnl_values_computed": 0,
    "funding_cash_values_computed": 0,
    "cagr_values_computed": 0,
    "mdd_values_computed": 0,
    "economic_rank_values_computed": 0,
    "candidate_metric_values_computed": 0,
    "overlap_metric_values_computed": 0,
}
CREATION_ZERO_COUNTER_NAMES = (
    "source_value_rows_opened",
    "pre2025_anchor_value_rows_opened",
    "runtime_modules_imported",
    "esdi_runtime_or_private_invocations",
    "model_files_loaded",
    "model_or_history_rows_opened",
    "market_rows_opened",
    "open_interest_rows_opened",
    "funding_rows_opened",
    "premium_rows_opened",
    "outcome_dependent_ohlc_rows_opened",
    "gross9_clock_rows_opened",
    "candidate_rows_opened",
    "comparator_clock_rows_opened",
)
CREATION_FALSE_DECLARATION_NAMES = (
    "portfolio_return_or_pnl_computed",
    "funding_cash_computed",
    "economic_metric_computed",
    "candidate_or_overlap_metric_computed",
)

SOURCE_COUNTER_NAMES = (
    "market_5m",
    "funding",
    "premium",
    "open_interest",
    "rex_taker_train",
    "rex_taker_test",
    "rex_taker_eval",
    "rex_veto_source",
    "rank7_hourly_history",
)
PER_SLEEVE_COUNTER_NAMES = (
    "signal_rows_evaluated",
    "intervals_emitted",
    "long_intervals",
    "short_intervals",
    "fixed_horizon_exits",
    "take_exits",
    "stop_exits",
    "outcome_dependent_ohlc_rows_examined",
)


def validate_zero_access_schema(payload: Mapping[str, Any]) -> None:
    creation = payload.get("creation_evidence_boundary")
    if not isinstance(creation, Mapping) or set(creation) != set(
        CREATION_EVIDENCE_BOUNDARY
    ):
        raise ValueError("creation evidence boundary schema differs")
    if creation.get("source_bytes_hashed") is not True:
        raise ValueError("source_bytes_hashed must be boolean true")
    for key in CREATION_ZERO_COUNTER_NAMES:
        value = creation.get(key)
        if type(value) is not int or value != 0:
            raise ValueError(f"{key}: expected exact integer zero")
    for key in CREATION_FALSE_DECLARATION_NAMES:
        if creation.get(key) is not False:
            raise ValueError(f"{key}: expected boolean false")

    prohibited = payload.get("permanent_prohibited_counters")
    if not isinstance(prohibited, Mapping) or set(prohibited) != set(
        PERMANENT_PROHIBITED_COUNTERS
    ):
        raise ValueError("permanent prohibited counter schema differs")
    for key, value in prohibited.items():
        if type(value) is not int or value != 0:
            raise ValueError(f"{key}: expected exact integer zero")

    anchor = payload.get("pre2025_anchor_boundary")
    if not isinstance(anchor, Mapping) or set(anchor) != {
        "pre2025_anchor_bytes_hashed",
        "pre2025_anchor_git_blob_authenticated",
        "pre2025_anchor_json_parsed",
        "pre2025_anchor_value_rows_opened",
    }:
        raise ValueError("pre-2025 anchor boundary schema differs")
    if anchor.get("pre2025_anchor_bytes_hashed") is not True:
        raise ValueError("pre-2025 anchor hash declaration differs")
    if anchor.get("pre2025_anchor_git_blob_authenticated") is not True:
        raise ValueError("pre-2025 anchor Git declaration differs")
    if anchor.get("pre2025_anchor_json_parsed") is not False:
        raise ValueError("pre-2025 anchor parse declaration differs")
    anchor_rows = anchor.get("pre2025_anchor_value_rows_opened")
    if type(anchor_rows) is not int or anchor_rows != 0:
        raise ValueError("pre-2025 anchor row counter differs")

    independence = payload.get("candidate_independence")
    if not isinstance(independence, Mapping) or set(independence) != {
        "candidate_identity_present",
        "candidate_artifacts_opened",
        "comparator_clock_rows_opened",
        "comparator_clocks_preseen_by_research_program",
    }:
        raise ValueError("candidate-independence schema differs")
    if independence.get("candidate_identity_present") is not False:
        raise ValueError("candidate identity declaration differs")
    if independence.get("candidate_artifacts_opened") is not False:
        raise ValueError("candidate artifact declaration differs")
    comparator_rows = independence.get("comparator_clock_rows_opened")
    if type(comparator_rows) is not int or comparator_rows != 0:
        raise ValueError("comparator clock row counter differs")
    if (
        independence.get("comparator_clocks_preseen_by_research_program")
        is not True
    ):
        raise ValueError("comparator research-context disclosure differs")

    allowed_locations = {
        ("permanent_prohibited_counters",),
        ("creation_evidence_boundary", "source_bytes_hashed"),
        *{
            ("creation_evidence_boundary", key)
            for key in CREATION_ZERO_COUNTER_NAMES
        },
        *{
            ("creation_evidence_boundary", key)
            for key in CREATION_FALSE_DECLARATION_NAMES
        },
        *{
            ("permanent_prohibited_counters", key)
            for key in PERMANENT_PROHIBITED_COUNTERS
        },
        (
            "pre2025_anchor_boundary",
            "pre2025_anchor_value_rows_opened",
        ),
        (
            "pre2025_anchor_boundary",
            "pre2025_anchor_bytes_hashed",
        ),
        ("candidate_independence", "comparator_clock_rows_opened"),
    }
    guarded_names = {
        "source_bytes_hashed",
        *CREATION_ZERO_COUNTER_NAMES,
        *CREATION_FALSE_DECLARATION_NAMES,
        *PERMANENT_PROHIBITED_COUNTERS,
    }
    guarded_suffixes = (
        "_computed",
        "_values_computed",
        "_rows_opened",
        "_rows_examined",
        "_files_loaded",
        "_files_opened",
        "_modules_imported",
        "_invocations",
        "_bytes_hashed",
        "_counter",
        "_counters",
    )

    def walk(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for raw_key, item in value.items():
                if not isinstance(raw_key, str):
                    raise ValueError("preregistration mapping key is not text")
                location = (*path, raw_key)
                if location in {
                    ("bindings", "failed_predecessor_attempts"),
                    ("bindings", "failed_predecessor_closures"),
                    (
                        "bindings",
                        "failed_predecessor_prepublication_closures",
                    ),
                }:
                    continue
                if (
                    raw_key in guarded_names
                    or raw_key.endswith(guarded_suffixes)
                ) and location not in allowed_locations:
                    raise ValueError(
                        f"{'.'.join(location)}: computed/counter key is misplaced"
                    )
                walk(item, location)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, (*path, str(index)))

    walk(payload)


def repository_path(
    path: str | os.PathLike[str], repository_root: Path = REPOSITORY_ROOT
) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repository_root / candidate
    return candidate


def canonical_json_bytes(value: Any, *, trailing_lf: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if trailing_lf else b"")


def canonical_hash(
    value: Mapping[str, Any], excluded_key: str = "manifest_hash"
) -> str:
    body = dict(value)
    body.pop(excluded_key, None)
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


_ACTIVE_PREREGISTRATION_CACHE: Mapping[
    str, tuple[bytes, os.stat_result]
] | None = None
_ACTIVE_PREREGISTRATION_GIT_PAIRS: Mapping[
    str, tuple[str, str] | None
] | None = None
_ACTIVE_PREREGISTRATION_SNAPSHOT: _PreregistrationSnapshot | None = None
_ACTIVE_PREREGISTRATION_ROOT: Path | None = None


def _cache_key(
    path: str | os.PathLike[str], repository_root: Path
) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            return candidate.relative_to(repository_root).as_posix()
        except ValueError:
            return candidate.as_posix()
    return candidate.as_posix()


def _cached_file(
    path: str | os.PathLike[str], repository_root: Path
) -> tuple[bytes, os.stat_result] | None:
    if _ACTIVE_PREREGISTRATION_CACHE is None:
        return None
    root = (
        repository_root
        if _ACTIVE_PREREGISTRATION_ROOT is None
        else _ACTIVE_PREREGISTRATION_ROOT
    )
    key = _cache_key(path, root)
    cached = _ACTIVE_PREREGISTRATION_CACHE.get(key)
    if cached is None:
        raise ValueError(f"{key}: path was not included in the bootstrap snapshot")
    return cached


def sha256_file(
    path: str | os.PathLike[str], repository_root: Path = REPOSITORY_ROOT
) -> str:
    cached = _cached_file(path, repository_root)
    if cached is not None:
        return hashlib.sha256(cached[0]).hexdigest()
    digest = hashlib.sha256()
    with repository_path(path, repository_root).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_type(
    path: Path, repository_root: Path = REPOSITORY_ROOT
) -> str:
    cached = _cached_file(path, repository_root)
    if cached is not None:
        mode = cached[1].st_mode
    else:
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            return "missing"
    if stat.S_ISREG(mode):
        return "regular_file"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISBLK(mode):
        return "block_device"
    if stat.S_ISCHR(mode):
        return "character_device"
    return "other"


def validate_file(
    path: str | os.PathLike[str],
    expected_sha256: str,
    *,
    expected_type: str = "regular_file",
    expected_size: int | None = None,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    absolute = repository_path(path, repository_root)
    actual_type = _path_type(absolute, repository_root)
    if actual_type != expected_type:
        raise ValueError(f"{path}: expected {expected_type}, found {actual_type}")
    cached = _cached_file(absolute, repository_root)
    size = cached[1].st_size if cached is not None else absolute.stat().st_size
    if expected_size is not None and size != expected_size:
        raise ValueError(f"{path}: expected {expected_size} bytes, found {size}")
    digest = sha256_file(absolute, repository_root)
    if digest != expected_sha256:
        raise ValueError(f"{path}: SHA-256 mismatch")
    return {
        "path": str(path),
        "path_type": actual_type,
        "size_bytes": size,
        "sha256": digest,
    }


def _run_git(
    arguments: Sequence[str], repository_root: Path = REPOSITORY_ROOT
) -> str:
    completed = _git_result(arguments, repository_root)
    completed.check_returncode()
    if not isinstance(completed.stdout, str):
        raise TypeError("text Git command returned non-text stdout")
    return completed.stdout.rstrip("\n")


def _run_git_bytes(
    arguments: Sequence[str], repository_root: Path = REPOSITORY_ROOT
) -> bytes:
    completed = _git_result(arguments, repository_root, text=False)
    completed.check_returncode()
    if not isinstance(completed.stdout, bytes):
        raise TypeError("binary Git command returned non-bytes stdout")
    return completed.stdout


def _tracked_results_top_level_entries(output: str) -> set[str]:
    """Project normalized tracked results paths onto entries below results/."""
    entries: set[str] = set()
    for path_text in output.splitlines():
        components = path_text.split("/")
        if (
            len(components) < 2
            or components[0] != "results"
            or any(component in {"", ".", ".."} for component in components)
            or "\\" in path_text
        ):
            raise ValueError("malformed tracked results path")
        entries.add(components[1])
    return entries


def git_blob(
    path: str | os.PathLike[str], repository_root: Path = REPOSITORY_ROOT
) -> tuple[str, str]:
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            candidate = candidate.relative_to(repository_root)
        except ValueError as error:
            raise ValueError(f"{path}: not a repository path") from error
    output = _run_git(
        ["ls-tree", "HEAD", "--", candidate.as_posix()], repository_root
    )
    fields = output.split(None, 3)
    if len(fields) != 4 or fields[1] != "blob":
        raise ValueError(f"{candidate}: not a tracked Git blob at HEAD")
    return fields[2], fields[0]


def validate_git_blob(
    path: str | os.PathLike[str],
    expected_blob: str,
    *,
    expected_mode: str = "100644",
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, str]:
    path_text = _cache_key(path, repository_root)
    if _ACTIVE_PREREGISTRATION_GIT_PAIRS is not None:
        pair = _ACTIVE_PREREGISTRATION_GIT_PAIRS.get(path_text)
        if pair is None:
            raise ValueError(f"{path_text}: tracked Git pair is absent")
        blob, mode = pair
    else:
        blob, mode = git_blob(path, repository_root)
    if blob != expected_blob:
        raise ValueError(f"{path}: Git blob mismatch")
    if mode != expected_mode:
        raise ValueError(f"{path}: expected Git mode {expected_mode}, found {mode}")
    return {"git_blob": blob, "git_mode": mode}


def _tracked_binding(
    path: str | os.PathLike[str],
    *,
    repository_root: Path = REPOSITORY_ROOT,
    expected_sha256: str | None = None,
    expected_blob: str | None = None,
) -> dict[str, Any]:
    absolute = repository_path(path, repository_root)
    if _path_type(absolute, repository_root) != "regular_file":
        raise ValueError(f"{path}: tracked binding must be a regular file")
    digest = sha256_file(absolute)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"{path}: SHA-256 mismatch")
    blob, mode = git_blob(path, repository_root)
    if expected_blob is not None and blob != expected_blob:
        raise ValueError(f"{path}: Git blob mismatch")
    if mode != "100644":
        raise ValueError(f"{path}: expected Git mode 100644, found {mode}")
    return {
        "path": Path(path).as_posix(),
        "path_type": "regular_file",
        "sha256": digest,
        "git_blob": blob,
        "git_mode": mode,
    }


def _single_parent(
    commit: str, repository_root: Path = REPOSITORY_ROOT
) -> str:
    fields = _run_git(
        ["rev-list", "--parents", "-n", "1", commit], repository_root
    ).split()
    if len(fields) != 2 or fields[0] != commit:
        raise ValueError(f"{commit}: expected exactly one Git parent")
    return fields[1]


def _commit_diff(
    parent: str,
    child: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[str, ...]:
    output = _run_git(
        [
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "-r",
            parent,
            child,
        ],
        repository_root,
    )
    return tuple(line for line in output.splitlines() if line)


def _addition_commits(
    path: Path, repository_root: Path = REPOSITORY_ROOT
) -> tuple[str, ...]:
    output = _run_git(
        [
            "log",
            "--format=%H",
            "--diff-filter=A",
            "--",
            path.as_posix(),
        ],
        repository_root,
    )
    return tuple(line for line in output.splitlines() if line)


def _require_ancestor(
    ancestor: str,
    descendant: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> None:
    try:
        _run_git(
            ["merge-base", "--is-ancestor", ancestor, descendant],
            repository_root,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            f"{ancestor}: not an ancestor of {descendant}"
        ) from exc


def validate_historical_preregistration_topology(
    repository_root: Path = REPOSITORY_ROOT,
) -> None:
    if (
        _single_parent(
            HISTORICAL_PREREGISTRATION_SEAL_COMMIT, repository_root
        )
        != HISTORICAL_PROTOCOL_PARENT_COMMIT
    ):
        raise ValueError("historical preregistration parent differs")
    expected = (
        "A\t"
        f"{HISTORICAL_PREREGISTRATION_PATH.as_posix()}",
    )
    if _commit_diff(
        HISTORICAL_PROTOCOL_PARENT_COMMIT,
        HISTORICAL_PREREGISTRATION_SEAL_COMMIT,
        repository_root,
    ) != expected:
        raise ValueError("historical preregistration seal diff differs")
    if _addition_commits(
        HISTORICAL_PREREGISTRATION_PATH, repository_root
    ) != (HISTORICAL_PREREGISTRATION_SEAL_COMMIT,):
        raise ValueError("historical preregistration addition history differs")
    head = _run_git(["rev-parse", "HEAD"], repository_root)
    _require_ancestor(
        HISTORICAL_PREREGISTRATION_SEAL_COMMIT,
        head,
        repository_root,
    )


def validate_failed_v2_preregistration_topology(
    repository_root: Path = REPOSITORY_ROOT,
) -> None:
    if (
        _single_parent(
            PREREGISTRATION_CORRECTION_AMENDMENT_COMMIT,
            repository_root,
        )
        != HISTORICAL_PREREGISTRATION_SEAL_COMMIT
    ):
        raise ValueError("G9CB-1C authority parent differs")
    if _commit_diff(
        HISTORICAL_PREREGISTRATION_SEAL_COMMIT,
        PREREGISTRATION_CORRECTION_AMENDMENT_COMMIT,
        repository_root,
    ) != G9CB1_CORRECTION_AUTHORITY_DIFF:
        raise ValueError("G9CB-1C authority diff differs")
    if (
        _single_parent(
            FAILED_V2_PROTOCOL_IMPLEMENTATION_COMMIT,
            repository_root,
        )
        != PREREGISTRATION_CORRECTION_AMENDMENT_COMMIT
    ):
        raise ValueError("failed G9CB-1 v2 implementation parent differs")
    if _commit_diff(
        PREREGISTRATION_CORRECTION_AMENDMENT_COMMIT,
        FAILED_V2_PROTOCOL_IMPLEMENTATION_COMMIT,
        repository_root,
    ) != G9CB1_CORRECTION_PROTOCOL_DIFF:
        raise ValueError("failed G9CB-1 v2 implementation diff differs")
    if (
        _single_parent(
            FAILED_V2_PREREGISTRATION_SEAL_COMMIT,
            repository_root,
        )
        != FAILED_V2_PROTOCOL_IMPLEMENTATION_COMMIT
    ):
        raise ValueError("failed G9CB-1 v2 seal parent differs")
    if _commit_diff(
        FAILED_V2_PROTOCOL_IMPLEMENTATION_COMMIT,
        FAILED_V2_PREREGISTRATION_SEAL_COMMIT,
        repository_root,
    ) != FAILED_V2_PREREGISTRATION_DIFF:
        raise ValueError("failed G9CB-1 v2 seal diff differs")
    if _addition_commits(
        FAILED_V2_PREREGISTRATION_PATH, repository_root
    ) != (FAILED_V2_PREREGISTRATION_SEAL_COMMIT,):
        raise ValueError("failed G9CB-1 v2 addition history differs")
    head = _run_git(["rev-parse", "HEAD"], repository_root)
    _require_ancestor(
        FAILED_V2_PREREGISTRATION_SEAL_COMMIT,
        head,
        repository_root,
    )


def validate_protocol_commit_topology(
    repository_root: Path = REPOSITORY_ROOT,
) -> str:
    validate_failed_v2_preregistration_topology(repository_root)
    if (
        _single_parent(G9CB2_AUTHORITY_DECISION_COMMIT, repository_root)
        != FAILED_V2_PREREGISTRATION_SEAL_COMMIT
        or _commit_diff(
            FAILED_V2_PREREGISTRATION_SEAL_COMMIT,
            G9CB2_AUTHORITY_DECISION_COMMIT,
            repository_root,
        )
        != G9CB2_SUCCESSOR_AUTHORITY_DIFF
    ):
        raise ValueError("G9CB-2 authority topology differs")
    if _addition_commits(
        G9CB2_AUTHORITY_DECISION_PATH, repository_root
    ) != (G9CB2_AUTHORITY_DECISION_COMMIT,):
        raise ValueError("G9CB-2 authority addition history differs")
    predecessor_steps = (
        (
            G9CB2_AUTHORITY_DECISION_COMMIT,
            G9CB2_PROTOCOL_IMPLEMENTATION_COMMIT,
            G9CB2_SUCCESSOR_PROTOCOL_DIFF,
            "protocol implementation",
        ),
        (
            G9CB2_PROTOCOL_IMPLEMENTATION_COMMIT,
            G9CB2_PREREGISTRATION_SEAL_COMMIT,
            G9CB2_ACTIVE_PREREGISTRATION_DIFF,
            "preregistration",
        ),
        (
            G9CB2_PREREGISTRATION_SEAL_COMMIT,
            G9CB2_CLAIM_COMMIT,
            G9CB2_CLAIM_DIFF,
            "claim",
        ),
        (
            G9CB2_CLAIM_COMMIT,
            G9CB3_AUTHORITY_DECISION_COMMIT,
            G9CB3_SUCCESSOR_AUTHORITY_DIFF,
            "successor authority",
        ),
        (
            G9CB3_AUTHORITY_DECISION_COMMIT,
            G9CB2_TERMINAL_EVIDENCE_COMMIT,
            G9CB2_TERMINAL_EVIDENCE_DIFF,
            "terminal evidence",
        ),
        (
            G9CB2_TERMINAL_EVIDENCE_COMMIT,
            G9CB3_PROTOCOL_IMPLEMENTATION_COMMIT,
            G9CB3_PROTOCOL_DIFF,
            "G9CB-3 protocol implementation",
        ),
        (
            G9CB3_PROTOCOL_IMPLEMENTATION_COMMIT,
            G9CB3_PREREGISTRATION_SEAL_COMMIT,
            G9CB3_ACTIVE_PREREGISTRATION_DIFF,
            "G9CB-3 preregistration",
        ),
        (
            G9CB3_PREREGISTRATION_SEAL_COMMIT,
            G9CB3_CLAIM_COMMIT,
            G9CB3_CLAIM_DIFF,
            "G9CB-3 claim",
        ),
        (
            G9CB3_CLAIM_COMMIT,
            G9CB4_AUTHORITY_DECISION_COMMIT,
            G9CB4_SUCCESSOR_AUTHORITY_DIFF,
            "G9CB-4 successor authority",
        ),
        (
            G9CB4_AUTHORITY_DECISION_COMMIT,
            G9CB3_TERMINAL_EVIDENCE_COMMIT,
            TERMINAL_EVIDENCE_DIFF,
            "G9CB-3 terminal evidence",
        ),
        (
            G9CB3_TERMINAL_EVIDENCE_COMMIT,
            G9CB4_PROTOCOL_IMPLEMENTATION_COMMIT,
            SUCCESSOR_PROTOCOL_DIFF,
            "G9CB-4 protocol implementation",
        ),
        (
            G9CB4_PROTOCOL_IMPLEMENTATION_COMMIT,
            G9CB4_PREREGISTRATION_SEAL_COMMIT,
            G9CB4_ACTIVE_PREREGISTRATION_DIFF,
            "G9CB-4 preregistration",
        ),
        (
            G9CB4_PREREGISTRATION_SEAL_COMMIT,
            G9CB5_AUTHORITY_DECISION_COMMIT,
            G9CB5_SUCCESSOR_AUTHORITY_DIFF,
            "G9CB-5 successor authority",
        ),
        (
            G9CB5_AUTHORITY_DECISION_COMMIT,
            G9CB5_PROTOCOL_IMPLEMENTATION_COMMIT,
            SUCCESSOR_PROTOCOL_DIFF,
            "G9CB-5 protocol implementation",
        ),
        (
            G9CB5_PROTOCOL_IMPLEMENTATION_COMMIT,
            G9CB6_AUTHORITY_DECISION_COMMIT,
            G9CB6_SUCCESSOR_AUTHORITY_DIFF,
            "G9CB-6 successor authority",
        ),
        (
            G9CB6_AUTHORITY_DECISION_COMMIT,
            G9CB6_PROTOCOL_IMPLEMENTATION_COMMIT,
            SUCCESSOR_PROTOCOL_DIFF,
            "G9CB-6 protocol implementation",
        ),
        (
            G9CB6_PROTOCOL_IMPLEMENTATION_COMMIT,
            AUTHORITY_DECISION_COMMIT,
            SUCCESSOR_AUTHORITY_DIFF,
            "G9CB-7 successor authority",
        ),
    )
    for parent, child, expected_diff, label in predecessor_steps:
        observed_parent = _single_parent(child, repository_root)
        if (
            observed_parent != parent
            or _commit_diff(parent, child, repository_root) != expected_diff
        ):
            raise ValueError(f"{label} topology differs")
    if _addition_commits(
        G9CB4_AUTHORITY_DECISION_PATH, repository_root
    ) != (G9CB4_AUTHORITY_DECISION_COMMIT,):
        raise ValueError("G9CB-4 authority addition history differs")
    if _addition_commits(
        G9CB5_AUTHORITY_DECISION_PATH, repository_root
    ) != (G9CB5_AUTHORITY_DECISION_COMMIT,):
        raise ValueError("G9CB-5 authority addition history differs")
    if _addition_commits(
        G9CB6_AUTHORITY_DECISION_PATH, repository_root
    ) != (G9CB6_AUTHORITY_DECISION_COMMIT,):
        raise ValueError("G9CB-6 authority addition history differs")
    if _addition_commits(
        ACTIVE_AUTHORITY_DECISION_PATH, repository_root
    ) != (AUTHORITY_DECISION_COMMIT,):
        raise ValueError("G9CB-7 authority addition history differs")
    if _addition_commits(
        G9CB3_AUTHORITY_DECISION_PATH, repository_root
    ) != (G9CB3_AUTHORITY_DECISION_COMMIT,):
        raise ValueError("G9CB-3 authority addition history differs")

    head = _run_git(["rev-parse", "HEAD"], repository_root)
    additions = _addition_commits(PREREGISTRATION_PATH, repository_root)
    if additions:
        if len(additions) != 1:
            raise ValueError("active preregistration addition history differs")
        preregistration_seal = additions[0]
        implementation = _single_parent(
            preregistration_seal, repository_root
        )
        if _commit_diff(
            implementation,
            preregistration_seal,
            repository_root,
        ) != ACTIVE_PREREGISTRATION_DIFF:
            raise ValueError("active preregistration seal diff differs")
        _require_ancestor(preregistration_seal, head, repository_root)
    else:
        implementation = head

    implementation_parent = AUTHORITY_DECISION_COMMIT
    if _single_parent(implementation, repository_root) != implementation_parent:
        raise ValueError("protocol implementation parent differs")
    if _commit_diff(
        implementation_parent,
        implementation,
        repository_root,
    ) != SUCCESSOR_PROTOCOL_DIFF:
        raise ValueError("protocol implementation diff differs")
    _require_ancestor(implementation, head, repository_root)
    return implementation


def validate_failed_predecessor_attempt_topology(
    repository_root: Path = REPOSITORY_ROOT,
) -> None:
    validate_failed_v2_preregistration_topology(repository_root)
    if (
        _single_parent(G9CB2_AUTHORITY_DECISION_COMMIT, repository_root)
        != FAILED_V2_PREREGISTRATION_SEAL_COMMIT
        or _commit_diff(
            FAILED_V2_PREREGISTRATION_SEAL_COMMIT,
            G9CB2_AUTHORITY_DECISION_COMMIT,
            repository_root,
        )
        != G9CB2_SUCCESSOR_AUTHORITY_DIFF
    ):
        raise ValueError("G9CB-2 authority topology differs")
    steps = (
        (
            G9CB2_AUTHORITY_DECISION_COMMIT,
            G9CB2_PROTOCOL_IMPLEMENTATION_COMMIT,
            G9CB2_SUCCESSOR_PROTOCOL_DIFF,
        ),
        (
            G9CB2_PROTOCOL_IMPLEMENTATION_COMMIT,
            G9CB2_PREREGISTRATION_SEAL_COMMIT,
            G9CB2_ACTIVE_PREREGISTRATION_DIFF,
        ),
        (
            G9CB2_PREREGISTRATION_SEAL_COMMIT,
            G9CB2_CLAIM_COMMIT,
            G9CB2_CLAIM_DIFF,
        ),
        (
            G9CB2_CLAIM_COMMIT,
            G9CB3_AUTHORITY_DECISION_COMMIT,
            G9CB3_SUCCESSOR_AUTHORITY_DIFF,
        ),
        (
            G9CB3_AUTHORITY_DECISION_COMMIT,
            G9CB2_TERMINAL_EVIDENCE_COMMIT,
            G9CB2_TERMINAL_EVIDENCE_DIFF,
        ),
        (
            G9CB2_TERMINAL_EVIDENCE_COMMIT,
            G9CB3_PROTOCOL_IMPLEMENTATION_COMMIT,
            G9CB3_PROTOCOL_DIFF,
        ),
        (
            G9CB3_PROTOCOL_IMPLEMENTATION_COMMIT,
            G9CB3_PREREGISTRATION_SEAL_COMMIT,
            G9CB3_ACTIVE_PREREGISTRATION_DIFF,
        ),
        (
            G9CB3_PREREGISTRATION_SEAL_COMMIT,
            G9CB3_CLAIM_COMMIT,
            G9CB3_CLAIM_DIFF,
        ),
        (
            G9CB3_CLAIM_COMMIT,
            G9CB4_AUTHORITY_DECISION_COMMIT,
            G9CB4_SUCCESSOR_AUTHORITY_DIFF,
        ),
        (
            G9CB4_AUTHORITY_DECISION_COMMIT,
            G9CB3_TERMINAL_EVIDENCE_COMMIT,
            TERMINAL_EVIDENCE_DIFF,
        ),
    )
    for parent, child, expected_diff in steps:
        if (
            _single_parent(child, repository_root) != parent
            or _commit_diff(parent, child, repository_root) != expected_diff
        ):
            raise ValueError("failed predecessor attempt topology differs")
    if _addition_commits(
        Path(expected_failed_predecessor_attempts()[0]["attempt_sentinel"]["path"]),
        repository_root,
    ) != (G9CB2_TERMINAL_EVIDENCE_COMMIT,):
        raise ValueError("G9CB-2 terminal evidence addition history differs")
    g9cb3 = expected_failed_predecessor_attempts()[1]["terminal_evidence"]
    for binding in g9cb3.values():
        if _addition_commits(
            Path(binding["path"]), repository_root
        ) != (G9CB3_TERMINAL_EVIDENCE_COMMIT,):
            raise ValueError(
                "G9CB-3 terminal evidence addition history differs"
            )


def _historical_v1_authority_amendments() -> list[dict[str, str]]:
    return [
        {
            "identity": "G9CB-1A",
            "path": RANK7_AUTHORITY_AMENDMENT_PATH.as_posix(),
            "path_type": "regular_file",
            "sha256": RANK7_AUTHORITY_AMENDMENT_SHA256,
            "git_blob": RANK7_AUTHORITY_AMENDMENT_GIT_BLOB,
            "git_mode": "100644",
            "authority_commit": RANK7_AUTHORITY_AMENDMENT_COMMIT,
        },
        {
            "identity": "G9CB-1B",
            "path": RUNTIME_ISOLATION_AMENDMENT_PATH.as_posix(),
            "path_type": "regular_file",
            "sha256": RUNTIME_ISOLATION_AMENDMENT_SHA256,
            "git_blob": RUNTIME_ISOLATION_AMENDMENT_GIT_BLOB,
            "git_mode": "100644",
            "authority_commit": RUNTIME_ISOLATION_AMENDMENT_COMMIT,
        },
    ]


def _historical_v2_authority_amendments() -> list[dict[str, str]]:
    return [
        *_historical_v1_authority_amendments(),
        {
            "identity": "G9CB-1C",
            "path": PREREGISTRATION_CORRECTION_AMENDMENT_PATH.as_posix(),
            "path_type": "regular_file",
            "sha256": PREREGISTRATION_CORRECTION_AMENDMENT_SHA256,
            "git_blob": PREREGISTRATION_CORRECTION_AMENDMENT_GIT_BLOB,
            "git_mode": "100644",
            "authority_commit": PREREGISTRATION_CORRECTION_AMENDMENT_COMMIT,
        },
    ]


def expected_superseded_preregistration_binding() -> dict[str, str]:
    return {
        "path": HISTORICAL_PREREGISTRATION_PATH.as_posix(),
        "path_type": "regular_file",
        "sha256": HISTORICAL_PREREGISTRATION_SHA256,
        "git_blob": HISTORICAL_PREREGISTRATION_GIT_BLOB,
        "git_mode": "100644",
        "filesystem_mode_octal": "0444",
        "seal_commit": HISTORICAL_PREREGISTRATION_SEAL_COMMIT,
        "protocol_parent_commit": HISTORICAL_PROTOCOL_PARENT_COMMIT,
        "protocol_version": HISTORICAL_PROTOCOL_VERSION,
        "manifest_hash": HISTORICAL_PREREGISTRATION_MANIFEST_HASH,
        "status": "historical_nonoperative_preclaim_validation_failure",
    }


def validate_superseded_preregistration(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, str]:
    binding = _tracked_binding(
        HISTORICAL_PREREGISTRATION_PATH,
        repository_root=repository_root,
        expected_sha256=HISTORICAL_PREREGISTRATION_SHA256,
        expected_blob=HISTORICAL_PREREGISTRATION_GIT_BLOB,
    )
    path = repository_path(
        HISTORICAL_PREREGISTRATION_PATH, repository_root
    )
    cached = _cached_file(path, repository_root)
    info = cached[1] if cached is not None else path.stat()
    if stat.S_IMODE(info.st_mode) != 0o444:
        raise ValueError("historical preregistration filesystem mode differs")
    raw = cached[0] if cached is not None else path.read_bytes()
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError("historical preregistration trailing LF differs")
    payload = json.loads(raw)
    if raw != canonical_json_bytes(payload, trailing_lf=True):
        raise ValueError("historical preregistration JSON is not canonical")
    if payload.get("protocol_version") != HISTORICAL_PROTOCOL_VERSION:
        raise ValueError("historical preregistration protocol version differs")
    if (
        payload.get("manifest_hash")
        != HISTORICAL_PREREGISTRATION_MANIFEST_HASH
        or canonical_hash(payload)
        != HISTORICAL_PREREGISTRATION_MANIFEST_HASH
    ):
        raise ValueError("historical preregistration manifest hash differs")
    if (
        payload.get("bindings", {}).get("authority_amendments")
        != _historical_v1_authority_amendments()
    ):
        raise ValueError("historical preregistration amendments differ")
    validate_historical_preregistration_topology(repository_root)
    binding.update(
        {
            "filesystem_mode_octal": "0444",
            "seal_commit": HISTORICAL_PREREGISTRATION_SEAL_COMMIT,
            "protocol_parent_commit": HISTORICAL_PROTOCOL_PARENT_COMMIT,
            "protocol_version": HISTORICAL_PROTOCOL_VERSION,
            "manifest_hash": HISTORICAL_PREREGISTRATION_MANIFEST_HASH,
            "status": "historical_nonoperative_preclaim_validation_failure",
        }
    )
    expected = expected_superseded_preregistration_binding()
    if binding != expected:
        raise ValueError("historical preregistration binding differs")
    return binding


def expected_failed_v2_preregistration_binding() -> dict[str, str]:
    return {
        "path": FAILED_V2_PREREGISTRATION_PATH.as_posix(),
        "path_type": "regular_file",
        "sha256": FAILED_V2_PREREGISTRATION_SHA256,
        "git_blob": FAILED_V2_PREREGISTRATION_GIT_BLOB,
        "git_mode": "100644",
        "filesystem_mode_octal": "0444",
        "seal_commit": FAILED_V2_PREREGISTRATION_SEAL_COMMIT,
        "protocol_implementation_commit": (
            FAILED_V2_PROTOCOL_IMPLEMENTATION_COMMIT
        ),
        "protocol_version": FAILED_V2_PROTOCOL_VERSION,
        "manifest_hash": FAILED_V2_PREREGISTRATION_MANIFEST_HASH,
        "status": (
            "historical_nonoperative_preclaim_git_metadata_contract_failure"
        ),
    }


def validate_failed_v2_preregistration(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, str]:
    binding = _tracked_binding(
        FAILED_V2_PREREGISTRATION_PATH,
        repository_root=repository_root,
        expected_sha256=FAILED_V2_PREREGISTRATION_SHA256,
        expected_blob=FAILED_V2_PREREGISTRATION_GIT_BLOB,
    )
    path = repository_path(
        FAILED_V2_PREREGISTRATION_PATH, repository_root
    )
    cached = _cached_file(path, repository_root)
    info = cached[1] if cached is not None else path.stat()
    if stat.S_IMODE(info.st_mode) != 0o444:
        raise ValueError("failed G9CB-1 v2 filesystem mode differs")
    raw = cached[0] if cached is not None else path.read_bytes()
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError("failed G9CB-1 v2 trailing LF differs")
    payload = json.loads(raw)
    if raw != canonical_json_bytes(payload, trailing_lf=True):
        raise ValueError("failed G9CB-1 v2 JSON is not canonical")
    if (
        payload.get("identity") != "G9CB-1"
        or payload.get("protocol_version") != FAILED_V2_PROTOCOL_VERSION
        or payload.get("protocol_implementation_commit")
        != FAILED_V2_PROTOCOL_IMPLEMENTATION_COMMIT
    ):
        raise ValueError("failed G9CB-1 v2 protocol identity differs")
    if (
        payload.get("manifest_hash")
        != FAILED_V2_PREREGISTRATION_MANIFEST_HASH
        or canonical_hash(payload)
        != FAILED_V2_PREREGISTRATION_MANIFEST_HASH
    ):
        raise ValueError("failed G9CB-1 v2 manifest hash differs")
    if (
        payload.get("bindings", {}).get("authority_amendments")
        != _historical_v2_authority_amendments()
    ):
        raise ValueError("failed G9CB-1 v2 amendments differ")
    validate_failed_v2_preregistration_topology(repository_root)
    binding.update(
        {
            "filesystem_mode_octal": "0444",
            "seal_commit": FAILED_V2_PREREGISTRATION_SEAL_COMMIT,
            "protocol_implementation_commit": (
                FAILED_V2_PROTOCOL_IMPLEMENTATION_COMMIT
            ),
            "protocol_version": FAILED_V2_PROTOCOL_VERSION,
            "manifest_hash": FAILED_V2_PREREGISTRATION_MANIFEST_HASH,
            "status": (
                "historical_nonoperative_preclaim_git_metadata_contract_failure"
            ),
        }
    )
    expected = expected_failed_v2_preregistration_binding()
    if binding != expected:
        raise ValueError("failed G9CB-1 v2 binding differs")
    return binding


def expected_failed_predecessor_preregistration_bindings(
) -> list[dict[str, str]]:
    return [
        expected_superseded_preregistration_binding(),
        expected_failed_v2_preregistration_binding(),
    ]


def validate_failed_predecessor_preregistrations(
    repository_root: Path = REPOSITORY_ROOT,
) -> list[dict[str, str]]:
    observed = [
        validate_superseded_preregistration(repository_root),
        validate_failed_v2_preregistration(repository_root),
    ]
    expected = expected_failed_predecessor_preregistration_bindings()
    if observed != expected:
        raise ValueError("failed predecessor preregistration bindings differ")
    return observed


def expected_failed_predecessor_attempts() -> list[dict[str, Any]]:
    return [
        {
            "access_claim": {
                "claim_hash": "28faeb0a7f9662c3264374785b7e53376c4d6500817f26e7d94e0afeab25979d",
                "filesystem_mode_octal": "0444",
                "git_blob": "05adab10031399c3599168c9673923812cecdc09",
                "git_mode": "100644",
                "path": "results/gross9_structural_clock_bundle_g9cb2_access_claim_2026-07-31.json",
                "path_type": "regular_file",
                "protocol_parent_commit": G9CB2_PREREGISTRATION_SEAL_COMMIT,
                "seal_commit": G9CB2_CLAIM_COMMIT,
                "sha256": "0d0cea614cc8ddc51106989c0d68362ed27684d1ce46b1d4daea41c6bfb0be23",
            },
            "attempt_sentinel": {
                "claim_commit": G9CB2_CLAIM_COMMIT,
                "filesystem_mode_octal": "0444",
                "git_blob": "8263cd1be0f061f349d7e93a8fb49ce2c72c08dd",
                "git_mode": "100644",
                "manifest_hash": "100ae22658c5dda3351761f3ea09db406ac47377c619fb39803f70af8646a3b5",
                "path": "results/gross9_structural_clock_bundle_g9cb2_attempt_consumed_2026-07-31.json",
                "path_type": "regular_file",
                "protocol_version": "gross9_structural_clock_bundle_g9cb2_v1",
                "resume_allowed": False,
                "seal_commit": G9CB2_TERMINAL_EVIDENCE_COMMIT,
                "sha256": "3bfb5c3259d398b4b16e029d180c72390fd3f835acffa1010fac7b9c40eeac83",
                "size_bytes": 3288,
                "status": "attempt_consumed_before_runtime_or_value_access",
                "retry_allowed": False,
            },
            "authority_decision": {
                "authority_commit": G9CB2_AUTHORITY_DECISION_COMMIT,
                "git_blob": G9CB2_AUTHORITY_DECISION_GIT_BLOB,
                "git_mode": "100644",
                "path": G9CB2_AUTHORITY_DECISION_PATH.as_posix(),
                "path_type": "regular_file",
                "sha256": G9CB2_AUTHORITY_DECISION_SHA256,
            },
            "classification": (
                "terminal_guarded_worker_git_subprocess_rejected_before_"
                "capability_or_value_access"
            ),
            "failure_counters": {
                "candidate_rows_opened": 0,
                "comparator_clock_rows_opened": 0,
                "generic_runtime_modules_imported": 0,
                "pre2025_anchor_value_rows_opened": 0,
                "source_value_rows_opened": 0,
                "worker_capabilities_consumed": 0,
                "worker_git_children_launched": 0,
                "worker_ledgers_published": 0,
            },
            "identity": "G9CB-2",
            "permanently_absent_outputs": [
                "results/gross9_structural_clock_bundle_g9cb2_worker_capability_consumed_pass1_2026-07-31.json",
                "results/gross9_structural_clock_bundle_g9cb2_worker_capability_consumed_pass2_2026-07-31.json",
                "results/gross9_structural_clock_bundle_g9cb2_2026-07-31.csv.gz",
                "results/gross9_structural_clock_bundle_g9cb2_manifest_2026-07-31.json",
            ],
            "preregistration": {
                "filesystem_mode_octal": "0444",
                "git_blob": "31bd51bdca5cec5da428b9ae3db067635d6d04b2",
                "git_mode": "100644",
                "manifest_hash": "070b0dded30c1ffcf8744c232ddf37ef22985f80546ad8d6ce4d0d3c72b84b0b",
                "path": "results/gross9_structural_clock_bundle_g9cb2_preregistration_2026-07-31.json",
                "path_type": "regular_file",
                "protocol_implementation_commit": G9CB2_PROTOCOL_IMPLEMENTATION_COMMIT,
                "protocol_version": "gross9_structural_clock_bundle_g9cb2_preregistration_v1",
                "seal_commit": G9CB2_PREREGISTRATION_SEAL_COMMIT,
                "sha256": "84cea282bda82270d5c1f10c2606f78ac8fddd40527598c3e2aaafa6089b38ec",
            },
            "protocol_implementation": {
                "commit": G9CB2_PROTOCOL_IMPLEMENTATION_COMMIT,
                "files": [
                    {"git_blob": "ef5b6e0f480fad3fd221e290c4b4f88b75dc4395", "git_mode": "100644", "path": "tests/test_build_gross9_structural_clock_bundle.py", "sha256": "d50cbcd24ac0af74a5c531161766ca99662af0d24e2f3c5ad4ae364d84165c33"},
                    {"git_blob": "03fc8aa8dd22bcd29fd6e8d51ad5d040bba1ce95", "git_mode": "100644", "path": "tests/test_gross9_structural_clock_bundle_preregistration_artifact.py", "sha256": "d50ad1eed00e0b1f8e55c09998dcfb576548885aaf5ec8b4650797fa20684700"},
                    {"git_blob": "d199ca4395cf3653860ea65b21a3ed573ee1b092", "git_mode": "100644", "path": "tests/test_preregister_gross9_structural_clock_bundle.py", "sha256": "21ef33be2aa1654b6a34d5771d57c829cc712d8ad0dc6178e81dd8c034042bb6"},
                    {"git_blob": "488a405fd39092c288a34b84a3a27811968f7050", "git_mode": "100644", "path": "training/build_gross9_structural_clock_bundle.py", "sha256": "48a94fe63ae1aeb040bef7fa632d87522d97550d754c112e069776a8b6692132"},
                    {"git_blob": "641db72d2c5f395147a844b38c27427c691a2d8d", "git_mode": "100644", "path": "training/preregister_gross9_structural_clock_bundle.py", "sha256": "07496f055b2e8ce2cdadd237f21eee0794679595ec82f4330bca1e82188c55bf"},
                ],
                "parent_commit": G9CB2_AUTHORITY_DECISION_COMMIT,
            },
            "protocol_version": "gross9_structural_clock_bundle_g9cb2_v1",
            "residue": {
                "slot1_stage": {
                    "committed": False,
                    "filesystem_mode_octal": "0700",
                    "path": "results/.gross9-structural-clock-worker-ca9ca670ffb0d1b377ed6aef",
                    "state": "empty_directory",
                },
                "slot2_stage": {
                    "committed": False,
                    "path": "results/.gross9-structural-clock-worker-2c9f266762f8864bf5e24691",
                    "state": "absent",
                },
            },
            "status": "historical_terminal_attempt_consumed_no_clock_authority",
            "topology": {
                "g9cb2_authority_commit": G9CB2_AUTHORITY_DECISION_COMMIT,
                "g9cb2_claim_commit": G9CB2_CLAIM_COMMIT,
                "g9cb2_preregistration_commit": G9CB2_PREREGISTRATION_SEAL_COMMIT,
                "g9cb2_protocol_commit": G9CB2_PROTOCOL_IMPLEMENTATION_COMMIT,
                "g9cb3_authority_commit": G9CB3_AUTHORITY_DECISION_COMMIT,
                "terminal_evidence_commit": G9CB2_TERMINAL_EVIDENCE_COMMIT,
            },
        },
        {
            "access_claim": {
                "claim_hash": "886fa3b177cd6d48c630020c2afef7d85b05126fd3805676bffdbec2899f2c6a",
                "filesystem_mode_octal": "0444",
                "git_blob": "e4464c15f103c760a89ada18b0e49f7a874d8578",
                "git_mode": "100644",
                "path": "results/gross9_structural_clock_bundle_g9cb3_access_claim_2026-07-31.json",
                "path_type": "regular_file",
                "protocol_parent_commit": G9CB3_PREREGISTRATION_SEAL_COMMIT,
                "seal_commit": G9CB3_CLAIM_COMMIT,
                "sha256": "6d59505ef14ce92bf5958d32725b7e5e1ffe8ce4065f215fdd092176a44bbe98",
            },
            "authority_decision": {
                "authority_commit": G9CB3_AUTHORITY_DECISION_COMMIT,
                "git_blob": G9CB3_AUTHORITY_DECISION_GIT_BLOB,
                "git_mode": "100644",
                "path": G9CB3_AUTHORITY_DECISION_PATH.as_posix(),
                "path_type": "regular_file",
                "sha256": G9CB3_AUTHORITY_DECISION_SHA256,
            },
            "classification": (
                "terminal_missing_physical_domain_end_value_row_after_pass1_"
                "source_decode_before_features_or_model_access"
            ),
            "diagnostic": {
                "elapsed_seconds_approximate": "59.35",
                "max_rss_kb": 1352228,
                "parent_terminal_message": (
                    "fresh worker failed with PID 2087327 and status 1"
                ),
                "status": 1,
                "stderr_sha256": "6c3f838c364acda081f576f3d379c05c71ff8cce3c8424bb8de5c59d9e7e8d88",
                "stderr_size_bytes": 2894,
                "stdout_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "stdout_size_bytes": 0,
                "worker_terminal_message": (
                    "bound market lacks the canonical domain-end boundary row"
                ),
            },
            "exposure": {
                "candidate_rows_opened": 0,
                "comparator_clock_rows_opened": 0,
                "decoded_and_handed_off": [
                    "market",
                    "funding",
                    "premium",
                    "open_interest",
                ],
                "exact_decoded_and_handoff_counts_recoverable": False,
                "features_constructed": False,
                "isolated_runtime_roots_imported": 2,
                "overlap_metric_values_computed": 0,
                "portfolio_economic_values_computed": 0,
                "pre2025_anchor_value_rows_opened": 0,
                "rank7_model_or_history_opened": False,
                "rex_jsonl_opened": False,
                "schedules_reached": False,
                "sleeve_intervals_reached": False,
                "worker_capabilities_consumed": {"pass1": 1, "pass2": 0},
                "worker_ledgers_published": {"pass1": 1, "pass2": 0},
            },
            "identity": "G9CB-3",
            "permanently_absent_outputs": [
                "results/gross9_structural_clock_bundle_g9cb3_worker_capability_consumed_pass2_2026-07-31.json",
                "results/gross9_structural_clock_bundle_g9cb3_2026-07-31.csv.gz",
                "results/gross9_structural_clock_bundle_g9cb3_manifest_2026-07-31.json",
            ],
            "preregistration": {
                "filesystem_mode_octal": "0444",
                "git_blob": "e4b2488783f0cfa88a8478b3ff0d94c99fbece6e",
                "git_mode": "100644",
                "manifest_hash": "e100e54557a9654923374dec433feff494d85b0956129157231c2fb8551ec5f2",
                "path": "results/gross9_structural_clock_bundle_g9cb3_preregistration_2026-07-31.json",
                "path_type": "regular_file",
                "protocol_implementation_commit": G9CB3_PROTOCOL_IMPLEMENTATION_COMMIT,
                "protocol_version": "gross9_structural_clock_bundle_g9cb3_preregistration_v1",
                "seal_commit": G9CB3_PREREGISTRATION_SEAL_COMMIT,
                "sha256": "1d967a590e50822dd83c20838c8796523d6594f0fcbd22aad137b0a97c1a982c",
            },
            "protocol_implementation": {
                "commit": G9CB3_PROTOCOL_IMPLEMENTATION_COMMIT,
                "files": [
                    {"git_blob": "ce6fec59bd8c81dcef4a1be8c04d0b6e8457aa2c", "git_mode": "100644", "path": "tests/test_build_gross9_structural_clock_bundle.py", "sha256": "b59b808b3fcd55aac3278f6914e3022d91688880bb605e8fe635543028438a77"},
                    {"git_blob": "3158c711293fdb110f0a27b960aa0fa060eda9f6", "git_mode": "100644", "path": "tests/test_gross9_structural_clock_bundle_preregistration_artifact.py", "sha256": "19add878f1171aa9d22f18f7547cb2d0274d8c90666bcd812d7c7e1c40769a8c"},
                    {"git_blob": "e63a1578f434d716adc260c7d26c433ce8c5dcab", "git_mode": "100644", "path": "tests/test_preregister_gross9_structural_clock_bundle.py", "sha256": "6487f6f4601e8e137846c9504373e2e76247d321c1e5b24e1e2340e1b151e67a"},
                    {"git_blob": "0e5b96802affa0ab221f75ac1387fd84fec2867a", "git_mode": "100644", "path": "training/build_gross9_structural_clock_bundle.py", "sha256": "50cece625b4af12f81842ba501d376ed0c50589d33187b54efb2139194000e27"},
                    {"git_blob": "a11e63d31e37ab391051bb456091401577f43b2f", "git_mode": "100644", "path": "training/preregister_gross9_structural_clock_bundle.py", "sha256": "71415da68e4a0a37cc2ff1d562f9014f74c297f62124cacba43cdf4ab375b113"},
                ],
                "parent_commit": G9CB2_TERMINAL_EVIDENCE_COMMIT,
            },
            "protocol_version": "gross9_structural_clock_bundle_g9cb3_v1",
            "residue": {
                "bytecode_cache": {
                    "path": "results/.g9cb3-bytecode-cache-disabled",
                    "state": "absent",
                },
                "slot1_stage": {
                    "committed": False,
                    "filesystem_mode_octal": "0700",
                    "path": "results/.gross9-structural-clock-g9cb3-worker-a3dffd3cbec3afd582638a23",
                    "staged_core_state": "absent",
                    "staged_csv_state": "absent",
                    "staged_receipt_state": "absent",
                    "state": "empty_directory",
                },
                "slot2_stage": {
                    "committed": False,
                    "path": "results/.gross9-structural-clock-g9cb3-worker-26e64bf0a62646afad3d77e6",
                    "state": "absent",
                },
            },
            "root_cause": {
                "domain_end": "2026-06-01T00:00:00Z",
                "domain_end_is_exclusive_boundary": True,
                "fabricated_boundary_value_authorized": False,
                "open_interest_left_merge_can_remove_market_rows": False,
                "source_hash_mismatch_at_authentication_ruled_out": True,
                "wrong_requirement": (
                    "exclusive domain-end boundary required as physical market "
                    "value-row open"
                ),
            },
            "status": "historical_terminal_attempt_consumed_no_clock_authority",
            "terminal_evidence": {
                "attempt_sentinel": {
                    "claim_commit": G9CB3_CLAIM_COMMIT,
                    "filesystem_mode_octal": "0444",
                    "git_blob": "672a973ae09495d38253991df90c7962c6f9c2ee",
                    "git_mode": "100644",
                    "manifest_hash": "aab875ecc91a07619b3006967efca42664cc77d90d0bb9e45da20e93ea16c973",
                    "path": "results/gross9_structural_clock_bundle_g9cb3_attempt_consumed_2026-07-31.json",
                    "path_type": "regular_file",
                    "protocol_version": "gross9_structural_clock_bundle_g9cb3_v1",
                    "resume_allowed": False,
                    "retry_allowed": False,
                    "seal_commit": G9CB3_TERMINAL_EVIDENCE_COMMIT,
                    "sha256": "7cec9807503c271e993f8bf456ca772d7ded9cec9619d37cbd4ac015d14c9f69",
                    "size_bytes": 3300,
                    "status": "attempt_consumed_before_runtime_or_value_access",
                },
                "pass1_worker_ledger": {
                    "filesystem_mode_octal": "0444",
                    "git_blob": "0e77d77fcdc70379d94a2c34e8dee19f9f45d0a9",
                    "git_mode": "100644",
                    "path": "results/gross9_structural_clock_bundle_g9cb3_worker_capability_consumed_pass1_2026-07-31.json",
                    "path_type": "regular_file",
                    "seal_commit": G9CB3_TERMINAL_EVIDENCE_COMMIT,
                    "sha256": "45b27b24074c53eeaaa9a1074c7888c22c4078e5fb5922007d31ad0baf704f66",
                    "size_bytes": 1766,
                    "slot": 1,
                    "status": "consumed_before_runtime_or_value_access",
                },
            },
            "topology": {
                "g9cb3_authority_commit": G9CB3_AUTHORITY_DECISION_COMMIT,
                "g9cb3_claim_commit": G9CB3_CLAIM_COMMIT,
                "g9cb3_preregistration_commit": G9CB3_PREREGISTRATION_SEAL_COMMIT,
                "g9cb3_protocol_commit": G9CB3_PROTOCOL_IMPLEMENTATION_COMMIT,
                "g9cb4_authority_commit": G9CB4_AUTHORITY_DECISION_COMMIT,
                "terminal_evidence_commit": G9CB3_TERMINAL_EVIDENCE_COMMIT,
            },
        },
    ]


def expected_failed_predecessor_closures() -> list[dict[str, Any]]:
    return [
        {
            "authority_decision": {
                "authority_commit": G9CB4_AUTHORITY_DECISION_COMMIT,
                "git_blob": G9CB4_AUTHORITY_DECISION_GIT_BLOB,
                "git_mode": "100644",
                "path": G9CB4_AUTHORITY_DECISION_PATH.as_posix(),
                "path_type": "regular_file",
                "sha256": G9CB4_AUTHORITY_DECISION_SHA256,
            },
            "classification": (
                "pre_access_claim_pre_sentinel_keyword_only_call_contract_failure"
            ),
            "exposure": {
                "bindings_authenticated": 63,
                "candidate_rows_opened": 0,
                "claim_files_published": 0,
                "economics_or_overlap_computed": False,
                "features_constructed": False,
                "gzip_csv_jsonl_npz_values_decoded_or_loaded": False,
                "historical_metadata_json_decoded": True,
                "model_or_history_values_opened": 0,
                "pre2025_anchor_and_rank7_manifest_bytes": 14680,
                "production_invocations": None,
                "rank7_history_and_model_bytes": 2121609,
                "runtime_python_ast_parsed": True,
                "schedules_reached": False,
                "source_files": 8,
                "source_files_bytes": 100551601,
                "source_values_opened": 0,
                "unique_paths_authenticated": 55,
                "worker_capabilities_consumed": 0,
                "workers_started": 0,
            },
            "failure": {
                "claim_payload_constructed": False,
                "claim_write_attempted": False,
                "exception": (
                    "TypeError: _validate_git_pair_preflight() takes 2 "
                    "positional arguments but 4 positional arguments (and 1 "
                    "keyword-only argument) were given"
                ),
                "official_production_invocations": None,
                "raw_capture_recoverable": False,
                "sentinel_published": False,
            },
            "identity": "G9CB-4",
            "permanently_absent_outputs": sorted(
                [
                    "results/gross9_structural_clock_bundle_g9cb4_2026-07-31.csv.gz",
                    "results/gross9_structural_clock_bundle_g9cb4_access_claim_2026-07-31.json",
                    "results/gross9_structural_clock_bundle_g9cb4_attempt_consumed_2026-07-31.json",
                    "results/gross9_structural_clock_bundle_g9cb4_manifest_2026-07-31.json",
                    "results/gross9_structural_clock_bundle_g9cb4_worker_capability_consumed_pass1_2026-07-31.json",
                    "results/gross9_structural_clock_bundle_g9cb4_worker_capability_consumed_pass2_2026-07-31.json",
                ]
            ),
            "preregistration": {
                "filesystem_mode_octal": "0444",
                "git_blob": "76f9011d5752282c058feb531442b203a0bbdb0d",
                "git_mode": "100644",
                "manifest_hash": (
                    "fa3dab6f7e6ab86428c03fc5c3d7b005e0a165cd76662bba9a7c3cd5941beeed"
                ),
                "path": (
                    "results/gross9_structural_clock_bundle_g9cb4_"
                    "preregistration_2026-07-31.json"
                ),
                "path_type": "regular_file",
                "protocol_implementation_commit": (
                    G9CB4_PROTOCOL_IMPLEMENTATION_COMMIT
                ),
                "protocol_version": (
                    "gross9_structural_clock_bundle_g9cb4_preregistration_v1"
                ),
                "seal_commit": G9CB4_PREREGISTRATION_SEAL_COMMIT,
                "sha256": (
                    "f65aaf5fd2219f90421912e6fc9065ddffb54f5adf881196986f25185fe7342e"
                ),
                "size_bytes": 41289,
            },
            "protocol_implementation": {
                "builder_path": (
                    "training/build_gross9_structural_clock_bundle.py"
                ),
                "builder_sha256": (
                    "c7c3bf1f9971e058e719139b50379c356f45a0fcc8f62c12aab100f70fa64c63"
                ),
                "commit": G9CB4_PROTOCOL_IMPLEMENTATION_COMMIT,
            },
            "protocol_version": "gross9_structural_clock_bundle_g9cb4_v1",
            "residue": {
                "bytecode_cache": {
                    "path": "results/.g9cb4-bytecode-cache-disabled",
                    "state": "absent",
                },
                "publication_stages": {
                    "glob": (
                        "results/.gross9_structural_clock_bundle_g9cb4_*.stage-*"
                    ),
                    "state": "absent",
                },
                "worker_stages": {
                    "glob": (
                        "results/.gross9-structural-clock-g9cb4-worker-*"
                    ),
                    "state": "absent",
                },
            },
            "status": (
                "historical_pre_access_pre_sentinel_closure_no_attempt_"
                "no_clock_authority"
            ),
            "topology": {
                "g9cb4_authority_commit": G9CB4_AUTHORITY_DECISION_COMMIT,
                "g9cb4_preregistration_commit": (
                    G9CB4_PREREGISTRATION_SEAL_COMMIT
                ),
                "g9cb4_protocol_commit": G9CB4_PROTOCOL_IMPLEMENTATION_COMMIT,
                "g9cb5_authority_commit": G9CB5_AUTHORITY_DECISION_COMMIT,
                "terminal_evidence_commit": None,
            },
        }
    ]


def expected_failed_predecessor_prepublication_closures() -> list[dict[str, Any]]:
    rows = [{'authority_decision': {'authority_commit': '1ca718d9dab1077b041e753f3b011fbf5b23f047',
                             'git_blob': 'e0bb4b1d26a67c4baf681d8a48e988307c92f9f5',
                             'git_mode': '100644',
                             'path': 'docs/gross9-structural-clock-bundle-g9cb5-successor-authority-decision-2026-07-31.md',
                             'path_type': 'regular_file',
                             'sha256': 'd0b2e14417b4cd46213708597220067c2195d22308da9eb95921bcb59da27385'},
      'classification': 'pre_preregistration_publication_missing_runtime_input_bootstrap_failure',
      'failure': {'bytes_opened': 21801778,
                  'exception': 'FileNotFoundError: [Errno 2] No such file or directory: '
                               "'binance_um_aux_btc_2020_2026'",
                  'exit_status': 1,
                  'manifest_constructed': False,
                  'normalized_invocation': 'PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv '
                                           'run python -B -m '
                                           'training.preregister_gross9_structural_clock_bundle',
                  'observed_preregistration_invocations': 2,
                  'official_production_invocations': None,
                  'paths_opened': 16,
                  'preregistration_published': False,
                  'publication_capability_probe_started': False,
                  'snapshot_final_recheck_completed': False,
                  'status': 'authorized_first_invocation_closed_identity'},
      'identity': 'G9CB-5',
      'permanently_absent_outputs': ['results/gross9_structural_clock_bundle_g9cb5_2026-07-31.csv.gz',
                                     'results/gross9_structural_clock_bundle_g9cb5_access_claim_2026-07-31.json',
                                     'results/gross9_structural_clock_bundle_g9cb5_attempt_consumed_2026-07-31.json',
                                     'results/gross9_structural_clock_bundle_g9cb5_manifest_2026-07-31.json',
                                     'results/gross9_structural_clock_bundle_g9cb5_preregistration_2026-07-31.json',
                                     'results/gross9_structural_clock_bundle_g9cb5_worker_capability_consumed_pass1_2026-07-31.json',
                                     'results/gross9_structural_clock_bundle_g9cb5_worker_capability_consumed_pass2_2026-07-31.json'],
      'post_closure_incident': {'bytes_opened': 105499876,
                                'exception': 'FileExistsError: Q5 exact results inventory '
                                             'differs',
                                'exit_status': 1,
                                'manifest_constructed': True,
                                'metadata_json_decoded': True,
                                'normalized_invocation': 'PYTHONPATH=$PWD '
                                                         'PYTHONDONTWRITEBYTECODE=1 uv run '
                                                         'python -B -m '
                                                         'training.preregister_gross9_structural_clock_bundle',
                                'paths_opened': 57,
                                'preregistration_published': False,
                                'publication_capability_probe_started': False,
                                'runtime_python_ast_parsed': True,
                                'snapshot_final_recheck_completed': False,
                                'source_model_or_history_values_decoded_or_loaded': False,
                                'status': 'unauthorized_post_closure_invocation_no_publication',
                                'unauthorized_after_closure': True},
      'protocol_implementation': {'builder_git_blob': '8af92fbdf7200b2e67275d9b41d3e40ebc1449a8',
                                  'builder_path': 'training/build_gross9_structural_clock_bundle.py',
                                  'builder_sha256': 'd7edaa3277b581c675f81b2364421d862c1897e89cc149335d912753bb182802',
                                  'commit': '02c3c83a5253684057f44f51ee96bcb089b40b2f',
                                  'preregistration_git_blob': '1f74ddbb8fa019884f674466a29cf0bfb5ec9af1',
                                  'preregistration_path': 'training/preregister_gross9_structural_clock_bundle.py',
                                  'preregistration_sha256': '2c989f97f8046154d8a479d541c1d4b3cb8f70ab1394d2e610fc203207854e1f'},
      'protocol_version': 'gross9_structural_clock_bundle_g9cb5_v1',
      'recovery_exposure': {'candidate_or_economic_metric_computed': False,
                            'frozen_open_interest_reconstruction': {'occurred_before_first_invocation': True,
                                                                    'output': {'columns': ['date',
                                                                                           'open_interest'],
                                                                               'path': '/tmp/btcusdt_open_interest_5m_2020_2026.csv',
                                                                               'rows': 674785,
                                                                               'sha256': 'e08f93033e56959e8e7a9c1e21f27c5f01efc8d06fa6b4fbbfe7354697122b31',
                                                                               'size_bytes': 19657777},
                                                                    'pandas_version': '2.3.3',
                                                                    'selected_columns': ['date',
                                                                                         'open_interest'],
                                                                    'source': {'data_rows_traversed': 674785,
                                                                               'gzip_stream_decompressed': True,
                                                                               'header': ['date',
                                                                                          'open',
                                                                                          'high',
                                                                                          'low',
                                                                                          'close',
                                                                                          'volume',
                                                                                          'quote_asset_volume',
                                                                                          'number_of_trades',
                                                                                          'taker_buy_base',
                                                                                          'taker_buy_quote',
                                                                                          'tic',
                                                                                          'day',
                                                                                          'dxy',
                                                                                          'kimchi_premium',
                                                                                          'usdkrw',
                                                                                          'btckrw',
                                                                                          'dxy_available',
                                                                                          'kimchi_available',
                                                                                          'usdkrw_available',
                                                                                          'external_any_available',
                                                                                          'dxy_zscore',
                                                                                          'dxy_momentum',
                                                                                          'kimchi_premium_zscore',
                                                                                          'kimchi_premium_change',
                                                                                          'usdkrw_zscore',
                                                                                          'usdkrw_momentum',
                                                                                          'open_interest',
                                                                                          'open_interest_value',
                                                                                          'cmc_circulating_supply',
                                                                                          'open_interest_available'],
                                                                               'header_columns_decoded': 30,
                                                                               'logical_path': 'data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz',
                                                                               'resolved_path': '/home/pakchu/rllm/data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz',
                                                                               'sha256': 'dbc9e53b09551b469168fe19cc750c5c3ea86278db3055d079103f7654050192',
                                                                               'size_bytes': 72898508},
                                                                    'transform': 'pd.read_csv(source, '
                                                                                 "usecols=['date', "
                                                                                 "'open_interest']).to_csv(target, "
                                                                                 'index=False)'},
                            'historical_filesystem_state': {'empty_mode_0700_directories': ['results/.gross9-structural-clock-g9cb3-worker-a3dffd3cbec3afd582638a23',
                                                                                            'results/.gross9-structural-clock-worker-ca9ca670ffb0d1b377ed6aef'],
                                                            'mode_0444_files': ['results/gross9_structural_clock_bundle_g9cb2_access_claim_2026-07-31.json',
                                                                                'results/gross9_structural_clock_bundle_g9cb2_attempt_consumed_2026-07-31.json',
                                                                                'results/gross9_structural_clock_bundle_g9cb2_preregistration_2026-07-31.json',
                                                                                'results/gross9_structural_clock_bundle_g9cb3_access_claim_2026-07-31.json',
                                                                                'results/gross9_structural_clock_bundle_g9cb3_attempt_consumed_2026-07-31.json',
                                                                                'results/gross9_structural_clock_bundle_g9cb3_preregistration_2026-07-31.json',
                                                                                'results/gross9_structural_clock_bundle_g9cb3_worker_capability_consumed_pass1_2026-07-31.json',
                                                                                'results/gross9_structural_clock_bundle_g9cb4_preregistration_2026-07-31.json',
                                                                                'results/gross9_structural_clock_bundle_preregistration_2026-07-31.json',
                                                                                'results/gross9_structural_clock_bundle_preregistration_v2_2026-07-31.json'],
                                                            'tracked_bytes_changed': False},
                            'opaque_regular_file_restoration_destination_root': '/tmp/rllm-alpha-orthogonal-20260718',
                            'opaque_regular_file_restoration_source_root': '/home/pakchu/rllm',
                            'opaque_regular_file_restorations': [{'path': 'data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz',
                                                                  'sha256': '4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7',
                                                                  'size_bytes': 89326},
                                                                 {'path': 'data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz',
                                                                  'sha256': 'b45fcc5a3cf75c8e594effe61a698c4652f841b1d304107e9669524e0fc9d0d7',
                                                                  'size_bytes': 1196481},
                                                                 {'path': 'data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz',
                                                                  'sha256': 'a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c',
                                                                  'size_bytes': 66696659},
                                                                 {'path': 'data/rex_pullback_reclaim_q075_h144_ranker_eval_2025_2026h1.jsonl',
                                                                  'sha256': 'bbe13d845d8dffcbb3e6c9b0f348390bd9d089c2d7b7bd6bccbafb91e75d9ce7',
                                                                  'size_bytes': 1029745},
                                                                 {'path': 'data/rex_pullback_reclaim_q075_h144_ranker_test_2024.jsonl',
                                                                  'sha256': 'b1f5abf59c901ac109823a50063665ef455e75e70e90135acda77755ab8e5371',
                                                                  'size_bytes': 1253048},
                                                                 {'path': 'data/rex_pullback_reclaim_q075_h144_ranker_train_2021_2023.jsonl',
                                                                  'sha256': '07f6c4bb43ac92b341ce1a1b54ea6a429983611000148ad6966b81ea4a086df0',
                                                                  'size_bytes': 6128620}]},
      'residue': {'bytecode_cache': {'path': 'results/.g9cb5-bytecode-cache-disabled',
                                     'state': 'absent'},
                  'publication_stages': {'glob': 'results/.gross9_structural_clock_bundle_g9cb5_*.stage-*',
                                         'state': 'absent'},
                  'worker_stages': {'glob': 'results/.gross9-structural-clock-g9cb5-worker-*',
                                    'state': 'absent'}},
      'status': 'historical_prepublication_closure_no_preregistration_no_attempt_no_clock_authority',
      'topology': {'g9cb5_authority_commit': '1ca718d9dab1077b041e753f3b011fbf5b23f047',
                   'g9cb5_protocol_commit': '02c3c83a5253684057f44f51ee96bcb089b40b2f',
                   'g9cb6_authority_commit': '2695ee61fbb9b5e053dbb9da597ebe2729aad361',
                   'preregistration_commit': None,
                   'terminal_evidence_commit': None}}]
    rows.append(
        {
            "authority_decision": {
                "authority_commit": G9CB6_AUTHORITY_DECISION_COMMIT,
                "git_blob": G9CB6_AUTHORITY_DECISION_GIT_BLOB,
                "git_mode": "100644",
                "path": G9CB6_AUTHORITY_DECISION_PATH.as_posix(),
                "path_type": "regular_file",
                "sha256": G9CB6_AUTHORITY_DECISION_SHA256,
            },
            "classification": (
                "pre_preregistration_publication_bootstrap_manifest_"
                "bound_path_set_mismatch"
            ),
            "failure": {
                "bytes_opened": 105_571_805,
                "exception": (
                    "ValueError: manifest bound path set differs from bootstrap"
                ),
                "exit_status": 1,
                "manifest_constructed": True,
                "metadata_json_decoded": True,
                "normalized_invocation": (
                    "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run "
                    "python -B -m "
                    "training.preregister_gross9_structural_clock_bundle"
                ),
                "observed_preregistration_invocations": 1,
                "official_production_invocations": None,
                "paths_opened": 58,
                "preregistration_published": False,
                "publication_capability_probe_started": False,
                "runtime_python_ast_parsed": True,
                "snapshot_final_recheck_completed": False,
                "source_model_or_history_values_decoded_or_loaded": False,
                "status": "authorized_first_invocation_closed_identity",
            },
            "identity": "G9CB-6",
            "input_materialization": {
                "authority_order": "after_clean_pushed_A7_before_Q7",
                "destination": {
                    "git_blob": None,
                    "git_mode": None,
                    "mode_octal": "0444",
                    "path": FROZEN_OPEN_INTEREST_GZIP_PATH.as_posix(),
                    "path_type": "regular_file",
                    "sha256": FROZEN_OPEN_INTEREST_GZIP_SHA256,
                    "size_bytes": FROZEN_OPEN_INTEREST_GZIP_SIZE,
                },
                "source": {
                    "absolute_path": (
                        "/home/pakchu/rllm/data/"
                        "cache_market_ext_5m_wavefull_2020-01-01_"
                        "2026-06-01_oi.csv.gz"
                    ),
                    "expected_sha256": FROZEN_OPEN_INTEREST_GZIP_SHA256,
                    "path_type": "regular_file",
                    "size_bytes": FROZEN_OPEN_INTEREST_GZIP_SIZE,
                },
                "source_values_decoded": False,
                "status": (
                    "opaque_byte_identical_symlink_replaced_by_regular_file"
                ),
            },
            "permanently_absent_outputs": [
                "results/gross9_structural_clock_bundle_g9cb6_2026-07-31.csv.gz",
                "results/gross9_structural_clock_bundle_g9cb6_access_claim_2026-07-31.json",
                "results/gross9_structural_clock_bundle_g9cb6_attempt_consumed_2026-07-31.json",
                "results/gross9_structural_clock_bundle_g9cb6_manifest_2026-07-31.json",
                "results/gross9_structural_clock_bundle_g9cb6_preregistration_2026-07-31.json",
                "results/gross9_structural_clock_bundle_g9cb6_worker_capability_consumed_pass1_2026-07-31.json",
                "results/gross9_structural_clock_bundle_g9cb6_worker_capability_consumed_pass2_2026-07-31.json",
            ],
            "protocol_implementation": {
                "builder_git_blob": G9CB6_BUILDER_GIT_BLOB,
                "builder_path": BUILDER_SOURCE.as_posix(),
                "builder_sha256": G9CB6_BUILDER_SHA256,
                "commit": G9CB6_PROTOCOL_IMPLEMENTATION_COMMIT,
                "preregistration_git_blob": G9CB6_PREREGISTRATION_GIT_BLOB,
                "preregistration_path": PREREGISTRATION_SOURCE.as_posix(),
                "preregistration_sha256": G9CB6_PREREGISTRATION_SHA256,
            },
            "protocol_version": "gross9_structural_clock_bundle_g9cb6_v1",
            "residue": {
                "bytecode_cache": {
                    "path": "results/.g9cb6-bytecode-cache-disabled",
                    "state": "absent",
                },
                "capability_probes": {
                    "glob": "results/.g9cb6-otmpfile-probe-*",
                    "state": "absent",
                },
                "publication_stages": {
                    "glob": (
                        "results/.gross9_structural_clock_bundle_g9cb6_"
                        "*.stage-*"
                    ),
                    "state": "absent",
                },
                "worker_stages": {
                    "glob": (
                        "results/.gross9-structural-clock-g9cb6-worker-*"
                    ),
                    "state": "absent",
                },
            },
            "root_cause": {
                "bootstrap_bound_path_count": 58,
                "bootstrap_missing_container": (
                    "failed_prepublication_closures"
                ),
                "bootstrap_minus_manifest": [],
                "manifest_bound_path_count": 59,
                "manifest_minus_bootstrap": [
                    {
                        "path": FROZEN_OPEN_INTEREST_GZIP_PATH.as_posix(),
                        "sha256": FROZEN_OPEN_INTEREST_GZIP_SHA256,
                        "size_bytes": FROZEN_OPEN_INTEREST_GZIP_SIZE,
                    }
                ],
                "publication_state_validation_started": False,
                "set_comparison_location": (
                    "write_once_retained_snapshot_before_results_parent_lookup"
                ),
            },
            "status": (
                "historical_prepublication_closure_no_preregistration_"
                "no_attempt_no_clock_authority"
            ),
            "topology": {
                "g9cb6_authority_commit": G9CB6_AUTHORITY_DECISION_COMMIT,
                "g9cb6_protocol_commit": G9CB6_PROTOCOL_IMPLEMENTATION_COMMIT,
                "g9cb7_authority_commit": AUTHORITY_DECISION_COMMIT,
                "preregistration_commit": None,
                "terminal_evidence_commit": None,
            },
        }
    )
    return rows


def expected_successor_preregistration_bindings() -> list[dict[str, Any]]:
    attempts = expected_failed_predecessor_attempts()
    closure = expected_failed_predecessor_closures()[0]
    return [
        {
            "identity": attempts[0]["identity"],
            "preregistration": attempts[0]["preregistration"],
        },
        {
            "identity": attempts[1]["identity"],
            "preregistration": attempts[1]["preregistration"],
        },
        {
            "identity": closure["identity"],
            "preregistration": closure["preregistration"],
        },
    ]


def _predecessor_state_bindings(
    repository_root: Path, *, authenticate: bool
) -> dict[str, Any]:
    return {
        "failed_predecessor_attempts": (
            validate_failed_predecessor_attempts(repository_root)
            if authenticate
            else expected_failed_predecessor_attempts()
        ),
        "failed_predecessor_closures": (
            validate_failed_predecessor_closures(repository_root)
            if authenticate
            else expected_failed_predecessor_closures()
        ),
        "failed_predecessor_prepublication_closures": (
            validate_failed_predecessor_prepublication_closures(
                repository_root
            )
            if authenticate
            else expected_failed_predecessor_prepublication_closures()
        ),
        "successor_preregistrations": (
            expected_successor_preregistration_bindings()
        ),
    }


def validate_failed_predecessor_closures(
    repository_root: Path = REPOSITORY_ROOT,
) -> list[dict[str, Any]]:
    closures = expected_failed_predecessor_closures()
    row = closures[0]
    binding = row["preregistration"]
    path_text = str(binding["path"])
    if _ACTIVE_PREREGISTRATION_GIT_PAIRS is None:
        staged = _git_result(
            ["ls-files", "--stage", "--", path_text], repository_root
        )
        tree = _git_result(["ls-tree", "HEAD", "--", path_text], repository_root)
        matched = _git_result(
            ["ls-files", "--error-unmatch", "--", path_text], repository_root
        )
        _historical_attempt_git_pair(binding, staged, tree, matched)
    elif _ACTIVE_PREREGISTRATION_GIT_PAIRS.get(path_text) != (
        binding["git_blob"],
        binding["git_mode"],
    ):
        raise ValueError("G9CB-4 closed Git pair differs")
    cached = _cached_file(path_text, repository_root)
    if cached is None:
        raw, info = _read_no_follow_once(
            repository_path(path_text, repository_root)
        )
    else:
        raw, info = cached
    if (
        hashlib.sha256(raw).hexdigest() != binding["sha256"]
        or hashlib.sha1(
            f"blob {len(raw)}\0".encode("ascii") + raw
        ).hexdigest()
        != binding["git_blob"]
        or info.st_size != binding["size_bytes"]
        or stat.S_IMODE(info.st_mode) != 0o444
    ):
        raise ValueError("G9CB-4 closed preregistration binding differs")
    if _ACTIVE_PREREGISTRATION_SNAPSHOT is not None:
        _validate_predecessor_inventory_from_snapshot(
            _ACTIVE_PREREGISTRATION_SNAPSHOT,
            expected_failed_predecessor_attempts(),
            closures,
            expected_failed_predecessor_prepublication_closures(),
        )
    else:
        for absent in row["permanently_absent_outputs"]:
            if _path_type(repository_path(absent, repository_root)) != "missing":
                raise ValueError(f"reserved G9CB-4 output exists: {absent}")
        residue = row["residue"]
        if _path_type(
            repository_path(residue["bytecode_cache"]["path"], repository_root)
        ) != "missing":
            raise ValueError("G9CB-4 bytecode residue differs")
        results = repository_path("results", repository_root)
        if list(results.glob(".gross9_structural_clock_bundle_g9cb4_*.stage-*")):
            raise ValueError("G9CB-4 publication-stage residue differs")
        if list(results.glob(".gross9-structural-clock-g9cb4-worker-*")):
            raise ValueError("G9CB-4 worker-stage residue differs")
    return closures


def validate_failed_predecessor_prepublication_closures(
    repository_root: Path = REPOSITORY_ROOT,
) -> list[dict[str, Any]]:
    rows = expected_failed_predecessor_prepublication_closures()
    specifications = (
        (
            "G9CB-5",
            G9CB5_AUTHORITY_DECISION_PATH,
            G9CB5_AUTHORITY_DECISION_SHA256,
            G9CB5_AUTHORITY_DECISION_GIT_BLOB,
            G9CB5_PROTOCOL_IMPLEMENTATION_COMMIT,
        ),
        (
            "G9CB-6",
            G9CB6_AUTHORITY_DECISION_PATH,
            G9CB6_AUTHORITY_DECISION_SHA256,
            G9CB6_AUTHORITY_DECISION_GIT_BLOB,
            G9CB6_PROTOCOL_IMPLEMENTATION_COMMIT,
        ),
    )
    if tuple(row.get("identity") for row in rows) != tuple(
        specification[0] for specification in specifications
    ):
        raise ValueError("prepublication closure identity order differs")
    for row, specification in zip(rows, specifications, strict=True):
        identity, authority_path, authority_sha, authority_blob, commit = (
            specification
        )
        authority = row["authority_decision"]
        observed_authority = _tracked_binding(
            authority_path,
            repository_root=repository_root,
            expected_sha256=authority_sha,
            expected_blob=authority_blob,
        )
        expected_authority = {
            key: authority[key]
            for key in (
                "path",
                "path_type",
                "sha256",
                "git_blob",
                "git_mode",
            )
        }
        if observed_authority != expected_authority:
            raise ValueError(f"{identity} authority closure binding differs")
        protocol = row["protocol_implementation"]
        for prefix, path in (
            ("preregistration", PREREGISTRATION_SOURCE),
            ("builder", BUILDER_SOURCE),
        ):
            raw = _run_git_bytes(
                ["show", f"{commit}:{path.as_posix()}"],
                repository_root,
            )
            if (
                hashlib.sha256(raw).hexdigest()
                != protocol[f"{prefix}_sha256"]
                or hashlib.sha1(
                    f"blob {len(raw)}\0".encode("ascii") + raw
                ).hexdigest()
                != protocol[f"{prefix}_git_blob"]
            ):
                raise ValueError(
                    f"{identity} historical {prefix} binding differs"
                )
    materialized = rows[1]["input_materialization"]["destination"]
    materialized_path = str(materialized["path"])
    if (
        _classify_git_pair_only(
            repository_root,
            materialized_path,
            {"git_blob": None, "git_mode": None},
        )
        is not None
    ):
        raise ValueError("G9CB-6 materialized input Git state differs")
    cached = _cached_file(materialized_path, repository_root)
    if cached is None:
        raw, info = _read_no_follow_once(
            repository_path(materialized_path, repository_root)
        )
    else:
        raw, info = cached
    if (
        hashlib.sha256(raw).hexdigest() != materialized["sha256"]
        or info.st_size != materialized["size_bytes"]
        or not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o444
        or info.st_nlink != 1
    ):
        raise ValueError("G9CB-6 materialized input binding differs")
    results_root = repository_root / "results"
    if results_root.exists():
        names = {candidate.name for candidate in results_root.iterdir()}
        for row in rows:
            identity = str(row["identity"])
            suffix = identity.lower().replace("-", "")
            for path_text in row["permanently_absent_outputs"]:
                if Path(path_text).name in names:
                    raise ValueError(
                        f"reserved {identity} output exists: {path_text}"
                    )
            residue = row["residue"]
            if Path(residue["bytecode_cache"]["path"]).name in names:
                raise ValueError(f"{identity} bytecode residue differs")
            if any(
                name.startswith(
                    f".gross9_structural_clock_bundle_{suffix}_"
                )
                and ".stage-" in name
                for name in names
            ):
                raise ValueError(f"{identity} publication-stage residue differs")
            if any(
                name.startswith(f".gross9-structural-clock-{suffix}-worker-")
                for name in names
            ):
                raise ValueError(f"{identity} worker-stage residue differs")
            capability = residue.get("capability_probes")
            if isinstance(capability, Mapping) and any(
                name.startswith(f".{suffix}-otmpfile-probe-") for name in names
            ):
                raise ValueError(f"{identity} capability-probe residue differs")
    return rows


def _historical_attempt_git_pair(
    binding: Mapping[str, Any],
    staged: subprocess.CompletedProcess[str],
    tree: subprocess.CompletedProcess[str],
    matched: subprocess.CompletedProcess[str],
) -> None:
    path_text = str(binding["path"])
    for completed in (staged, tree, matched):
        if completed.returncode != 0:
            raise ValueError(
                f"predecessor Git classification differs: {path_text}"
            )
    staged_lines = staged.stdout.splitlines()
    tree_lines = tree.stdout.splitlines()
    if (
        len(staged_lines) != 1
        or len(tree_lines) != 1
        or matched.stdout.rstrip("\n") != path_text
    ):
        raise ValueError(f"predecessor Git classification differs: {path_text}")
    staged_fields = staged_lines[0].split()
    tree_fields = tree_lines[0].split()
    if (
        len(staged_fields) < 4
        or staged_fields[0] != binding["git_mode"]
        or staged_fields[1] != binding["git_blob"]
        or staged_fields[2] != "0"
        or staged_fields[-1] != path_text
        or len(tree_fields) < 4
        or tree_fields[0] != binding["git_mode"]
        or tree_fields[1] != "blob"
        or tree_fields[2] != binding["git_blob"]
        or tree_fields[-1] != path_text
    ):
        raise ValueError(f"predecessor index/HEAD pair differs: {path_text}")


def _read_no_follow_once(path: Path) -> tuple[bytes, os.stat_result]:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{path}: expected a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if len(raw) != info.st_size:
        raise ValueError(f"{path}: changed during authentication")
    return raw, info


def _validate_failed_attempt_current_files(
    repository_root: Path,
) -> None:
    attempts = expected_failed_predecessor_attempts()
    bindings = [
        attempts[0][key]
        for key in (
            "authority_decision",
            "preregistration",
            "access_claim",
            "attempt_sentinel",
        )
    ]
    bindings.extend(
        attempts[1][key]
        for key in ("authority_decision", "preregistration", "access_claim")
    )
    bindings.extend(attempts[1]["terminal_evidence"].values())
    paths = [str(binding["path"]) for binding in bindings]
    if len(paths) != len(set(paths)):
        raise ValueError("predecessor bound paths are not unique")
    if _ACTIVE_PREREGISTRATION_GIT_PAIRS is None:
        classifications = []
        for binding in bindings:
            path_text = str(binding["path"])
            classifications.append(
                (
                    binding,
                    _git_result(
                        ["ls-files", "--stage", "--", path_text],
                        repository_root,
                    ),
                    _git_result(
                        ["ls-tree", "HEAD", "--", path_text],
                        repository_root,
                    ),
                    _git_result(
                        ["ls-files", "--error-unmatch", "--", path_text],
                        repository_root,
                    ),
                )
            )
        for binding, staged, tree, matched in classifications:
            _historical_attempt_git_pair(binding, staged, tree, matched)
    else:
        for binding in bindings:
            if _ACTIVE_PREREGISTRATION_GIT_PAIRS.get(
                str(binding["path"])
            ) != (binding["git_blob"], binding["git_mode"]):
                raise ValueError(
                    f"predecessor Git pair differs: {binding['path']}"
                )
    observed_objects: set[tuple[int, int]] = set()
    for binding in bindings:
        cached = _cached_file(binding["path"], repository_root)
        if cached is None:
            path = repository_path(binding["path"], repository_root)
            raw, info = _read_no_follow_once(path)
        else:
            raw, info = cached
        object_key = (info.st_dev, info.st_ino)
        if object_key in observed_objects:
            raise ValueError("predecessor bound paths alias one filesystem object")
        observed_objects.add(object_key)
        if (
            hashlib.sha256(raw).hexdigest() != binding["sha256"]
            or hashlib.sha1(
                f"blob {len(raw)}\0".encode("ascii") + raw
            ).hexdigest()
            != binding["git_blob"]
        ):
            raise ValueError(
                f"predecessor byte binding differs: {binding['path']}"
            )
        if (
            "filesystem_mode_octal" in binding
            and stat.S_IMODE(info.st_mode)
            != int(binding["filesystem_mode_octal"], 8)
        ):
            raise ValueError(
                f"predecessor filesystem mode differs: {binding['path']}"
            )
        if "size_bytes" in binding and info.st_size != binding["size_bytes"]:
            raise ValueError(f"predecessor size differs: {binding['path']}")
        if binding in (
            attempts[0]["authority_decision"],
            attempts[1]["authority_decision"],
        ):
            continue
        payload = json.loads(raw)
        if raw != canonical_json_bytes(payload, trailing_lf=True):
            raise ValueError(
                f"predecessor canonical bytes differ: {binding['path']}"
            )
        hash_field = None
        if binding in (attempts[0]["access_claim"], attempts[1]["access_claim"]):
            hash_field = "claim_hash"
        elif binding in (
            attempts[0]["preregistration"],
            attempts[0]["attempt_sentinel"],
            attempts[1]["preregistration"],
            attempts[1]["terminal_evidence"]["attempt_sentinel"],
        ):
            hash_field = "manifest_hash"
        if (
            hash_field is not None
            and canonical_hash(payload, hash_field) != binding[hash_field]
        ):
            raise ValueError(
                f"predecessor internal hash differs: {binding['path']}"
            )
        if binding in attempts[1]["terminal_evidence"].values() and (
            payload.get("identity") != "G9CB-3"
            or payload.get("protocol_version")
            != "gross9_structural_clock_bundle_g9cb3_v1"
        ):
            raise ValueError(
                f"predecessor identity/protocol differs: {binding['path']}"
            )
        for key in (
            "claim_commit",
            "claim_hash",
            "identity",
            "manifest_hash",
            "parent_pid",
            "protocol_version",
            "slot",
            "stage_directory",
            "status",
        ):
            if key in binding and payload.get(key) != binding[key]:
                raise ValueError(
                    f"predecessor embedded binding differs: {binding['path']}"
                )


def validate_failed_predecessor_attempt_history(
    repository_root: Path = REPOSITORY_ROOT,
) -> None:
    for row in expected_failed_predecessor_attempts():
        commit = row["protocol_implementation"]["commit"]
        for file_binding in row["protocol_implementation"]["files"]:
            completed = _git_result(
                ["show", f"{commit}:{file_binding['path']}"],
                repository_root,
            )
            completed.check_returncode()
            output = completed.stdout.encode("utf-8")
            if (
                hashlib.sha256(output).hexdigest() != file_binding["sha256"]
                or hashlib.sha1(
                    f"blob {len(output)}\0".encode("ascii") + output
                ).hexdigest()
                != file_binding["git_blob"]
            ):
                raise ValueError(
                    f"{row['identity']} historical protocol file differs: "
                    f"{file_binding['path']}"
                )
    validate_failed_predecessor_attempt_topology(repository_root)


def validate_failed_predecessor_attempts(
    repository_root: Path = REPOSITORY_ROOT,
) -> list[dict[str, Any]]:
    attempts = expected_failed_predecessor_attempts()
    _validate_failed_attempt_current_files(repository_root)
    if _ACTIVE_PREREGISTRATION_SNAPSHOT is not None:
        _validate_predecessor_inventory_from_snapshot(
            _ACTIVE_PREREGISTRATION_SNAPSHOT,
            attempts,
            expected_failed_predecessor_closures(),
            expected_failed_predecessor_prepublication_closures(),
        )
        validate_failed_predecessor_attempt_history(repository_root)
        return attempts
    for row in attempts:
        for path_text in row["permanently_absent_outputs"]:
            if (
                _path_type(repository_path(path_text, repository_root))
                != "missing"
            ):
                raise ValueError(
                    f"reserved {row['identity']} output exists: {path_text}"
                )
        residue = row["residue"]
        bytecode = residue.get("bytecode_cache")
        if bytecode is not None and _path_type(
            repository_path(bytecode["path"], repository_root)
        ) != "missing":
            raise ValueError(f"{row['identity']} bytecode residue differs")
        slot1_row = residue["slot1_stage"]
        slot1 = repository_path(slot1_row["path"], repository_root)
        if (
            _path_type(slot1) != "directory"
            or stat.S_IMODE(slot1.stat().st_mode) != 0o700
            or any(slot1.iterdir())
        ):
            raise ValueError(f"{row['identity']} slot-1 residue state differs")
        slot2 = repository_path(
            residue["slot2_stage"]["path"], repository_root
        )
        if _path_type(slot2) != "missing":
            raise ValueError(f"{row['identity']} slot-2 residue state differs")
    validate_failed_predecessor_attempt_history(repository_root)
    return attempts


def _validate_predecessor_inventory_from_snapshot(
    snapshot: _PreregistrationSnapshot,
    attempts: Sequence[Mapping[str, Any]],
    closures: Sequence[Mapping[str, Any]],
    prepublication_closures: Sequence[Mapping[str, Any]] = (),
) -> None:
    if not prepublication_closures:
        prepublication_closures = (
            expected_failed_predecessor_prepublication_closures()
        )
    results_fd = snapshot.directories.get(("repo", ("results",)))
    if results_fd is None:
        raise ValueError("retained results descriptor is absent")
    names = set(os.listdir(results_fd))

    def require_absent(path_text: str, message: str) -> None:
        leaf = Path(path_text).name
        try:
            os.stat(leaf, dir_fd=results_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise ValueError(message)

    for row in attempts:
        identity = str(row["identity"])
        for path_text in row["permanently_absent_outputs"]:
            require_absent(
                str(path_text),
                f"reserved {identity} output exists: {path_text}",
            )
        bytecode = row["residue"].get("bytecode_cache")
        if isinstance(bytecode, Mapping):
            require_absent(
                str(bytecode["path"]),
                f"{identity} bytecode residue differs",
            )
        snapshot.retain_empty_directory(
            str(row["residue"]["slot1_stage"]["path"]),
            mode=0o700,
        )
        require_absent(
            str(row["residue"]["slot2_stage"]["path"]),
            f"{identity} slot-2 residue state differs",
        )
    for row in closures:
        for path_text in row["permanently_absent_outputs"]:
            require_absent(
                str(path_text),
                f"reserved G9CB-4 output exists: {path_text}",
            )
        residue = row["residue"]
        require_absent(
            str(residue["bytecode_cache"]["path"]),
            "G9CB-4 bytecode residue differs",
        )
    if any(
        name.startswith(".gross9_structural_clock_bundle_g9cb4_")
        and ".stage-" in name
        for name in names
    ):
        raise ValueError("G9CB-4 publication-stage residue differs")
    if any(
        name.startswith(".gross9-structural-clock-g9cb4-worker-")
        for name in names
    ):
        raise ValueError("G9CB-4 worker-stage residue differs")
    if tuple(row.get("identity") for row in prepublication_closures) != (
        "G9CB-5",
        "G9CB-6",
    ):
        raise ValueError("prepublication closure identity order differs")
    for row in prepublication_closures:
        identity = str(row["identity"])
        for path_text in row["permanently_absent_outputs"]:
            require_absent(
                str(path_text),
                f"reserved {identity} output exists: {path_text}",
            )
        require_absent(
            str(row["residue"]["bytecode_cache"]["path"]),
            f"{identity} bytecode residue differs",
        )
        suffix = identity.lower().replace("-", "")
        if any(
            name.startswith(f".gross9_structural_clock_bundle_{suffix}_")
            and ".stage-" in name
            for name in names
        ):
            raise ValueError(f"{identity} publication-stage residue differs")
        if any(
            name.startswith(f".gross9-structural-clock-{suffix}-worker-")
            for name in names
        ):
            raise ValueError(f"{identity} worker-stage residue differs")
        if isinstance(row["residue"].get("capability_probes"), Mapping) and any(
            name.startswith(f".{suffix}-otmpfile-probe-") for name in names
        ):
            raise ValueError(f"{identity} capability-probe residue differs")


def validate_git_seal(
    repository_root: Path = REPOSITORY_ROOT,
    expected_branch: str = EXPECTED_BRANCH,
) -> dict[str, Any]:
    branch = _run_git(["branch", "--show-current"], repository_root)
    if branch != expected_branch:
        raise ValueError(f"expected branch {expected_branch}, found {branch}")
    upstream_name = _run_git(
        ["rev-parse", "--abbrev-ref", "@{upstream}"], repository_root
    )
    head = _run_git(["rev-parse", "HEAD"], repository_root)
    upstream = _run_git(["rev-parse", "@{upstream}"], repository_root)
    if head != upstream:
        raise ValueError("HEAD does not equal upstream")
    if _run_git(["status", "--porcelain=v1"], repository_root):
        raise ValueError("worktree or index is not clean")
    return {
        "expected_branch": expected_branch,
        "expected_upstream": f"origin/{expected_branch}",
        "required_head_equals_upstream": True,
        "required_worktree_and_index_clean": True,
        "observed_upstream_name": upstream_name,
    }


def normalized_distribution_inventory() -> dict[str, str]:
    inventory: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            raise ValueError("installed distribution has no Name metadata")
        name = re.sub(r"[-_.]+", "-", raw_name).lower()
        version = distribution.version
        previous = inventory.get(name)
        if previous is not None and previous != version:
            raise ValueError(f"conflicting installed versions for {name}")
        inventory[name] = version
    return dict(sorted(inventory.items()))


def environment_inventory() -> dict[str, Any]:
    inventory = normalized_distribution_inventory()
    libc_name, libc_version = platform.libc_ver()
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "libc": f"{libc_name} {libc_version}",
        "zlib_compile": zlib.ZLIB_VERSION,
        "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
        "selected_distributions": {
            name: inventory.get(name, "absent")
            for name in FROZEN_ENVIRONMENT["selected_distributions"]
        },
        "distribution_count": len(inventory),
        "distribution_inventory_sha256": hashlib.sha256(
            canonical_json_bytes(inventory)
        ).hexdigest(),
        "distribution_inventory": inventory,
    }


def worker_process_environment(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, str]:
    environment = dict(WORKER_PROCESS_ENVIRONMENT)
    root = repository_root.resolve()
    environment["PYTHONPATH"] = root.as_posix()
    environment["PYTHONPYCACHEPREFIX"] = (
        root / "results/.g9cb7-bytecode-cache-disabled"
    ).as_posix()
    return environment


def validate_environment(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    actual = environment_inventory()
    for key, expected in FROZEN_ENVIRONMENT.items():
        if actual[key] != expected:
            raise ValueError(
                f"environment mismatch for {key}: expected {expected!r}, "
                f"found {actual[key]!r}"
            )
    actual["worker_process_environment"] = worker_process_environment(
        repository_root
    )
    return actual


def _relative_module_name(path: Path) -> tuple[str, bool]:
    initializer = path.name == "__init__.py"
    parts = list(path.with_suffix("").parts)
    if initializer:
        parts.pop()
    return ".".join(parts), initializer


def _module_files(module: str, repository_root: Path) -> list[Path]:
    if not module:
        return []
    parts = module.split(".")
    discovered: list[Path] = []
    for index in range(1, len(parts)):
        initializer = Path(*parts[:index]) / "__init__.py"
        if (repository_root / initializer).is_file():
            discovered.append(initializer)
    module_file = Path(*parts).with_suffix(".py")
    package_file = Path(*parts) / "__init__.py"
    if (repository_root / module_file).is_file():
        discovered.append(module_file)
    elif (repository_root / package_file).is_file():
        discovered.append(package_file)
    return discovered


def _imported_local_paths(
    node: ast.Import | ast.ImportFrom,
    current_path: Path,
    repository_root: Path,
) -> set[Path]:
    modules: set[str] = set()
    current_module, current_is_initializer = _relative_module_name(current_path)
    current_package = (
        current_module
        if current_is_initializer
        else current_module.rpartition(".")[0]
    )
    if isinstance(node, ast.Import):
        modules.update(alias.name for alias in node.names)
    else:
        if node.level:
            package_parts = current_package.split(".") if current_package else []
            remove = node.level - 1
            if remove > len(package_parts):
                return set()
            prefix_parts = package_parts[: len(package_parts) - remove]
            if node.module:
                prefix_parts.extend(node.module.split("."))
            base = ".".join(prefix_parts)
        else:
            base = node.module or ""
        if base:
            modules.add(base)
        for alias in node.names:
            if alias.name != "*":
                modules.add(f"{base}.{alias.name}" if base else alias.name)
    paths: set[Path] = set()
    for module in modules:
        paths.update(_module_files(module, repository_root))
    return paths


def discover_import_closure(
    entry_paths: Iterable[str | os.PathLike[str]],
    repository_root: Path = REPOSITORY_ROOT,
) -> list[Path]:
    pending = {Path(path) for path in entry_paths}
    discovered: set[Path] = set()
    while pending:
        current = min(pending, key=lambda item: item.as_posix())
        pending.remove(current)
        if current in discovered:
            continue
        absolute = repository_path(current, repository_root)
        if _path_type(absolute) != "regular_file":
            raise ValueError(f"{current}: import-closure member is not a regular file")
        cached = _cached_file(absolute, repository_root)
        source = (
            cached[0].decode("utf-8")
            if cached is not None
            else absolute.read_text(encoding="utf-8")
        )
        tree = ast.parse(source, filename=current.as_posix())
        discovered.add(current)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                pending.update(
                    _imported_local_paths(node, current, repository_root) - discovered
                )
    return sorted(discovered, key=lambda item: item.as_posix())


def import_closure_inventory(
    entry_paths: Iterable[str | os.PathLike[str]],
    repository_root: Path = REPOSITORY_ROOT,
) -> list[dict[str, Any]]:
    inventory = []
    for path in discover_import_closure(entry_paths, repository_root):
        binding = _tracked_binding(path, repository_root=repository_root)
        binding["package_initializer"] = path.name == "__init__.py"
        inventory.append(binding)
    return inventory


def validate_import_closure(
    expected: Sequence[Mapping[str, Any]],
    entry_paths: Iterable[str | os.PathLike[str]],
    repository_root: Path = REPOSITORY_ROOT,
) -> list[dict[str, Any]]:
    actual = import_closure_inventory(entry_paths, repository_root)
    if actual != list(expected):
        raise ValueError("repository-local static import closure mismatch")
    return actual


def _load_json_metadata(path: Path, repository_root: Path) -> Any:
    if path == Path(
        "results/gross9_pre2025_authoritative_anchor_2026-07-28.json"
    ):
        raise ValueError("the pre-2025 anchor is hash-only and must not be parsed")
    absolute = repository_path(path, repository_root)
    cached = _cached_file(absolute, repository_root)
    if cached is not None:
        return json.loads(cached[0].decode("utf-8"))
    with absolute.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_config_metadata(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    portfolio = _load_json_metadata(
        Path(DIRECT_AUTHORITY_BINDINGS[0][1]), repository_root
    )
    expected_weights = {sleeve["name"]: sleeve["configured_weight"] for sleeve in SLEEVES}
    if portfolio.get("weights") != expected_weights or portfolio.get(
        "gross_weight"
    ) != 9.0:
        raise ValueError("Gross9 portfolio weights do not match the frozen contract")

    base = _load_json_metadata(
        Path(DIRECT_AUTHORITY_BINDINGS[1][1]), repository_root
    )
    expected_base_sources = {
        "cand_rex_veto_7": "configs/live/rex_veto_7_candidate.json",
        "fresh_kimchi_fx": "configs/shadow/fresh_kimchi_fx_2026-07-16.json",
        "frozen_annual_rank7": "configs/shadow/frozen_annual_rank7_2026-07-16.json",
        "markov_transition_long": (
            "configs/shadow/markov_transition_long_2026-07-16.json"
        ),
        "rex_taker_low_range_position": (
            "configs/shadow/rex_taker_low_range_position_2026-07-16.json"
        ),
    }
    observed_base_sources = {
        item["name"]: item["source"] for item in base.get("base_sleeves", [])
    }
    if observed_base_sources != expected_base_sources:
        raise ValueError("base portfolio sleeve-source bindings mismatch")

    checks = {
        "cand_rex_veto_7": {
            "path": Path("configs/live/rex_veto_7_candidate.json"),
            "side": "AUTO",
            "hold_bars": 144,
            "entry_delay_bars": 1,
        },
        "fresh_kimchi_fx": {
            "path": Path("configs/shadow/fresh_kimchi_fx_2026-07-16.json"),
            "side": "AUTO",
            "hold_bars": 288,
            "entry_delay_bars": 1,
            "take_bps": 400,
            "stop_bps": 250,
        },
        "frozen_annual_rank7": {
            "path": Path("configs/shadow/frozen_annual_rank7_2026-07-16.json"),
            "side": "LONG",
            "hold_bars": 576,
            "entry_delay_bars": 1,
            "bundle_manifest_hash": RANK7_BUNDLE_MANIFEST_HASH,
        },
        "markov_transition_long": {
            "path": Path("configs/shadow/markov_transition_long_2026-07-16.json"),
            "side": "LONG",
            "hold_bars": 576,
            "entry_delay_bars": 1,
        },
        "rex_taker_low_range_position": {
            "path": Path(
                "configs/shadow/rex_taker_low_range_position_2026-07-16.json"
            ),
            "side": "AUTO",
            "hold_bars": 144,
            "entry_delay_bars": 1,
        },
    }
    for name, contract in checks.items():
        metadata = _load_json_metadata(contract["path"], repository_root)
        for key, expected in contract.items():
            if key != "path" and metadata.get(key) != expected:
                raise ValueError(f"{name}: config metadata mismatch for {key}")
    return {
        "gross_weight": 9.0,
        "portfolio_weights": expected_weights,
        "base_sleeve_sources": expected_base_sources,
        "sleeve_contracts_authenticated": list(checks),
    }


def _rank7_declared_files(
    repository_root: Path = REPOSITORY_ROOT,
) -> list[tuple[str, str]]:
    metadata = _load_json_metadata(RANK7_BUNDLE_MANIFEST_PATH, repository_root)
    if metadata.get("bundle_manifest_hash") != RANK7_BUNDLE_MANIFEST_HASH:
        raise ValueError("Rank7 internal manifest hash mismatch")
    declared = [
        (
            str(metadata.get("hourly_history", {}).get("path")),
            str(metadata.get("hourly_history", {}).get("sha256")),
        )
    ]
    declared.extend(
        (str(model.get("path")), str(model.get("sha256")))
        for model in metadata.get("models", [])
    )
    expected = [(path, digest) for path, digest, _blob in RANK7_FILE_BINDINGS]
    if declared != expected:
        raise ValueError("Rank7 manifest-declared file inventory mismatch")
    expected_exits = {
        "funding": {"hold_bars": 576, "stop_bps": 1_000_000, "take_bps": 400},
        "premium": {"hold_bars": 144, "stop_bps": 300, "take_bps": 1_000_000},
    }
    if metadata.get("exits_by_source") != expected_exits:
        raise ValueError("Rank7 source-routed exit metadata mismatch")
    return declared


def validate_rank7_bundle(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    _rank7_declared_files(repository_root)
    files = []
    for relative, digest, blob in RANK7_FILE_BINDINGS:
        path = RANK7_BUNDLE_ROOT / relative
        files.append(
            _tracked_binding(
                path,
                repository_root=repository_root,
                expected_sha256=digest,
                expected_blob=blob,
            )
        )
    return {
        "bundle_manifest_hash": RANK7_BUNDLE_MANIFEST_HASH,
        "declared_files": files,
    }


def _declared_sources(
    repository_root: Path = REPOSITORY_ROOT,
) -> list[tuple[str, str, str]]:
    metadata = _load_json_metadata(SOURCE_MANIFEST_PATH, repository_root)
    if metadata.get("schema_version") != 1:
        raise ValueError("source manifest schema mismatch")
    declared = [
        (str(item.get("name")), str(item.get("path")), str(item.get("sha256")))
        for item in metadata.get("sources", [])
    ]
    if declared != list(SOURCE_BINDINGS):
        raise ValueError("source manifest ordered inventory mismatch")
    return declared


def _git_result(
    arguments: Sequence[str], repository_root: Path, *, text: bool = True
) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
    )


def _require_no_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if stat.S_ISLNK(os.lstat(current).st_mode):
            raise ValueError(f"{path}: path traverses a symlink")


def _normalized_git_candidate(
    path: str, repository_root: Path
) -> tuple[Path, bool]:
    root = Path(os.path.abspath(repository_root))
    candidate = Path(path)
    if candidate.is_absolute():
        normalized = Path(os.path.normpath(path))
        if (
            path != normalized.as_posix()
            or "//" in path
            or any(part in ("", ".", "..") for part in path[1:].split("/"))
        ):
            raise ValueError(f"{path}: absolute path is not canonical")
        if normalized == root or root in normalized.parents:
            raise ValueError(
                f"{path}: absolute repository path must be repository-relative"
            )
        return normalized, False

    if "\\" in path:
        raise ValueError(f"{path}: repository path is not POSIX text")
    components = path.split("/")
    if any(component in ("", ".", "..") for component in components):
        raise ValueError(f"{path}: repository path is not normalized")
    if Path(path).as_posix() != path:
        raise ValueError(f"{path}: repository path is not normalized")
    return root.joinpath(*components), True


def _parse_stage_zero_entry(output: str, path: str) -> tuple[str, str]:
    lines = output.splitlines()
    if len(lines) != 1:
        raise ValueError(f"{path}: expected exactly one Git index entry")
    metadata, separator, observed_path = lines[0].partition("\t")
    fields = metadata.split()
    if (
        not separator
        or observed_path != path
        or len(fields) != 3
        or fields[2] != "0"
    ):
        raise ValueError(f"{path}: Git index entry is not exact stage zero")
    mode, blob, _stage = fields
    return blob, mode


def _parse_head_tree_entry(output: str, path: str) -> tuple[str, str]:
    lines = output.splitlines()
    if len(lines) != 1:
        raise ValueError(f"{path}: expected exactly one HEAD tree entry")
    metadata, separator, observed_path = lines[0].partition("\t")
    fields = metadata.split()
    if (
        not separator
        or observed_path != path
        or len(fields) != 3
        or fields[1] != "blob"
    ):
        raise ValueError(f"{path}: HEAD tree entry is not an exact blob")
    mode, _kind, blob = fields
    return blob, mode


def _optional_git_metadata(
    path: str, repository_root: Path
) -> dict[str, str | None]:
    _candidate, repository_relative = _normalized_git_candidate(
        path, repository_root
    )
    if not repository_relative:
        return {"git_blob": None, "git_mode": None}
    if _ACTIVE_PREREGISTRATION_GIT_PAIRS is not None:
        pair = _ACTIVE_PREREGISTRATION_GIT_PAIRS.get(path)
        if pair is None:
            return {"git_blob": None, "git_mode": None}
        cached = _cached_file(path, repository_root)
        if cached is None:
            raise ValueError(f"{path}: Git pair/cache state differs")
        raw, _info = cached
        if hashlib.sha1(
            f"blob {len(raw)}\0".encode("ascii") + raw
        ).hexdigest() != pair[0]:
            raise ValueError(f"{path}: cached worktree Git blob differs")
        return {"git_blob": pair[0], "git_mode": pair[1]}

    staged = _git_result(
        ["ls-files", "--stage", "--", path], repository_root
    )
    tree = _git_result(["ls-tree", "HEAD", "--", path], repository_root)
    matched = _git_result(
        ["ls-files", "--error-unmatch", "--", path], repository_root
    )
    if not staged.stdout and not tree.stdout:
        if (
            staged.returncode != 0
            or tree.returncode != 0
            or matched.returncode != 1
            or matched.stdout
        ):
            raise ValueError(f"{path}: untracked Git classification differs")
        return {"git_blob": None, "git_mode": None}
    if staged.returncode or tree.returncode or matched.returncode:
        raise ValueError(f"{path}: tracked Git classification is incomplete")
    if matched.stdout.rstrip("\n") != path:
        raise ValueError(f"{path}: tracked Git path differs")
    index_blob, index_mode = _parse_stage_zero_entry(staged.stdout, path)
    tree_blob, tree_mode = _parse_head_tree_entry(tree.stdout, path)
    if (
        index_blob != tree_blob
        or index_mode != tree_mode
        or index_mode != "100644"
        or not re.fullmatch(r"[0-9a-f]{40}", index_blob)
    ):
        raise ValueError(f"{path}: index and HEAD Git metadata differ")
    cached = _cached_file(path, repository_root)
    if cached is None:
        raw, _info = _read_no_follow_once(
            repository_path(path, repository_root)
        )
    else:
        raw, _info = cached
    worktree_blob = hashlib.sha1(
        f"blob {len(raw)}\0".encode("ascii") + raw
    ).hexdigest()
    if worktree_blob != index_blob:
        raise ValueError(f"{path}: worktree Git blob differs")
    return {"git_blob": index_blob, "git_mode": index_mode}


def validate_sources(
    repository_root: Path = REPOSITORY_ROOT,
) -> list[dict[str, Any]]:
    _declared_sources(repository_root)
    bindings = []
    for name, logical_path, digest in SOURCE_BINDINGS:
        absolute = repository_path(logical_path, repository_root)
        path_type = _path_type(absolute, repository_root)
        if path_type != "regular_file":
            raise ValueError(f"{logical_path}: invalid preregistration path type")
        resolved = absolute
        actual = sha256_file(absolute)
        if actual != digest:
            raise ValueError(f"{logical_path}: source SHA-256 mismatch")
        cached = _cached_file(resolved, repository_root)
        size = cached[1].st_size if cached is not None else resolved.stat().st_size
        binding = {
            "name": name,
            "logical_path": logical_path,
            "resolved_path": (
                str(resolved)
                if Path(logical_path).is_absolute()
                else Path(logical_path).as_posix()
            ),
            "path_type": path_type,
            "resolved_path_type": "regular_file",
            "size_bytes": size,
            "bytes_read_for_sha256_preclaim": size,
            "sha256": actual,
        }
        binding.update(_optional_git_metadata(logical_path, repository_root))
        bindings.append(binding)
    return bindings


def source_preclaim_disclosures(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    frozen = repository_path(FROZEN_OPEN_INTEREST_GZIP_PATH, repository_root)
    cached = _cached_file(frozen, repository_root)
    if cached is None:
        raw, info = _read_no_follow_once(frozen)
    else:
        raw, info = cached
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o444
        or info.st_nlink != 1
        or info.st_size != FROZEN_OPEN_INTEREST_GZIP_SIZE
        or hashlib.sha256(raw).hexdigest()
        != FROZEN_OPEN_INTEREST_GZIP_SHA256
    ):
        raise ValueError("frozen open-interest gzip binding differs")
    validate_file(
        OPEN_INTEREST_PATH,
        OPEN_INTEREST_SHA256,
        expected_size=OPEN_INTEREST_SIZE,
        repository_root=repository_root,
    )
    return {
        "frozen_open_interest_gzip_logical_path": str(
            FROZEN_OPEN_INTEREST_GZIP_PATH
        ),
        "frozen_open_interest_gzip_resolved_path": str(
            FROZEN_OPEN_INTEREST_GZIP_RESOLVED_PATH
        ),
        "frozen_open_interest_gzip_path_type": "regular_file",
        "frozen_open_interest_gzip_mode_octal": "0444",
        "frozen_open_interest_gzip_size_bytes": FROZEN_OPEN_INTEREST_GZIP_SIZE,
        "frozen_open_interest_gzip_sha256": FROZEN_OPEN_INTEREST_GZIP_SHA256,
        "frozen_open_interest_gzip_opaque_bytes_opened_preclaim": True,
        "frozen_open_interest_gzip_decompressed_preclaim": False,
        "frozen_open_interest_gzip_headers_decoded_preclaim": 0,
        "frozen_open_interest_gzip_rows_decoded_preclaim": 0,
        "frozen_open_interest_gzip_fields_or_values_opened_preclaim": 0,
        "open_interest_logical_path": str(OPEN_INTEREST_PATH),
        "open_interest_artifact_size_bytes": OPEN_INTEREST_SIZE,
        "open_interest_artifact_bytes_read_for_sha256_preclaim": (
            OPEN_INTEREST_SIZE
        ),
        "open_interest_sha256_preclaim": OPEN_INTEREST_SHA256,
        "open_interest_headers_decoded_preclaim": 0,
        "open_interest_rows_decoded_preclaim": 0,
        "open_interest_fields_or_values_opened_preclaim": 0,
    }


def _direct_authority_inventory(
    repository_root: Path = REPOSITORY_ROOT,
) -> list[dict[str, Any]]:
    records = []
    for name, path, digest, blob in DIRECT_AUTHORITY_BINDINGS:
        record = _tracked_binding(
            path,
            repository_root=repository_root,
            expected_sha256=digest,
            expected_blob=blob,
        )
        record["name"] = name
        records.append(record)
    return records


def _protocol_inventory(
    repository_root: Path = REPOSITORY_ROOT,
    *,
    preauthenticated: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    records = []
    known = {} if preauthenticated is None else preauthenticated
    for path in sorted(PROTOCOL_PATHS, key=lambda item: item.as_posix()):
        path_text = path.as_posix()
        if path_text in known:
            records.append(
                {
                    key: known[path_text][key]
                    for key in (
                        "path",
                        "path_type",
                        "sha256",
                        "git_blob",
                        "git_mode",
                    )
                }
            )
        else:
            records.append(
                _tracked_binding(path, repository_root=repository_root)
            )
    return records


def _authority_decision_binding(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    binding = _tracked_binding(
        ACTIVE_AUTHORITY_DECISION_PATH,
        repository_root=repository_root,
        expected_sha256=AUTHORITY_DECISION_SHA256,
        expected_blob=AUTHORITY_DECISION_GIT_BLOB,
    )
    binding["authority_commit"] = AUTHORITY_DECISION_COMMIT
    return binding


def _active_authority_decision_binding(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    binding = _tracked_binding(
        ACTIVE_AUTHORITY_DECISION_PATH,
        repository_root=repository_root,
        expected_sha256=AUTHORITY_DECISION_SHA256,
        expected_blob=AUTHORITY_DECISION_GIT_BLOB,
    )
    binding["authority_commit"] = AUTHORITY_DECISION_COMMIT
    return binding


def _authority_amendment_binding(
    *,
    identity: str,
    path: Path,
    sha256: str,
    git_blob: str,
    authority_commit: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    binding = _tracked_binding(
        path,
        repository_root=repository_root,
        expected_sha256=sha256,
        expected_blob=git_blob,
    )
    binding = {"identity": identity, **binding}
    binding["authority_commit"] = authority_commit
    return binding


def _authority_amendment_bindings(
    repository_root: Path = REPOSITORY_ROOT,
) -> list[dict[str, Any]]:
    return [
        _authority_amendment_binding(
            identity="G9CB-1A",
            path=RANK7_AUTHORITY_AMENDMENT_PATH,
            sha256=RANK7_AUTHORITY_AMENDMENT_SHA256,
            git_blob=RANK7_AUTHORITY_AMENDMENT_GIT_BLOB,
            authority_commit=RANK7_AUTHORITY_AMENDMENT_COMMIT,
            repository_root=repository_root,
        ),
        _authority_amendment_binding(
            identity="G9CB-1B",
            path=RUNTIME_ISOLATION_AMENDMENT_PATH,
            sha256=RUNTIME_ISOLATION_AMENDMENT_SHA256,
            git_blob=RUNTIME_ISOLATION_AMENDMENT_GIT_BLOB,
            authority_commit=RUNTIME_ISOLATION_AMENDMENT_COMMIT,
            repository_root=repository_root,
        ),
        _authority_amendment_binding(
            identity="G9CB-1C",
            path=PREREGISTRATION_CORRECTION_AMENDMENT_PATH,
            sha256=PREREGISTRATION_CORRECTION_AMENDMENT_SHA256,
            git_blob=PREREGISTRATION_CORRECTION_AMENDMENT_GIT_BLOB,
            authority_commit=PREREGISTRATION_CORRECTION_AMENDMENT_COMMIT,
            repository_root=repository_root,
        ),
    ]


def _manifest_without_hash(
    repository_root: Path, *, require_git_seal: bool
) -> dict[str, Any]:
    protocol_implementation_commit = (
        validate_protocol_commit_topology(repository_root)
        if require_git_seal
        else UNSEALED_PROTOCOL_IMPLEMENTATION_COMMIT
    )
    git_seal = (
        validate_git_seal(repository_root)
        if require_git_seal
        else {
            "expected_branch": EXPECTED_BRANCH,
            "expected_upstream": f"origin/{EXPECTED_BRANCH}",
            "required_head_equals_upstream": True,
            "required_worktree_and_index_clean": True,
            "observed_upstream_name": f"origin/{EXPECTED_BRANCH}",
        }
    )
    runtime_closure = import_closure_inventory(
        RUNTIME_IMPORT_ROOTS, repository_root
    )
    environment = validate_environment(repository_root)
    config_evidence = validate_config_metadata(repository_root)
    predecessor_state = _predecessor_state_bindings(
        repository_root,
        authenticate=require_git_seal,
    )
    failed_attempts = predecessor_state["failed_predecessor_attempts"]
    failed_prepublication_closures = predecessor_state[
        "failed_predecessor_prepublication_closures"
    ]
    failed_authorities = {
        row["authority_decision"]["path"]: row["authority_decision"]
        for row in (*failed_attempts, *failed_prepublication_closures)
    }
    protocol_inventory = _protocol_inventory(
        repository_root,
        preauthenticated=failed_authorities,
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "identity": IDENTITY,
        "protocol_implementation_commit": protocol_implementation_commit,
        "authority_decision": _active_authority_decision_binding(
            repository_root
        ),
        "direct_authority_verification_commit": (
            DIRECT_AUTHORITY_VERIFICATION_COMMIT
        ),
        "git_seal": git_seal,
        "candidate_independence": {
            "candidate_identity_present": False,
            "candidate_artifacts_opened": False,
            "comparator_clock_rows_opened": 0,
            "comparator_clocks_preseen_by_research_program": True,
        },
        "domain": {
            "start_inclusive": "2023-06-01T00:00:00Z",
            "end_exclusive": "2026-06-01T00:00:00Z",
            "bar_seconds": 300,
            "interval_semantics": "half_open",
        },
        "sleeves": list(SLEEVES),
        "configured_weight_sum": 9.0,
        "interval_geometry": {
            "timestamps": "YYYY-MM-DDTHH:MM:SSZ",
            "epoch_alignment_seconds": 300,
            "strict_entry_order_within_sleeve": True,
            "duplicate_entry_within_sleeve_forbidden": True,
            "per_sleeve_non_overlap": True,
            "touching_intervals_allowed": True,
            "cross_sleeve_overlap_allowed": True,
            "complete_intervals_only": True,
            "barrier_exit": (
                "first_5m_boundary_after_first_occupied_touching_bar"
            ),
        },
        "serialization": {
            "json": {
                "encoding": "UTF-8",
                "sort_keys": True,
                "separators": [",", ":"],
                "ensure_ascii": True,
                "allow_nan": False,
                "file_trailing_lf": True,
                "manifest_hash": (
                    "SHA256 canonical compact JSON excluding manifest_hash "
                    "without trailing LF"
                ),
            },
            "csv": {
                "columns": [
                    "identity",
                    "sleeve",
                    "sleeve_order",
                    "configured_weight",
                    "interval_index",
                    "entry_time_utc",
                    "exit_time_utc",
                    "side",
                ],
                "encoding": "UTF-8",
                "bom": False,
                "dialect": "RFC-4180",
                "delimiter": ",",
                "line_ending": "LF",
                "blank_lines": False,
                "final_lf": True,
                "row_order": ["sleeve_order", "interval_index"],
                "equivalent_sort": [
                    "sleeve_order",
                    "entry_time_utc",
                    "exit_time_utc",
                    "side",
                ],
                "allowed_sides": [1, -1],
            },
            "gzip": {
                "members": 1,
                "compression_level": 9,
                "original_filename": "",
                "mtime": 0,
                "comment": None,
                "extra_field": None,
                "xfl": 2,
                "os_byte": 255,
                "prefix_hex": "1f8b08000000000002ff",
            },
        },
        "bindings": {
            "protocol": protocol_inventory,
            "authority_amendments": _authority_amendment_bindings(
                repository_root
            ),
            "failed_predecessor_preregistrations": (
                validate_failed_predecessor_preregistrations(
                    repository_root
                )
            ),
            **predecessor_state,
            "direct_authority": _direct_authority_inventory(repository_root),
            "config_metadata_evidence": config_evidence,
            "runtime_import_roots": [
                path.as_posix() for path in RUNTIME_IMPORT_ROOTS
            ],
            "runtime_import_closure": runtime_closure,
            "rank7_bundle": validate_rank7_bundle(repository_root),
            "source_manifest_ordered_inventory": validate_sources(repository_root),
            "environment": environment,
        },
        "pre2025_anchor_boundary": {
            "pre2025_anchor_bytes_hashed": True,
            "pre2025_anchor_git_blob_authenticated": True,
            "pre2025_anchor_json_parsed": False,
            "pre2025_anchor_value_rows_opened": 0,
        },
        "source_preclaim_disclosures": source_preclaim_disclosures(
            repository_root
        ),
        "creation_evidence_boundary": dict(CREATION_EVIDENCE_BOUNDARY),
        "output_paths": {
            "preregistration": PREREGISTRATION_PATH.as_posix(),
            "access_claim": ACCESS_CLAIM_PATH.as_posix(),
            "attempt_sentinel": ATTEMPT_SENTINEL_PATH.as_posix(),
            "worker_capability_consumption_ledgers": [
                path.as_posix()
                for path in WORKER_CAPABILITY_CONSUMPTION_LEDGER_PATHS
            ],
            "canonical_csv_gzip": BUNDLE_PATH.as_posix(),
            "final_manifest": FINAL_MANIFEST_PATH.as_posix(),
        },
        "pre_access_claim_contract": {
            "parent_protocol_commit": "P",
            "claim_commit": "C",
            "claim_only_direct_child": True,
            "claim_hash_excludes_claim_hash": True,
            "zero_value_access": True,
            "retry_allowed": False,
        },
        "attempt_sentinel_contract": {
            "status": "attempt_consumed_before_runtime_or_value_access",
            "one_shot": True,
            "retry_allowed": False,
            "resume_allowed": False,
            "zero_runtime_imports": True,
            "zero_value_row_access": True,
            "canonical_mode_octal": "0444",
            "publish": "same_directory_fsync_hard_link_create_only",
            "next_operation": (
                "single_slot1_popen_with_anonymous_pipe_capability"
            ),
            "worker_capability_kind": "anonymous_pipe_v1",
            "worker_capability_slots": [1, 2],
        },
        "access_counter_names": {
            "file_access": [
                "bytes_read_by_logical_source",
                "source_files_opened",
                "model_files_opened",
                "runtime_modules_imported",
            ],
            "rows_decoded": list(SOURCE_COUNTER_NAMES),
            "rows_used": [
                "causal_feature_rows_by_source",
                "prediction_rows_scored",
                "outcome_dependent_ohlc_rows_examined",
                "rank7_training_trades_replayed",
                "rank7_net_labels_computed",
                "rank7_adverse_labels_computed",
                "rank7_price_factor_values_used",
                "rank7_funding_factor_values_used",
                "rank7_funding_debit_factor_values_used",
                "rank7_adverse_price_factor_values_used",
                "rank7_fee_factor_values_used",
                "rank7_bundle_activation_rows_scored",
                "rank7_bundle_parity_rows_compared",
            ],
            "per_sleeve": list(PER_SLEEVE_COUNTER_NAMES),
        },
        "permanent_prohibited_counters": dict(PERMANENT_PROHIBITED_COUNTERS),
        "two_pass_protocol": {
            "fresh_subprocesses": 2,
            "independent_runtime_imports": True,
            "independent_input_reads": True,
            "separate_same_filesystem_staging_directories": True,
            "cross_pass_state_forbidden": True,
            "compressed_csv_bytes_identical": True,
            "decompressed_csv_bytes_identical": True,
            "core_json_bytes_identical": True,
        },
        "publication_protocol": {
            "manifest_last": True,
            "canonical_mode_octal": "0444",
            "publish": "same_directory_fsync_hard_link_create_only",
            "csv_without_manifest_is_authority": False,
            "network_access": False,
        },
        "forbidden_computations": [
            "portfolio_return",
            "portfolio_pnl",
            "funding_cash",
            "cagr",
            "mdd",
            "economic_rank",
            "candidate_metric",
            "overlap_metric",
        ],
        "one_shot_policy": {
            "one_shot": True,
            "retry_allowed": False,
            "resume_allowed": False,
            "repair_allowed": False,
            "terminal_failure_action": (
                "TERMINAL_G9CB7_ATTEMPT_CONSUMED_NO_RETRY"
            ),
        },
    }


def build_manifest(
    repository_root: Path = REPOSITORY_ROOT,
    *,
    require_git_seal: bool = True,
) -> dict[str, Any]:
    manifest = _manifest_without_hash(
        repository_root, require_git_seal=require_git_seal
    )
    manifest["manifest_hash"] = canonical_hash(manifest)
    validate_manifest(
        manifest,
        repository_root=repository_root,
        verify_files=False,
        verify_environment=False,
        verify_git_seal=False,
    )
    return manifest


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    repository_root: Path = REPOSITORY_ROOT,
    verify_files: bool = True,
    verify_environment: bool = True,
    verify_git_seal: bool = True,
) -> None:
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("preregistration protocol version mismatch")
    if manifest.get("identity") != IDENTITY:
        raise ValueError("preregistration identity mismatch")
    implementation = manifest.get("protocol_implementation_commit")
    if not isinstance(implementation, str) or not re.fullmatch(
        r"[0-9a-f]{40}", implementation
    ):
        raise ValueError("protocol implementation commit is invalid")
    if manifest.get("manifest_hash") != canonical_hash(manifest):
        raise ValueError("preregistration manifest_hash mismatch")
    validate_zero_access_schema(manifest)
    bindings = manifest.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("preregistration bindings object is absent")
    amendments = bindings.get("authority_amendments")
    expected_amendments = _historical_v2_authority_amendments()
    if amendments != expected_amendments:
        raise ValueError("authority amendment bindings mismatch")
    expected_predecessors = (
        validate_failed_predecessor_preregistrations(repository_root)
        if verify_files
        else expected_failed_predecessor_preregistration_bindings()
    )
    if (
        bindings.get("failed_predecessor_preregistrations")
        != expected_predecessors
    ):
        raise ValueError("failed predecessor preregistration bindings mismatch")
    expected_attempts = (
        validate_failed_predecessor_attempts(repository_root)
        if verify_files
        else expected_failed_predecessor_attempts()
    )
    if bindings.get("failed_predecessor_attempts") != expected_attempts:
        raise ValueError("failed predecessor attempt binding mismatch")
    expected_closures = (
        validate_failed_predecessor_closures(repository_root)
        if verify_files
        else expected_failed_predecessor_closures()
    )
    if bindings.get("failed_predecessor_closures") != expected_closures:
        raise ValueError("failed predecessor closure binding mismatch")
    expected_prepublication_closures = (
        validate_failed_predecessor_prepublication_closures(repository_root)
        if verify_files
        else expected_failed_predecessor_prepublication_closures()
    )
    if (
        bindings.get("failed_predecessor_prepublication_closures")
        != expected_prepublication_closures
    ):
        raise ValueError("failed predecessor prepublication closure binding mismatch")
    if (
        bindings.get("successor_preregistrations")
        != expected_successor_preregistration_bindings()
    ):
        raise ValueError("successor preregistration bindings mismatch")
    if manifest.get("source_preclaim_disclosures", {}).get(
        "frozen_open_interest_gzip_opaque_bytes_opened_preclaim"
    ) is not True:
        raise ValueError("accidental opaque gzip disclosure is missing")
    if manifest.get("source_preclaim_disclosures", {}).get(
        "frozen_open_interest_gzip_decompressed_preclaim"
    ) is not False:
        raise ValueError("gzip preclaim decompression disclosure mismatch")
    if verify_git_seal:
        actual_implementation = validate_protocol_commit_topology(
            repository_root
        )
        if implementation != actual_implementation:
            raise ValueError("protocol implementation commit differs")
        validate_git_seal(repository_root)
    if verify_environment:
        actual_environment = validate_environment(repository_root)
        if manifest.get("bindings", {}).get("environment") != actual_environment:
            raise ValueError("manifest environment inventory mismatch")
    if verify_files:
        rebuilt = build_manifest(
            repository_root,
            require_git_seal=verify_git_seal,
        )
        if rebuilt != dict(manifest):
            raise ValueError("preregistration does not match authenticated metadata")


def _validate_output_path(
    output: Path, repository_root: Path = REPOSITORY_ROOT
) -> Path:
    canonical = repository_path(PREREGISTRATION_PATH, repository_root)
    resolved_output = repository_path(output, repository_root)
    if resolved_output != canonical:
        raise ValueError(f"only canonical preregistration path is allowed: {canonical}")
    if resolved_output.parent != repository_path("results", repository_root):
        raise ValueError("preregistration must be a singleton under results")
    return resolved_output


def _pread_complete(descriptor: int, size: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)
    raw = b"".join(chunks)
    if len(raw) != size:
        raise RuntimeError("same-descriptor read was incomplete")
    return raw


def _descriptor_token(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        stat.S_IMODE(info.st_mode),
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _iter_sha_bindings(payload: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        aliases = [
            key
            for key in ("path", "logical_path", "repository_path")
            if key in payload
        ]
        if isinstance(payload.get("sha256"), str) and aliases:
            values = [payload[key] for key in aliases]
            if any(not isinstance(value, str) for value in values):
                raise ValueError("bound path alias is not text")
            if len(set(values)) != 1:
                raise ValueError("bound path aliases conflict")
            yield payload
        for key in sorted(payload):
            if key != "protocol_implementation":
                yield from _iter_sha_bindings(payload[key])
    elif isinstance(payload, list):
        for value in payload:
            yield from _iter_sha_bindings(value)


def _binding_text(binding: Mapping[str, Any]) -> str:
    values = [
        binding[key]
        for key in ("path", "logical_path", "repository_path")
        if isinstance(binding.get(key), str)
    ]
    if not values or len(set(values)) != 1:
        raise ValueError("bound path aliases differ")
    return str(values[0])


def _classify_git_pair_only(
    root: Path,
    path_text: str,
    declaration: Mapping[str, Any],
) -> tuple[str, str] | None:
    repository_relative = not path_text.startswith("/")
    blob = declaration.get("git_blob")
    mode = declaration.get("git_mode")
    require_tracked = declaration.get("require_tracked") is True
    if not repository_relative:
        if blob is not None or mode is not None:
            raise ValueError("absolute binding declares Git metadata")
        return None
    staged = _git_result(["ls-files", "--stage", "--", path_text], root)
    tree = _git_result(["ls-tree", "HEAD", "--", path_text], root)
    matched = _git_result(
        ["ls-files", "--error-unmatch", "--", path_text], root
    )
    if blob is None and mode is None and not require_tracked:
        if (
            staged.returncode != 0
            or tree.returncode != 0
            or staged.stdout
            or tree.stdout
            or matched.returncode != 1
            or matched.stdout
        ):
            raise ValueError(f"{path_text}: paired-null Git state differs")
        return None
    if staged.returncode or tree.returncode or matched.returncode:
        raise ValueError(f"{path_text}: tracked Git state differs")
    index = _parse_stage_zero_entry(staged.stdout, path_text)
    head = _parse_head_tree_entry(tree.stdout, path_text)
    if index != head or (
        not require_tracked and index != (blob, mode)
    ):
        raise ValueError(f"{path_text}: index/HEAD Git pair differs")
    return index


class _PreregistrationSnapshot(dict[str, tuple[bytes, os.stat_result]]):
    def __init__(self, root: Path) -> None:
        super().__init__()
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        self.root = root
        self.directories: dict[tuple[str, tuple[str, ...]], int] = {
            ("repo", ()): os.open(root, flags),
            ("absolute", ()): os.open("/", flags),
        }
        self.directory_tokens = {
            key: _descriptor_token(os.fstat(descriptor))
            for key, descriptor in self.directories.items()
        }
        self.files: dict[str, int] = {}
        self.file_tokens: dict[str, tuple[int, ...]] = {}
        self.file_edges: dict[str, tuple[int, str]] = {}
        self.directory_edges: dict[
            tuple[str, tuple[str, ...]], tuple[int, str]
        ] = {}
        self.directory_entries: dict[
            tuple[str, tuple[str, ...]], tuple[str, ...]
        ] = {}
        self.final_verified = False

    def _parent(self, path_text: str) -> tuple[int, str]:
        absolute = path_text.startswith("/")
        if absolute:
            if (
                path_text not in _ABSOLUTE_BINDING_ALLOWLIST
                or "//" in path_text
                or any(
                    part in ("", ".", "..")
                    for part in path_text[1:].split("/")
                )
            ):
                raise ValueError("absolute binding is not exactly allowlisted")
            namespace = "absolute"
            components = path_text[1:].split("/")
        else:
            if "\\" in path_text or any(
                part in ("", ".", "..") for part in path_text.split("/")
            ):
                raise ValueError("repository binding is not normalized")
            namespace = "repo"
            components = path_text.split("/")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        prefix: tuple[str, ...] = ()
        descriptor = self.directories[(namespace, prefix)]
        for component in components[:-1]:
            prefix += (component,)
            key = (namespace, prefix)
            if key not in self.directories:
                opened = _open_component_no_follow(
                    descriptor, component, directory=True
                )
                info = os.fstat(opened)
                if not stat.S_ISDIR(info.st_mode):
                    os.close(opened)
                    raise ValueError(f"{path_text}: parent is not a directory")
                self.directories[key] = opened
                self.directory_tokens[key] = _descriptor_token(info)
                self.directory_edges[key] = (descriptor, component)
            descriptor = self.directories[key]
        return descriptor, components[-1]

    def retain_empty_directory(self, path_text: str, *, mode: int) -> None:
        components = tuple(path_text.split("/"))
        if not components or any(part in ("", ".", "..") for part in components):
            raise ValueError("retained directory path is not normalized")
        key = ("repo", components)
        if key in self.directories:
            if self.directory_entries.get(key) != ():
                raise ValueError("retained directory declaration differs")
            return
        parent, leaf = self._parent(path_text)
        descriptor = _open_component_no_follow(parent, leaf, directory=True)
        try:
            info = os.fstat(descriptor)
            entries = tuple(sorted(os.listdir(descriptor)))
            if (
                not stat.S_ISDIR(info.st_mode)
                or stat.S_IMODE(info.st_mode) != mode
                or entries
            ):
                raise ValueError(f"{path_text}: retained directory state differs")
            self.directories[key] = descriptor
            self.directory_tokens[key] = _descriptor_token(info)
            self.directory_edges[key] = (parent, leaf)
            self.directory_entries[key] = entries
        except BaseException:
            os.close(descriptor)
            raise

    def open_initial(self, path_text: str) -> tuple[bytes, os.stat_result]:
        if path_text in self:
            return self[path_text]
        parent, leaf = self._parent(path_text)
        descriptor = _open_component_no_follow(
            parent, leaf, directory=False
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            os.close(descriptor)
            raise ValueError(f"{path_text}: leaf is not a regular file")
        raw = _pread_complete(descriptor, before.st_size)
        after = os.fstat(descriptor)
        if _descriptor_token(before) != _descriptor_token(after):
            os.close(descriptor)
            raise ValueError(f"{path_text}: changed during initial read")
        identity = (after.st_dev, after.st_ino)
        if any(
            (token[0], token[1]) == identity
            for token in self.file_tokens.values()
        ):
            os.close(descriptor)
            raise ValueError("distinct bound paths hard-link alias")
        self.files[path_text] = descriptor
        self.file_tokens[path_text] = _descriptor_token(after)
        self.file_edges[path_text] = (parent, leaf)
        self[path_text] = (raw, after)
        return raw, after

    def verify_final(self) -> None:
        if self.final_verified:
            raise RuntimeError("snapshot final recheck repeated")
        for path_text in sorted(self.files):
            descriptor = self.files[path_text]
            before = os.fstat(descriptor)
            raw = _pread_complete(descriptor, before.st_size)
            after = os.fstat(descriptor)
            if (
                _descriptor_token(before) != self.file_tokens[path_text]
                or _descriptor_token(after) != self.file_tokens[path_text]
                or raw != self[path_text][0]
            ):
                raise RuntimeError(f"{path_text}: final same-FD recheck differs")
            parent, leaf = self.file_edges[path_text]
            path_info = os.stat(
                leaf, dir_fd=parent, follow_symlinks=False
            )
            if _descriptor_token(path_info) != self.file_tokens[path_text]:
                raise RuntimeError(f"{path_text}: leaf component changed")
        for key, descriptor in self.directories.items():
            if _descriptor_token(os.fstat(descriptor)) != self.directory_tokens[key]:
                raise RuntimeError(f"{key}: directory graph changed")
            expected_entries = self.directory_entries.get(key)
            if (
                expected_entries is not None
                and tuple(sorted(os.listdir(descriptor))) != expected_entries
            ):
                raise RuntimeError(f"{key}: retained directory inventory changed")
            edge = self.directory_edges.get(key)
            if edge is not None:
                parent, component = edge
                path_info = os.stat(
                    component, dir_fd=parent, follow_symlinks=False
                )
                if _descriptor_token(path_info) != self.directory_tokens[key]:
                    raise RuntimeError(f"{key}: parent component changed")
        self.final_verified = True

    def rebaseline_directory(self, identity: tuple[int, int]) -> None:
        for key, descriptor in self.directories.items():
            info = os.fstat(descriptor)
            if (info.st_dev, info.st_ino) == identity:
                self.directory_tokens[key] = _descriptor_token(info)

    def close(self) -> None:
        for descriptor in self.files.values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        for descriptor in self.directories.values():
            try:
                os.close(descriptor)
            except OSError:
                pass


def _open_component_no_follow(
    parent_fd: int, component: str, *, directory: bool
) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    if directory:
        flags |= getattr(os, "O_DIRECTORY", 0)
    else:
        flags |= getattr(os, "O_NONBLOCK", 0)
    return os.open(component, flags, dir_fd=parent_fd)


def _require_clean_pushed_branch(repository_root: Path) -> str:
    validate_git_seal(repository_root)
    return _run_git(["rev-parse", "HEAD"], repository_root)


def _validate_q7_publication_topology(repository_root: Path) -> str:
    head = _require_clean_pushed_branch(repository_root)
    if _single_parent(head, repository_root) != AUTHORITY_DECISION_COMMIT:
        raise ValueError("Q7 is not the direct child of A7")
    if (
        _commit_diff(AUTHORITY_DECISION_COMMIT, head, repository_root)
        != SUCCESSOR_PROTOCOL_DIFF
    ):
        raise ValueError("Q7 implementation diff differs")
    return head


def _validate_closed_path_state(
    results_fd: int,
    phase: str,
    *,
    snapshot: _PreregistrationSnapshot,
    validate_predecessors: bool = True,
    preregistration: bool,
    claim: bool,
    worker_stage: bool,
    fixed_pycache: bool,
) -> tuple[str, ...]:
    if phase != Q7_PREREGISTRATION_PUBLICATION:
        raise ValueError("preregistration publication phase differs")
    names = tuple(sorted(os.listdir(results_fd)))
    required_absence = {
        ACCESS_CLAIM_PATH.name,
        ATTEMPT_SENTINEL_PATH.name,
        *(path.name for path in WORKER_CAPABILITY_CONSUMPTION_LEDGER_PATHS),
        BUNDLE_PATH.name,
        FINAL_MANIFEST_PATH.name,
        ".g9cb7-bytecode-cache-disabled",
    }
    if not preregistration:
        required_absence.add(PREREGISTRATION_PATH.name)
    if claim or worker_stage or fixed_pycache:
        raise ValueError("Q7 entry-point flags differ")
    named_staging_prefixes = tuple(
        f".{path.name}.stage-"
        for path in (
            PREREGISTRATION_PATH,
            ACCESS_CLAIM_PATH,
            ATTEMPT_SENTINEL_PATH,
            *WORKER_CAPABILITY_CONSUMPTION_LEDGER_PATHS,
            BUNDLE_PATH,
            FINAL_MANIFEST_PATH,
        )
    )
    if required_absence & set(names) or any(
        name.startswith(
            (
                ".gross9-structural-clock-g9cb7-worker-",
                ".g9cb7-otmpfile-probe-",
            )
        )
        or name.startswith(named_staging_prefixes)
        or (
            name.startswith(".gross9_structural_clock_bundle_g9cb7_")
            and ".stage-" in name
        )
        for name in names
    ):
        raise FileExistsError("Q7 preregistration path-state is not closed")
    if preregistration:
        info = os.stat(
            PREREGISTRATION_PATH.name,
            dir_fd=results_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o444
        ):
            raise ValueError("Q7 preregistration final state differs")
    if (snapshot.root / ".git").exists():
        tracked_results = _tracked_results_top_level_entries(
            _run_git(["ls-files", "--", "results"], snapshot.root)
        )
        residue_names = {
            Path(str(row["residue"]["slot1_stage"]["path"])).name
            for row in expected_failed_predecessor_attempts()
        }
        active = {PREREGISTRATION_PATH.name} if preregistration else set()
        if set(names) != tracked_results | residue_names | active:
            raise FileExistsError("Q7 exact results inventory differs")
    if validate_predecessors:
        _validate_predecessor_inventory_from_snapshot(
            snapshot,
            expected_failed_predecessor_attempts(),
            expected_failed_predecessor_closures(),
            expected_failed_predecessor_prepublication_closures(),
        )
    return names


def _snapshot_declarations(
    payload: Any,
) -> tuple[
    dict[str, str | None],
    dict[str, dict[str, Any]],
]:
    declarations: dict[str, dict[str, Any]] = {}
    digests: dict[str, str | None] = {}
    for binding in _iter_sha_bindings(payload):
        path_text = _binding_text(binding)
        digest = str(binding["sha256"])
        if path_text in digests and digests[path_text] != digest:
            raise ValueError(f"{path_text}: conflicting SHA-256 declarations")
        digests[path_text] = digest
        declaration = declarations.setdefault(path_text, {})
        for key in ("git_blob", "git_mode"):
            if key in binding:
                if key in declaration and declaration[key] != binding[key]:
                    raise ValueError(f"{path_text}: conflicting Git declaration")
                declaration[key] = binding[key]
    return digests, declarations


def _prepare_declared_snapshot(
    digests: Mapping[str, str | None],
    declarations: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> tuple[
    _PreregistrationSnapshot,
    dict[str, tuple[str, str] | None],
]:
    pairs = {
        path_text: _classify_git_pair_only(
            root, path_text, declarations.get(path_text, {})
        )
        for path_text in sorted(digests)
    }
    snapshot = _PreregistrationSnapshot(root)
    try:
        for path_text in sorted(digests):
            raw, info = snapshot.open_initial(path_text)
            digest = digests[path_text]
            if (
                digest is not None
                and hashlib.sha256(raw).hexdigest() != digest
            ):
                raise ValueError(f"{path_text}: SHA-256 differs")
            pair = pairs[path_text]
            if pair is not None and (
                hashlib.sha1(
                    f"blob {len(raw)}\0".encode("ascii") + raw
                ).hexdigest()
                != pair[0]
                or ("100755" if info.st_mode & 0o111 else "100644")
                != pair[1]
            ):
                raise ValueError(f"{path_text}: cached Git binding differs")
        return snapshot, pairs
    except BaseException:
        snapshot.close()
        raise


def _prepare_preregistration_snapshot(
    manifest: Mapping[str, Any], root: Path
) -> tuple[
    _PreregistrationSnapshot,
    dict[str, tuple[str, str] | None],
]:
    digests, declarations = _snapshot_declarations(manifest)
    return _prepare_declared_snapshot(digests, declarations, root)


def _bootstrap_declarations(
    predecessor: Mapping[str, Any], root: Path
) -> tuple[dict[str, str | None], dict[str, dict[str, Any]]]:
    bootstrap_payload = {
        "predecessor": predecessor,
        **_predecessor_state_bindings(root, authenticate=False),
    }
    digests, declarations = _snapshot_declarations(bootstrap_payload)
    previous_protocol = {
        str(row.get("path"))
        for row in predecessor.get("bindings", {}).get("protocol", [])
        if isinstance(row, Mapping)
    }
    for path in PROTOCOL_PATHS:
        path_text = path.as_posix()
        digests[path_text] = None
        declarations[path_text] = {"require_tracked": True}
    for path_text in previous_protocol - {
        path.as_posix() for path in PROTOCOL_PATHS
    }:
        digests.pop(path_text, None)
        declarations.pop(path_text, None)
    return digests, declarations


def _bootstrap_q7_snapshot(
    root: Path,
) -> tuple[
    _PreregistrationSnapshot,
    dict[str, tuple[str, str] | None],
]:
    predecessor_path = (
        "results/"
        "gross9_structural_clock_bundle_g9cb4_preregistration_2026-07-31.json"
    )
    completed = _git_result(["show", f"HEAD:{predecessor_path}"], root)
    completed.check_returncode()
    predecessor = json.loads(completed.stdout)
    digests, declarations = _bootstrap_declarations(predecessor, root)
    return _prepare_declared_snapshot(digests, declarations, root)


def _link_prepared_publication(
    unnamed_fd: int, results_fd: int, leaf: str
) -> None:
    try:
        os.link(
            f"/proc/self/fd/{unnamed_fd}",
            leaf,
            dst_dir_fd=results_fd,
            follow_symlinks=True,
        )
    except FileExistsError as exc:
        raise FileExistsError(f"write-once path exists: {leaf}") from exc


def _prepare_unnamed_publication(
    results_fd: int, raw: bytes
) -> tuple[int, os.stat_result]:
    if not getattr(os, "O_TMPFILE", 0):
        raise RuntimeError("O_TMPFILE is unavailable")
    descriptor = os.open(
        ".",
        os.O_RDWR | os.O_TMPFILE | getattr(os, "O_CLOEXEC", 0),
        0o600,
        dir_fd=results_fd,
    )
    try:
        view = memoryview(raw)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise RuntimeError("unnamed preregistration write stalled")
            offset += written
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o444
            or info.st_size != len(raw)
            or _pread_complete(descriptor, info.st_size) != raw
        ):
            raise RuntimeError("unnamed preregistration verification failed")
        return descriptor, info
    except BaseException:
        os.close(descriptor)
        raise


def _probe_preregistration_publication(
    results_fd: int, snapshot: _PreregistrationSnapshot
) -> None:
    baseline = tuple(sorted(os.listdir(results_fd)))
    leaf = f".g9cb7-otmpfile-probe-{os.getpid()}-{os.urandom(8).hex()}"
    raw = b"G9CB7 preregistration capability probe\n"
    descriptor, unnamed_info = _prepare_unnamed_publication(
        results_fd, raw
    )
    canonical_fd = -1
    try:
        _link_prepared_publication(descriptor, results_fd, leaf)
        added = tuple(sorted((*baseline, leaf)))
        if tuple(sorted(os.listdir(results_fd))) != added:
            raise RuntimeError("preregistration probe addition delta differs")
        canonical_fd = os.open(
            leaf,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=results_fd,
        )
        canonical_info = os.fstat(canonical_fd)
        canonical_raw = _pread_complete(canonical_fd, canonical_info.st_size)
        if (
            (canonical_info.st_dev, canonical_info.st_ino)
            != (unnamed_info.st_dev, unnamed_info.st_ino)
            or not stat.S_ISREG(canonical_info.st_mode)
            or stat.S_IMODE(canonical_info.st_mode) != 0o444
            or canonical_info.st_size != len(raw)
            or canonical_raw != raw
            or hashlib.sha256(canonical_raw).digest()
            != hashlib.sha256(raw).digest()
        ):
            raise RuntimeError("preregistration capability probe differs")
        os.fsync(results_fd)
        results_info = os.fstat(results_fd)
        snapshot.rebaseline_directory(
            (results_info.st_dev, results_info.st_ino)
        )
        os.unlink(leaf, dir_fd=results_fd)
        if tuple(sorted(os.listdir(results_fd))) != baseline:
            raise RuntimeError("preregistration probe removal delta differs")
        os.fsync(results_fd)
        results_info = os.fstat(results_fd)
        snapshot.rebaseline_directory(
            (results_info.st_dev, results_info.st_ino)
        )
        if tuple(sorted(os.listdir(results_fd))) != baseline:
            raise RuntimeError("preregistration capability probe left residue")
    finally:
        if canonical_fd >= 0:
            os.close(canonical_fd)
        os.close(descriptor)


def write_once(
    manifest: Mapping[str, Any],
    output: Path = PREREGISTRATION_PATH,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    _snapshot: _PreregistrationSnapshot | None = None,
    _pairs: Mapping[str, tuple[str, str] | None] | None = None,
) -> bool:
    validate_manifest(
        manifest,
        repository_root=repository_root,
        verify_files=False,
        verify_environment=False,
        verify_git_seal=False,
    )
    target = _validate_output_path(output, repository_root)
    expected_bytes = canonical_json_bytes(dict(manifest), trailing_lf=True)
    canonical_git = (repository_root / ".git").exists()
    if canonical_git:
        _validate_q7_publication_topology(repository_root)
        _require_clean_pushed_branch(repository_root)
    owns_snapshot = _snapshot is None
    if _snapshot is None:
        snapshot, initial_pairs = _prepare_preregistration_snapshot(
            manifest, repository_root
        )
    else:
        if _pairs is None:
            raise ValueError("retained preregistration snapshot lacks Git pairs")
        snapshot = _snapshot
        initial_pairs = dict(_pairs)
        manifest_digests, _manifest_declarations = _snapshot_declarations(
            manifest
        )
        manifest_paths = set(manifest_digests)
        bootstrap_paths = set(snapshot)
        if manifest_paths != bootstrap_paths:
            raise ValueError(
                "manifest bound path set differs from bootstrap: "
                f"manifest_minus_bootstrap="
                f"{sorted(manifest_paths - bootstrap_paths)!r}; "
                f"bootstrap_minus_manifest="
                f"{sorted(bootstrap_paths - manifest_paths)!r}"
            )
        for path_text, digest in manifest_digests.items():
            if (
                digest is not None
                and hashlib.sha256(snapshot[path_text][0]).hexdigest()
                != digest
            ):
                raise ValueError(f"{path_text}: manifest cache digest differs")
    snapshot._parent(PREREGISTRATION_PATH.as_posix())
    results_fd = snapshot.directories.get(("repo", ("results",)))
    if results_fd is None:
        raise ValueError("bootstrap snapshot lacks the retained results descriptor")
    unnamed_fd = canonical_fd = -1
    try:
        if not canonical_git and PREREGISTRATION_PATH.name in os.listdir(
            results_fd
        ):
            try:
                canonical_fd = _open_component_no_follow(
                    results_fd,
                    PREREGISTRATION_PATH.name,
                    directory=False,
                )
            except OSError as exc:
                raise FileExistsError(
                    "existing preregistration is not a regular file"
                ) from exc
            canonical_info = os.fstat(canonical_fd)
            if (
                not stat.S_ISREG(canonical_info.st_mode)
                or stat.S_IMODE(canonical_info.st_mode) != 0o444
                or _pread_complete(canonical_fd, canonical_info.st_size)
                != expected_bytes
            ):
                raise FileExistsError(
                    "existing preregistration has other bytes or mode"
                )
            snapshot.verify_final()
            return False
        initial_entries = _validate_closed_path_state(
            results_fd,
            Q7_PREREGISTRATION_PUBLICATION,
            snapshot=snapshot,
            preregistration=False,
            claim=False,
            worker_stage=False,
            fixed_pycache=False,
        )

        _probe_preregistration_publication(results_fd, snapshot)
        if tuple(sorted(os.listdir(results_fd))) != initial_entries:
            raise RuntimeError("capability probe did not restore results inventory")
        publication_flags = getattr(os, "O_TMPFILE", 0)
        if not publication_flags:
            raise RuntimeError("O_TMPFILE is unavailable")
        unnamed_fd, unnamed_info = _prepare_unnamed_publication(
            results_fd, expected_bytes
        )
        final_recheck = snapshot.verify_final
        final_recheck()
        for path_text, initial_pair in initial_pairs.items():
            observed = _classify_git_pair_only(
                repository_root,
                path_text,
                (
                    {
                        "git_blob": initial_pair[0],
                        "git_mode": initial_pair[1],
                    }
                    if initial_pair is not None
                    else {"git_blob": None, "git_mode": None}
                ),
            )
            if observed != initial_pair:
                raise RuntimeError(f"{path_text}: final Git pair differs")
        if canonical_git:
            _require_clean_pushed_branch(repository_root)
        if tuple(sorted(os.listdir(results_fd))) != initial_entries:
            raise RuntimeError("results inventory drifted before preregistration link")
        _link_prepared_publication(
            unnamed_fd, results_fd, target.name
        )
        canonical_fd = os.open(
            target.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=results_fd,
        )
        canonical_info = os.fstat(canonical_fd)
        if (
            (canonical_info.st_dev, canonical_info.st_ino)
            != (unnamed_info.st_dev, unnamed_info.st_ino)
            or stat.S_IMODE(canonical_info.st_mode) != 0o444
            or _pread_complete(canonical_fd, canonical_info.st_size)
            != expected_bytes
        ):
            raise RuntimeError("canonical preregistration verification failed")
        expected_entries = tuple(sorted((*initial_entries, target.name)))
        if tuple(sorted(os.listdir(results_fd))) != expected_entries:
            raise RuntimeError("preregistration one-leaf delta differs")
        os.fsync(results_fd)
        _validate_closed_path_state(
            results_fd,
            Q7_PREREGISTRATION_PUBLICATION,
            snapshot=snapshot,
            validate_predecessors=False,
            preregistration=True,
            claim=False,
            worker_stage=False,
            fixed_pycache=False,
        )
        snapshot.rebaseline_directory(
            (os.fstat(results_fd).st_dev, os.fstat(results_fd).st_ino)
        )
        return True
    finally:
        if canonical_fd >= 0:
            os.close(canonical_fd)
        if unnamed_fd >= 0:
            os.close(unnamed_fd)
        if owns_snapshot:
            snapshot.close()


def main(argv: Sequence[str] | None = None) -> int:
    global _ACTIVE_PREREGISTRATION_CACHE
    global _ACTIVE_PREREGISTRATION_GIT_PAIRS
    global _ACTIVE_PREREGISTRATION_SNAPSHOT
    global _ACTIVE_PREREGISTRATION_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    _validate_q7_publication_topology(REPOSITORY_ROOT)
    _require_clean_pushed_branch(REPOSITORY_ROOT)
    publication_phase = Q7_PREREGISTRATION_PUBLICATION
    if publication_phase != Q7_PREREGISTRATION_PUBLICATION:
        raise RuntimeError("Q7 publication phase differs")
    snapshot, pairs = _bootstrap_q7_snapshot(REPOSITORY_ROOT)
    try:
        _ACTIVE_PREREGISTRATION_CACHE = snapshot
        _ACTIVE_PREREGISTRATION_GIT_PAIRS = pairs
        _ACTIVE_PREREGISTRATION_SNAPSHOT = snapshot
        _ACTIVE_PREREGISTRATION_ROOT = REPOSITORY_ROOT
        manifest = build_manifest()
        created = write_once(
            manifest,
            _snapshot=snapshot,
            _pairs=pairs,
        )
    finally:
        _ACTIVE_PREREGISTRATION_SNAPSHOT = None
        _ACTIVE_PREREGISTRATION_GIT_PAIRS = None
        _ACTIVE_PREREGISTRATION_CACHE = None
        _ACTIVE_PREREGISTRATION_ROOT = None
        snapshot.close()
    print(
        f"{'created' if created else 'verified'} {PREREGISTRATION_PATH} "
        f"{manifest['manifest_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
