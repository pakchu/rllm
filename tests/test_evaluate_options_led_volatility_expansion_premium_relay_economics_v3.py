from __future__ import annotations

import inspect
import json
import os
import subprocess
from pathlib import Path

from training import (
    evaluate_options_led_volatility_expansion_premium_relay_economics_v2 as v2,
)
from training import (
    evaluate_options_led_volatility_expansion_premium_relay_economics_v3 as v3,
)


SCRIPT = Path(
    "training/evaluate_options_led_volatility_expansion_premium_relay_economics_v3.py"
)


def test_v3_preserves_the_frozen_v2_accounting_engine() -> None:
    assert inspect.getsource(v3.simulate) == inspect.getsource(v2.simulate)
    assert inspect.getsource(v3.evaluate_primary) == inspect.getsource(v2.evaluate_primary)
    assert v3.BASE_COST == v2.BASE_COST
    assert v3.STRESS_COST == v2.STRESS_COST
    assert v3.LEVERAGE == v2.LEVERAGE
    assert v3.STAGES == v2.STAGES


def test_direct_cli_preflight_reaches_verifier_without_opening_outcomes() -> None:
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run(
        [
            str(Path(".venv/bin/python")),
            "-B",
            str(SCRIPT),
            "--stage",
            "train",
            "--verify-only",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert json.loads(completed.stdout) == {
        "stage": "train",
        "verified": True,
        "outcomes_opened": False,
    }
