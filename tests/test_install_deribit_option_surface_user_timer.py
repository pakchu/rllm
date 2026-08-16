from pathlib import Path
from unittest.mock import patch

from scripts.install_deribit_option_surface_user_timer import install


def test_installer_substitutes_absolute_paths_and_enables_timer(tmp_path):
    repo = tmp_path / "repo"
    unit_dir = repo / "ops/systemd"
    unit_dir.mkdir(parents=True)
    (unit_dir / "rllm-deribit-option-surface.service.in").write_text(
        "WorkingDirectory=@REPO@\nExecStart=@PYTHON@ -B collector.py\n"
    )
    (unit_dir / "rllm-deribit-option-surface.timer").write_text("[Timer]\nPersistent=true\n")
    python = tmp_path / "python"
    python.write_text("")
    user_dir = tmp_path / "units"
    with patch("subprocess.run") as run:
        install(repo, python, user_dir)
    service = (user_dir / "rllm-deribit-option-surface.service").read_text()
    assert str(repo.resolve()) in service
    assert str(python.resolve()) in service
    assert "@REPO@" not in service and "@PYTHON@" not in service
    assert run.call_count == 2
    assert run.call_args_list[-1].args[0] == [
        "systemctl",
        "--user",
        "enable",
        "--now",
        "rllm-deribit-option-surface.timer",
    ]
