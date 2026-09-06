"""Install the preregistered Deribit option-surface systemd user timer."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def install(repo: Path, python: Path, user_dir: Path) -> None:
    repo = repo.resolve()
    python = python.resolve()
    service_template = repo / "ops/systemd/rllm-deribit-option-surface.service.in"
    timer_source = repo / "ops/systemd/rllm-deribit-option-surface.timer"
    if not service_template.is_file() or not timer_source.is_file() or not python.is_file():
        raise FileNotFoundError("timer template, timer, or Python executable missing")
    user_dir.mkdir(parents=True, exist_ok=True)
    service = service_template.read_text().replace("@REPO@", str(repo)).replace("@PYTHON@", str(python))
    (user_dir / "rllm-deribit-option-surface.service").write_text(service)
    (user_dir / "rllm-deribit-option-surface.timer").write_bytes(timer_source.read_bytes())
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(
        ["systemctl", "--user", "enable", "--now", "rllm-deribit-option-surface.timer"],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--python", type=Path, default=Path("/home/pakchu/rllm/.venv/bin/python"))
    parser.add_argument(
        "--user-dir", type=Path, default=Path.home() / ".config/systemd/user"
    )
    args = parser.parse_args()
    install(args.repo, args.python, args.user_dir)


if __name__ == "__main__":
    main()
