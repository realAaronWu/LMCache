# SPDX-License-Identifier: Apache-2.0
"""Exercise the experiment's public command boundary."""

# Standard
import subprocess
import sys

# Third Party
import pytest


def test_help_names_only_available_commands() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.pooled_kv_routing.run", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for command in ("preflight", "generate", "analyze", "smoke"):
        assert command in result.stdout


def test_unknown_command_is_not_success() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.pooled_kv_routing.run", "qualify-all"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr


@pytest.mark.parametrize(
    "command, option",
    [("serve", "--worker-url"), ("preflight", "--config"), ("analyze", "--protocol")],
)
def test_subcommand_help_reaches_concrete_parser(command: str, option: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.pooled_kv_routing.run", command, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert option in result.stdout
