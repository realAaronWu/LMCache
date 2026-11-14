# SPDX-License-Identifier: Apache-2.0
"""Exercise the experiment's public command boundary."""

# Standard
import subprocess
import sys


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
