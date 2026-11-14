# SPDX-License-Identifier: Apache-2.0
"""Exercise the experiment's public command boundary."""

# Standard
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import subprocess
import sys

# Third Party
import pytest

# First Party
from benchmarks.pooled_kv_routing.tests import test_workloads
from benchmarks.pooled_kv_routing.workloads import TraceSpec

spec = test_workloads.spec


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


def test_generate_subprocess_preserves_command_and_failure_status(
    spec: TraceSpec, tmp_path: Path
) -> None:
    """Reuse the corpus fixture while exercising the actual top-level process."""
    raw = json.dumps(
        asdict(spec), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    config = tmp_path / "config.json"
    config.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    output = tmp_path / "trace.json"
    argv = [
        sys.executable,
        "-m",
        "benchmarks.pooled_kv_routing.run",
        "generate",
        "--config",
        str(config),
        "--config-sha256",
        sha,
        "--seed",
        "101",
        "--output",
        str(output),
    ]
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    trace = json.loads(output.read_text())
    assert trace["seed"] == 101 and trace["config_sha256"] == sha
    assert {row["request_class"] for row in trace["requests"]} == {
        "shared",
        "background",
    }
    assert all(
        len(row["input_token_ids"]) == 4352 and row["output_tokens"] == 128
        for row in trace["requests"]
    )
    failed_output = tmp_path / "invalid-trace.json"
    argv[argv.index("--config-sha256") + 1] = "0" * 64
    argv[-1] = str(failed_output)
    failed = subprocess.run(argv, capture_output=True, text=True, check=False)
    assert failed.returncode == 1 and "config SHA mismatch" in failed.stderr
    assert not failed_output.exists()
