# SPDX-License-Identifier: Apache-2.0
"""CPU fixtures qualify the checker, never hardware or model execution."""

# Standard
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

# Third Party
import pytest

# First Party
from benchmarks.pooled_kv_routing.preflight import main, preflight

BASE = "bcafb031401d404a5dabae4853f5e23bf9e5685c"


def artifact(tmp_path: Path, name: str, value: object) -> dict:
    data = json.dumps(value, allow_nan=False).encode()
    path = tmp_path / name
    path.write_bytes(data)
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


@pytest.fixture
def config(tmp_path: Path) -> dict:
    cfg = {
        "schema_version": 1,
        "scope": "functional_single_gpu",
        "source": {
            "base": BASE,
            "head": BASE,
            "patch_sha256": "0" * 64,
            "artifact": artifact(tmp_path, "source-archive", {"fixture": BASE}),
        },
        "model": {
            "id": "fixture/model",
            "revision": "a" * 40,
            "tokenizer_revision": "b" * 40,
            "files": [artifact(tmp_path, "weights", [1, 2])],
        },
        "environment": {
            "python": "3.12.0",
            "vllm": "0.12.0",
            "torch": "2.9.0",
            "cuda": "13.0",
            "driver": "595.84",
            "nixl": "0.7.0",
            "image_digest": "sha256:" + "c" * 64,
            "artifact": artifact(tmp_path, "environment-lock", {"wheel": "e" * 64}),
        },
        "nodes": [],
        "native_evidence": None,
    }
    node = {
        "host_id": "machine-A",
        "gpu_uuid": "GPU-A",
        "gpu_model": "fixture-GPU",
        "gpu_memory_bytes": 16 * 1024**3,
        "rdma_device": "mlx5_0",
        "rdma_gid": "gid-A",
        "link_gbps": 100,
        "source": cfg["source"],
        "model": cfg["model"],
        "environment": cfg["environment"],
        "raw": [artifact(tmp_path, "inventory-raw", {"collector": "fixture"})],
    }
    cfg["nodes"] = [artifact(tmp_path, "node-A", node)]
    bind_native(tmp_path, cfg)
    return cfg


def bind_native(tmp_path: Path, cfg: dict, cases: list | None = None) -> None:
    provenance = {k: cfg[k] for k in ("source", "model", "environment", "nodes")}
    probe = {
        "provenance_sha256": hashlib.sha256(
            json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "cases": cases
        if cases is not None
        else [
            {
                "name": "single_gpu",
                "returncode": 0,
                "request_id": "req-1",
                "host_ids": ["machine-A"],
                "transport": "local",
                "remote_bytes": 0,
                "external_fetches": 0,
                "expected_token_ids": [1, 2],
                "actual_token_ids": [1, 2],
                "loader_extra_args": {},
                "raw": [artifact(tmp_path, "probe-raw", {"tokens": [1, 2]})],
            }
        ],
    }
    cfg["native_evidence"] = artifact(tmp_path, "native", probe)


def test_functional_is_not_primary(config: dict) -> None:
    report = preflight(config)
    assert report["evidence_status"] == "PASS"
    assert report["functional_status"] == "PASS"
    assert report["eligible_for_primary"] is False
    assert report["performance_status"] == "NOT RUN"
    assert report["checks"]["rdma"]["status"] == "NOT RUN"


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("scope", "primary"),
        ("certified", True),
        ("nodes", []),
        ("nodes", [True]),
        ("native_evidence", True),
    ],
)
def test_reject_bad_config(config: dict, field: str, value: object) -> None:
    config[field] = value
    assert preflight(config)["evidence_status"] == "FAIL"


@pytest.mark.parametrize("value", ["main", "", True, "a" * 39])
def test_model_revision_must_be_immutable(config: dict, value: object) -> None:
    config["model"]["revision"] = value
    assert preflight(config)["evidence_status"] == "FAIL"


def test_missing_and_tampered_evidence(config: dict) -> None:
    Path(config["native_evidence"]["path"]).write_text("PASS")
    assert preflight(config)["evidence_status"] == "FAIL"
    Path(config["native_evidence"]["path"]).unlink()
    assert preflight(config)["evidence_status"] == "FAIL"


def test_tampered_valid_json_rejected(config: dict) -> None:
    path = Path(config["native_evidence"]["path"])
    probe = json.loads(path.read_text())
    probe["cases"][0]["request_id"] = "rewritten-request"
    path.write_text(json.dumps(probe))
    assert preflight(config)["evidence_status"] == "FAIL"


def test_missing_native_is_not_run(config: dict) -> None:
    config["native_evidence"] = None
    report = preflight(config)
    assert report["evidence_status"] == "NOT RUN"
    assert report["functional_status"] == "NOT RUN"


@pytest.mark.parametrize(
    "field,value",
    [
        ("returncode", 1),
        ("returncode", False),
        ("actual_token_ids", []),
        ("actual_token_ids", [1, 3]),
        ("actual_token_ids", [True, 2]),
        ("remote_bytes", -1),
        ("remote_bytes", float("nan")),
        ("raw", []),
        ("success", True),
        ("host_ids", ["unknown"]),
    ],
)
def test_native_bad_evidence(
    config: dict, tmp_path: Path, field: str, value: object
) -> None:
    probe = json.loads(Path(config["native_evidence"]["path"]).read_text())
    probe["cases"][0][field] = value
    data = json.dumps(probe).encode()
    path = tmp_path / "native-bad"
    path.write_bytes(data)
    config["native_evidence"] = {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    assert preflight(config)["evidence_status"] == "FAIL"


def test_provenance_drift(config: dict) -> None:
    config["environment"]["torch"] = "2.8.0"
    assert preflight(config)["evidence_status"] == "FAIL"


def test_primary_missing_second_node_fails(config: dict) -> None:
    config["scope"] = "primary_two_node_rdma"
    report = preflight(config)
    assert report["eligible_for_primary"] is False
    assert report["evidence_status"] == "FAIL"


def primary_config(tmp_path: Path, cfg: dict) -> dict:
    cfg = deepcopy(cfg)
    cfg["scope"] = "primary_two_node_rdma"
    node = json.loads(Path(cfg["nodes"][0]["path"]).read_text())
    node.update(host_id="machine-B", gpu_uuid="GPU-B", rdma_gid="gid-B")
    cfg["nodes"].append(artifact(tmp_path, "node-B", node))
    case = json.loads(Path(cfg["native_evidence"]["path"]).read_text())["cases"][0]
    cases = []
    for name in (
        "A_to_B",
        "B_to_A",
        "shared_prefix",
        "partial_hit",
        "miss",
        "skip_load",
    ):
        item = deepcopy(case)
        item.update(
            name=name,
            request_id=name,
            host_ids=["machine-A", "machine-B"],
            transport="rdma",
            remote_bytes=4096,
            external_fetches=1,
        )
        if name == "B_to_A":
            item["host_ids"].reverse()
        if name in ("miss", "skip_load"):
            item.update(remote_bytes=0, external_fetches=0)
        if name == "skip_load":
            item["loader_extra_args"] = {"lmcache.skip_load": True}
        cases.append(item)
    bind_native(tmp_path, cfg, cases)
    return cfg


def test_known_unsafe_base_cannot_be_certified(config: dict, tmp_path: Path) -> None:
    cfg = primary_config(tmp_path, config)
    report = preflight(cfg)
    assert report["checks"]["native"]["status"] == "PASS"
    assert report["checks"]["rdma"]["status"] == "PASS"
    assert report["checks"]["resource_safety"]["status"] == "FAIL"
    assert report["eligible_for_primary"] is False
    path, output = tmp_path / "complete-primary", tmp_path / "primary-report"
    path.write_text(json.dumps(cfg))
    assert main(["--config", str(path), "--output", str(output)]) == 1
    assert json.loads(output.read_text())["evidence_status"] == "PASS"


def test_functional_does_not_require_rdma(config: dict, tmp_path: Path) -> None:
    node = json.loads(Path(config["nodes"][0]["path"]).read_text())
    node.update(rdma_device=None, rdma_gid=None, link_gbps=None)
    config["nodes"][0] = artifact(tmp_path, "non-rdma-node", node)
    bind_native(tmp_path, config)
    report = preflight(config)
    assert report["functional_status"] == "PASS"
    assert report["checks"]["rdma"]["status"] == "NOT RUN"


def test_bare_metal_environment_uses_binary_manifest(
    config: dict, tmp_path: Path
) -> None:
    config["environment"]["image_digest"] = None
    config["environment"]["artifact"] = artifact(
        tmp_path, "binary-manifest", {"wheel_sha256": "e" * 64}
    )
    node = json.loads(Path(config["nodes"][0]["path"]).read_text())
    node["environment"] = config["environment"]
    config["nodes"][0] = artifact(tmp_path, "bare-metal", node)
    bind_native(tmp_path, config)
    assert preflight(config)["evidence_status"] == "PASS"
    Path(config["environment"]["artifact"]["path"]).unlink()
    assert preflight(config)["evidence_status"] == "FAIL"


@pytest.mark.parametrize(
    "field,value",
    [
        ("host_id", "machine-A"),
        ("gpu_uuid", "GPU-A"),
        ("rdma_gid", "gid-A"),
        ("gpu_memory_bytes", True),
        ("link_gbps", float("inf")),
        ("rdma_device", ""),
        ("gpu_model", "different"),
        ("raw", []),
    ],
)
def test_primary_inventory_rejected(
    config: dict, tmp_path: Path, field: str, value: object
) -> None:
    cfg = primary_config(tmp_path, config)
    node = json.loads(Path(cfg["nodes"][1]["path"]).read_text())
    node[field] = value
    data = json.dumps(node).encode()
    path = tmp_path / "bad-node"
    path.write_bytes(data)
    cfg["nodes"][1] = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}
    report = preflight(cfg)
    assert report["checks"]["inventory"]["status"] == "FAIL"


@pytest.mark.parametrize(
    "mutation", ["tcp", "no_bytes", "skip_ignored", "missing_case"]
)
def test_rdma_and_skip_require_observations(
    config: dict, tmp_path: Path, mutation: str
) -> None:
    cfg = primary_config(tmp_path, config)
    probe = json.loads(Path(cfg["native_evidence"]["path"]).read_text())
    if mutation == "tcp":
        probe["cases"][0]["transport"] = "tcp"
    elif mutation == "no_bytes":
        probe["cases"][0]["remote_bytes"] = 0
    elif mutation == "skip_ignored":
        probe["cases"][-1]["loader_extra_args"] = {}
    else:
        probe["cases"].pop()
    cfg["native_evidence"] = artifact(tmp_path, "bad-native", probe)
    assert preflight(cfg)["checks"]["native"]["status"] == "FAIL"


def test_unknown_new_head_remains_unqualified(config: dict, tmp_path: Path) -> None:
    cfg = primary_config(tmp_path, config)
    cfg["source"]["head"] = "d" * 40
    assert preflight(cfg)["eligible_for_primary"] is False


def test_cli_actual_exit_and_report(config: dict, tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    output = tmp_path / "report.json"
    path.write_text(json.dumps(config))
    command = [
        sys.executable,
        "-m",
        "benchmarks.pooled_kv_routing.preflight",
        "--config",
        str(path),
        "--output",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["eligible_for_primary"] is False
    config["scope"] = "primary_two_node_rdma"
    path.write_text(json.dumps(config))
    result = subprocess.run(command, capture_output=True, timeout=10)
    assert result.returncode != 0
    assert json.loads(output.read_text())["evidence_status"] == "FAIL"


@pytest.mark.parametrize(
    "content",
    ['{"scope":1,"scope":2}', '{"value":NaN}', "[" * 1000, '"' + "x" * (1024**2) + '"'],
    ids=["duplicate", "nan", "depth", "size"],
)
def test_cli_malformed_bounded_json(tmp_path: Path, content: str) -> None:
    path, output = tmp_path / "config", tmp_path / "output"
    path.write_text(content)
    assert main(["--config", str(path), "--output", str(output)]) != 0
    assert json.loads(output.read_text())["evidence_status"] == "FAIL"


def test_artifact_fifo_rejected_without_blocking(config: dict, tmp_path: Path) -> None:
    path = tmp_path / "fifo"
    os.mkfifo(path)
    config["native_evidence"]["path"] = str(path)
    assert preflight(config)["evidence_status"] == "FAIL"


def test_source_payload_required(config: dict) -> None:
    Path(config["source"]["artifact"]["path"]).unlink()
    assert preflight(config)["evidence_status"] == "FAIL"


@pytest.mark.parametrize(
    "field,value",
    [
        ("rdma_device", True),
        ("rdma_gid", []),
        ("link_gbps", True),
        ("link_gbps", -1),
        pytest.param("link_gbps", 10**400, id="overflow-speed"),
    ],
)
def test_functional_inventory_strict_types(
    config: dict, tmp_path: Path, field: str, value: object
) -> None:
    node = json.loads(Path(config["nodes"][0]["path"]).read_text())
    node[field] = value
    config["nodes"][0] = artifact(tmp_path, "bad-functional-node", node)
    assert preflight(config)["checks"]["inventory"]["status"] == "FAIL"
