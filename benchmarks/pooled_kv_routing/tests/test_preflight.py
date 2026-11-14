# SPDX-License-Identifier: Apache-2.0
"""CPU fixtures qualify the checker, never hardware or model execution."""

# Standard
from copy import deepcopy
from pathlib import Path
from typing import cast
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


@pytest.mark.parametrize(
    "capture",
    [
        '{"actual_token_ids":[',
        '{"returncode":23,"stdout":"PASS","complete":true}',
        '{"tokens":[1,2],"request_id":"other-request"}',
    ],
)
def test_cli_retained_capture_cannot_be_overridden(
    config: dict, tmp_path: Path, capture: str
) -> None:
    native = json.loads(Path(config["native_evidence"]["path"]).read_text())
    path = tmp_path / "retained"
    path.write_text(capture)
    native["cases"][0]["probe"] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    config["native_evidence"] = artifact(tmp_path, "index", native)
    cfg, out = tmp_path / "config", tmp_path / "report"
    cfg.write_text(json.dumps(config))
    child = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.pooled_kv_routing.preflight",
            "--config",
            str(cfg),
            "--output",
            str(out),
        ],
        capture_output=True,
        timeout=10,
    )
    assert child.returncode == 1
    assert json.loads(out.read_text())["functional_status"] != "PASS"


def artifact(tmp_path: Path, name: str, value: object) -> dict:
    data = json.dumps(value).encode()
    path = tmp_path / name
    path.write_bytes(data)
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def capture(
    tmp_path: Path, name: str, role: str, identity: dict, observation: dict
) -> dict:
    return artifact(
        tmp_path,
        name,
        {
            "schema_version": 2,
            "role": role,
            "identity": identity,
            "terminal": "completed",
            "returncode": 0,
            "observation": observation,
            "logs": [artifact(tmp_path, name + "-stdout", "PASS")],
        },
    )


def load(ref: dict) -> dict:
    return json.loads(Path(ref["path"]).read_text())


def node_artifact(tmp_path: Path, name: str, cfg: dict, node: dict) -> dict:
    return capture(
        tmp_path,
        name,
        "inventory",
        {"run_id": cfg["run_id"], "source_head": cfg["source"]["head"]},
        node,
    )


def freeze_environment(tmp_path: Path, cfg: dict) -> None:
    env = cfg["environment"]
    env["artifact"] = capture(
        tmp_path,
        "environment",
        "environment",
        {"run_id": cfg["run_id"], "source_head": cfg["source"]["head"]},
        {
            **{k: v for k, v in env.items() if k != "artifact"},
            "binaries": [artifact(tmp_path, "installed-binary", "fixture-binary")],
            "argv": ["python", "native-probe.py"],
        },
    )


@pytest.fixture
def config(tmp_path: Path) -> dict:
    cfg = {
        "schema_version": 2,
        "run_id": "fixture-run-1",
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
        },
        "nodes": [],
        "native_evidence": None,
    }
    freeze_environment(tmp_path, cfg)
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
    }
    cfg["nodes"] = [node_artifact(tmp_path, "node-A", cfg, node)]
    bind_native(tmp_path, cfg)
    return cfg


def bind_native(tmp_path: Path, cfg: dict, cases: list | None = None) -> None:
    provenance = {
        k: cfg[k] for k in ("run_id", "source", "model", "environment", "nodes")
    }
    probe: dict[str, object] = {
        "provenance_sha256": hashlib.sha256(
            json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "cases": cases
        if cases is not None
        else [
            {
                "name": "single_gpu",
                "request_id": "req-1",
                "host_ids": ["machine-A"],
                "transport": "local",
                "remote_bytes": 0,
                "external_fetches": 0,
                "expected_token_ids": [1, 2],
                "actual_token_ids": [1, 2],
                "loader_extra_args": {},
            }
        ],
    }
    records = []
    for item in cast(list[dict[str, object]], probe["cases"]):
        case = {k: item[k] for k in ("name", "request_id", "host_ids")}
        identity = dict(
            case, run_id=cfg["run_id"], provenance_sha256=probe["provenance_sha256"]
        )
        case["reference"] = capture(
            tmp_path,
            cast(str, item["name"]) + "-reference",
            "reference",
            identity,
            {
                "token_ids": item["expected_token_ids"],
                "token_count": len(cast(list[int], item["expected_token_ids"])),
            },
        )
        case["probe"] = capture(
            tmp_path,
            cast(str, item["name"]) + "-probe",
            "probe",
            identity,
            {
                "token_ids": item["actual_token_ids"],
                "token_count": len(cast(list[int], item["actual_token_ids"])),
                **{
                    k: item[k]
                    for k in (
                        "transport",
                        "remote_bytes",
                        "external_fetches",
                        "loader_extra_args",
                    )
                },
            },
        )
        records.append(case)
    probe["cases"] = records
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
    case = probe["cases"][0]
    record = load(case["probe"])
    if field == "host_ids":
        case[field] = value
    elif field in ("returncode", "raw"):
        record["logs" if field == "raw" else field] = value
    else:
        record["observation"]["token_ids" if field == "actual_token_ids" else field] = (
            value
        )
    case["probe"] = artifact(tmp_path, "bad-probe", record)
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
    node = load(cfg["nodes"][0])["observation"]
    node.update(host_id="machine-B", gpu_uuid="GPU-B", rdma_gid="gid-B")
    cfg["nodes"].append(node_artifact(tmp_path, "node-B", cfg, node))
    case: dict[str, object] = {
        "expected_token_ids": [1, 2],
        "actual_token_ids": [1, 2],
        "loader_extra_args": {},
    }
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
            cast(list[str], item["host_ids"]).reverse()
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
    node = load(config["nodes"][0])["observation"]
    node.update(rdma_device=None, rdma_gid=None, link_gbps=None)
    config["nodes"][0] = node_artifact(tmp_path, "non-rdma-node", config, node)
    bind_native(tmp_path, config)
    report = preflight(config)
    assert report["functional_status"] == "PASS"
    assert report["checks"]["rdma"]["status"] == "NOT RUN"


def test_bare_metal_environment_uses_binary_manifest(
    config: dict, tmp_path: Path
) -> None:
    config["environment"]["image_digest"] = None
    freeze_environment(tmp_path, config)
    node = load(config["nodes"][0])["observation"]
    node["environment"] = config["environment"]
    config["nodes"][0] = node_artifact(tmp_path, "bare-metal", config, node)
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
    record = load(cfg["nodes"][1])
    node = record["observation"]
    if field == "raw":
        record["logs"] = value
    else:
        node[field] = value
    data = json.dumps(record).encode()
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
    case = probe["cases"][-1 if mutation == "skip_ignored" else 0]
    record = load(case["probe"])
    if mutation == "tcp":
        record["observation"]["transport"] = "tcp"
    elif mutation == "no_bytes":
        record["observation"]["remote_bytes"] = 0
    elif mutation == "skip_ignored":
        record["observation"]["loader_extra_args"] = {}
    else:
        probe["cases"].pop()
    case["probe"] = artifact(tmp_path, "mutated-capture", record)
    cfg["native_evidence"] = artifact(tmp_path, "bad-native", probe)
    assert preflight(cfg)["checks"]["native"]["status"] == "FAIL"


def test_unknown_new_head_remains_unqualified(config: dict, tmp_path: Path) -> None:
    config["source"]["head"] = "d" * 40
    freeze_environment(tmp_path, config)
    node = load(config["nodes"][0])["observation"]
    node.update(source=config["source"], environment=config["environment"])
    config["nodes"][0] = node_artifact(tmp_path, "new-head-node", config, node)
    bind_native(tmp_path, config)
    report = preflight(primary_config(tmp_path, config))
    assert report["evidence_status"] == "PASS"
    assert report["checks"]["resource_safety"]["status"] == "NOT RUN"
    assert report["eligible_for_primary"] is False


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
    node = load(config["nodes"][0])["observation"]
    node[field] = value
    config["nodes"][0] = node_artifact(tmp_path, "bad-functional-node", config, node)
    assert preflight(config)["checks"]["inventory"]["status"] == "FAIL"


@pytest.mark.parametrize("role", ["environment", "inventory", "reference", "probe"])
@pytest.mark.parametrize("fault", ["exit23", "partial", "stale", "missing"])
def test_cli_authoritative_capture_failures(
    config: dict, tmp_path: Path, role: str, fault: str
) -> None:
    native = load(config["native_evidence"])
    case = native["cases"][0]
    ref = (
        config["environment"]["artifact"]
        if role == "environment"
        else config["nodes"][0]
        if role == "inventory"
        else case[role]
    )
    record = load(ref)
    if fault == "exit23":
        record["returncode"] = 23  # Diagnostic stdout still contains PASS.
    elif fault == "partial":
        record["terminal"] = "collecting"
    elif fault == "stale":
        record["identity"]["run_id"] = "previous-run"
    else:
        del record["observation"]
    changed = artifact(tmp_path, "bad-" + role, record)
    if role == "environment":
        config["environment"]["artifact"] = changed
    elif role == "inventory":
        config["nodes"][0] = changed
    else:
        case[role] = changed
        config["native_evidence"] = artifact(tmp_path, "bad-index", native)
    cfg, out = tmp_path / "cli-config", tmp_path / "cli-report"
    cfg.write_text(json.dumps(config))
    child = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.pooled_kv_routing.preflight",
            "--config",
            str(cfg),
            "--output",
            str(out),
        ],
        capture_output=True,
        timeout=10,
    )
    report = json.loads(out.read_text())
    assert child.returncode == 1
    assert report["evidence_status"] == "FAIL"
    stage = (
        "configuration"
        if role == "environment"
        else "inventory"
        if role == "inventory"
        else "native"
    )
    assert report["checks"][stage]["status"] == "FAIL"
    assert report["functional_status"] != "PASS"


@pytest.mark.parametrize(
    "fault",
    [
        "truncated_tokens",
        "case_identity",
        "missing_reference",
        "unexpected_summary",
        "counter_drift",
    ],
)
def test_cli_capture_inventory_consistency(
    config: dict, tmp_path: Path, fault: str
) -> None:
    native = load(config["native_evidence"])
    case = native["cases"][0]
    record = load(case["probe"])
    if fault == "truncated_tokens":
        record["observation"]["token_ids"] = [1]
    elif fault == "case_identity":
        case["request_id"] = "wrong-request"
    elif fault == "missing_reference":
        del case["reference"]
    elif fault == "unexpected_summary":
        case["returncode"] = 0
    else:
        record["observation"]["external_fetches"] = 1
    case["probe"] = artifact(tmp_path, "changed-probe", record)
    config["native_evidence"] = artifact(tmp_path, "changed-index", native)
    cfg, out = tmp_path / "cli-config", tmp_path / "cli-report"
    cfg.write_text(json.dumps(config))
    child = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.pooled_kv_routing.preflight",
            "--config",
            str(cfg),
            "--output",
            str(out),
        ],
        capture_output=True,
        timeout=10,
    )
    assert child.returncode == 1
    assert json.loads(out.read_text())["checks"]["native"]["status"] == "FAIL"


@pytest.mark.parametrize(
    "field,value", [("binaries", []), ("argv", []), ("torch", "2.8.0")]
)
def test_environment_capture_semantics(
    config: dict, tmp_path: Path, field: str, value: object
) -> None:
    record = load(config["environment"]["artifact"])
    record["observation"][field] = value
    config["environment"]["artifact"] = artifact(tmp_path, "bad-env", record)
    report = preflight(config)
    assert report["checks"]["configuration"]["status"] == "FAIL"
