# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
from typing import Any
import copy
import hashlib
import json
import subprocess
import sys

# Third Party
import pytest

# First Party
from benchmarks.pooled_kv_routing import analysis


def packet(tmp_path: Path, improvement: float = 0.10) -> tuple[dict[str, Any], Path]:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    protocol: dict[str, Any] = {
        "version": 1,
        "baseline": "LEW",
        "config_sha256": "a" * 64,
        "source_sha": "b" * 40,
        "corpus_sha256": "c" * 64,
        "primary_blocks": 12,
        "cases": [],
        "argv": {"LEW": ["runner", "LEW"], "Cost": ["runner", "Cost"]},
    }
    inventory = {"version": 1, "files": []}
    for scenario, count in (
        ("primary", 12),
        ("equal-load", 8),
        ("unique", 8),
        ("slow-path", 8),
        ("low-load", 4),
        ("cost-observe", 4),
    ):
        for block in range(count):
            case_id = f"{scenario}-{block}"
            requests = [
                {
                    "request_id": "shared",
                    "request_class": "shared",
                    "family": "f1",
                    "affinity": None,
                    "arrival_ns": 0,
                    "input_token_ids": [1, 2],
                    "expected_token_ids": [5, 6],
                },
                {
                    "request_id": "background",
                    "request_class": "background",
                    "family": "f2",
                    "affinity": "A",
                    "arrival_ns": 100_000_000,
                    "input_token_ids": [3, 4],
                    "expected_token_ids": [7, 8],
                },
            ]
            case = {
                "case_id": case_id,
                "scenario": scenario,
                "block": block,
                "seed": 101 + block,
                "order": ["LEW", "Cost"] if block % 2 == 0 else ["Cost", "LEW"],
                "requests": requests,
            }
            protocol["cases"].append(case)
            for arm in ("LEW", "Cost"):
                ttft = int(
                    200_000_000
                    * (
                        1 - improvement
                        if arm == "Cost" and scenario == "primary"
                        else 1
                    )
                )
                records = []
                for request in requests:
                    arrival = request["arrival_ns"]
                    records.append(
                        {
                            "request_id": request["request_id"],
                            "arrival_ns": arrival,
                            "ingress_ns": arrival,
                            "dispatch_ns": arrival,
                            "token_times_ns": [
                                arrival + ttft,
                                arrival + ttft + 10_000_000,
                            ],
                            "end_ns": arrival + ttft + 10_000_000,
                            "status": "complete",
                            "input_token_ids": request["input_token_ids"],
                            "token_ids": request["expected_token_ids"],
                            "usage": {"prompt_tokens": 2, "completion_tokens": 2},
                        }
                    )
                data = {
                    "version": 1,
                    "case_id": case_id,
                    "arm": arm,
                    "protocol_sha256": hashlib.sha256(
                        json.dumps(
                            protocol, sort_keys=True, separators=(",", ":")
                        ).encode()
                    ).hexdigest(),
                    "config_sha256": protocol["config_sha256"],
                    "source_sha": protocol["source_sha"],
                    "argv": protocol["argv"][arm],
                    "returncode": 0,
                    "process_status": "complete",
                    "stdout": "PASS",
                    "records": records,
                }
                name = f"{case_id}-{arm}.json"
                (artifacts / name).write_text(json.dumps(data))
                inventory["files"].append(
                    {"case_id": case_id, "arm": arm, "path": name, "sha256": ""}
                )
    # Seal only after the complete pre-registered case inventory exists.
    digest = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    for item in inventory["files"]:
        path = artifacts / item["path"]
        data = json.loads(path.read_text())
        data["protocol_sha256"] = digest
        path.write_text(json.dumps(data))
        item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (artifacts / "inventory.json").write_text(json.dumps(inventory))
    return protocol, artifacts


def mutate(artifacts: Path, field: str) -> None:
    inventory = json.loads((artifacts / "inventory.json").read_text())
    item = inventory["files"][0]
    path = artifacts / item["path"]
    data = json.loads(path.read_text())
    record = data["records"][0]
    if field == "missingarm":
        inventory["files"].pop(0)
        path.unlink()
    elif field == "missingrequest":
        data["records"].pop()
    elif field == "mismatch":
        record["token_ids"][0] = 999
    elif field == "booltimestamp":
        record["ingress_ns"] = False
    elif field == "nan":
        record["end_ns"] = float("nan")
    elif field == "schedule":
        record["arrival_ns"] += 1
    elif field == "partial":
        record["status"] = "partial"
    elif field == "crash":
        data["returncode"] = 1
    elif field == "drift":
        data["argv"].append("--changed")
    elif field == "sha":
        data["config_sha256"] = "d" * 64
    if field != "missingarm":
        path.write_text(json.dumps(data))
        item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (artifacts / "inventory.json").write_text(json.dumps(inventory))


def test_paired_statistics_and_qualification_boundaries(tmp_path: Path) -> None:
    protocol, artifacts = packet(tmp_path)
    report = analysis.analyze(protocol, artifacts)
    assert report["evidence_status"] == "PASS"
    assert report["packet_checks"]["ttft_threshold"] == "PASS"
    assert report["packet_checks"]["control_regressions"] == "PASS"
    assert report["packet_checks"]["streaming"] == "PASS"
    assert set(report["acceptance"].values()) == {"NOT RUN"}
    assert report["metrics"]["primary"]["ttft"]["estimate"] == pytest.approx(0.10)
    assert report["metrics"]["primary"]["ttft"]["lower"] == pytest.approx(0.10)
    assert report["overall_status"] == "NOT RUN"
    assert report["acceptance"]["AC0"] == "NOT RUN"
    assert report["acceptance"]["AC5"] == "NOT RUN"


def test_four_percent_fails_packet_threshold_not_benchmark_qualification(
    tmp_path: Path,
) -> None:
    protocol, artifacts = packet(tmp_path, 0.04)
    report = analysis.analyze(protocol, artifacts)
    assert report["evidence_status"] == "PASS"
    assert report["packet_checks"]["ttft_threshold"] == "FAIL"


@pytest.mark.parametrize(
    "field",
    [
        "missingarm",
        "missingrequest",
        "mismatch",
        "booltimestamp",
        "nan",
        "schedule",
        "partial",
        "crash",
        "drift",
        "sha",
    ],
)
def test_public_cli_rejects_bad_packets(tmp_path: Path, field: str) -> None:
    protocol, artifacts = packet(tmp_path)
    mutate(artifacts, field)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    output = tmp_path / "acceptance.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.pooled_kv_routing.analysis",
            "--protocol",
            str(protocol_path),
            "--artifacts",
            str(artifacts),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stderr
    report = json.loads(output.read_text())
    assert report["overall_status"] == "FAIL"
    assert report["acceptance"]["AC1"] != "PASS"
    assert report["diagnostics"]


def test_missing_controls_and_boolean_attestations_cannot_qualify(
    tmp_path: Path,
) -> None:
    protocol, artifacts = packet(tmp_path)
    altered = copy.deepcopy(protocol)
    altered["rdma_eligible"] = True
    with pytest.raises(ValueError):
        analysis.analyze(altered, artifacts)
    # No source-side flag can promote the external safety/mechanism gates.
    assert analysis.analyze(protocol, artifacts)["acceptance"]["AC6"] == "NOT RUN"


def reseal(protocol: dict[str, Any], artifacts: Path) -> None:
    inventory = json.loads((artifacts / "inventory.json").read_text())
    digest = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    keep = {c["case_id"] for c in protocol["cases"]}
    retained = []
    for item in inventory["files"]:
        path = artifacts / item["path"]
        if item["case_id"] not in keep:
            path.unlink()
            continue
        data = json.loads(path.read_text())
        data["protocol_sha256"] = digest
        path.write_text(json.dumps(data))
        item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        retained.append(item)
    inventory["files"] = retained
    (artifacts / "inventory.json").write_text(json.dumps(inventory))


def test_controls_cannot_hide_missing_diagnostics(tmp_path: Path) -> None:
    protocol, artifacts = packet(tmp_path)
    # Three good negative controls cannot replace low-load/observation diagnostics.
    protocol["cases"] = [
        c
        for c in protocol["cases"]
        if c["scenario"] not in ("low-load", "cost-observe")
    ]
    reseal(protocol, artifacts)
    assert (
        analysis.analyze(protocol, artifacts)["packet_checks"]["control_regressions"]
        == "NOT RUN"
    )


def test_absent_control_is_not_run_and_unlisted_evidence_rejected(
    tmp_path: Path,
) -> None:
    protocol, artifacts = packet(tmp_path)
    protocol["cases"] = [c for c in protocol["cases"] if c["scenario"] != "slow-path"]
    reseal(protocol, artifacts)
    assert (
        analysis.analyze(protocol, artifacts)["packet_checks"]["control_regressions"]
        == "NOT RUN"
    )
    (artifacts / "hidden.log").write_text("PASS")
    with pytest.raises(ValueError, match="unlisted"):
        analysis.analyze(protocol, artifacts)


def test_valid_cli_persists_not_run_and_malformed_json_persists_failure(
    tmp_path: Path,
) -> None:
    protocol, artifacts = packet(tmp_path)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    output = tmp_path / "result.json"
    args = [
        "--protocol",
        str(path),
        "--artifacts",
        str(artifacts),
        "--output",
        str(output),
    ]
    assert analysis.main(args) == 2
    assert json.loads(output.read_text())["overall_status"] == "NOT RUN"
    path.write_text('{"version": 1, "version": 1}')
    assert analysis.main(args) == 1
    assert "duplicate JSON key" in json.loads(output.read_text())["diagnostics"][0]


def test_ratio_of_block_means_and_paired_interval_are_reproducible(
    tmp_path: Path,
) -> None:
    protocol, artifacts = packet(tmp_path)
    for path in artifacts.glob("primary-*.json"):
        data = json.loads(path.read_text())
        block = int(data["case_id"].split("-")[1])
        ttft = 200_000_000 if block % 2 == 0 else 400_000_000
        if data["arm"] == "Cost":
            ttft = 180_000_000 if block % 2 == 0 else 320_000_000
        for r in data["records"]:
            r["token_times_ns"] = [
                r["arrival_ns"] + ttft,
                r["arrival_ns"] + ttft + 10_000_000,
            ]
            r["end_ns"] = r["token_times_ns"][-1]
        path.write_text(json.dumps(data))
    reseal(protocol, artifacts)
    report = analysis.analyze(protocol, artifacts)
    metric = report["metrics"]["primary"]["ttft"]
    assert metric["estimate"] == pytest.approx(1 - 0.5 / 0.6)
    assert metric["lower"] < metric["estimate"] < metric["upper"]
    assert report == analysis.analyze(protocol, artifacts)


def test_exact_five_percent_does_not_pass_strict_threshold(tmp_path: Path) -> None:
    protocol, artifacts = packet(tmp_path, 0.05)
    assert (
        analysis.analyze(protocol, artifacts)["packet_checks"]["ttft_threshold"]
        == "FAIL"
    )


@pytest.mark.parametrize("metric", ["itl", "drain", "goodput"])
def test_streaming_drain_or_goodput_regression(tmp_path: Path, metric: str) -> None:
    protocol, artifacts = packet(tmp_path)
    if metric == "drain":
        for case in protocol["cases"]:
            for r in case["requests"]:
                r["arrival_ns"] += 599_500_000_000
    for path in artifacts.glob("*.json"):
        if path.name == "inventory.json":
            continue
        data = json.loads(path.read_text())
        for r in data["records"]:
            if metric == "drain":
                for key in ("arrival_ns", "ingress_ns", "dispatch_ns", "end_ns"):
                    r[key] += 599_500_000_000
                r["token_times_ns"] = [t + 599_500_000_000 for t in r["token_times_ns"]]
                if data["arm"] == "Cost":
                    r["end_ns"] = r["arrival_ns"] + 9_000_000_000
            elif data["arm"] == "Cost" and data["case_id"].startswith("primary-"):
                if metric == "itl":
                    r["token_times_ns"][-1] += 10_000_000
                else:
                    r["token_times_ns"] = [
                        r["arrival_ns"] + 600_000_000,
                        r["arrival_ns"] + 610_000_000,
                    ]
                r["end_ns"] = r["token_times_ns"][-1]
        path.write_text(json.dumps(data))
    reseal(protocol, artifacts)
    report = analysis.analyze(protocol, artifacts)
    if metric == "drain":
        row = report["metrics"]["primary"]["arms"]["Cost"][0]
        assert row["drain_s"] > 0
        assert row["throughput"] < 2 / 600
    else:
        assert report["packet_checks"]["streaming"] == "FAIL"


def test_background_class_remains_in_report(tmp_path: Path) -> None:
    protocol, artifacts = packet(tmp_path)
    row = analysis.analyze(protocol, artifacts)["metrics"]["primary"]["arms"]["Cost"][0]
    assert row["classes"]["background"]["requests"] == 1
    assert row["classes"]["shared"]["requests"] == 1


def test_frozen_balanced_order_not_forced_to_alternate(tmp_path: Path) -> None:
    protocol, artifacts = packet(tmp_path)
    for case in protocol["cases"]:
        case["order"] = case["order"][::-1]
    reseal(protocol, artifacts)
    assert analysis.analyze(protocol, artifacts)["evidence_status"] == "PASS"
    for case in protocol["cases"]:
        case["order"] = ["LEW", "Cost"]
    reseal(protocol, artifacts)
    with pytest.raises(ValueError, match="unbalanced"):
        analysis.analyze(protocol, artifacts)


def invoke_analysis(
    protocol_path: Path, artifacts: Path, output: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.pooled_kv_routing.analysis",
            "--protocol",
            str(protocol_path),
            "--artifacts",
            str(artifacts),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("cohort", ["short-tokens", "no-background"])
def test_cli_fixture_statistics_never_qualify_benchmark(
    tmp_path: Path, cohort: str
) -> None:
    protocol, artifacts = packet(tmp_path)
    if cohort == "no-background":
        for case in protocol["cases"]:
            case["requests"] = case["requests"][:1]
        for path in artifacts.glob("*-*.json"):
            data = json.loads(path.read_text())
            data["records"] = data["records"][:1]
            path.write_text(json.dumps(data))
        reseal(protocol, artifacts)
    source = tmp_path / "protocol.json"
    source.write_text(json.dumps(protocol))
    output = tmp_path / "report.json"
    result = invoke_analysis(source, artifacts, output)
    assert result.returncode == 2
    report = json.loads(output.read_text())
    assert set(report["acceptance"].values()) == {"NOT RUN"}
    assert report["performance_status"] == "NOT RUN"
    assert report["scope"] == "packet-statistics-only"
    assert report["packet_checks"]["ttft_threshold"] == "PASS"


@pytest.mark.parametrize(
    "alias", ["protocol", "symlink", "hardlink", "arm-hardlink", "inventory-hardlink"]
)
def test_cli_preserves_protected_inputs(tmp_path: Path, alias: str) -> None:
    protocol, artifacts = packet(tmp_path)
    source = tmp_path / "protocol.json"
    source.write_text(json.dumps(protocol))
    target = (
        artifacts / "primary-0-LEW.json"
        if alias == "arm-hardlink"
        else artifacts / "inventory.json"
        if alias == "inventory-hardlink"
        else source
    )
    before = target.read_bytes()
    output = source if alias == "protocol" else tmp_path / "alias.json"
    if alias == "symlink":
        output.symlink_to(target)
    elif "hardlink" in alias:
        output.hardlink_to(target)
    result = invoke_analysis(source, artifacts, output)
    assert result.returncode == 1
    assert target.read_bytes() == before
    assert "output" in result.stderr


@pytest.mark.parametrize(
    "payload",
    [b'{"x":' + b"[" * 20000 + b"0" + b"]" * 20000 + b"}", b"\xff", b"{"],
    ids=["depth", "unicode", "syntax"],
)
@pytest.mark.parametrize("location", ["protocol", "arm"])
def test_cli_decoder_failure_replaces_stale_report(
    tmp_path: Path, payload: bytes, location: str
) -> None:
    protocol, artifacts = packet(tmp_path)
    source = tmp_path / "protocol.json"
    source.write_text(json.dumps(protocol))
    output = tmp_path / "result.json"
    assert invoke_analysis(source, artifacts, output).returncode == 2
    before = output.read_bytes()
    if location == "protocol":
        source.write_bytes(payload)
    else:
        inventory = json.loads((artifacts / "inventory.json").read_text())
        item = inventory["files"][0]
        (artifacts / item["path"]).write_bytes(payload)
        item["sha256"] = hashlib.sha256(payload).hexdigest()
        (artifacts / "inventory.json").write_text(json.dumps(inventory))
    result = invoke_analysis(source, artifacts, output)
    assert result.returncode == 1 and "Traceback" not in result.stderr
    report = json.loads(output.read_text())
    assert output.read_bytes() != before
    assert report["evidence_status"] == report["overall_status"] == "FAIL"
    assert report["diagnostics"]


def test_cli_does_not_overwrite_unrelated_existing_file(tmp_path: Path) -> None:
    protocol, artifacts = packet(tmp_path)
    source = tmp_path / "protocol.json"
    source.write_text(json.dumps(protocol))
    output = tmp_path / "other.json"
    output.write_text('{"licensed_corpus": "keep"}')
    before = output.read_bytes()
    assert invoke_analysis(source, artifacts, output).returncode == 1
    assert output.read_bytes() == before


def test_cli_never_refreshes_report_when_used_as_protocol(tmp_path: Path) -> None:
    protocol, artifacts = packet(tmp_path)
    source = tmp_path / "protocol.json"
    source.write_text(json.dumps(protocol))
    report = tmp_path / "result.json"
    assert invoke_analysis(source, artifacts, report).returncode == 2
    original = report.read_bytes()
    result = invoke_analysis(report, artifacts, report)
    assert result.returncode == 1
    assert report.read_bytes() == original
