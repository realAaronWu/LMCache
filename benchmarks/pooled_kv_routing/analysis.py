# SPDX-License-Identifier: Apache-2.0
"""Fail-closed, offline pooled-routing evidence reader (packet version 1).

This bounded interchange format is NOT a raw proxy journal or RDMA certificate.
The parent adapter must retain its raw journals separately and construct:

* protocol JSON: version=1, baseline=LEW|Static, config_sha256, source_sha (40
  hex), corpus_sha256, primary_blocks (even 12..24), argv (baseline/Cost lists),
  cases. Every case has case_id, scenario (primary/equal-load/unique/slow-path/
  low-load/cost-observe; the last two are separate diagnostics),
  block (zero based), seed (101+block), order (balanced AB/BA), requests.
* Every request has request_id, request_class (shared/background), family,
  affinity (A/B/null), arrival_ns (relative barrier, [0,600s)), input_token_ids,
  expected_token_ids (uncached greedy reference; do not derive from treatment).
* artifacts/inventory.json: version=1, files: [{case_id, arm, path, sha256}].
  Exactly one file per registered case/arm, no unlisted files, no symlinks;
  flat filenames only. All hashes cover raw bytes except protocol_sha256, which
  covers sorted compact JSON (ensure_ascii=True, allow_nan=False).
* Each arm file: version=1, case_id, arm, protocol_sha256, config_sha256,
  source_sha, argv, returncode (integer), process_status (complete), stdout
  (diagnostic only), records. Every record: request_id, arrival_ns, ingress_ns,
  dispatch_ns, token_times_ns, end_ns, status, input_token_ids, token_ids,
  usage={prompt_tokens, completion_tokens}. All times integer nanoseconds on
  the same client monotonic clock relative to the arm barrier. Times must be
  ordered, generated-token times strictly increasing (batched/unknown boundaries
  cannot certify ITL); end <= arrival+10s. Exact generated IDs and usage required.

Unknown fields, JSON duplicate keys, bool-as-int, nonfinite values and missing
inventory fail. Max 144 cases, 100k requests/case, 128 MiB/JSON file. Frozen
protocol must be sealed externally before outcomes; this reader cannot prove
pre-registration, tokenizer semantics, raw-adapter fidelity, safety, causality,
resource cleanup, baseline tuning or the complete planned workload contract.
All benchmark ACs therefore stay NOT RUN. Report v2 separates packet_checks
(pedagogical/statistical checks only) from benchmark acceptance; neither short
fixtures nor a resealed truncated cohort can certify a benchmark component.
No eligible boolean can promote them. CLI exit 1 means FAIL, 2 NOT RUN
(argparse also uses 2 for grammar errors). A report is written on input failure.
"""

# Standard
from pathlib import Path
from typing import Any, TypedDict, cast
import argparse
import hashlib
import json
import math
import sys
import tempfile

# Third Party
import numpy as np


class RequestSpec(TypedDict):
    """Frozen request input/reference and evaluation schedule (relative ns)."""

    request_id: str
    request_class: str
    family: str
    affinity: str | None
    arrival_ns: int
    input_token_ids: list[int]
    expected_token_ids: list[int]


class CaseSpec(TypedDict):
    """One paired block; both arms must execute the same request cohort."""

    case_id: str
    scenario: str
    block: int
    seed: int
    order: list[str]
    requests: list[RequestSpec]


class Protocol(TypedDict):
    """Version 1 frozen experiment contract; field semantics are above."""

    version: int
    baseline: str
    config_sha256: str
    source_sha: str
    corpus_sha256: str
    primary_blocks: int
    argv: dict[str, list[str]]
    cases: list[CaseSpec]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _fields(value: Any, fields: str) -> None:
    _require(
        isinstance(value, dict) and set(value) == set(fields.split()),
        f"invalid fields: expected {fields}",
    )


def _integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and minimum <= value <= 2**63 - 1


def _hash(value: Any, size: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == size
        and all(c in "0123456789abcdef" for c in value)
    )


def _ids(value: Any) -> bool:
    return (
        isinstance(value, list)
        and 0 < len(value) <= 100_000
        and all(_integer(t) for t in value)
    )


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read(path: Path) -> dict[str, Any]:
    _require(path.stat().st_size <= 128 * 1024 * 1024, "JSON file exceeds size bound")
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=_pairs)
    except (json.JSONDecodeError, UnicodeError, RecursionError) as exc:
        raise ValueError(
            f"invalid JSON in {path.name}: {type(exc).__name__}: {exc}"
        ) from exc
    _require(isinstance(value, dict), "JSON root must be object")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _status(values: list[str]) -> str:
    return "FAIL" if "FAIL" in values else "NOT RUN" if "NOT RUN" in values else "PASS"


def _report(diagnostic: str) -> dict[str, Any]:
    return {
        "version": 2,
        "scope": "packet-statistics-only",
        "evidence_status": "FAIL",
        "correctness_status": "NOT RUN",
        "performance_status": "NOT RUN",
        "overall_status": "FAIL",
        "acceptance": {f"AC{i}": "NOT RUN" for i in range(7)},
        "metrics": {},
        "packet_checks": {},
        "diagnostics": [diagnostic],
    }


def _write_report(
    output: Path, protocol: Path, artifacts: Path, report: dict[str, Any]
) -> None:
    # Exclusive creation protects unknown files; only our own reports may be
    # atomically refreshed, so a failed decode cannot leave a stale success.
    _require(
        not output.resolve().is_relative_to(artifacts.resolve()),
        "output must be outside raw artifacts",
    )
    _require(
        not output.is_symlink() and output.resolve() != protocol.resolve(),
        "output aliases a protected input",
    )
    if output.exists():
        _require(
            output.is_file() and output.stat().st_nlink == 1,
            "output must not be a hardlink or non-file",
        )
        previous = _read(output)
        _require(
            previous.get("scope") == "packet-statistics-only"
            and previous.get("version") == 2,
            "output exists and is not a v2 analysis report",
        )
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=output.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
            temporary.replace(output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    else:
        with output.open("x") as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")


def _interval(baseline: list[float], cost: list[float]) -> dict[str, float]:
    a, b = np.asarray(baseline), np.asarray(cost)
    _require(
        bool(np.all(np.isfinite(a)))
        and bool(np.all(np.isfinite(b)))
        and bool(np.all(a > 0)),
        "undefined/nonfinite ratio metric",
    )
    indices = np.random.default_rng(42).integers(0, len(a), size=(10_000, len(a)))
    samples = 1 - b[indices].mean(axis=1) / a[indices].mean(axis=1)
    lower = float(np.quantile(samples, 0.025))
    upper = float(np.quantile(samples, 0.975))
    return {
        "estimate": float(1 - b.mean() / a.mean()),
        "lower": float(lower),
        "upper": float(upper),
    }


def _summarize(rows: list[tuple[float, float, float]]) -> dict[str, float]:
    ttfts, itls, ends = zip(*rows, strict=True)
    return {
        "requests": len(rows),
        "ttft": float(np.mean(ttfts)),
        "itl": float(np.quantile(itls, 0.95)),
        "throughput": len(rows) / max(600.0, max(ends)),
        "goodput": sum(t <= 0.5 and i <= 0.1 for t, i, _ in rows) / 600.0,
        "window_complete": sum(e <= 600 for e in ends),
        "drain_s": max(0.0, max(ends) - 600),
    }


def _validate_protocol(protocol: Protocol) -> None:
    _fields(
        protocol,
        "version baseline config_sha256 source_sha corpus_sha256 "
        "primary_blocks argv cases",
    )
    _require(
        type(protocol["version"]) is int and protocol["version"] == 1,
        "unsupported protocol version",
    )
    _require(
        protocol["baseline"] in ("LEW", "Static"),
        "baseline must be frozen LEW or Static",
    )
    for field in ("config_sha256", "corpus_sha256"):
        _require(_hash(protocol[field]), f"invalid {field}")
    _require(_hash(protocol["source_sha"], 40), "invalid source_sha")
    n = protocol["primary_blocks"]
    _require(
        _integer(n, 12) and n <= 24 and n % 2 == 0, "primary blocks must be even 12..24"
    )
    _fields(protocol["argv"], f"{protocol['baseline']} Cost")
    for argv in protocol["argv"].values():
        _require(
            isinstance(argv, list)
            and bool(argv)
            and all(isinstance(a, str) and bool(a) for a in argv),
            "invalid argv",
        )
    cases = protocol["cases"]
    _require(
        isinstance(cases, list) and 0 < len(cases) <= 144,
        "empty/oversized case inventory",
    )
    seen: set[str] = set()
    blocks: dict[str, list[int]] = {}
    baseline_first: dict[str, int] = {}
    for case in cases:
        _fields(case, "case_id scenario block seed order requests")
        case_id, scenario, block = case["case_id"], case["scenario"], case["block"]
        _require(
            isinstance(case_id, str) and bool(case_id) and case_id not in seen,
            "duplicate/invalid case ID",
        )
        seen.add(case_id)
        _require(
            scenario
            in (
                "primary",
                "equal-load",
                "unique",
                "slow-path",
                "low-load",
                "cost-observe",
            ),
            "unknown scenario",
        )
        _require(
            _integer(block)
            and block < 24
            and type(case["seed"]) is int
            and case["seed"] == 101 + block,
            "block/seed drift",
        )
        order = [protocol["baseline"], "Cost"]
        _require(
            case["order"] in (order, order[::-1]),
            "invalid frozen arm order",
        )
        blocks.setdefault(scenario, []).append(block)
        baseline_first[scenario] = baseline_first.get(scenario, 0) + (
            case["order"] == order
        )
        requests = case["requests"]
        _require(
            isinstance(requests, list) and 0 < len(requests) <= 100_000,
            "invalid request inventory",
        )
        ids: set[str] = set()
        for r in requests:
            _fields(
                r,
                "request_id request_class family affinity arrival_ns "
                "input_token_ids expected_token_ids",
            )
            _require(
                isinstance(r["request_id"], str)
                and bool(r["request_id"])
                and r["request_id"] not in ids,
                "duplicate/invalid request ID",
            )
            ids.add(r["request_id"])
            _require(
                r["request_class"] in ("shared", "background")
                and isinstance(r["family"], str)
                and bool(r["family"]),
                "invalid class/family",
            )
            _require(r["affinity"] in (None, "A", "B"), "invalid affinity")
            _require(
                r["request_class"] != "background" or r["affinity"] in ("A", "B"),
                "background affinity missing",
            )
            _require(
                _integer(r["arrival_ns"]) and r["arrival_ns"] < 600_000_000_000,
                "invalid arrival_ns",
            )
            _require(
                _ids(r["input_token_ids"]) and _ids(r["expected_token_ids"]),
                "invalid reference token IDs",
            )
    _require(
        sorted(blocks.get("primary", [])) == list(range(n)),
        "primary block inventory mismatch",
    )
    for scenario, numbers in blocks.items():
        _require(
            abs(2 * baseline_first[scenario] - len(numbers)) <= 1,
            f"unbalanced arm order in {scenario}",
        )
        _require(
            sorted(numbers) == list(range(len(numbers))),
            f"duplicate/missing blocks in {scenario}",
        )


def _arm_metrics(
    data: dict[str, Any],
    case: CaseSpec,
    arm: str,
    protocol: Protocol,
    protocol_sha: str,
) -> dict[str, Any]:
    _fields(
        data,
        "version case_id arm protocol_sha256 config_sha256 source_sha argv "
        "returncode process_status stdout records",
    )
    _require(
        type(data["version"]) is int
        and data["version"] == 1
        and data["case_id"] == case["case_id"]
        and data["arm"] == arm,
        "arm identity mismatch",
    )
    _require(
        data["protocol_sha256"] == protocol_sha
        and data["config_sha256"] == protocol["config_sha256"]
        and data["source_sha"] == protocol["source_sha"],
        "provenance SHA drift",
    )
    _require(data["argv"] == protocol["argv"][arm], "actual argv drift")
    _require(
        type(data["returncode"]) is int
        and data["returncode"] == 0
        and data["process_status"] == "complete",
        "process failed/partial despite stdout",
    )
    _require(isinstance(data["stdout"], str), "stdout must be diagnostic text")
    records = data["records"]
    _require(
        isinstance(records, list) and len(records) == len(case["requests"]),
        "request inventory mismatch",
    )
    expected = {r["request_id"]: r for r in case["requests"]}
    seen: set[str] = set()
    observations: list[tuple[float, float, float]] = []
    classes: dict[str, list[tuple[float, float, float]]] = {}
    for r in records:
        _fields(
            r,
            "request_id arrival_ns ingress_ns dispatch_ns token_times_ns end_ns "
            "status input_token_ids token_ids usage",
        )
        rid = r["request_id"]
        _require(
            isinstance(rid, str) and rid in expected and rid not in seen,
            "request ID inventory mismatch",
        )
        seen.add(rid)
        reference = expected[rid]
        _require(r["status"] == "complete", f"request {rid} failed/partial")
        _require(
            r["arrival_ns"] == reference["arrival_ns"] and _integer(r["arrival_ns"]),
            "schedule drift",
        )
        _require(
            _ids(r["input_token_ids"])
            and r["input_token_ids"] == reference["input_token_ids"],
            "input token drift",
        )
        _require(
            _ids(r["token_ids"]) and r["token_ids"] == reference["expected_token_ids"],
            "output token mismatch",
        )
        _fields(r["usage"], "prompt_tokens completion_tokens")
        for field, tokens in (
            ("prompt_tokens", r["input_token_ids"]),
            ("completion_tokens", r["token_ids"]),
        ):
            _require(
                _integer(r["usage"][field]) and r["usage"][field] == len(tokens),
                "usage/token count mismatch",
            )
        times = r["token_times_ns"]
        _require(
            isinstance(times, list)
            and len(times) == len(r["token_ids"])
            and len(times) >= 2
            and all(_integer(t) for t in times),
            "missing/invalid token timestamps",
        )
        for field in ("ingress_ns", "dispatch_ns", "end_ns"):
            _require(_integer(r[field]), f"invalid finite integer {field}")
        _require(
            r["arrival_ns"]
            <= r["ingress_ns"]
            <= r["dispatch_ns"]
            <= times[0]
            <= times[-1]
            <= r["end_ns"]
            <= r["arrival_ns"] + 10_000_000_000,
            "timestamp/deadline ordering failure",
        )
        _require(
            all(a < b for a, b in zip(times, times[1:], strict=False)),
            "batched/nonmonotonic token timing cannot qualify ITL",
        )
        ttft = (times[0] - r["arrival_ns"]) / 1e9
        itl = max(b - a for a, b in zip(times, times[1:], strict=False)) / 1e9
        _require(ttft > 0, "TTFT must be positive")
        row = (ttft, itl, r["end_ns"] / 1e9)
        observations.append(row)
        classes.setdefault(reference["request_class"], []).append(row)
    return {
        **_summarize(observations),
        "classes": {key: _summarize(rows) for key, rows in classes.items()},
    }


def analyze(protocol: Protocol, artifacts: Path) -> dict[str, Any]:
    """Validate a v1 packet and compute paired-block statistics, without executing it.

    Args:
        protocol: Externally frozen v1 contract described in the module docstring.
        artifacts: Directory containing inventory.json and exactly its arm files.
    Returns:
        Report v2 with packet_checks, metrics (improvement ratios, not percent),
        diagnostics and evidence status. All benchmark AC0..6 and performance
        qualification remain NOT RUN until the complete benchmark contract is
        independently validated outside this packet-statistics reader.
        Bootstrap is 10,000 paired resamples, NumPy default_rng seed 42, linear
        2.5/97.5 percentiles. Blocks, not requests, receive equal weight.
    Raises:
        ValueError: Any malformed, incomplete, drifting or unsuccessful packet.
        OSError: Required file cannot be read. Such failures become FAIL in CLI.
    """
    _validate_protocol(protocol)
    protocol_sha = _digest(protocol)
    inventory = _read(artifacts / "inventory.json")
    _fields(inventory, "version files")
    _require(
        type(inventory["version"]) is int
        and inventory["version"] == 1
        and isinstance(inventory["files"], list),
        "invalid inventory version/files",
    )
    expected = {
        (c["case_id"], a)
        for c in protocol["cases"]
        for a in (protocol["baseline"], "Cost")
    }
    files: dict[tuple[str, str], Path] = {}
    names = {"inventory.json"}
    for item in inventory["files"]:
        _fields(item, "case_id arm path sha256")
        _require(
            isinstance(item["case_id"], str) and isinstance(item["arm"], str),
            "invalid inventory IDs",
        )
        key = (item["case_id"], item["arm"])
        name = item["path"]
        _require(
            key in expected and key not in files, "duplicate/unexpected arm inventory"
        )
        _require(
            isinstance(name, str)
            and name.endswith(".json")
            and Path(name).name == name
            and name not in names,
            "unsafe/duplicate packet path",
        )
        path = artifacts / name
        _require(not path.is_symlink() and path.is_file(), "missing/symlink artifact")
        _require(path.stat().st_size <= 128 * 1024 * 1024, "oversized arm artifact")
        _require(
            _hash(item["sha256"])
            and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"],
            "artifact SHA mismatch",
        )
        files[key] = path
        names.add(name)
    _require(set(files) == expected, "missing arm inventory")
    _require(
        not (artifacts / "inventory.json").is_symlink()
        and {p.name for p in artifacts.iterdir()} == names,
        "unlisted artifact inventory",
    )
    groups: dict[str, dict[str, list[dict[str, float]]]] = {}
    for case in protocol["cases"]:
        group = groups.setdefault(
            case["scenario"], {protocol["baseline"]: [], "Cost": []}
        )
        for arm in (protocol["baseline"], "Cost"):
            group[arm].append(
                _arm_metrics(
                    _read(files[(case["case_id"], arm)]),
                    case,
                    arm,
                    protocol,
                    protocol_sha,
                )
            )
    metrics: dict[str, Any] = {}
    for scenario, group in groups.items():
        metrics[scenario] = {
            metric: _interval(
                [r[metric] for r in group[protocol["baseline"]]],
                [r[metric] for r in group["Cost"]],
            )
            for metric in ("ttft", "itl", "throughput")
        }
        metrics[scenario]["arms"] = group
    primary = metrics["primary"]
    checks = {}
    checks["ttft_threshold"] = (
        "PASS"
        if all(
            value > 0.05 and not math.isclose(value, 0.05, rel_tol=0, abs_tol=1e-12)
            for value in (primary["ttft"]["estimate"], primary["ttft"]["lower"])
        )
        else "FAIL"
    )
    checks["token_equality"] = "PASS"
    controls = []
    for scenario in ("equal-load", "unique", "slow-path"):
        if scenario not in metrics or len(groups[scenario]["Cost"]) < 8:
            controls.append("NOT RUN")
        else:
            controls.append(
                "PASS" if metrics[scenario]["ttft"]["lower"] >= -0.03 else "FAIL"
            )
    # Diagnostic reporting is required but has no invented efficacy threshold.
    for scenario in ("low-load", "cost-observe"):
        if scenario not in metrics or len(groups[scenario]["Cost"]) < 4:
            controls.append("NOT RUN")
    checks["control_regressions"] = _status(controls)
    baseline_good = [r["goodput"] for r in groups["primary"][protocol["baseline"]]]
    cost_good = [r["goodput"] for r in groups["primary"]["Cost"]]
    # A zero baseline goodput is valid, but cannot define a ratio CI.
    good_delta = np.asarray(cost_good) - np.asarray(baseline_good)
    idx = np.random.default_rng(42).integers(
        0, len(good_delta), size=(10_000, len(good_delta))
    )
    good_lower = float(np.quantile(good_delta[idx].mean(axis=1), 0.025))
    primary["goodput_delta_lower"] = good_lower
    checks["streaming"] = (
        "PASS"
        if primary["itl"]["lower"] >= -0.03
        and primary["throughput"]["upper"] <= 0.02
        and good_lower >= 0
        else "FAIL"
    )
    return {
        "version": 2,
        "scope": "packet-statistics-only",
        "protocol_sha256": protocol_sha,
        "bootstrap": {"seed": 42, "resamples": 10_000, "unit": "paired block"},
        "evidence_status": "PASS",
        "correctness_status": "NOT RUN",
        "performance_status": "NOT RUN",
        "overall_status": "NOT RUN",
        "acceptance": {f"AC{i}": "NOT RUN" for i in range(7)},
        "packet_checks": checks,
        "metrics": metrics,
        "diagnostics": [
            "Packet statistics are not benchmark qualification. The complete "
            "planned workload, deployment, safety, mechanism and independent "
            "checker evidence have not been validated; all ACs remain NOT RUN."
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """Analyze --protocol FILE --artifacts DIR --output FILE; persist even FAIL.

    Returns 1 FAIL or 2 NOT RUN. Output must live outside raw artifacts and not
    alias protocol. New paths are exclusive; existing v2 analysis reports alone
    may be atomically refreshed. Other existing files/aliases are preserved.
    Input errors retain a fresh structured diagnostic report at a safe output.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = analyze(cast(Protocol, _read(args.protocol)), args.artifacts)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        report = _report(str(exc))
    try:
        _write_report(args.output, args.protocol, args.artifacts, report)
    except (OSError, ValueError) as exc:
        print(f"analysis: cannot write report: {exc}", file=sys.stderr)
        return 1
    print(f"analysis: {report['overall_status']}", file=sys.stderr)
    return 1 if report["overall_status"] == "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
