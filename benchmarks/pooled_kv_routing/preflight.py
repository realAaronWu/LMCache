# SPDX-License-Identifier: Apache-2.0
"""Offline, fail-closed qualification of frozen native probe evidence.

This module never launches a probe, trusts a PASS flag, or certifies performance.
See preflight-contract.md for the input schema and evidence trust boundary.
"""

# Standard
from pathlib import Path
from typing import TypedDict, cast
import argparse
import hashlib
import json
import os
import re
import stat

BASE = "bcafb031401d404a5dabae4853f5e23bf9e5685c"
JSON_LIMIT = 1024**2
RAW_LIMIT = 64 * 1024**2
MODEL_LIMIT = 64 * 1024**3
PRIMARY = "primary_two_node_rdma"
ENV_KEYS = {
    "python",
    "vllm",
    "torch",
    "cuda",
    "driver",
    "nixl",
    "image_digest",
    "artifact",
}


class PreflightReport(TypedDict):
    """Serialized qualification result; statuses retain their runtime checks."""

    schema_version: int
    evidence_status: str
    functional_status: str
    eligible_for_primary: bool
    performance_status: str
    checks: dict[str, dict[str, str]]
    diagnostics: list[str]


def _object(value: object, keys: set[str]) -> dict:
    """Require exactly the documented fields at an untrusted JSON boundary."""
    if type(value) is not dict or value.keys() != keys:
        raise ValueError(f"expected object fields: {sorted(keys)}")
    return value


def _text(value: object) -> str:
    """Require a bounded nonblank identifier, not a coerced scalar."""
    if type(value) is not str or not value.strip() or len(value) > 4096:
        raise ValueError("expected nonempty bounded string")
    return value


def _hex(value: object, length: int) -> None:
    """Require a full lowercase immutable identifier."""
    if not re.fullmatch(f"[0-9a-f]{{{length}}}", _text(value)):
        raise ValueError(f"expected {length}-digit immutable hex digest")


def _integer(value: object, minimum: int = 0, maximum: int = 2**63 - 1) -> None:
    """Reject booleans and out-of-range integral counters."""
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("expected bounded integer, not bool")


def _sequence(value: object, minimum: int, maximum: int) -> list:
    """Bound inventories before processing their contents."""
    if type(value) is not list or not minimum <= len(value) <= maximum:
        raise ValueError("invalid list size/type")
    return value


def _pairs(pairs: list[tuple[str, object]]) -> dict:
    """Reject duplicate JSON fields rather than silently overwriting evidence."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    """Reject non-JSON numeric constants accepted by Python's decoder."""
    raise ValueError(f"nonfinite JSON number: {value}")


def _json(data: bytes) -> object:
    """Decode bounded JSON, preserving invalid-input failures for the caller."""
    if len(data) > JSON_LIMIT:
        raise ValueError("JSON exceeds 1 MiB limit")
    return json.loads(data, object_pairs_hook=_pairs, parse_constant=_constant)


def _read(path: str, limit: int, digest: str | None = None) -> bytes:
    """Read regular files only, streaming large hashed payloads in bounded memory.

    Without a digest, returns bytes for JSON parsing. With one, returns no
    payload and verifies its SHA256. Nonblocking open prevents FIFO hangs.
    """
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
            raise ValueError("evidence must be a nonempty bounded regular file")
        hasher = hashlib.sha256()
        chunks = []
        total = 0
        while chunk := stream.read(min(JSON_LIMIT, limit + 1 - total)):
            total += len(chunk)
            if total > limit:
                raise ValueError("evidence grew beyond resource limit")
            hasher.update(chunk)
            if digest is None:
                chunks.append(chunk)
        if total != info.st_size:
            raise ValueError("evidence size changed during read")
        if digest is not None and hasher.hexdigest() != digest:
            raise ValueError("evidence SHA256 mismatch")
        return b"".join(chunks)


def _ref(ref: object, limit: int = RAW_LIMIT) -> bytes:
    """Verify a local artifact reference; return small JSON evidence bytes."""
    ref = _object(ref, {"path", "sha256"})
    path = _text(ref["path"])
    if not Path(path).is_absolute():
        raise ValueError("artifact path must be absolute")
    _hex(ref["sha256"], 64)
    if limit == JSON_LIMIT:
        data = _read(path, limit)
        if hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise ValueError("evidence SHA256 mismatch")
        return data
    return _read(path, limit, ref["sha256"])


def _capture(ref: object, role: str, identity: dict) -> dict:
    """Read one authoritative collector record, never a success summary.

    The collector writes terminal only after child exit and complete structured
    output collection. Diagnostic logs are retained but cannot override it.
    """
    record = _object(
        _json(_ref(ref, JSON_LIMIT)),
        {
            "schema_version",
            "role",
            "identity",
            "terminal",
            "returncode",
            "observation",
            "logs",
        },
    )
    _integer(record["schema_version"], 2, 2)
    if record["role"] != role or record["identity"] != identity:
        raise ValueError(f"misassociated {role} capture identity")
    if record["terminal"] != "completed":
        raise ValueError(f"incomplete {role} capture")
    _integer(record["returncode"], 0, 0)
    for log in _sequence(record["logs"], 1, 16):
        _ref(log)
    if type(record["observation"]) is not dict:
        raise ValueError("capture observation must be an object")
    return record["observation"]


def _configuration(config: object) -> dict:
    """Validate the frozen provenance schema and verify listed model payloads."""
    cfg = _object(
        config,
        {
            "schema_version",
            "run_id",
            "scope",
            "source",
            "model",
            "environment",
            "nodes",
            "native_evidence",
        },
    )
    _integer(cfg["schema_version"], 2, 2)
    _text(cfg["run_id"])
    if cfg["scope"] not in ("functional_single_gpu", PRIMARY):
        raise ValueError("unknown qualification scope")
    source = _object(cfg["source"], {"base", "head", "patch_sha256", "artifact"})
    _hex(source["base"], 40)
    _hex(source["head"], 40)
    _hex(source["patch_sha256"], 64)
    _ref(source["artifact"])
    if source["base"] != BASE:
        raise ValueError("source base differs from frozen candidate")
    model = _object(cfg["model"], {"id", "revision", "tokenizer_revision", "files"})
    _text(model["id"])
    _hex(model["revision"], 40)
    _hex(model["tokenizer_revision"], 40)
    files = _sequence(model["files"], 1, 256)
    paths = set()
    for ref in files:
        _ref(ref, MODEL_LIMIT)
        if ref["path"] in paths:
            raise ValueError("duplicate model payload")
        paths.add(ref["path"])
    env = _object(cfg["environment"], ENV_KEYS)
    observed = _capture(
        env["artifact"],
        "environment",
        {"run_id": cfg["run_id"], "source_head": source["head"]},
    )
    _object(observed, ENV_KEYS - {"artifact"} | {"binaries", "argv"})
    if {k: observed[k] for k in ENV_KEYS - {"artifact"}} != {
        k: v for k, v in env.items() if k != "artifact"
    }:
        raise ValueError("environment capture differs from frozen versions")
    binaries = _sequence(observed["binaries"], 1, 256)
    binary_paths = set()
    for binary in binaries:
        _ref(binary, MODEL_LIMIT)
        if binary["path"] in binary_paths:
            raise ValueError("duplicate environment binary")
        binary_paths.add(binary["path"])
    for arg in _sequence(observed["argv"], 1, 256):
        _text(arg)
    for key in ENV_KEYS - {"image_digest", "artifact"}:
        value = env[key]
        _text(value)
        if not re.fullmatch(r"[0-9][0-9A-Za-z.+_-]*", value):
            raise ValueError("environment versions must be fixed, not aliases")
    if env["image_digest"] is not None:
        if not _text(env["image_digest"]).startswith("sha256:"):
            raise ValueError("immutable image digest required")
        _hex(env["image_digest"][7:], 64)
    _sequence(cfg["nodes"], 1, 2)
    return cfg


def _inventory(cfg: dict) -> list[dict]:
    """Check observed host/device identity and node-to-node provenance equality."""
    nodes = []
    for ref in cfg["nodes"]:
        node = _object(
            _capture(
                ref,
                "inventory",
                {"run_id": cfg["run_id"], "source_head": cfg["source"]["head"]},
            ),
            {
                "host_id",
                "gpu_uuid",
                "gpu_model",
                "gpu_memory_bytes",
                "rdma_device",
                "rdma_gid",
                "link_gbps",
                "source",
                "model",
                "environment",
            },
        )
        for field in ("host_id", "gpu_uuid", "gpu_model"):
            _text(node[field])
        _integer(node["gpu_memory_bytes"], 1, 2**50)
        for field in ("rdma_device", "rdma_gid"):
            if node[field] is not None:
                _text(node[field])
        speed = node["link_gbps"]
        if speed is not None and (
            type(speed) not in (int, float) or not 0 < speed <= 10000
        ):
            raise ValueError("invalid measured RDMA link speed")
        for key in ("source", "model", "environment"):
            if node[key] != cfg[key]:
                raise ValueError(f"observed {key} differs from frozen provenance")
        nodes.append(node)
    if cfg["scope"] == PRIMARY:
        if len(nodes) != 2:
            raise ValueError("primary requires two distinct physical nodes")
        for field in ("host_id", "gpu_uuid", "rdma_gid"):
            if len({_text(n[field]) for n in nodes}) != 2:
                raise ValueError(f"primary requires distinct {field}")
        if nodes[0]["gpu_model"] != nodes[1]["gpu_model"]:
            raise ValueError("primary requires same GPU model")
        for node in nodes:
            _text(node["rdma_device"])
            if node["link_gbps"] is None:
                raise ValueError("primary requires measured RDMA link speed")
    elif len(nodes) != 1:
        raise ValueError("functional_single_gpu requires exactly one node")
    return nodes


def _native(cfg: dict, nodes: list[dict]) -> None:
    """Check native outputs, measured transfers and loader skip propagation."""
    probe = _object(
        _json(_ref(cfg["native_evidence"], JSON_LIMIT)), {"provenance_sha256", "cases"}
    )
    provenance = {
        key: cfg[key] for key in ("run_id", "source", "model", "environment", "nodes")
    }
    expected = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if probe["provenance_sha256"] != expected:
        raise ValueError("native probe provenance drift")
    primary = cfg["scope"] == PRIMARY
    required = (
        {"A_to_B", "B_to_A", "shared_prefix", "partial_hit", "miss", "skip_load"}
        if primary
        else {"single_gpu"}
    )
    seen = set()
    request_ids = set()
    for case in _sequence(probe["cases"], len(required), len(required)):
        case = _object(
            case,
            {
                "name",
                "request_id",
                "host_ids",
                "reference",
                "probe",
            },
        )
        name = _text(case["name"])
        request_id = _text(case["request_id"])
        if name not in required or name in seen or request_id in request_ids:
            raise ValueError("missing/duplicate native case or request")
        seen.add(name)
        request_ids.add(request_id)
        hosts = [node["host_id"] for node in nodes]
        if name == "B_to_A":
            hosts.reverse()
        if case["host_ids"] != hosts:
            raise ValueError("native case host identity/direction mismatch")
        identity = {
            "run_id": cfg["run_id"],
            "provenance_sha256": expected,
            "name": name,
            "request_id": request_id,
            "host_ids": hosts,
        }
        reference = _object(
            _capture(case["reference"], "reference", identity),
            {"token_ids", "token_count"},
        )
        observation = _object(
            _capture(case["probe"], "probe", identity),
            {
                "token_ids",
                "token_count",
                "transport",
                "remote_bytes",
                "external_fetches",
                "loader_extra_args",
            },
        )
        for output in (reference, observation):
            tokens = _sequence(output["token_ids"], 1, 65536)
            _integer(output["token_count"], 1, 65536)
            if output["token_count"] != len(tokens):
                raise ValueError("incomplete token output inventory")
            for token in tokens:
                _integer(token, 0, 2**31 - 1)
        if reference["token_ids"] != observation["token_ids"]:
            raise ValueError("native output token mismatch")
        for field in ("remote_bytes", "external_fetches"):
            _integer(observation[field])
        _object(
            observation["loader_extra_args"],
            {"lmcache.skip_load"} if name == "skip_load" else set(),
        )
        if (
            name == "skip_load"
            and observation["loader_extra_args"]["lmcache.skip_load"] is not True
        ):
            raise ValueError("loader did not observe skip flag")
        if observation["transport"] != ("rdma" if primary else "local"):
            raise ValueError("native transport is not qualified for requested scope")
        positive = primary and name not in ("miss", "skip_load")
        if positive:
            if observation["remote_bytes"] <= 0 or observation["external_fetches"] <= 0:
                raise ValueError("remote read not observed")
        elif observation["remote_bytes"] != 0 or observation["external_fetches"] != 0:
            raise ValueError("unexpected external fetch for local/miss/skip")


def preflight(config: dict[str, object]) -> PreflightReport:
    """Validate a frozen config and return JSON-serializable evidence statuses.

    Args:
        config: Strict schema documented in preflight-contract.md. File paths
            must be absolute. The caller must retain trusted collector artifacts.
    Returns:
        evidence_status (PASS/FAIL/NOT RUN), functional_status,
        eligible_for_primary, performance_status, checks and diagnostics.
        Invalid config/evidence returns FAIL, not an exception or empty result.
        No currently audited source head has qualified primary resource safety.
    """
    checks: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    report: PreflightReport = {
        "schema_version": 2,
        "evidence_status": "FAIL",
        "functional_status": "NOT RUN",
        "eligible_for_primary": False,
        "performance_status": "NOT RUN",
        "checks": checks,
        "diagnostics": errors,
    }
    stage = "configuration"
    try:
        cfg = _configuration(config)
        checks[stage] = {"status": "PASS"}
        stage = "inventory"
        nodes = _inventory(cfg)
        checks[stage] = {"status": "PASS"}
        primary = cfg["scope"] == PRIMARY
        checks["resource_safety"] = {
            "status": "FAIL"
            if primary and cfg["source"]["head"] == BASE
            else "NOT RUN",
            "detail": (
                "Pinned P2P timeout frees buffers before quiescence; "
                "other heads need independent source qualification."
            ),
        }
        checks["rdma"] = {"status": "NOT RUN"}
        stage = "native"
        if cfg["native_evidence"] is None:
            checks[stage] = {"status": "NOT RUN"}
            report["evidence_status"] = "NOT RUN"
            return report
        _native(cfg, nodes)
        checks[stage] = {"status": "PASS"}
        checks["rdma"] = {"status": "PASS" if primary else "NOT RUN"}
        report["functional_status"] = "PASS"
        report["evidence_status"] = "PASS"
        # Artifact completeness cannot override a source-confirmed safety defect.
        if primary:
            errors.append(checks["resource_safety"]["detail"])
        return report
    except (ValueError, TypeError, OSError, RecursionError) as exc:
        checks[stage] = {"status": "FAIL", "detail": str(exc)}
        errors.append(f"{stage}: {exc}")
        return report


def main(argv: list[str] | None = None) -> int:
    """Create a fresh report; return nonzero when the scope is unqualified.

    Args:
        argv: CLI arguments, or None for sys.argv; accepts --config and --output.
            Output must not exist. Reruns require a new output path, including
            failed reruns; existing reports and evidence are never overwritten.
    Returns:
        0 for qualified functional evidence; 1 for failed/unrun qualification.
        Argument parsing and output I/O failures retain conventional CLI errors.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        # Preflight validates this untrusted value before any CLI indexing.
        config = cast(dict[str, object], _json(_read(args.config, JSON_LIMIT)))
        report = preflight(config)
    except (ValueError, OSError, RecursionError) as exc:
        report = preflight({})
        report["diagnostics"] = [f"config read: {exc}"]
    with Path(args.output).open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return (
        0 if report["evidence_status"] == "PASS" and config["scope"] != PRIMARY else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
