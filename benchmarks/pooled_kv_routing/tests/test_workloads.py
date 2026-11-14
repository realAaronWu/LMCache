# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import asdict, replace
from pathlib import Path
from typing import TypedDict
import hashlib
import json
import subprocess
import sys

# Third Party
import pytest

# First Party
from benchmarks.pooled_kv_routing import workloads


class CorpusFamily(TypedDict):
    family: str
    split: str
    prefix_token_ids: list[int]
    suffixes: list[list[int]]


class Corpus(TypedDict):
    version: int
    source: str
    license: str
    tokenizer_revision: str
    families: list[CorpusFamily]


class TraceSpecChanges(TypedDict, total=False):
    corpus_sha256: str
    background_fraction: float
    prefix_tokens: int
    arrival_rate: float
    tokenizer_revision: str


@pytest.fixture
def spec(tmp_path: Path) -> workloads.TraceSpec:
    # Synthetic token IDs test serialization, not semantic workload efficacy.
    corpus: Corpus = {
        "version": 1,
        "source": "test fixture",
        "license": "CC0",
        "tokenizer_revision": "a" * 40,
        "families": [],
    }
    for split_index, split in enumerate(("train", "validation", "test")):
        for i in range(16):
            corpus["families"].append(
                {
                    "family": f"{split}-{i}",
                    "split": split,
                    "prefix_token_ids": [split_index * 16 + i + 1] * 4096,
                    "suffixes": [[100 + j] * 256 for j in range(80)],
                }
            )
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(corpus))
    return workloads.TraceSpec(
        corpus_path=str(path),
        corpus_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        tokenizer_revision="a" * 40,
        duration_s=30,
        arrival_rate=2.0,
    )


def test_trace_is_content_split_seeded_and_background_counted(
    spec: workloads.TraceSpec,
) -> None:
    trace = workloads.generate(spec, 101)
    assert asdict(trace) == asdict(workloads.generate(spec, 101))
    assert asdict(trace) != asdict(workloads.generate(spec, 102))
    assert trace.requests
    assert all(r.family.startswith("test-") for r in trace.requests)
    assert all(
        len(r.input_token_ids) == 4352 and r.output_tokens == 128
        for r in trace.requests
    )
    assert [r.arrival_ns for r in trace.requests] == sorted(
        r.arrival_ns for r in trace.requests
    )
    assert len({r.request_id for r in trace.requests}) == len(trace.requests)
    background = [r for r in trace.requests if r.request_class == "background"]
    assert background and all(r.affinity == "A" for r in background)
    assert all(
        r.affinity == "B"
        for r in workloads.generate(spec, 102).requests
        if r.request_class == "background"
    )
    assert len(workloads.development_grid()) == 36
    assert len(trace.generator_sha256) == 64


@pytest.mark.parametrize(
    "change",
    [
        {"corpus_sha256": "0" * 64},
        {"background_fraction": 0.4},
        {"prefix_tokens": 123},
        {"arrival_rate": float("nan")},
        {"tokenizer_revision": "b" * 40},
    ],
)
def test_trace_rejects_unfrozen_inputs(
    spec: workloads.TraceSpec, change: TraceSpecChanges
) -> None:
    with pytest.raises(ValueError):
        workloads.generate(replace(spec, **change), 101)


def test_generate_cli_checks_config_seed_and_corpus(
    spec: workloads.TraceSpec, tmp_path: Path
) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps(asdict(spec)))
    config_sha = hashlib.sha256(
        json.dumps(asdict(spec), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output = tmp_path / "trace.json"
    command = [
        sys.executable,
        "-m",
        "benchmarks.pooled_kv_routing.workloads",
        "generate",
        "--config",
        str(config),
        "--config-sha256",
        config_sha,
        "--seed",
        "101",
        "--output",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["seed"] == 101
    changed = asdict(spec)
    changed["arrival_rate"] = 3.0
    config.write_text(json.dumps(changed))
    assert subprocess.run(command, capture_output=True).returncode != 0
    config.write_text(json.dumps(asdict(spec)))
    Path(spec.corpus_path).write_text("{}")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and "SHA" in result.stderr
    command[command.index("101")] = "11"
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_content_leakage_and_duplicate_suffixes_rejected(
    spec: workloads.TraceSpec,
) -> None:
    path = Path(spec.corpus_path)
    corpus = json.loads(path.read_text())
    corpus["families"][16]["prefix_token_ids"] = corpus["families"][0][
        "prefix_token_ids"
    ]
    path.write_text(json.dumps(corpus))
    leaked = replace(spec, corpus_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="reused prefix"):
        workloads.generate(leaked, 101)


@pytest.mark.parametrize(
    "target_name", ["config", "corpus", "symlink", "hardlink", "existing"]
)
def test_generate_cli_never_overwrites_inputs_or_existing_output(
    spec: workloads.TraceSpec, tmp_path: Path, target_name: str
) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps(asdict(spec)))
    sha = hashlib.sha256(
        json.dumps(asdict(spec), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    corpus = Path(spec.corpus_path)
    output = tmp_path / "output.json"
    if target_name == "config":
        output = config
    elif target_name == "corpus":
        output = corpus
    elif target_name == "symlink":
        output.symlink_to(corpus)
    elif target_name == "hardlink":
        output.hardlink_to(corpus)
    else:
        output.write_text("previous output")
    originals = {path: path.read_bytes() for path in (config, corpus, output)}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.pooled_kv_routing.workloads",
            "generate",
            "--config",
            str(config),
            "--config-sha256",
            sha,
            "--seed",
            "101",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert all(path.read_bytes() == raw for path, raw in originals.items())
