# SPDX-License-Identifier: Apache-2.0
"""Deterministic corpus-token traces; no runtime, placement or oracle dependency.

Corpus v1 JSON has exactly version, source, license, tokenizer_revision, families.
Each family has family, split (train/validation/test), prefix_token_ids, suffixes
(a list of distinct 256-token sequences). Each split has 16 disjoint families;
prefix content cannot cross splits. Tokens must be supplied by an actual pinned
tokenizer over licensed text; this generator checks identity/shape, not semantics.
No tokens are invented, padded, repeated to length, or selected by future value.
The parent supplies warmup/barriers separately; every generated request is part
of the evaluation cohort, including pinned background traffic.
"""

# Standard
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import argparse
import hashlib
import itertools
import json
import math
import random
import sys


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _tokens(value: Any, length: int | None = None) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(t) is not int or t < 0 for t in value)
    ):
        raise ValueError("corpus token IDs must be nonempty nonnegative integer lists")
    if length is not None and len(value) != length:
        raise ValueError("corpus token length mismatch")
    return tuple(value)


@dataclass(frozen=True)
class TraceSpec:
    """Frozen trace inputs; SHA is of raw corpus bytes, rates are requests/second.

    split selects disjoint content and seed space. duration_s is the arrival
    window. prefix_tokens/background_fraction select the finite development
    grid. arrival_rate is frozen from a common train capacity reference; neither
    this module nor a router infers capacity or imposes an offload cap.
    """

    corpus_path: str
    corpus_sha256: str
    tokenizer_revision: str
    split: str = "test"
    prefix_tokens: int = 4096
    background_fraction: float = 0.30
    duration_s: float = 600.0
    arrival_rate: float = 1.0


@dataclass(frozen=True)
class TraceRequest:
    """One counted inference request; arrival_ns is relative to the barrier."""

    request_id: str
    request_class: str
    family: str
    affinity: str | None
    arrival_ns: int
    input_token_ids: tuple[int, ...]
    output_tokens: int = 128


@dataclass(frozen=True)
class TraceManifest:
    """Versioned trace with reproducible corpus/config provenance and all arrivals."""

    version: int
    seed: int
    spec: TraceSpec
    config_sha256: str
    generator_sha256: str
    requests: tuple[TraceRequest, ...]


def development_grid() -> tuple[tuple[int, float, float, float], ...]:
    """Return 36 (prefix tokens, background share, capacity share, L1/hot bytes) cells.

    These are exploration candidates, not measured capacities or efficacy claims.
    """
    return tuple(
        itertools.product(
            (4096, 8192, 16384), (0.2, 0.3), (0.55, 0.70, 0.85), (0.75, 1.25)
        )
    )


def generate(spec: TraceSpec, seed: int) -> TraceManifest:
    """Generate a fixed open-loop trace from supplied corpus tokens.

    Args:
        spec: Frozen corpus identity, split, grid selection, rate and duration.
        seed: Train 11/12, validation 21/22, or test 101..124.
    Returns:
        Manifest including both independent Poisson shared arrivals and thinned
        30-second on/off background arrivals, pinned A/B by seed parity. Zipf
        ranks reverse halfway through the window; no phase label is exported.
    Raises:
        ValueError: Invalid configuration, provenance, split, or insufficient
            distinct corpus suffixes. OSError: Corpus cannot be read.
    """
    seeds = {"train": (11, 12), "validation": (21, 22), "test": tuple(range(101, 125))}
    if (
        type(seed) is not int
        or spec.split not in seeds
        or seed not in seeds[spec.split]
    ):
        raise ValueError("seed does not belong to content split")
    if type(spec.prefix_tokens) is not int or spec.prefix_tokens not in (
        4096,
        8192,
        16384,
    ):
        raise ValueError("prefix length outside finite grid")
    if type(spec.background_fraction) not in (
        float,
        int,
    ) or spec.background_fraction not in (0.2, 0.3):
        raise ValueError("background fraction outside finite grid")
    for value in (spec.duration_s, spec.arrival_rate):
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError("duration and rate must be finite positive numbers")
    if spec.duration_s > 600 or spec.duration_s * spec.arrival_rate > 100_000:
        raise ValueError("trace exceeds bounded 600s/100000 expected requests")
    raw = Path(spec.corpus_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec.corpus_sha256:
        raise ValueError("corpus SHA mismatch")
    corpus = json.loads(raw)
    if set(corpus) != {
        "version",
        "source",
        "license",
        "tokenizer_revision",
        "families",
    }:
        raise ValueError("invalid corpus v1 fields")
    if type(corpus["version"]) is not int or corpus["version"] != 1:
        raise ValueError("invalid corpus version")
    if (
        corpus["tokenizer_revision"] != spec.tokenizer_revision
        or not spec.tokenizer_revision
    ):
        raise ValueError("tokenizer revision mismatch")
    if not all(
        isinstance(corpus[k], str) and corpus[k].strip() for k in ("source", "license")
    ):
        raise ValueError("source and license required")
    families: dict[str, list[dict[str, Any]]] = {key: [] for key in seeds}
    names: set[str] = set()
    contents: set[tuple[int, ...]] = set()
    for family in corpus["families"]:
        if set(family) != {"family", "split", "prefix_token_ids", "suffixes"}:
            raise ValueError("invalid family fields")
        name, split = family["family"], family["split"]
        if not isinstance(name, str) or not name or name in names or split not in seeds:
            raise ValueError("duplicate/invalid family or split")
        prefix = _tokens(family["prefix_token_ids"])
        # At least the first supported prefix must differ, including across splits.
        if len(prefix) < spec.prefix_tokens or prefix[:4096] in contents:
            raise ValueError("insufficient or reused prefix content")
        suffixes = [_tokens(s, 256) for s in family["suffixes"]]
        if not suffixes or len(set(suffixes)) != len(suffixes):
            raise ValueError("suffixes must be distinct corpus sequences")
        names.add(name)
        contents.add(prefix[:4096])
        families[split].append(family)
    if any(len(items) != 16 for items in families.values()):
        raise ValueError("each content split requires 16 families")
    selected = sorted(families[spec.split], key=lambda item: item["family"])
    rng = random.Random(seed)
    arrivals: list[tuple[float, str]] = []
    for kind, rate in (
        ("shared", spec.arrival_rate * (1 - spec.background_fraction)),
        ("background", spec.arrival_rate * spec.background_fraction * 2),
    ):
        elapsed = rng.expovariate(rate)
        while elapsed < spec.duration_s:
            if kind == "shared" or rng.random() < (1 if elapsed % 30 < 10 else 0.25):
                arrivals.append((elapsed, kind))
            elapsed += rng.expovariate(rate)
    used = [0] * 16
    requests = []
    weights = [1 / (rank + 1) ** 1.1 for rank in range(16)]
    for index, (arrival, kind) in enumerate(sorted(arrivals)):
        rank = rng.choices(range(16), weights=weights)[0]
        family_index = rank if arrival < spec.duration_s / 2 else 15 - rank
        family = selected[family_index]
        suffix_index = used[family_index]
        if suffix_index >= len(family["suffixes"]):
            raise ValueError("insufficient unique corpus suffixes for frozen trace")
        used[family_index] += 1
        tokens = tuple(
            family["prefix_token_ids"][: spec.prefix_tokens]
            + family["suffixes"][suffix_index]
        )
        requests.append(
            TraceRequest(
                f"{spec.split}-{seed}-{index}",
                kind,
                family["family"],
                ("A" if seed % 2 else "B") if kind == "background" else None,
                int(arrival * 1e9),
                tokens,
            )
        )
    return TraceManifest(
        1,
        seed,
        spec,
        hashlib.sha256(_canonical(asdict(spec))).hexdigest(),
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        tuple(requests),
    )


def main(argv: list[str] | None = None) -> int:
    """Run generate with --config FILE --config-sha256 SHA --seed N --output FILE.

    Config is exactly TraceSpec's JSON fields; SHA covers sorted compact JSON
    with all defaults explicit (ensure_ascii=True, allow_nan=False).
    Returns 0 or 1. Invalid input is diagnosed on
    stderr; argparse returns its usual exit 2 for invalid command grammar.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["generate"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        trace = generate(TraceSpec(**json.loads(args.config.read_text())), args.seed)
        if trace.config_sha256 != args.config_sha256:
            raise ValueError("config SHA mismatch")
        args.output.write_bytes(_canonical(asdict(trace)) + b"\n")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"generate: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
