# SPDX-License-Identifier: Apache-2.0
"""Public entry point for the opt-in pooled routing experiment."""

# Standard
import argparse

# Local
from . import analysis, gpu_smoke, preflight, proxy, workloads


def main(argv: list[str] | None = None) -> int:
    """Delegate to a concrete tool without importing the LMCache runtime.

    Args:
        argv: Command and arguments, or the process command line.

    Returns:
        The selected tool's actual exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("preflight", "generate", "analyze", "smoke", "serve")
    )
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    remaining = args.arguments
    if args.command == "preflight":
        return preflight.main(remaining)
    elif args.command == "generate":
        return workloads.main([args.command, *remaining])
    elif args.command == "analyze":
        return analysis.main(remaining)
    elif args.command == "serve":
        return proxy.main(remaining)
    else:
        return gpu_smoke.main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
