"""Task 3 raw acquisition, canonical input construction and structural QA only."""

import argparse
import json
from pathlib import Path

from spot_perp_basis.data.final_sources import inventory
from spot_perp_basis.data.final_inputs import build, qa, structural_only, identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Isolated final-inputs-v1 directory")
    parser.add_argument("--reuse-root", type=Path, help="Optional existing Binance raw archive tree; never modified")
    parser.add_argument("--acquire", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--qa", action="store_true")
    args = parser.parse_args(argv)
    if args.offline and args.acquire:
        parser.error("--offline forbids --acquire")
    if not any((args.acquire, args.verify, args.build, args.qa)):
        parser.error("Choose explicit raw-input actions; no action runs by default")
    root = args.data_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with structural_only():
        identity()
        if args.acquire or args.verify:
            report = inventory(root, acquire=args.acquire, offline=args.offline, reuse_root=args.reuse_root)
            print(json.dumps({"archives_verified": sum("receipt" in r for r in report["files"]),
                              "archives_required": len(report["files"]), "complete": report["complete"]}), flush=True)
            if not report["complete"]:
                print("FINAL_DATA_STRUCTURAL_QA_PARTIAL — see inventory.json for exact failures")
                return 2
        if args.build:
            build(root)
        if args.qa:
            print(qa(root)["status"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
