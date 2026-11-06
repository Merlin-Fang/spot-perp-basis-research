"""Read-only final-input integration; never invokes numerical evaluation."""

import argparse
import json
from pathlib import Path

from spot_perp_basis.data.input_preflight import inspect_inputs, FORMAT
from spot_perp_basis.data.forecast_inputs import FINANCING
from spot_perp_basis.data.final_sources import ASSETS
from spot_perp_basis.data.final_sources import atomic, json_bytes
from spot_perp_basis.data.final_inputs import ROOT


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--history-root", type=Path, required=True, help="Root of the explicitly cataloged historical input bundle")
    p.add_argument("--input-catalog", type=Path, default=ROOT/"configs/forecast_input_integration_v1.json")
    p.add_argument("--public-root", type=Path, default=ROOT, help="Root containing the pinned final-data public receipt")
    p.add_argument("--output", type=Path, help="Optional new immutable metadata-only report")
    args = p.parse_args(argv)
    try:
        report = inspect_inputs(args.data_root, args.history_root, json.loads(args.input_catalog.read_text()), args.public_root)
    except (FileNotFoundError, KeyError, ValueError) as e:
        report = {"schema_version": FORMAT, "status": "FINAL_FORECASTING_INPUT_INTEGRATION_BLOCKED",
                  "error": f"{type(e).__name__}: {e}", "statistical_support": "not_evaluated",
                  "requested_cells": 24, "available_cells": 0,
                  "cells": [{"asset": a, "year": y, "financing": r, "input_status": "unavailable",
                             "diagnostic": f"{type(e).__name__}: {e}", "statistical_support": "not_evaluated"}
                            for a in ASSETS for y in (2023,2024,2025) for r in FINANCING]}
    if args.output: atomic(args.output, json_bytes(report))
    print(json.dumps({k:report[k] for k in ("status", "available_cells", "requested_cells", "error") if k in report}, sort_keys=True))
    return 0 if report["status"] == "FINAL_FORECASTING_INPUT_INTEGRATION_READY" else 2


if __name__ == "__main__": raise SystemExit(main())
