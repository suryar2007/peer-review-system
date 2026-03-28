"""CLI entrypoint for the peer-review verification pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import ConfigurationError, get_settings
from pipeline import build_graph
from pipeline.state import initial_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify citations and claims in an academic PDF.")
    parser.add_argument("pdf", type=Path, help="Path to the paper PDF")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print final state as JSON to stdout",
    )
    args = parser.parse_args()

    try:
        get_settings()
    except ConfigurationError as exc:
        raise SystemExit(str(exc)) from exc

    pdf_path = args.pdf.resolve()
    if not pdf_path.is_file():
        raise SystemExit(f"PDF not found: {pdf_path}")

    app = build_graph()
    start = initial_state(str(pdf_path))
    final_state = app.invoke(start)

    if args.json:
        print(json.dumps(final_state, default=str, indent=2))
    else:
        print("Pipeline complete.")
        print(f"  Phase: {final_state.get('current_phase')}")
        print(f"  Citations extracted: {len(final_state.get('citations') or [])}")
        print(f"  Claims extracted: {len(final_state.get('claims') or [])}")
        print(f"  Resolved citations: {len(final_state.get('resolved_citations') or [])}")
        print(f"  Verification results: {len(final_state.get('verification_results') or [])}")
        print(f"  Statistical audits: {len(final_state.get('statistical_audit_results') or [])}")
        print(f"  Summary — supported claims: {final_state.get('supported_claims')}, flagged: {final_state.get('flagged_claims')}")
        if final_state.get("hex_run_id"):
            print(f"  Hex run id: {final_state['hex_run_id']}")
        if final_state.get("dashboard_url"):
            print(f"  Dashboard: {final_state['dashboard_url']}")
        errs = final_state.get("errors") or []
        if errs:
            print("  Errors:")
            for e in errs:
                print(f"    - {e}")


if __name__ == "__main__":
    main()
