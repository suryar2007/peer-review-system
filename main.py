"""CLI entrypoint for the peer-review verification pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from config import ConfigurationError, get_settings
from pipeline.graph import run_pipeline


def _print_summary(state: dict) -> None:
    """Print a formatted summary of the pipeline results."""
    title = state.get("paper_title") or "(unknown title)"
    citations = state.get("citations") or []
    resolved = state.get("resolved_citations") or []
    verifications = state.get("verification_results") or []
    audits = state.get("statistical_audit_results") or []

    total = len(citations)
    resolved_count = sum(1 for c in resolved if getattr(c, "resolved", False))
    unresolvable = total - resolved_count
    verified = len(verifications)

    # Breakdown of flagged verdicts
    overstated = sum(1 for v in verifications if getattr(v, "verdict", "") == "overstated")
    contradicted = sum(1 for v in verifications if getattr(v, "verdict", "") == "contradicted")
    out_of_scope = sum(1 for v in verifications if getattr(v, "verdict", "") == "out_of_scope")
    supported = sum(1 for v in verifications if getattr(v, "verdict", "") == "supported")
    flagged = overstated + contradicted + out_of_scope

    stat_issues = sum(
        1 for a in audits
        if not getattr(a, "is_internally_consistent", True)
    )

    print(f"\n{'=' * 60}")
    print(f"  Paper: {title}")
    print(f"{'=' * 60}")
    print(f"  Total citations found:       {total}")
    print(f"  Citations resolved:           {resolved_count}/{total}")
    print(f"  Unresolvable citations:       {unresolvable} (possible hallucinations)")
    print(f"  Claims verified:              {verified}")
    print(f"    Supported:                  {supported}")
    print(f"    Flagged:                    {flagged}")
    if flagged:
        print(f"      - Overstated:             {overstated}")
        print(f"      - Contradicted:           {contradicted}")
        print(f"      - Out of scope:           {out_of_scope}")
    print(f"  Statistical issues found:     {stat_issues}")

    dashboard = state.get("dashboard_url")
    if dashboard:
        print(f"  Dashboard URL:                {dashboard}")

    errs = state.get("errors") or []
    if errs:
        print(f"\n  Errors ({len(errs)}):")
        for e in errs:
            print(f"    - {e}")
    print()


def quick_test() -> None:
    """
    Run the pipeline on a hardcoded test PDF for development.

    Usage: python main.py --quick-test
    Expects tests/sample_paper.pdf to exist.
    """
    test_pdf = Path(__file__).parent / "tests" / "sample_paper.pdf"
    if not test_pdf.is_file():
        print(f"Quick test PDF not found: {test_pdf}", file=sys.stderr)
        print("Download an arXiv preprint and save it as tests/sample_paper.pdf", file=sys.stderr)
        sys.exit(1)
    print(f"Running quick test with: {test_pdf}")
    state = run_pipeline(str(test_pdf))
    _print_summary(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify citations and claims in an academic PDF.")
    parser.add_argument("--paper", type=Path, help="Path to the paper PDF", default=None)
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Save the full pipeline state as JSON to this path",
        default=None,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed logs during each phase",
    )
    parser.add_argument(
        "--skip-reasoning",
        action="store_true",
        help="Skip claim verification (faster, useful for testing extraction only)",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run pipeline on tests/sample_paper.pdf",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print final state as JSON to stdout",
    )
    # Positional PDF argument (backward compatible)
    parser.add_argument("pdf", type=Path, nargs="?", help="Path to the paper PDF (positional)")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    if args.quick_test:
        try:
            get_settings()
        except ConfigurationError as exc:
            raise SystemExit(str(exc)) from exc
        quick_test()
        return

    # Resolve PDF path from --paper or positional argument
    pdf = args.paper or args.pdf
    if pdf is None:
        parser.error("the following arguments are required: --paper or pdf")

    try:
        get_settings()
    except ConfigurationError as exc:
        raise SystemExit(str(exc)) from exc

    pdf_path = pdf.resolve()
    if not pdf_path.is_file():
        raise SystemExit(f"PDF not found: {pdf_path}")

    print(f"\n  Peer Review Verification Pipeline")
    print(f"  Paper: {pdf_path.name}")
    print()

    if args.skip_reasoning:
        import os
        os.environ["SKIP_REASONING"] = "1"
        get_settings.cache_clear()

    final_state = run_pipeline(str(pdf_path))

    if args.json:
        print(json.dumps(final_state, default=str, indent=2))
    else:
        _print_summary(final_state)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(final_state, f, default=str, indent=2)
        print(f"  Full state saved to: {args.output_json}")

    # Exit code based on success
    errs = final_state.get("errors") or []
    phase = final_state.get("current_phase", "")
    if phase == "extraction_failed" or (not final_state.get("citations") and not errs):
        sys.exit(1)


if __name__ == "__main__":
    main()
