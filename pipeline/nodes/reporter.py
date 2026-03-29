"""Reporter node: compute summary stats and generate local HTML report."""

from __future__ import annotations

import html
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.state import PipelineState
from utils.hex_client import HexClient

logger = logging.getLogger(__name__)

_VERDICT_COLORS = {
    "supported": "#22c55e",
    "overstated": "#f59e0b",
    "contradicted": "#ef4444",
    "out_of_scope": "#a855f7",
    "unverifiable": "#6b7280",
    "paper_mill_journal": "#f97316",
}

_VERDICT_EMOJI = {
    "supported": "&#10003;",
    "overstated": "&#9888;",
    "contradicted": "&#10007;",
    "out_of_scope": "&#8631;",
    "unverifiable": "?",
    "paper_mill_journal": "&#9873;",
}


def _build_payload(state: PipelineState) -> dict[str, Any]:
    citations = state.get("citations") or []
    claims = state.get("claims") or []
    resolved = state.get("resolved_citations") or []
    verifications = state.get("verification_results") or []
    audits = state.get("statistical_audit_results") or []
    stats = state.get("statistical_assertions") or []
    return {
        "citations": [c.model_dump(mode="json") for c in citations],
        "claims": [c.model_dump(mode="json") for c in claims],
        "statistical_assertions": [s.model_dump(mode="json") for s in stats],
        "resolved_citations": [c.model_dump(mode="json") for c in resolved],
        "verification_results": [v.model_dump(mode="json") for v in verifications],
        "statistical_audit_results": [a.model_dump(mode="json") for a in audits],
        "summary": {
            "total_citations": state.get("total_citations", len(citations)),
            "resolved_count": state.get("resolved_count", 0),
            "hallucinated_count": state.get("hallucinated_count", 0),
            "supported_claims": state.get("supported_claims", 0),
            "flagged_claims": state.get("flagged_claims", 0),
        },
        "errors": list(state.get("errors") or []),
    }


def _generate_html_report(state: PipelineState, summary: dict[str, int]) -> str:
    """Generate a standalone HTML report with all verification results."""
    title = html.escape(state.get("paper_title") or "Unknown Paper")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    verifications = state.get("verification_results") or []
    audits = state.get("statistical_audit_results") or []
    resolved = state.get("resolved_citations") or []
    errors = state.get("errors") or []

    tc = summary["total_citations"]
    rc = summary["resolved_count"]
    hc = summary["hallucinated_count"]
    sc = summary["supported_claims"]
    fc = summary["flagged_claims"]
    uv = sum(1 for v in verifications if v.verdict == "unverifiable")

    res_pct = f"{rc / tc * 100:.0f}" if tc else "0"

    verification_rows = ""
    for v in verifications:
        color = _VERDICT_COLORS.get(v.verdict, "#6b7280")
        emoji = _VERDICT_EMOJI.get(v.verdict, "?")
        claim = html.escape(v.claim_text[:300])
        explanation = html.escape(v.explanation[:500])
        passage = html.escape(v.relevant_passage[:300]) if v.relevant_passage else "<em>none</em>"
        conf = f"{v.confidence:.0%}"
        verification_rows += f"""
        <tr>
            <td style="max-width:400px">{claim}</td>
            <td><span style="color:{color};font-weight:700">{emoji} {v.verdict}</span></td>
            <td>{conf}</td>
            <td style="max-width:400px;font-size:0.85em">{explanation}</td>
            <td style="max-width:250px;font-size:0.8em;color:#555">{passage}</td>
        </tr>"""

    audit_rows = ""
    for a in audits:
        consistent = "Yes" if a.is_internally_consistent else "No"
        color = "#22c55e" if a.is_internally_consistent else "#ef4444"
        issues = "; ".join(html.escape(i) for i in a.issues) if a.issues else "&mdash;"
        text = html.escape(a.assertion_text[:200])
        audit_rows += f"""
        <tr>
            <td style="max-width:400px">{text}</td>
            <td><span style="color:{color};font-weight:700">{consistent}</span></td>
            <td>{a.severity}</td>
            <td style="max-width:300px;font-size:0.85em">{issues}</td>
        </tr>"""

    unresolved_rows = ""
    for i, c in enumerate(resolved):
        if not c.resolved:
            raw = html.escape((c.raw_text or "")[:200])
            t = html.escape(c.title or "")
            unresolved_rows += f"<tr><td>{i}</td><td>{t}</td><td style='font-size:0.8em;color:#888'>{raw}</td></tr>"

    error_section = ""
    if errors:
        items = "".join(f"<li>{html.escape(str(e))}</li>" for e in errors)
        error_section = f'<h2>Pipeline Errors</h2><ul style="color:#ef4444">{items}</ul>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Peer Review Report — {title}</title>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
       background:#f8fafc;color:#1e293b;line-height:1.5;padding:2rem}}
  h1{{font-size:1.5rem;margin-bottom:.25rem}}
  h2{{font-size:1.15rem;margin:2rem 0 .75rem;border-bottom:2px solid #e2e8f0;padding-bottom:.25rem}}
  .meta{{color:#64748b;font-size:.85rem;margin-bottom:1.5rem}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1rem;margin-bottom:2rem}}
  .card{{background:#fff;border-radius:.5rem;padding:1rem;box-shadow:0 1px 3px rgba(0,0,0,.1)}}
  .card .num{{font-size:1.8rem;font-weight:700}}
  .card .label{{font-size:.8rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em}}
  table{{width:100%;border-collapse:collapse;background:#fff;border-radius:.5rem;overflow:hidden;
         box-shadow:0 1px 3px rgba(0,0,0,.1);margin-bottom:1.5rem}}
  th{{background:#f1f5f9;padding:.6rem .75rem;text-align:left;font-size:.8rem;
      text-transform:uppercase;letter-spacing:.04em;color:#475569}}
  td{{padding:.5rem .75rem;border-top:1px solid #e2e8f0;vertical-align:top;font-size:.9rem}}
  tr:hover{{background:#f8fafc}}
</style>
</head>
<body>
<h1>Peer Review Verification Report</h1>
<p class="meta"><strong>{title}</strong> &mdash; Generated {ts}</p>

<div class="cards">
  <div class="card"><div class="num">{tc}</div><div class="label">Citations Found</div></div>
  <div class="card"><div class="num" style="color:#22c55e">{rc} <span style="font-size:.7em">({res_pct}%)</span></div><div class="label">Resolved</div></div>
  <div class="card"><div class="num" style="color:#ef4444">{hc}</div><div class="label">Hallucinated</div></div>
  <div class="card"><div class="num" style="color:#22c55e">{sc}</div><div class="label">Claims Supported</div></div>
  <div class="card"><div class="num" style="color:#f59e0b">{fc}</div><div class="label">Claims Flagged</div></div>
  <div class="card"><div class="num" style="color:#6b7280">{uv}</div><div class="label">Unverifiable</div></div>
</div>

<h2>Claim Verification Results ({len(verifications)})</h2>
<table>
<thead><tr><th>Claim</th><th>Verdict</th><th>Conf.</th><th>Explanation</th><th>Relevant Passage</th></tr></thead>
<tbody>{verification_rows if verification_rows else "<tr><td colspan=5>No claims verified</td></tr>"}</tbody>
</table>

<h2>Statistical Audit ({len(audits)})</h2>
<table>
<thead><tr><th>Assertion</th><th>Consistent?</th><th>Severity</th><th>Issues</th></tr></thead>
<tbody>{audit_rows if audit_rows else "<tr><td colspan=4>No statistical assertions audited</td></tr>"}</tbody>
</table>

<h2>Unresolved Citations ({tc - rc})</h2>
<table>
<thead><tr><th>#</th><th>Title</th><th>Raw Text</th></tr></thead>
<tbody>{unresolved_rows if unresolved_rows else "<tr><td colspan=3>All citations resolved</td></tr>"}</tbody>
</table>

{error_section}
</body></html>"""


def reporter_node(state: PipelineState) -> dict[str, Any]:
    """
    Phase 4: Compute summary stats, generate local HTML report, and optionally
    trigger Hex dashboard if configured.
    """
    citations = state.get("citations") or []
    resolved_citations = state.get("resolved_citations") or []
    verification_results = state.get("verification_results") or []

    total_citations = len(citations)
    resolved_count = sum(1 for c in resolved_citations if c.resolved)
    hallucinated_count = sum(1 for c in resolved_citations if c.exists is False)
    supported_claims = sum(1 for v in verification_results if v.verdict == "supported")
    flagged_claims = sum(
        1
        for v in verification_results
        if v.verdict in ("overstated", "contradicted", "out_of_scope", "paper_mill_journal")
    )

    summary = {
        "total_citations": total_citations,
        "resolved_count": resolved_count,
        "hallucinated_count": hallucinated_count,
        "supported_claims": supported_claims,
        "flagged_claims": flagged_claims,
    }

    # Generate local HTML report
    report_path: str | None = None
    try:
        html_content = _generate_html_report(state, summary)
        paper_path = state.get("paper_path") or ""
        stem = Path(paper_path).stem if paper_path else "paper"
        out_dir = Path(paper_path).parent if paper_path else Path(".")
        report_file = out_dir / f"{stem}_review_report.html"
        report_file.write_text(html_content, encoding="utf-8")
        report_path = str(report_file)
        logger.info("HTML report saved to %s", report_path)
    except Exception as exc:
        logger.warning("Failed to generate HTML report: %s", exc)

    # Save JSON payload alongside
    json_path: str | None = None
    try:
        payload = _build_payload(state)
        payload["summary"] = summary
        payload["paper_text"] = state.get("paper_text") or ""
        payload["title"] = state.get("paper_title") or ""
        paper_path = state.get("paper_path") or ""
        stem = Path(paper_path).stem if paper_path else "paper"
        out_dir = Path(paper_path).parent if paper_path else Path(".")
        json_file = out_dir / f"{stem}_review_data.json"
        json_file.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        json_path = str(json_file)
    except Exception as exc:
        logger.warning("Failed to save JSON data: %s", exc)

    # Try Hex dashboard if configured
    dashboard_url: str | None = None
    run_id: str | None = None
    try:
        client = HexClient.from_config()
        if client._is_configured():
            payload = _build_payload(state)
            dashboard_url = client.trigger_and_wait(payload)
            if dashboard_url:
                run_id = dashboard_url.split("run=")[-1] if "run=" in dashboard_url else None
    except Exception as exc:
        logger.debug("Hex dashboard skipped: %s", exc)

    print(f"\nPipeline complete.")
    print(f"  Citations: {total_citations} total, {resolved_count} resolved ({summary['resolved_count']}/{total_citations})")
    print(f"  Hallucinated citations: {hallucinated_count}")
    print(f"  Claims: {supported_claims} supported, {flagged_claims} flagged")
    if report_path:
        print(f"  HTML report: {report_path}")
    if json_path:
        print(f"  JSON data:   {json_path}")
    if dashboard_url:
        print(f"  Dashboard:   {dashboard_url}")

    return {
        "hex_run_id": run_id,
        "dashboard_url": dashboard_url or report_path,
        "total_citations": total_citations,
        "resolved_count": resolved_count,
        "hallucinated_count": hallucinated_count,
        "supported_claims": supported_claims,
        "flagged_claims": flagged_claims,
        "current_phase": "reporting_complete",
    }
