"""FastAPI web app — Grammarly-style peer-review annotation on actual PDFs."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from web.annotator import build_pdf_annotations, compute_score

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
TEMPLATES_DIR = BASE_DIR / "templates"
DATA_DIR = PROJECT_ROOT / "tests"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Peer Review UI")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Background pipeline jobs: upload_id → {status, phase, error}
_jobs: dict[str, dict[str, Any]] = {}


# ── Helpers ──────────────────────────────────────────────────────────


def _scan_reviews(directory: Path) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    for f in sorted(directory.glob("*_review_data.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            summary = data.get("summary", {})
            name = f.stem.replace("_review_data", "")
            pdf_exists = (f.parent / f"{name}.pdf").exists()
            reviews.append({
                "name": name,
                "display_name": name.replace("_", " ").replace("-", " ").title(),
                "filename": f.name,
                "directory": str(f.parent),
                "summary": summary,
                "pdf_available": pdf_exists,
                "verification_count": len(data.get("verification_results", [])),
                "audit_count": len(data.get("statistical_audit_results", [])),
            })
        except Exception as exc:
            logger.warning("Skipping %s: %s", f, exc)
    return reviews


def _resolve_paths(directory: str, filename: str) -> tuple[Path, Path] | None:
    """Resolve JSON and PDF paths from the route parameters."""
    if directory == "tests":
        base = DATA_DIR
    elif directory == "uploads":
        base = UPLOADS_DIR
    else:
        return None
    json_path = base / filename
    stem = filename.replace("_review_data.json", "")
    pdf_path = base / f"{stem}.pdf"
    return json_path, pdf_path


class _PhaseTracker(logging.Handler):
    """Capture pipeline log messages to update job phase in real time."""

    def __init__(self, job: dict[str, Any]) -> None:
        super().__init__(level=logging.INFO)
        self._job = job
        self._keywords = {
            "extract": "Extracting citations & claims...",
            "resolv": "Resolving citations...",
            "reason": "Verifying claims...",
            "report": "Generating report...",
            "hermes": "Extracting citations & claims...",
            "citation_resolver": "Resolving citations...",
            "k2": "Verifying claims...",
            "reporter": "Generating report...",
        }

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage().lower()
        name = record.name.lower()
        for kw, phase in self._keywords.items():
            if kw in msg or kw in name:
                self._job["phase"] = phase
                return


def _run_pipeline_thread(upload_id: str, pdf_path: str) -> None:
    """Run the full pipeline in a background thread."""
    job = _jobs[upload_id]
    tracker = _PhaseTracker(job)
    root_logger = logging.getLogger()
    root_logger.addHandler(tracker)
    if root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)

    try:
        job["phase"] = "Loading configuration..."
        from config import get_settings
        get_settings()

        job["phase"] = "Extracting citations & claims..."
        from pipeline.graph import run_pipeline
        run_pipeline(pdf_path)

        job["status"] = "done"
        job["phase"] = "Complete"
    except Exception as exc:
        logger.exception("Pipeline failed for %s", upload_id)
        job["status"] = "error"
        job["error"] = str(exc)
    finally:
        root_logger.removeHandler(tracker)


# ── Routes ───────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    reviews = _scan_reviews(DATA_DIR) + _scan_reviews(UPLOADS_DIR)
    return templates.TemplateResponse(
        request, "index.html", {"reviews": reviews},
    )


@app.get("/pdf/{directory}/{stem}")
async def serve_pdf(directory: str, stem: str):
    """Serve a PDF file for the in-browser viewer."""
    if directory == "tests":
        path = DATA_DIR / f"{stem}.pdf"
    elif directory == "uploads":
        path = UPLOADS_DIR / f"{stem}.pdf"
    else:
        return HTMLResponse("Not found", status_code=404)
    if not path.exists():
        return HTMLResponse("PDF not found", status_code=404)
    return FileResponse(path, media_type="application/pdf")


@app.get("/review/{directory}/{filename}", response_class=HTMLResponse)
async def review(request: Request, directory: str, filename: str):
    paths = _resolve_paths(directory, filename)
    if not paths:
        return HTMLResponse("Not found", status_code=404)
    json_path, pdf_path = paths

    if not json_path.exists():
        return HTMLResponse("Review data not found", status_code=404)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    title = data.get("title", "")
    stem = filename.replace("_review_data.json", "")

    # Build annotations with PDF coordinates
    annotations: list[dict] = []
    has_pdf = pdf_path.exists()
    if has_pdf:
        annotations = build_pdf_annotations(str(pdf_path), data)

    if not title and has_pdf:
        from utils.pdf_parser import PaperParser
        try:
            title = PaperParser(str(pdf_path)).parse().get("title", "")
        except Exception:
            pass
    if not title:
        title = stem.replace("_", " ").replace("-", " ").title()

    score = compute_score(annotations)
    verdict_counts: dict[str, int] = {}
    for ann in annotations:
        v = ann["verdict"]
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    pdf_url = f"/pdf/{directory}/{stem}" if has_pdf else ""
    annotations_json = json.dumps(annotations, default=str)

    return templates.TemplateResponse(
        request, "review.html", {
            "title": title,
            "pdf_url": pdf_url,
            "has_pdf": has_pdf,
            "annotations_json": annotations_json,
            "annotations": annotations,
            "score": score,
            "summary": summary,
            "verdict_counts": verdict_counts,
            "data": data,
        },
    )


# ── Upload & pipeline execution ─────────────────────────────────────


@app.post("/upload")
async def upload(
    pdf: UploadFile = File(None),
    results_json: UploadFile = File(None),
):
    if not pdf and not results_json:
        return RedirectResponse("/", status_code=303)

    upload_id = uuid.uuid4().hex[:8]

    # If JSON results provided → instant view
    if results_json and results_json.filename:
        json_dest = UPLOADS_DIR / f"{upload_id}_review_data.json"
        json_dest.write_bytes(await results_json.read())
        if pdf and pdf.filename:
            pdf_dest = UPLOADS_DIR / f"{upload_id}.pdf"
            pdf_dest.write_bytes(await pdf.read())
        return RedirectResponse(
            f"/review/uploads/{json_dest.name}", status_code=303,
        )

    # PDF only → run the pipeline
    if pdf and pdf.filename:
        pdf_dest = UPLOADS_DIR / f"{upload_id}.pdf"
        pdf_dest.write_bytes(await pdf.read())
        _jobs[upload_id] = {
            "status": "running",
            "phase": "Starting pipeline...",
            "error": None,
        }
        thread = threading.Thread(
            target=_run_pipeline_thread,
            args=(upload_id, str(pdf_dest)),
            daemon=True,
        )
        thread.start()
        return RedirectResponse(f"/processing/{upload_id}", status_code=303)

    return RedirectResponse("/", status_code=303)


@app.get("/processing/{upload_id}", response_class=HTMLResponse)
async def processing_page(request: Request, upload_id: str):
    return templates.TemplateResponse(
        request, "processing.html", {"upload_id": upload_id},
    )


@app.get("/api/status/{upload_id}")
async def pipeline_status(upload_id: str):
    job = _jobs.get(upload_id)
    if not job:
        return JSONResponse({"status": "not_found"})

    # Also detect completion by output file
    json_path = UPLOADS_DIR / f"{upload_id}_review_data.json"
    if json_path.exists() and job["status"] == "running":
        job["status"] = "done"

    result: dict[str, Any] = {
        "status": job["status"],
        "phase": job.get("phase", ""),
    }
    if job["status"] == "done":
        result["review_url"] = f"/review/uploads/{upload_id}_review_data.json"
    elif job["status"] == "error":
        result["error"] = job.get("error", "Unknown error")
    return JSONResponse(result)
