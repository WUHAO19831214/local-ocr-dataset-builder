from __future__ import annotations

import subprocess
from pathlib import Path
from threading import Thread

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .job_store import job_store
from .models import (
    HealthResponse,
    LogsResponse,
    OpenPathRequest,
    OpenPathResponse,
    StartJobRequest,
    StartJobResponse,
)
from .ocr_runner import OcrRunnerError, run_ocr_job, validate_request


ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"

app = FastAPI(title="Local OCR Dataset Builder")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True)


@app.post("/api/jobs/start", response_model=StartJobResponse)
def start_job(request: StartJobRequest) -> StartJobResponse:
    try:
        validate_request(request)
    except OcrRunnerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = job_store.create(request)
    thread = Thread(target=_run_background_job, args=(job.job_id,), daemon=True)
    thread.start()
    return StartJobResponse(job_id=job.job_id, status="running")


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    response = job_store.response(job_id)
    if response is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return response


@app.get("/api/jobs/{job_id}/logs", response_model=LogsResponse)
def get_logs(job_id: str) -> LogsResponse:
    logs = job_store.logs(job_id)
    if logs is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return LogsResponse(logs=logs)


@app.post("/api/system/open-path", response_model=OpenPathResponse)
def open_path(request: OpenPathRequest) -> OpenPathResponse:
    path = Path(request.path).expanduser()
    if not path.is_absolute() or not path.exists():
        raise HTTPException(status_code=400, detail="路径必须存在且为绝对路径")
    subprocess.Popen(["open", str(path)])
    return OpenPathResponse(ok=True)


def _run_background_job(job_id: str) -> None:
    job = job_store.get(job_id)
    if job is None:
        return

    job_store.set_status(job_id, status="running", progress_stage="queued")

    def log(message: str) -> None:
        job_store.append_log(job_id, message)

    def stage(progress_stage: str) -> None:
        job_store.set_status(job_id, progress_stage=progress_stage)

    try:
        output_path = run_ocr_job(job.request, log, stage)
        job_store.set_status(
            job_id,
            status="success",
            progress_stage="done",
            output_path=str(output_path),
        )
    except Exception as exc:
        job_store.append_log(job_id, f"错误：{exc}")
        job_store.set_status(job_id, status="failed", progress_stage="failed", error=str(exc))


if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.get("/{path:path}")
def serve_frontend(path: str):
    index_html = FRONTEND_DIST / "index.html"
    requested = FRONTEND_DIST / path
    if path and requested.exists() and requested.is_file():
        return FileResponse(requested)
    if index_html.exists():
        return FileResponse(index_html)
    raise HTTPException(status_code=404, detail="前端尚未构建，请先运行 npm run build")

