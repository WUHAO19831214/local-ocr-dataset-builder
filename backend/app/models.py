from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


JobStatus = Literal["queued", "running", "success", "failed"]
ProgressStage = Literal["queued", "markdown", "json", "normalize", "done", "failed"]
ProcessMode = Literal["normal", "formula"]


class HealthResponse(BaseModel):
    ok: bool = True


class StartJobRequest(BaseModel):
    pdf_path: str = Field(..., min_length=1)
    output_root: str = Field(..., min_length=1)
    output_name: str = Field(..., min_length=1)
    ocr_lang: str = Field(..., min_length=1)
    process_mode: ProcessMode = "normal"
    force_ocr: bool | None = None


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress_stage: ProgressStage
    output_path: Optional[str] = None
    error: Optional[str] = None


class StartJobResponse(BaseModel):
    job_id: str
    status: JobStatus


class LogsResponse(BaseModel):
    logs: List[str]


class OpenPathRequest(BaseModel):
    path: str = Field(..., min_length=1)


class OpenPathResponse(BaseModel):
    ok: bool
