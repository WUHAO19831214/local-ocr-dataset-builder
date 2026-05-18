from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional
from uuid import uuid4

from .models import JobResponse, JobStatus, ProgressStage, StartJobRequest


@dataclass
class Job:
    job_id: str
    request: StartJobRequest
    status: JobStatus = "queued"
    progress_stage: ProgressStage = "queued"
    output_path: Optional[str] = None
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = Lock()

    def create(self, request: StartJobRequest) -> Job:
        job = Job(job_id=uuid4().hex, request=request)
        with self._lock:
            self._jobs[job.job_id] = job
        self.append_log(job.job_id, "任务已创建")
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def response(self, job_id: str) -> Optional[JobResponse]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            return JobResponse(
                job_id=job.job_id,
                status=job.status,
                progress_stage=job.progress_stage,
                output_path=job.output_path,
                error=job.error,
            )

    def logs(self, job_id: str) -> Optional[List[str]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            return list(job.logs)

    def set_status(
        self,
        job_id: str,
        *,
        status: Optional[JobStatus] = None,
        progress_stage: Optional[ProgressStage] = None,
        output_path: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if status is not None:
                job.status = status
            if progress_stage is not None:
                job.progress_stage = progress_stage
            if output_path is not None:
                job.output_path = output_path
            if error is not None:
                job.error = error
            job.updated_at = datetime.now()

    def append_log(self, job_id: str, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            job = self._jobs[job_id]
            job.logs.append(f"[{timestamp}] {message}")
            job.updated_at = datetime.now()


job_store = JobStore()

