from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Callable, Iterable

import pymupdf

from .models import StartJobRequest
from .process_registry import register_process, unregister_process
from .word_exporter import export_word_files
from .formula_enhancer import PADDLE_PYTHON, enhance_formula_regions


OCR_PROJECT_DIR = Path("/Users/wuhao/my-pdf-tool/my-pdf-tool")
DOCLING_BIN = OCR_PROJECT_DIR / "venv" / "bin" / "docling"
JAVA_HOME = Path("/opt/homebrew/opt/openjdk@17")
ALLOWED_LANGS = {
    "ja-JP,zh-Hans",
    "zh-Hans",
    "ja-JP",
    "zh-Hant,ja-JP,zh-Hans",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
OUTPUT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
FORMULA_BLOCK_RE = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
DOCLING_TIMEOUT_SECONDS = 600
MATH_SPAN_RE = re.compile(r"(\$\$.*?\$\$|\$.*?\$)", re.DOTALL)
PA_POWER_RE = re.compile(
    r"(?<![\w$])(?:10\s*\^\s*(?P<caret>-?\s*\d{1,2})\s*Pa\b|"
    r"10\s+(?P<positive>\d{1,2})\s*Pa\b|"
    r"10\s*-\s*(?P<negative>\d{1,2})\s*Pa\b)"
)
VACUUM_KEYWORDS = ("真空一般指气压低于一个大气压", "粗真空", "低真空", "高真空", "超高真空", "极高真空")
VACUUM_CLASSIFICATION = "\n".join(
    [
        "粗真空（$10^5\\ \\mathrm{Pa}$～$10^3\\ \\mathrm{Pa}$）",
        "低真空（$10^3\\ \\mathrm{Pa}$～$10^{-1}\\ \\mathrm{Pa}$）",
        "高真空（$10^{-1}\\ \\mathrm{Pa}$～$10^{-6}\\ \\mathrm{Pa}$）",
        "超高真空（$10^{-6}\\ \\mathrm{Pa}$～$10^{-10}\\ \\mathrm{Pa}$）",
        "极高真空（低于 $10^{-12}\\ \\mathrm{Pa}$）",
    ]
)


class OcrRunnerError(RuntimeError):
    pass


def validate_request(request: StartJobRequest) -> tuple[Path, Path, Path]:
    pdf_path = Path(request.pdf_path).expanduser()
    output_root = Path(request.output_root).expanduser()

    if not pdf_path.is_absolute():
        raise OcrRunnerError("PDF 路径必须是绝对路径")
    if not output_root.is_absolute():
        raise OcrRunnerError("输出根目录必须是绝对路径")
    if not pdf_path.exists() or not pdf_path.is_file():
        raise OcrRunnerError(f"PDF 文件不存在：{pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise OcrRunnerError("输入文件后缀必须是 .pdf")
    if request.ocr_lang not in ALLOWED_LANGS:
        raise OcrRunnerError(f"OCR 语言不支持：{request.ocr_lang}")
    if not OUTPUT_NAME_RE.match(request.output_name):
        raise OcrRunnerError("输出名称只能包含英文、数字、点、下划线、短横线，且必须以英文或数字开头")
    if not DOCLING_BIN.exists():
        raise OcrRunnerError(f"docling 命令不存在：{DOCLING_BIN}")
    if request.process_mode == "formula_vl" and not PADDLE_PYTHON.is_file():
        raise OcrRunnerError(f"高精度公式模式缺少 PaddleOCR-VL 环境：{PADDLE_PYTHON}")

    output_root.mkdir(parents=True, exist_ok=True)
    target_dir = output_root / request.output_name

    if target_dir.exists() and any(target_dir.iterdir()):
        raise OcrRunnerError(f"输出目录已存在且非空，为避免覆盖请更换输出名称：{target_dir}")

    return pdf_path, output_root, target_dir


def run_ocr_job(request: StartJobRequest, log: Callable[[str], None], stage: Callable[[str], None]) -> Path:
    pdf_path, _, target_dir = validate_request(request)
    force_ocr = _effective_force_ocr(request)
    log(f"开始处理 PDF：{pdf_path}")
    log(f"OCR 引擎：ocrmac，语言：{request.ocr_lang}")
    mode_names = {"normal": "普通教材 OCR", "formula": "Docling 公式增强", "formula_vl": "高精度公式复核"}
    log(f"处理模式：{mode_names[request.process_mode]}")
    log(f"强制 OCR：{'开启' if force_ocr else '关闭'}")
    log(f"Docling 公式增强：{'开启' if request.process_mode == 'formula' else '关闭'}")
    if request.process_mode == "formula_vl":
        log("高精度模式使用基础 OCR 后由 PaddleOCR-VL 复核公式，跳过耗时过长的 Docling CodeFormulaV2")
    log(f"目标输出目录：{target_dir}")

    with tempfile.TemporaryDirectory(prefix="local-ocr-dataset-builder-") as tmp:
        tmp_root = Path(tmp)
        docling_dir = tmp_root / "docling"

        stage("markdown")
        log("开始解析 PDF，同时输出 Markdown 和 JSON")
        _run_docling(pdf_path, docling_dir, ("md", "json"), request.ocr_lang,
                     request.process_mode, force_ocr, log)

        stage("normalize")
        log("开始整理 md/json/images")
        _normalize_outputs(docling_dir, docling_dir, target_dir, request.output_name,
                           request.process_mode, log)

        if request.process_mode == "formula_vl":
            stage("formula_vl")
            log("开始高精度公式区域复核（PaddleOCR-VL，CPU 运行可能较慢）")
            enhance_formula_regions(pdf_path, target_dir, request.output_name, log)

        if request.export_word:
            stage("word")
            log("开始生成原版式和可编辑 Word")
            for word_path in export_word_files(pdf_path, target_dir, request.output_name):
                log(f"输出 Word：{word_path.name}")

    stage("done")
    log("完成")
    return target_dir


def _run_docling(
    pdf_path: Path,
    output_dir: Path,
    output_formats: tuple[str, ...],
    ocr_lang: str,
    process_mode: str,
    force_ocr: bool,
    log: Callable[[str], None],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timeout_seconds = _docling_timeout_seconds(pdf_path, process_mode)
    command = [
        str(DOCLING_BIN),
        str(pdf_path),
        "--from",
        "pdf",
        "--output",
        str(output_dir),
        "--image-export-mode",
        "referenced",
        "--ocr-engine",
        "ocrmac",
        "--ocr-lang",
        ocr_lang,
    ]
    for output_format in output_formats:
        command.extend(["--to", output_format])
    if force_ocr:
        command.append("--force-ocr")
    if process_mode == "formula":
        command.append("--enrich-formula")
    command.extend(["--document-timeout", str(timeout_seconds)])
    command.append("-v")

    env = os.environ.copy()
    env["JAVA_HOME"] = str(JAVA_HOME)
    env["PATH"] = f"{JAVA_HOME / 'bin'}:{DOCLING_BIN.parent}:{env.get('PATH', '')}"

    process = subprocess.Popen(
        command,
        cwd=str(OCR_PROJECT_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    register_process(process)
    try:
        return_code = _read_docling_output(process, log, timeout_seconds)
    finally:
        unregister_process(process)
    if return_code != 0:
        raise OcrRunnerError(f"docling 输出失败（处理上限 {timeout_seconds // 60} 分钟），退出码：{return_code}")


def _docling_timeout_seconds(pdf_path: Path, process_mode: str) -> int:
    if process_mode == "formula":
        return DOCLING_TIMEOUT_SECONDS
    try:
        with pymupdf.open(str(pdf_path)) as document:
            return max(DOCLING_TIMEOUT_SECONDS, min(3600, document.page_count * 90))
    except Exception:
        return DOCLING_TIMEOUT_SECONDS


def _read_docling_output(
    process: subprocess.Popen, log: Callable[[str], None], timeout_seconds: float | None = None,
) -> int:
    """Show a heartbeat and enforce a real timeout during silent model inference."""
    assert process.stdout is not None
    lines: Queue[str | None] = Queue()

    def read_lines() -> None:
        try:
            for line in process.stdout:
                lines.put(line.rstrip())
        finally:
            lines.put(None)

    Thread(target=read_lines, daemon=True).start()
    started = time.monotonic()
    timeout_seconds = timeout_seconds if timeout_seconds is not None else DOCLING_TIMEOUT_SECONDS
    while True:
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise OcrRunnerError(f"Docling 已运行 {int(timeout_seconds // 60)} 分钟仍未完成；已停止该次解析。请检查 PDF，或关闭强制 OCR 后重试。")
        try:
            line = lines.get(timeout=min(30, timeout_seconds - elapsed))
        except Empty:
            log(f"Docling 仍在计算，已耗时 {int(time.monotonic() - started)} 秒")
            continue
        if line is None:
            return process.wait()
        if line:
            log(line)


def _find_one(directory: Path, suffix: str) -> Path:
    files = sorted(directory.glob(f"*.{suffix}"))
    if not files:
        raise OcrRunnerError(f"未找到 {suffix} 文件：{directory}")
    return files[0]


def _artifact_dirs(*roots: Path) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in roots:
        for artifact_dir in sorted(root.glob("*_artifacts")):
            if artifact_dir.is_dir() and artifact_dir not in seen:
                seen.add(artifact_dir)
                yield artifact_dir


def _normalize_outputs(
    md_dir: Path,
    json_dir: Path,
    target_dir: Path,
    output_name: str,
    process_mode: str,
    log: Callable[[str], None],
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    images_dir = target_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    md_file = _find_one(md_dir, "md")
    json_file = _find_one(json_dir, "json")
    out_md = target_dir / f"{output_name}.md"
    out_json = target_dir / f"{output_name}.json"

    log(f"复制 JSON：{out_json.name}")
    shutil.copy2(json_file, out_json)

    copied_by_name: dict[str, str] = {}
    image_count = 0
    for artifact_dir in _artifact_dirs(md_dir, json_dir):
        for image in sorted(artifact_dir.iterdir()):
            if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
                destination = images_dir / image.name
                if not destination.exists():
                    shutil.copy2(image, destination)
                    image_count += 1
                copied_by_name[image.name] = f"images/{image.name}"

    log(f"整理 images：{image_count} 个图片文件")

    md_text = md_file.read_text(encoding="utf-8")
    md_text = _rewrite_markdown_images(md_text, copied_by_name, images_dir, log)
    if process_mode in ("formula", "formula_vl"):
        md_text = _postprocess_formula_markdown(md_text)
        log("完成物理/数学公式 Markdown 后处理")
    out_md.write_text(md_text, encoding="utf-8")
    log(f"输出 Markdown：{out_md.name}")

    if not out_json.exists() or not out_md.exists() or not images_dir.exists():
        raise OcrRunnerError("输出检查失败，缺少 md/json/images")


def _rewrite_markdown_images(
    md_text: str,
    copied_by_name: dict[str, str],
    images_dir: Path,
    log: Callable[[str], None],
) -> str:
    def replace_image(match: re.Match[str]) -> str:
        alt = match.group(1)
        path_text = match.group(2).strip()
        filename = Path(path_text).name

        if filename in copied_by_name:
            return f"![{alt}]({copied_by_name[filename]})"

        source = Path(path_text)
        if source.exists() and source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
            destination = images_dir / source.name
            if not destination.exists():
                shutil.copy2(source, destination)
                log(f"复制 Markdown 外部图片：{source.name}")
            return f"![{alt}](images/{source.name})"

        return match.group(0)

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace_image, md_text)


def _effective_force_ocr(request: StartJobRequest) -> bool:
    return request.force_ocr is True


def _postprocess_formula_markdown(md_text: str) -> str:
    def normalize_block(match: re.Match[str]) -> str:
        formula = " ".join(match.group(1).strip().split())
        return f"\n\n$${formula}$$\n\n"

    normalized = FORMULA_BLOCK_RE.sub(normalize_block, md_text)
    normalized = _replace_pa_powers(normalized)
    normalized = _repair_vacuum_classification(normalized)
    normalized = _repair_line_numbering(normalized)
    return re.sub(r"\n{3,}", "\n\n", normalized).strip() + "\n"


def _format_pa_power(exp: str) -> str:
    exp = exp.replace(" ", "")
    if exp.startswith("-"):
        return f"$10^{{{exp}}}\\ \\mathrm{{Pa}}$"
    return f"$10^{exp}\\ \\mathrm{{Pa}}$"


def _replace_pa_powers(md_text: str) -> str:
    parts = MATH_SPAN_RE.split(md_text)
    for index, part in enumerate(parts):
        if part.startswith("$"):
            continue
        parts[index] = PA_POWER_RE.sub(_replace_pa_match, part)
    return "".join(parts)


def _replace_pa_match(match: re.Match[str]) -> str:
    if match.group("caret") is not None:
        return _format_pa_power(match.group("caret"))
    if match.group("positive") is not None:
        return _format_pa_power(match.group("positive"))
    return _format_pa_power(f"-{match.group('negative')}")


def _repair_vacuum_classification(md_text: str) -> str:
    if not all(keyword in md_text for keyword in VACUUM_KEYWORDS):
        return md_text

    start = md_text.find("粗真空")
    end_anchor = md_text.find("极高真空", start)
    if start == -1 or end_anchor == -1:
        return md_text

    paragraph_end = md_text.find("\n\n", end_anchor)
    line_end = md_text.find("\n", end_anchor)
    if paragraph_end != -1 and paragraph_end - start < 2000:
        end = paragraph_end
    elif line_end != -1 and line_end - start < 2000:
        end = line_end
    else:
        end = end_anchor + len("极高真空")

    return f"{md_text[:start]}{VACUUM_CLASSIFICATION}{md_text[end:]}"


def _repair_line_numbering(md_text: str) -> str:
    repaired = re.sub(r"(?m)^-\s*(\d{1,2})\.([^\s\n])", r"\1. \2", md_text)
    return re.sub(r"(?m)^(\d{1,2})\.([^\s\n])", r"\1. \2", repaired)
