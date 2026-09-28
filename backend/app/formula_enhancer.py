"""Revisit likely mathematical regions with the locally cached PaddleOCR-VL model."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

import pymupdf


PADDLE_PYTHON = Path("/Users/wuhao/LocalProjects/Codex/macbook-air-m2/paddleocr-vl-benchmark/.venv/bin/python")
DOCLING_PYTHON = Path("/Users/wuhao/my-pdf-tool/my-pdf-tool/venv/bin/python")
MATH_MARKER = re.compile(r"\\(?:frac|sqrt|Delta|theta|phi|lambda|pi|sin|cos|mathrm|text|sum|int|begin)|[\^_=<>≤≥]", re.I)
CJK = re.compile(r"[\u3400-\u9fff]")
LATIN_OR_GREEK = re.compile(r"[A-Za-zα-ωΑ-Ω∆Δ]")
MIXED_MATH = re.compile(r"\d|[=<>＞＜＋+−*/^_]|[∆Δφθλπ]")


def _should_revisit(item: dict) -> bool:
    label = item.get("label")
    value = item.get("text", "").strip()
    if not value or not item.get("prov"):
        return False
    if label == "formula":
        return True
    if label not in {"text", "list_item"}:
        return False
    if label == "list_item" and len(value) <= 35 and LATIN_OR_GREEK.search(value):
        return True
    return bool(LATIN_OR_GREEK.search(value) and MIXED_MATH.search(value))


def _crop_regions(pdf_path: Path, items: list[dict], output_dir: Path, max_pages: int | None) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    with pymupdf.open(str(pdf_path)) as pdf:
        for index, item in enumerate(items):
            if not _should_revisit(item):
                continue
            provenance = item["prov"][0]
            page_number = provenance["page_no"]
            if max_pages is not None and page_number > max_pages:
                continue
            if not 1 <= page_number <= pdf.page_count:
                continue
            page = pdf[page_number - 1]
            def item_rect(value: dict) -> pymupdf.Rect:
                box = value["prov"][0]["bbox"]
                if box.get("coord_origin") == "BOTTOMLEFT":
                    top, bottom = page.rect.height - box["t"], page.rect.height - box["b"]
                else:
                    top, bottom = box["t"], box["b"]
                return pymupdf.Rect(box["l"], top, box["r"], bottom)

            rect = item_rect(item)
            crop_name = f"page-{page_number:02d}-text-{index:04d}.png"
            option = re.match(r"^\s*([A-F])[.．、]", item.get("orig", "") or item.get("text", ""))
            if option and len(item.get("text", "")) <= 35:
                peers: list[tuple[int, pymupdf.Rect]] = []
                for other_index, other in enumerate(items):
                    if other.get("label") != "list_item" or not other.get("prov") or len(other.get("text", "")) > 35:
                        continue
                    if other["prov"][0]["page_no"] != page_number:
                        continue
                    if not re.match(r"^\s*[A-F][.．、]", other.get("orig", "") or other.get("text", "")):
                        continue
                    other_rect = item_rect(other)
                    if abs((rect.y0 + rect.y1 - other_rect.y0 - other_rect.y1) / 2) <= 6:
                        peers.append((other_index, other_rect))
                if len(peers) >= 2:
                    rect = pymupdf.Rect(
                        min(value.x0 for _, value in peers), min(value.y0 for _, value in peers),
                        max(value.x1 for _, value in peers), max(value.y1 for _, value in peers),
                    )
                    crop_name = f"page-{page_number:02d}-options-{min(i for i, _ in peers):04d}.png"
            top_pad = max(4, rect.height * 0.15) if item.get("label") == "formula" else 2
            bottom_pad = max(14, rect.height * 0.55) if item.get("label") == "formula" else 2
            rect = pymupdf.Rect(rect.x0 - 3, rect.y0 - top_pad, rect.x1 + 4, rect.y1 + bottom_pad) & page.rect
            if rect.width < 8 or rect.height < 5:
                continue
            crop_path = output_dir / crop_name
            if not crop_path.is_file():
                page.get_pixmap(matrix=pymupdf.Matrix(4, 4), clip=rect, alpha=False).save(str(crop_path))
            manifest.append({"index": index, "page": page_number, "crop": str(crop_path)})
    return manifest


def _clean_candidate(raw: str, item: dict) -> str:
    candidate = raw.strip()
    candidate = re.sub(r"^#{1,6}\s*", "", candidate)
    candidate = " ".join(candidate.split())
    marker = re.match(r"^\s*([A-F])[.．、]", item.get("orig", "") or item.get("text", ""))
    choices = list(re.finditer(r"(?<![A-Za-z])([A-F])[.．、]\s*", candidate))
    if marker and len(choices) >= 2:
        matching = next((position for position, choice in enumerate(choices) if choice.group(1) == marker.group(1)), None)
        if matching is None:
            return ""
        start = choices[matching].start()
        end = choices[matching + 1].start() if matching + 1 < len(choices) else len(candidate)
        candidate = candidate[start:end].strip()
    if item.get("label") == "formula":
        candidate = re.sub(r"^\$\$?\s*|\s*\$\$?$", "", candidate).strip()
    else:
        # Docling list items can store the list marker in ``orig`` only. Its
        # Markdown exporter adds that marker, so do not also put it in text.
        marker = re.match(r"^\s*([A-F]|\d{1,2})[.．、]", item.get("orig", ""))
        old_text_has_marker = re.match(r"^\s*([A-F]|\d{1,2})[.．、]", item.get("text", ""))
        new_marker = re.match(r"^\s*([A-F]|\d{1,2})[.．、]\s*", candidate)
        if marker and not old_text_has_marker and new_marker and marker.group(1) == new_marker.group(1):
            candidate = candidate[new_marker.end():]
    return candidate


def _accept_candidate(old: str, new: str, label: str) -> tuple[bool, str]:
    if not new or len(new) > max(250, 4 * len(old)):
        return False, "结果为空或异常冗长"
    if new.count("$") % 2:
        return False, "数学分隔符不成对"
    if not MATH_MARKER.search(new):
        return False, "结果没有可复现的数学结构"
    old_cjk = "".join(CJK.findall(old))
    new_cjk = "".join(CJK.findall(new))
    if len(old_cjk) >= 4 and SequenceMatcher(None, old_cjk, new_cjk).ratio() < 0.62:
        return False, "中文主体与原 OCR 差异过大，需人工复核"
    if label != "formula" and "$" not in new:
        return False, "行内公式缺少数学分隔符"
    return True, "自动采用；仍建议对照裁图检查"


def enhance_formula_regions(
    pdf_path: Path,
    dataset_dir: Path,
    output_name: str,
    log: Callable[[str], None],
    *,
    max_pages: int | None = None,
) -> Path:
    if not PADDLE_PYTHON.is_file():
        raise RuntimeError(f"高精度公式模式缺少 PaddleOCR-VL 环境：{PADDLE_PYTHON}")
    if not DOCLING_PYTHON.is_file():
        raise RuntimeError(f"高精度公式模式缺少 Docling 环境：{DOCLING_PYTHON}")
    dataset_dir = dataset_dir.resolve()
    json_path = dataset_dir / f"{output_name}.json"
    md_path = dataset_dir / f"{output_name}.md"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    items = data["texts"]
    crops_dir = dataset_dir / "formula_regions"
    manifest = _crop_regions(pdf_path, items, crops_dir, max_pages)
    log(f"待复核区域：{len(manifest)} 处；裁图保存在 formula_regions/")
    report_path = dataset_dir / f"{output_name}_公式复核.json"
    if not manifest:
        report_path.write_text(json.dumps({"regions": []}, ensure_ascii=False, indent=2), encoding="utf-8")
        return report_path

    with tempfile.TemporaryDirectory(prefix="formula-vl-") as tmp:
        manifest_path = Path(tmp) / "manifest.json"
        result_path = Path(tmp) / "results.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        env = os.environ.copy()
        env["PADDLE_PDX_MODEL_SOURCE"] = "bos"
        env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        env["PYTHONUNBUFFERED"] = "1"
        command = [str(PADDLE_PYTHON), str(Path(__file__).with_name("formula_vl_worker.py")), str(manifest_path), str(result_path)]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if line.startswith("REGION "):
                log(line)
            elif line.startswith("ERROR "):
                log(line)
        if process.wait() != 0 or not result_path.is_file():
            raise RuntimeError("PaddleOCR-VL 公式复核失败；原 OCR 文件未被修改")
        results = json.loads(result_path.read_text(encoding="utf-8"))

    report: list[dict] = []
    accepted = 0
    for entry in manifest:
        index = entry["index"]
        item = items[index]
        original = item.get("text", "")
        raw = results.get(str(index), "")
        candidate = _clean_candidate(raw, item)
        use, reason = _accept_candidate(original, candidate, item.get("label", ""))
        if use:
            item["text"] = candidate
            accepted += 1
        report.append({
            "text_index": index,
            "page": entry["page"],
            "crop": str(Path(entry["crop"]).relative_to(dataset_dir)),
            "original": original,
            "candidate": candidate,
            "accepted": use,
            "reason": reason,
        })
    report_path.write_text(json.dumps({"regions": report, "accepted": accepted}, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"自动采用 {accepted}/{len(manifest)} 处；逐项记录：{report_path.name}")
    if accepted:
        shutil.copy2(json_path, dataset_dir / f"{output_name}_原始OCR.json")
        shutil.copy2(md_path, dataset_dir / f"{output_name}_原始OCR.md")
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        export = subprocess.run(
            [str(DOCLING_PYTHON), str(Path(__file__).with_name("formula_md_worker.py")), str(json_path), str(md_path)],
            capture_output=True, text=True, timeout=120,
        )
        if export.returncode:
            shutil.copy2(dataset_dir / f"{output_name}_原始OCR.json", json_path)
            shutil.copy2(dataset_dir / f"{output_name}_原始OCR.md", md_path)
            raise RuntimeError(f"公式 Markdown 重建失败：{export.stderr.strip()}")
        log("已更新 md/json，原始 OCR 已备份")
    return report_path
