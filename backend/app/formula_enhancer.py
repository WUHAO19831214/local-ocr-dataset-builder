"""Revisit likely mathematical regions with the locally cached PaddleOCR-VL model."""

from __future__ import annotations

import json
import os
import re
import signal
import shutil
import subprocess
import tempfile
import time
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Callable

import pymupdf

from .process_registry import register_process, unregister_process


PADDLE_PYTHON = Path("/Users/wuhao/LocalProjects/Codex/macbook-air-m2/paddleocr-vl-benchmark/.venv/bin/python")
DOCLING_PYTHON = Path("/Users/wuhao/my-pdf-tool/my-pdf-tool/venv/bin/python")
PADDLE_TIMEOUT_SECONDS = 2700
MATH_MARKER = re.compile(r"\\(?:frac|sqrt|Delta|theta|phi|lambda|pi|sin|cos|mathrm|text|sum|int|begin)|[\^_=<>≤≥]", re.I)
CJK = re.compile(r"[\u3400-\u9fff]")
LATIN_OR_GREEK = re.compile(r"[A-Za-zα-ωΑ-Ω∆Δ]")
EXPLICIT_MATH = re.compile(r"\$|\\(?:frac|sqrt|Delta|theta|phi|lambda|pi|sum|int)|[=<>≤≥^]")


def _should_revisit(item: dict) -> bool:
    label = item.get("label")
    value = item.get("text", "").strip()
    if not item.get("prov"):
        return False
    if label == "formula":
        return True
    if not value:
        return False
    if label not in {"text", "list_item"}:
        return False
    if label == "list_item" and len(value) <= 35 and re.match(r"^\s*[A-F][.．、]", value):
        return True
    return bool(len(value) <= 120 and LATIN_OR_GREEK.search(value) and EXPLICIT_MATH.search(value))


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
    # VL sometimes appends a layout crop as HTML after otherwise useful text.
    # Docling stores real figures separately; this temporary imgs/ reference
    # cannot be resolved by the dataset exporter.
    candidate = re.sub(r'<div\b[^>]*>\s*<img\b[^>]*src=["\']imgs/[^>]*>\s*</div>', '', candidate, flags=re.I)
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
        math_span = re.search(r"\$\$([^$]+)\$\$|\$([^$]+)\$", candidate)
        if math_span:
            candidate = (math_span.group(1) or math_span.group(2)).strip()
        else:
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
    if re.search(r"<\s*/?\s*(?:div|img)\b|&lt;|!\[[^]]*\]\(|img_in_image_box", new, re.I):
        return False, "识别结果混入图片或 HTML，需整行复核"
    if "$$" in new and label != "formula":
        return False, "行内公式混入块级数学分隔符"
    if new.count("$") % 2:
        return False, "数学分隔符不成对"
    if not MATH_MARKER.search(new):
        return False, "结果没有可复现的数学结构"
    if label == "formula" and CJK.search(new):
        return False, "公式候选混入正文，需人工复核"
    old_cjk = "".join(CJK.findall(old))
    new_cjk = "".join(CJK.findall(new))
    if len(old_cjk) >= 4 and SequenceMatcher(None, old_cjk, new_cjk).ratio() < 0.62:
        return False, "中文主体与原 OCR 差异过大，需人工复核"
    if label != "formula" and "$" not in new:
        return False, "行内公式缺少数学分隔符"
    return True, "自动采用；仍建议对照裁图检查"


def _option_label(item: dict) -> str | None:
    match = re.match(r"^\s*([A-D])[.．、]", item.get("orig", "") or item.get("text", ""))
    return match.group(1) if match else None


def _option_rows(pdf_path: Path, data: dict, output_dir: Path, max_pages: int | None) -> list[dict]:
    """Find one-line answer choices where Docling turned a missing choice into a picture."""
    rows = []
    items = data["texts"]
    with pymupdf.open(str(pdf_path)) as pdf:
        for page_number in range(1, pdf.page_count + 1):
            if max_pages is not None and page_number > max_pages:
                continue
            options = []
            for index, item in enumerate(items):
                label = _option_label(item)
                if label and item.get("prov") and item["prov"][0]["page_no"] == page_number:
                    box = item["prov"][0]["bbox"]
                    options.append((index, label, (box["t"] + box["b"]) / 2, box))
            seen = set()
            for index, label, center, _ in options:
                if label != "A" or index in seen:
                    continue
                peers = [entry for entry in options if abs(entry[2] - center) <= 14]
                labels = {entry[1] for entry in peers}
                if len(peers) < 2 or labels == set("ABCD") or len(labels) != len(peers):
                    continue
                left = min(entry[3]["l"] for entry in peers)
                right = max(entry[3]["r"] for entry in peers)
                pictures = []
                for picture_index, picture in enumerate(data.get("pictures", [])):
                    if not picture.get("prov") or picture["prov"][0]["page_no"] != page_number:
                        continue
                    box = picture["prov"][0]["bbox"]
                    if left < (box["l"] + box["r"]) / 2 < right and abs((box["t"] + box["b"]) / 2 - center) <= 17:
                        pictures.append((picture_index, box))
                if len(pictures) != 1 or len(labels) != 3:
                    continue
                all_boxes = [entry[3] for entry in peers] + [pictures[0][1]]
                page = pdf[page_number - 1]
                rect = pymupdf.Rect(
                    max(0, min(box["l"] for box in all_boxes) - 15),
                    max(0, page.rect.height - max(box["t"] for box in all_boxes) - 20),
                    min(page.rect.width, max(box["r"] for box in all_boxes) + 15),
                    min(page.rect.height, page.rect.height - min(box["b"] for box in all_boxes) + 20),
                )
                crop = output_dir / f"page-{page_number:02d}-option-row-{index:04d}.png"
                page.get_pixmap(matrix=pymupdf.Matrix(4, 4), clip=rect, alpha=False).save(str(crop))
                rows.append({"index": f"option-row-{index}", "page": page_number, "crop": str(crop),
                             "kind": "option_row", "peers": [entry[0] for entry in peers],
                             "picture": pictures[0][0]})
                seen.update(entry[0] for entry in peers)
    return rows


def _parse_option_row(raw: str) -> dict[str, str]:
    choices = list(re.finditer(r"(?<![A-Za-z])([A-D])[.．、]\s*", raw))
    if [match.group(1) for match in choices] != list("ABCD"):
        return {}
    parsed = {}
    for position, match in enumerate(choices):
        end = choices[position + 1].start() if position + 1 < len(choices) else len(raw)
        segment = raw[match.end():end]
        formula = re.search(r"\$([^$\n]+)\$", segment)
        if not formula:
            return {}
        value = f"{match.group(1)}. ${formula.group(1).strip()}$"
        if not _accept_candidate(match.group(1) + ".", value, "text")[0]:
            return {}
        parsed[match.group(1)] = value
    return parsed


def _apply_option_row(data: dict, row: dict, parsed: dict[str, str]) -> bool:
    """Replace split option nodes and the misclassified formula picture in reading order."""
    items = data["texts"]
    peers = row["peers"]
    by_label = {_option_label(items[index]): index for index in peers}
    missing = (set("ABCD") - set(by_label))
    if len(missing) != 1:
        return False
    picture_index = row["picture"]
    picture = data["pictures"][picture_index]
    parent_ref = items[by_label["A"]]["parent"]["$ref"]
    if parent_ref == "#/body":
        parent = data["body"]
    else:
        parent_kind, parent_index = parent_ref.strip("#/").split("/")
        parent = data[parent_kind][int(parent_index)]
    sibling_refs = {f"#/texts/{index}" for index in peers}
    for index in peers:
        item = items[index]
        old_parent_ref = item["parent"]["$ref"]
        kind, number = old_parent_ref.strip("#/").split("/") if old_parent_ref != "#/body" else ("body", "0")
        old_parent = data[kind][int(number)] if kind != "body" else data["body"]
        if old_parent is not parent:
            old_parent["children"] = [child for child in old_parent["children"] if child.get("$ref") != item["self_ref"]]
            sibling_refs.add(old_parent_ref)
        item.update(label="text", text=parsed[_option_label(item)], orig=parsed[_option_label(item)], parent={"$ref": parent_ref})
        item.pop("enumerated", None)
        item.pop("marker", None)
    new_index = len(items)
    new_item = deepcopy(items[by_label["A"]])
    new_item.update(self_ref=f"#/texts/{new_index}", orig=parsed[next(iter(missing))],
                    text=parsed[next(iter(missing))], prov=deepcopy(picture["prov"]))
    new_item["prov"][0]["charspan"] = [0, len(new_item["text"])]
    items.append(new_item)
    picture_parent_ref = picture["parent"]["$ref"]
    if picture_parent_ref == "#/body":
        picture_parent = data["body"]
    else:
        kind, number = picture_parent_ref.strip("#/").split("/")
        picture_parent = data[kind][int(number)]
    picture_parent["children"] = [child for child in picture_parent["children"] if child.get("$ref") != picture["self_ref"]]
    sibling_refs.add(picture["self_ref"])
    children = parent["children"]
    insert_at = next(i for i, child in enumerate(children) if child.get("$ref") == items[by_label["A"]]["self_ref"])
    parent["children"] = ([child for child in children[:insert_at] if child.get("$ref") not in sibling_refs]
                          + [{"$ref": items[by_label[label]]["self_ref"] if label in by_label else new_item["self_ref"]}
                             for label in "ABCD"]
                          + [child for child in children[insert_at:] if child.get("$ref") not in sibling_refs])
    return True


def _read_paddle_output(process: subprocess.Popen, log: Callable[[str], None]) -> int:
    assert process.stdout is not None
    lines: Queue[str | None] = Queue()

    def read_lines() -> None:
        try:
            for line in process.stdout:
                lines.put(line.strip())
        finally:
            lines.put(None)

    Thread(target=read_lines, daemon=True).start()
    started = time.monotonic()
    current_region = "模型加载中"
    while True:
        elapsed = time.monotonic() - started
        if elapsed >= PADDLE_TIMEOUT_SECONDS:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise RuntimeError("PaddleOCR-VL 超过 45 分钟仍未完成；已停止复核，原始 OCR 文件保留。")
        try:
            line = lines.get(timeout=min(30, PADDLE_TIMEOUT_SECONDS - elapsed))
        except Empty:
            log(f"PaddleOCR-VL 仍在运行：{current_region}，已耗时 {int(time.monotonic() - started)} 秒")
            continue
        if line is None:
            return process.wait()
        if line.startswith("REGION_START"):
            current_region = line.removeprefix("REGION_START").strip()
        if line.startswith(("MODEL_LOADING", "MODEL_READY", "REGION_START", "REGION ", "ERROR ")):
            log(line)


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
    manifest.extend(_option_rows(pdf_path, data, crops_dir, max_pages))
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
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, env=env, bufsize=1, start_new_session=True)
        register_process(process)
        try:
            return_code = _read_paddle_output(process, log)
        finally:
            unregister_process(process)
        if return_code != 0 or not result_path.is_file():
            raise RuntimeError("PaddleOCR-VL 公式复核失败；原 OCR 文件未被修改")
        results = json.loads(result_path.read_text(encoding="utf-8"))

    report: list[dict] = []
    accepted = 0
    for entry in manifest:
        if entry.get("kind") == "option_row":
            parsed = _parse_option_row(results.get(str(entry["index"]), ""))
            use = bool(parsed) and _apply_option_row(data, entry, parsed)
            accepted += int(use)
            report.append({"text_index": entry["index"], "page": entry["page"],
                           "crop": str(Path(entry["crop"]).relative_to(dataset_dir)),
                           "candidate": parsed, "accepted": use,
                           "reason": "整行恢复四个可编辑选项" if use else "整行公式未可靠识别，需人工复核"})
            continue
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
