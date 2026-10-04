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


def _leftmost_formula_ink(page: pymupdf.Page, marker: pymupdf.Rect) -> float | None:
    """Find an equation printed to the left of a narrow numbered formula box."""
    search = pymupdf.Rect(max(0, page.rect.width * 0.1), marker.y0 - 5,
                          marker.x0 - 8, marker.y1 + 5) & page.rect
    if search.width < 20:
        return None
    scale = 2
    pixels = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=search,
                             colorspace=pymupdf.csGRAY, alpha=False)
    for x in range(pixels.width):
        dark = sum(pixels.samples[y * pixels.stride + x] < 170 for y in range(pixels.height))
        if dark >= 3:
            return search.x0 + x / scale
    return None


def _formula_region_rects(page: pymupdf.Page, rect: pymupdf.Rect) -> tuple[pymupdf.Rect, pymupdf.Rect]:
    """Keep existing formula boxes; widen boxes that contain only the equation number."""
    if rect.width < max(32, page.rect.width * 0.07) and rect.x0 > page.rect.width * 0.7:
        leftmost = _leftmost_formula_ink(page, rect)
        if leftmost is not None and leftmost < rect.x0 - 12:
            left = max(0, leftmost - 6)
            equation = pymupdf.Rect(left, rect.y0 - 6, rect.x0 - 5, rect.y1 + 6) & page.rect
            visual = pymupdf.Rect(left, rect.y0 - 6, rect.x1 + 5, rect.y1 + 6) & page.rect
            return equation, visual
    top_pad = max(4, rect.height * 0.15)
    bottom_pad = max(14, rect.height * 0.55)
    visual = pymupdf.Rect(rect.x0 - 3, rect.y0 - top_pad,
                          rect.x1 + 4, rect.y1 + bottom_pad) & page.rect
    return visual, visual


def _code_formula_crop(page: pymupdf.Page, equation: pymupdf.Rect,
                       source_box: pymupdf.Rect, path: Path) -> bool:
    """Make a 120 DPI, vertically padded formula-only crop for CodeFormulaV2."""
    search = pymupdf.Rect(equation.x0, source_box.y0 - 12,
                          equation.x1, source_box.y1 + 12) & page.rect
    if search.width < 10 or search.height < 10:
        return False
    scale = 2
    pixels = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=search,
                             colorspace=pymupdf.csGRAY, alpha=False)
    ink_columns = [x for x in range(pixels.width)
                   if sum(pixels.samples[y * pixels.stride + x] < 180
                          for y in range(pixels.height)) >= 2]
    if not ink_columns:
        return False
    left = max(search.x0, search.x0 + min(ink_columns) / scale - 1)
    right = min(search.x1, max(search.x0 + max(ink_columns) / scale + 8, left + 135))
    crop = pymupdf.Rect(left, search.y0, right, search.y1)
    page.get_pixmap(matrix=pymupdf.Matrix(120 / 72, 120 / 72),
                    clip=crop, alpha=False).save(str(path))
    return True


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
            source_rect = rect
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
            fallback_rect = None
            if item.get("label") == "formula":
                rect, fallback_rect = _formula_region_rects(page, rect)
            else:
                rect = pymupdf.Rect(rect.x0 - 3, rect.y0 - 2, rect.x1 + 4, rect.y1 + 2) & page.rect
            if rect.width < 8 or rect.height < 5:
                continue
            crop_path = output_dir / crop_name
            page.get_pixmap(matrix=pymupdf.Matrix(4, 4), clip=rect, alpha=False).save(str(crop_path))
            entry = {"index": index, "page": page_number, "crop": str(crop_path)}
            if fallback_rect is not None:
                fallback_path = output_dir / crop_name.replace(".png", "-fallback.png")
                page.get_pixmap(matrix=pymupdf.Matrix(4, 4), clip=fallback_rect,
                                alpha=False).save(str(fallback_path))
                entry["fallback_crop"] = str(fallback_path)
                if not item.get("text", "").strip():
                    code_path = output_dir / crop_name.replace(".png", "-code.png")
                    if _code_formula_crop(page, rect, source_rect, code_path):
                        entry["code_crop"] = str(code_path)
            manifest.append(entry)
    return manifest


def _needs_inline_line_review(item: dict) -> bool:
    text = item.get("text", "")
    return (item.get("label") in {"text", "list_item"} and bool(item.get("prov"))
            and 20 <= len(text) <= 240 and text.count("_") >= 2
            and bool(LATIN_OR_GREEK.search(text)))


def _ink_line_ranges(page: pymupdf.Page, rect: pymupdf.Rect) -> list[tuple[float, float]]:
    scale = 2
    pixels = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=rect,
                             colorspace=pymupdf.csGRAY, alpha=False)
    threshold = max(8, pixels.width // 50)
    active = [y for y in range(pixels.height)
              if sum(pixels.samples[y * pixels.stride + x] < 170 for x in range(pixels.width)) >= threshold]
    ranges: list[list[int]] = []
    for y in active:
        if not ranges or y - ranges[-1][1] > 8:
            ranges.append([y, y])
        else:
            ranges[-1][1] = y
    return [(rect.y0 + start / scale, rect.y0 + end / scale)
            for start, end in ranges if end - start >= 8]


def _has_fill_blank(pixels: pymupdf.Pixmap) -> bool:
    """A long printed rule is a fill-in blank, not a short fraction bar."""
    minimum = max(70, pixels.width // 12)
    for y in range(pixels.height):
        run = 0
        for x in range(pixels.width):
            if pixels.samples[y * pixels.stride + x] < 160:
                run += 1
                if run >= minimum:
                    return True
            else:
                run = 0
    return False


def _inline_line_regions(pdf_path: Path, items: list[dict], output_dir: Path,
                         max_pages: int | None) -> list[dict]:
    """Review short printed lines when OCR has collapsed inline math into underscores."""
    output_dir.mkdir(parents=True, exist_ok=True)
    regions: list[dict] = []
    with pymupdf.open(str(pdf_path)) as pdf:
        for index, item in enumerate(items):
            if not _needs_inline_line_review(item):
                continue
            provenance = item["prov"][0]
            page_number = provenance["page_no"]
            if (max_pages is not None and page_number > max_pages) or not 1 <= page_number <= pdf.page_count:
                continue
            page = pdf[page_number - 1]
            box = provenance["bbox"]
            top, bottom = ((page.rect.height - box["t"], page.rect.height - box["b"])
                           if box.get("coord_origin") == "BOTTOMLEFT" else (box["t"], box["b"]))
            rect = pymupdf.Rect(box["l"], top, box["r"], bottom) & page.rect
            if rect.height < 30 or rect.width < 60:
                continue
            lines = _ink_line_ranges(page, rect)
            if not 2 <= len(lines) <= 10:
                continue
            for position, (line_top, line_bottom) in enumerate(lines):
                top_pad = 4 if position == 0 else 12  # include fraction numerators above the text baseline
                crop_rect = pymupdf.Rect(rect.x0 - 3, line_top - top_pad, rect.x1 + 3,
                                         line_bottom + 4) & page.rect
                crop_path = output_dir / f"page-{page_number:02d}-inline-{index:04d}-{position:02d}.png"
                pixels = page.get_pixmap(matrix=pymupdf.Matrix(4, 4), clip=crop_rect,
                                         colorspace=pymupdf.csGRAY, alpha=False)
                page.get_pixmap(matrix=pymupdf.Matrix(4, 4), clip=crop_rect,
                                alpha=False).save(str(crop_path))
                regions.append({"index": f"inline-{index}-{position}", "text_index": index,
                                "line_position": position, "page": page_number,
                                "crop": str(crop_path), "kind": "inline_line",
                                "has_fill_blank": _has_fill_blank(pixels)})
            if len(regions) >= 30:
                break
    return regions


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
            if CJK.search(candidate):
                prefix = candidate[:CJK.search(candidate).start()].strip()
                prefix = re.sub(r"[①-⑳].*$", "", prefix).strip()
                if "=" in prefix and len(prefix) >= 3 and _balanced_math(prefix):
                    candidate = prefix
    else:
        # Docling list items can store the list marker in ``orig`` only. Its
        # Markdown exporter adds that marker, so do not also put it in text.
        marker = re.match(r"^\s*([A-F]|\d{1,2})[.．、]", item.get("orig", ""))
        old_text_has_marker = re.match(r"^\s*([A-F]|\d{1,2})[.．、]", item.get("text", ""))
        new_marker = re.match(r"^\s*([A-F]|\d{1,2})[.．、]\s*", candidate)
        if marker and not old_text_has_marker and new_marker and marker.group(1) == new_marker.group(1):
            candidate = candidate[new_marker.end():]
    return candidate


def _combine_inline_lines(entries: list[dict], results: dict[str, str]) -> str:
    lines = []
    for entry in sorted(entries, key=lambda value: value["line_position"]):
        raw = results.get(str(entry["index"]), "")
        if re.search(r"<\s*/?\s*(?:div|img)\b|!\[[^]]*\]\(", raw, re.I):
            return ""
        line = re.sub(r"^\s*#{1,6}\s*", "", raw.strip())
        line = " ".join(line.split())
        if not line or line.count("$") % 2 or "$$" in line:
            return ""
        if entry.get("has_fill_blank") and "___" not in line and re.search(r"[A-Za-z]\s*$", line):
            line += "___"
        lines.append(line)
    return "".join(lines)


def _restore_dropped_chinese(original: str, candidate: str) -> str:
    """Keep multi-character prose that a line OCR omitted beside a blank rule."""
    old_chars = "".join(CJK.findall(original))
    new_chars = "".join(CJK.findall(candidate))
    positions = [match.start() for match in CJK.finditer(candidate)]
    additions: list[tuple[int, str]] = []
    for tag, old_start, old_end, new_start, _ in SequenceMatcher(None, old_chars, new_chars).get_opcodes():
        missing = old_chars[old_start:old_end]
        if tag == "delete" and len(missing) >= 2 and missing not in candidate:
            point = positions[new_start] if new_start < len(positions) else len(candidate)
            additions.append((point, missing))
    for point, missing in reversed(additions):
        candidate = candidate[:point] + missing + candidate[point:]
    return candidate


def _normalize_repeated_resistance_fraction(candidate: str) -> str:
    """Resolve OCR's I/r confusion only when the same resistance fraction repeats."""
    if ("电流表" in candidate and "电阻" in candidate and "电压" in candidate
            and candidate.count(r"\frac{U}{I}") == 1
            and candidate.count(r"\frac{U}{r}") == 1):
        return candidate.replace(r"\frac{U}{r}", r"\frac{U}{I}")
    return candidate


def _balanced_math(value: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for character in value:
        if character in "([{":
            stack.append(character)
        elif character in pairs:
            if not stack or stack.pop() != pairs[character]:
                return False
    return not stack


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
    if label == "formula" and not _balanced_math(new):
        return False, "公式括号或结构不完整，需人工复核"
    old_cjk = "".join(CJK.findall(old))
    new_cjk = "".join(CJK.findall(new))
    if len(old_cjk) >= 4 and SequenceMatcher(None, old_cjk, new_cjk).ratio() < 0.62:
        return False, "中文主体与原 OCR 差异过大，需人工复核"
    if label != "formula" and "$" not in new:
        return False, "行内公式缺少数学分隔符"
    return True, "自动采用；仍建议对照裁图检查"


def _accept_inline_line_candidate(old: str, new: str, label: str) -> tuple[bool, str]:
    accepted, reason = _accept_candidate(old, new, label)
    if accepted and old.count("_") >= 3 and new.count(r"\frac") < 2:
        return False, "段内多处疑似分式未完整识别，保留原文供复核"
    return accepted, reason


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


def _read_paddle_output(process: subprocess.Popen, log: Callable[[str], None],
                        timeout_seconds: float | None = None,
                        model_name: str = "PaddleOCR-VL") -> int:
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
    timeout_seconds = timeout_seconds if timeout_seconds is not None else PADDLE_TIMEOUT_SECONDS
    current_region = "模型加载中"
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
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise RuntimeError(f"{model_name} 超过 {int(timeout_seconds // 60)} 分钟仍未完成；已停止复核，原始 OCR 文件保留。")
        try:
            line = lines.get(timeout=min(30, timeout_seconds - elapsed))
        except Empty:
            log(f"{model_name} 仍在运行：{current_region}，已耗时 {int(time.monotonic() - started)} 秒")
            continue
        if line is None:
            return process.wait()
        if line.startswith("REGION_START"):
            current_region = line.removeprefix("REGION_START").strip()
        if line.startswith(("MODEL_LOADING", "MODEL_READY", "REGION_START", "REGION ", "ERROR ")):
            log(line)


def _retry_code_formula(manifest: list[dict], results: dict[str, str], items: list[dict],
                        log: Callable[[str], None]) -> dict[str, str]:
    unresolved = []
    for entry in manifest:
        index = entry.get("index")
        if not isinstance(index, int) or not entry.get("code_crop"):
            continue
        item = items[index]
        if item.get("label") != "formula" or item.get("text", "").strip():
            continue
        candidate = _clean_candidate(results.get(str(index), ""), item)
        if not _accept_candidate("", candidate, "formula")[0]:
            unresolved.append(entry)
    if not unresolved or not DOCLING_PYTHON.is_file():
        return {}
    log(f"CodeFormulaV2 单独复核未识别公式：{len(unresolved)} 处")
    with tempfile.TemporaryDirectory(prefix="formula-code-") as tmp:
        manifest_path = Path(tmp) / "manifest.json"
        result_path = Path(tmp) / "results.json"
        manifest_path.write_text(json.dumps(unresolved, ensure_ascii=False), encoding="utf-8")
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        command = [str(DOCLING_PYTHON), str(Path(__file__).with_name("formula_code_worker.py")),
                   str(manifest_path), str(result_path)]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, env=env, bufsize=1, start_new_session=True)
        register_process(process)
        try:
            return_code = _read_paddle_output(process, log, timeout_seconds=min(900, 180 + 120 * len(unresolved)),
                                              model_name="CodeFormulaV2")
        except RuntimeError as exc:
            log(str(exc))
            return {}
        finally:
            unregister_process(process)
        if return_code or not result_path.is_file():
            log("CodeFormulaV2 单独复核失败；继续保留公式原图")
            return {}
        return json.loads(result_path.read_text(encoding="utf-8"))


def _ordered_blank_formulas(data: dict) -> list[int]:
    ordered: list[int] = []

    def visit(children: list[dict]) -> None:
        for child in children:
            ref = child.get("$ref", "")
            parts = ref.strip("#/").split("/")
            if len(parts) != 2 or not parts[1].isdigit():
                continue
            kind, index = parts[0], int(parts[1])
            if kind == "texts":
                item = data["texts"][index]
                if item.get("label") == "formula" and not item.get("text", "").strip():
                    ordered.append(index)
            elif kind == "groups":
                visit(data["groups"][index].get("children", []))

    visit(data["body"].get("children", []))
    return ordered


def _root_position(data: dict, ref: str) -> tuple[int, float] | None:
    parts = ref.strip("#/").split("/")
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    kind, index = parts[0], int(parts[1])
    if kind == "groups":
        positions = [_root_position(data, child.get("$ref", ""))
                     for child in data["groups"][index].get("children", [])]
        positions = [position for position in positions if position is not None]
        return min(positions) if positions else None
    item = data.get(kind, [])[index]
    if not item.get("prov"):
        return None
    provenance = item["prov"][0]
    bbox = provenance.get("bbox", {})
    if "t" not in bbox:
        return None
    top = -bbox["t"] if bbox.get("coord_origin") == "BOTTOMLEFT" else bbox["t"]
    return provenance["page_no"], top


def _repair_direct_text_order(data: dict) -> list[int]:
    """Insert late OCR prose back between the formulas indicated by its PDF Y coordinate."""
    children = data["body"].get("children", [])
    moved: list[int] = []

    def formula_position(child: dict) -> tuple[int, float] | None:
        ref = child.get("$ref", "")
        if not ref.startswith("#/texts/"):
            return None
        index = int(ref.rsplit("/", 1)[1])
        if data["texts"][index].get("label") != "formula":
            return None
        return _root_position(data, ref)

    text_refs = [child["$ref"] for child in children if child.get("$ref", "").startswith("#/texts/")]
    for ref in text_refs:
        index = int(ref.rsplit("/", 1)[1])
        if data["texts"][index].get("label") != "text":
            continue
        current = next(position for position, child in enumerate(children) if child.get("$ref") == ref)
        source_position = _root_position(data, ref)
        if source_position is None:
            continue
        page, y = source_position
        # Only move prose that Docling placed after *both* neighbouring equations.
        # This avoids changing normal paragraph order on multi-column pages.
        above = [position for position, child in enumerate(children[:current])
                 if (anchor := formula_position(child)) is not None
                 and anchor[0] == page and anchor[1] < y - 10]
        below = [position for position, child in enumerate(children[:current])
                 if (anchor := formula_position(child)) is not None
                 and anchor[0] == page and anchor[1] > y + 10]
        if not above or not below or max(above) >= min(below):
            continue
        target = next((position for position, child in enumerate(children[:current])
                       if not child.get("$ref", "").startswith("#/pictures/")
                       and (other := _root_position(data, child.get("$ref", ""))) is not None
                       and other[0] == page and other[1] > y + 10), None)
        if target is not None:
            child = children.pop(current)
            children.insert(target, child)
            moved.append(index)
    return moved


def _insert_visual_formula_fallbacks(md_path: Path, data: dict, fallback_paths: dict[int, str]) -> int:
    """Keep an unrecognized equation visible without altering accepted native math."""
    markdown = md_path.read_text(encoding="utf-8")
    marker = "<!-- formula-not-decoded -->"
    ordered = _ordered_blank_formulas(data)
    if markdown.count(marker) != len(ordered):
        return 0
    inserted = 0
    for index in ordered:
        replacement = marker
        if index in fallback_paths:
            image_path = md_path.parent / fallback_paths[index]
            width_attribute = ""
            if image_path.is_file():
                width_attribute = f"{{width={pymupdf.Pixmap(str(image_path)).width / 4:.1f}pt}}"
            replacement = f"![未识别公式原图]({fallback_paths[index]}){width_attribute}"
            inserted += 1
        markdown = markdown.replace(marker, replacement, 1)
    if inserted:
        md_path.write_text(markdown, encoding="utf-8")
    return inserted


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
    reordered = _repair_direct_text_order(data)
    crops_dir = dataset_dir / "formula_regions"
    manifest = _crop_regions(pdf_path, items, crops_dir, max_pages)
    manifest.extend(_inline_line_regions(pdf_path, items, crops_dir, max_pages))
    manifest.extend(_option_rows(pdf_path, data, crops_dir, max_pages))
    log(f"待复核区域：{len(manifest)} 处；裁图保存在 formula_regions/")
    report_path = dataset_dir / f"{output_name}_公式复核.json"
    if not manifest:
        report_path.write_text(json.dumps({"regions": [], "accepted": 0,
                                           "reading_order_repaired": reordered},
                                          ensure_ascii=False, indent=2), encoding="utf-8")
        if reordered:
            shutil.copy2(json_path, dataset_dir / f"{output_name}_原始OCR.json")
            shutil.copy2(md_path, dataset_dir / f"{output_name}_原始OCR.md")
            json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            export = subprocess.run(
                [str(DOCLING_PYTHON), str(Path(__file__).with_name("formula_md_worker.py")),
                 str(json_path), str(md_path)], capture_output=True, text=True, timeout=120)
            if export.returncode:
                shutil.copy2(dataset_dir / f"{output_name}_原始OCR.json", json_path)
                shutil.copy2(dataset_dir / f"{output_name}_原始OCR.md", md_path)
                raise RuntimeError(f"正文顺序重建失败：{export.stderr.strip()}")
            log(f"已按原 PDF 坐标修复 {len(reordered)} 处正文顺序")
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

    code_results = _retry_code_formula(manifest, results, items, log)

    report: list[dict] = []
    accepted = 0
    fallback_paths: dict[int, str] = {}
    inline_groups: dict[int, list[dict]] = {}
    for entry in manifest:
        if entry.get("kind") == "inline_line":
            inline_groups.setdefault(entry["text_index"], []).append(entry)
            continue
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
        if not use and item.get("label") == "formula" and str(index) in code_results:
            code_candidate = _clean_candidate(code_results[str(index)], item)
            code_use, code_reason = _accept_candidate(original, code_candidate, "formula")
            if code_use:
                candidate, use, reason = code_candidate, True, "CodeFormulaV2 单条公式复核采用；仍建议对照裁图检查"
            else:
                reason = f"PaddleOCR-VL 与 CodeFormulaV2 未可靠识别；{code_reason}"
        if use:
            item["text"] = candidate
            accepted += 1
        elif item.get("label") == "formula" and not original.strip() and entry.get("fallback_crop"):
            fallback_paths[index] = str(Path(entry["fallback_crop"]).relative_to(dataset_dir))
        report.append({
            "text_index": index,
            "page": entry["page"],
            "crop": str(Path(entry["crop"]).relative_to(dataset_dir)),
            "original": original,
            "candidate": candidate,
            "accepted": use,
            "reason": reason,
            "visual_fallback": fallback_paths.get(index),
        })
    for index, entries in inline_groups.items():
        item = items[index]
        original = item.get("text", "")
        candidate = _combine_inline_lines(entries, results)
        if candidate:
            candidate = _restore_dropped_chinese(original, candidate)
            candidate = _normalize_repeated_resistance_fraction(candidate)
        use, reason = _accept_inline_line_candidate(original, candidate, item.get("label", ""))
        if use:
            item["text"] = candidate
            accepted += 1
        report.append({"text_index": index, "page": entries[0]["page"],
                       "crops": [str(Path(entry["crop"]).relative_to(dataset_dir)) for entry in entries],
                       "original": original, "candidate": candidate, "accepted": use,
                       "reason": reason})
    report_path.write_text(json.dumps({"regions": report, "accepted": accepted,
                                       "reading_order_repaired": reordered}, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"自动采用 {accepted}/{len(report)} 处；逐项记录：{report_path.name}")
    if reordered:
        log(f"已按原 PDF 坐标修复 {len(reordered)} 处正文顺序")
    if accepted or fallback_paths or reordered:
        shutil.copy2(json_path, dataset_dir / f"{output_name}_原始OCR.json")
        shutil.copy2(md_path, dataset_dir / f"{output_name}_原始OCR.md")
        if accepted or reordered:
            json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            export = subprocess.run(
                [str(DOCLING_PYTHON), str(Path(__file__).with_name("formula_md_worker.py")), str(json_path), str(md_path)],
                capture_output=True, text=True, timeout=120,
            )
            if export.returncode:
                shutil.copy2(dataset_dir / f"{output_name}_原始OCR.json", json_path)
                shutil.copy2(dataset_dir / f"{output_name}_原始OCR.md", md_path)
                raise RuntimeError(f"公式 Markdown 重建失败：{export.stderr.strip()}")
        inserted = _insert_visual_formula_fallbacks(md_path, data, fallback_paths)
        log(f"已更新 md/json，原始 OCR 已备份；未识别公式保留原图 {inserted} 处")
    return report_path
