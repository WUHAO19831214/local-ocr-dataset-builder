from __future__ import annotations

import unittest
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pymupdf

from backend.app.formula_enhancer import (_accept_candidate, _accept_inline_line_candidate,
                                          _apply_option_row, _clean_candidate,
                                          _combine_inline_lines, _formula_region_rects,
                                          _insert_visual_formula_fallbacks, _needs_inline_line_review,
                                          _normalize_repeated_resistance_fraction,
                                          _parse_option_row, _read_paddle_output,
                                          _restore_dropped_chinese, _should_revisit)


class FormulaEnhancerTests(unittest.TestCase):
    def test_number_only_formula_box_expands_to_the_equation(self) -> None:
        pdf = pymupdf.open()
        page = pdf.new_page(width=300, height=400)
        page.insert_text((110, 110), "E=mc^2")
        page.insert_text((276, 110), "3")
        equation, visual = _formula_region_rects(page, pymupdf.Rect(270, 96, 290, 115))
        self.assertLess(equation.x0, 115)
        self.assertLess(equation.x1, 270)
        self.assertGreater(visual.x1, 290)
        pdf.close()

    def test_existing_wide_formula_keeps_proven_crop_padding(self) -> None:
        pdf = pymupdf.open()
        page = pdf.new_page(width=595, height=841)
        original = pymupdf.Rect(240, 610, 516, 635)
        equation, _ = _formula_region_rects(page, original)
        self.assertEqual(equation.x0, 237)
        self.assertEqual(equation.x1, 520)
        self.assertEqual(equation.y1, 649)
        pdf.close()

    def test_unrecognized_formula_gets_image_without_changing_successful_math(self) -> None:
        data = {"body": {"children": [{"$ref": "#/texts/0"}, {"$ref": "#/texts/1"}]},
                "texts": [{"label": "formula", "text": "W=mgR"},
                          {"label": "formula", "text": ""}]}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.md"
            image = Path(temporary) / "formula_regions" / "missing.png"
            image.parent.mkdir()
            pdf = pymupdf.open()
            page = pdf.new_page(width=100, height=20)
            page.get_pixmap(matrix=pymupdf.Matrix(4, 4)).save(str(image))
            pdf.close()
            path.write_text("$$\nW=mgR\n$$\n\n<!-- formula-not-decoded -->\n", encoding="utf-8")
            count = _insert_visual_formula_fallbacks(path, data, {1: "formula_regions/missing.png"})
            result = path.read_text(encoding="utf-8")
        self.assertEqual(count, 1)
        self.assertIn("$$\nW=mgR\n$$", result)
        self.assertIn("![未识别公式原图](formula_regions/missing.png){width=100.0pt}", result)

    def test_inline_fraction_lines_restore_math_and_omitted_prose(self) -> None:
        original = "I和U。若将电流表内接，则 元件两端的电压，兰 _元件的电阻；若将电流表外接，则I_ 流过元件的电流，一 _元件的电阻。"
        entries = [{"index": str(index), "line_position": index, "has_fill_blank": index in (0, 2)}
                   for index in range(4)]
        results = {"0": "I和U。若将电流表内接，则U",
                   "1": r"元件两端的电压，$\frac{U}{I}$___元件的电阻；",
                   "2": "若将电流表外接，则I",
                   "3": r"元件的电流，$\frac{U}{I}$___元件的电阻。"}
        candidate = _restore_dropped_chinese(original, _combine_inline_lines(entries, results))
        self.assertIn(r"$\frac{U}{I}$", candidate)
        self.assertEqual(candidate.count(r"\frac{U}{I}"), 2)
        self.assertIn("I___流过元件的电流", candidate)
        self.assertTrue(_accept_inline_line_candidate(original, candidate, "text")[0])

    def test_inline_review_rejects_missing_second_fraction(self) -> None:
        original = "I和U。元件的电压___，U/I___电阻；I___流过元件的电流，U/I___电阻。"
        incomplete = "I和U。元件的电压___，$\\frac{U}{I}$___电阻；I___流过元件的电流，电阻。"
        accepted, reason = _accept_inline_line_candidate(original, incomplete, "text")
        self.assertFalse(accepted)
        self.assertIn("未完整识别", reason)

    def test_repeated_resistance_fraction_uses_consistent_current_symbol(self) -> None:
        candidate = (r"电流表测电压，$\frac{U}{r}$ 表示电阻；"
                     r"电流表测电流，$\frac{U}{I}$ 表示电阻。")
        self.assertEqual(_normalize_repeated_resistance_fraction(candidate).count(r"\frac{U}{I}"), 2)
        self.assertEqual(_normalize_repeated_resistance_fraction(r"$\frac{U}{r}$"), r"$\frac{U}{r}$")

    def test_inline_review_is_limited_to_corrupted_math_prose(self) -> None:
        self.assertTrue(_needs_inline_line_review({"label": "text", "text": "电压U ___ 电阻U/I ___ 的结果是否正确？", "prov": [{}]}))
        self.assertFalse(_needs_inline_line_review({"label": "text", "text": "电压U 的结果是否正确？", "prov": [{}]}))

    def test_silent_paddle_worker_is_stopped_at_hard_timeout(self) -> None:
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                   stdout=subprocess.PIPE, text=True, start_new_session=True)
        try:
            with patch("backend.app.formula_enhancer.PADDLE_TIMEOUT_SECONDS", 0.2):
                with self.assertRaises(RuntimeError):
                    _read_paddle_output(process, lambda _: None)
            self.assertIsNotNone(process.poll())
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
            if process.stdout is not None:
                process.stdout.close()

    def test_empty_formula_box_is_sent_to_paddle(self) -> None:
        self.assertTrue(_should_revisit({"label": "formula", "text": "", "prov": [{"page_no": 1}]}))

    def test_long_physics_prose_without_formula_structure_is_not_sent_to_paddle(self) -> None:
        item = {"label": "list_item", "text": "欲使P和Q断开后，弹簧的最大弹性势能等于2.2mgR，Q的质量应为多大？",
                "prov": [{"page_no": 1}]}
        self.assertFalse(_should_revisit(item))
        item["text"] = r"A. $\frac{U_1}{U_2}$"
        self.assertTrue(_should_revisit(item))

    def test_formula_crop_keeps_only_math_when_paddle_appends_page_text(self) -> None:
        raw = '$ W = mgR $\n<div style="text-align: center;"><img src="imgs/img_in_image_box_1.jpg" alt="Image" /></div>\n误入的正文'
        candidate = _clean_candidate(raw, {"label": "formula", "text": ""})
        self.assertEqual(candidate, "W = mgR")
        self.assertTrue(_accept_candidate("", candidate, "formula")[0])

    def test_truncated_formula_does_not_replace_good_equation(self) -> None:
        self.assertFalse(_accept_candidate("", r"\frac{\pm}{3}v_{0}=(", "formula")[0])
        self.assertTrue(_accept_candidate("", r"m\cdot\frac{1}{3}v_0=(m+M)v_1", "formula")[0])

    def test_formula_before_spurious_neighbor_prose_is_kept(self) -> None:
        candidate = _clean_candidate("M=4m 估的及体目 验等标有质V时", {"label": "formula", "text": ""})
        self.assertEqual(candidate, "M=4m")
        self.assertTrue(_accept_candidate("", candidate, "formula")[0])

    def test_option_row_extracts_target_and_keeps_docling_marker(self) -> None:
        row = (
            r"A. $^{1}_{0}n$ B. $3_{0}^{1}n$ "
            r"C. $^{3}_{1}H$ D. $3_{1}^{0}e$"
        )
        item = {"label": "list_item", "orig": "C. MH", "text": "MH"}
        self.assertEqual(_clean_candidate(row, item), r"$^{3}_{1}H$")

    def test_inline_math_is_accepted_when_chinese_remains(self) -> None:
        old = "下降高度 Ah，引力势能变化量为 AEpl"
        new = r"下降高度 $\Delta h$，引力势能变化量为 $\Delta E_{p1}$"
        self.assertTrue(_accept_candidate(old, new, "list_item")[0])

    def test_unrelated_chinese_is_not_written_back(self) -> None:
        old = "下降高度 Ah，引力势能变化量为 AEpl"
        new = r"光刻技术方案 $\Delta h$"
        self.assertFalse(_accept_candidate(old, new, "list_item")[0])

    def test_inline_formula_rejects_embedded_image_markup(self) -> None:
        broken = (r"B. \frac{U_{2}^{2}}{U_{1}} $$ "
                  '<div><img src="imgs/img_in_image_box.jpg" alt="Image" /></div>')
        self.assertFalse(_accept_candidate("B.9", broken, "list_item")[0])
        self.assertFalse(_accept_candidate("B.9", r"B. $\frac{U_2^2}{U_1}$$", "list_item")[0])

    def test_candidate_drops_temporary_layout_image_html(self) -> None:
        raw = (r"入射角为 $\theta$。 "
               '<div style="text-align: center;"><img src="imgs/crop.jpg" alt="Image" /></div>')
        self.assertEqual(_clean_candidate(raw, {"label": "text", "text": "入射角为0。"}),
                         r"入射角为 $\theta$。")

    def test_whole_option_row_restores_four_editable_formulas(self) -> None:
        row = ("①电压 $U_3$ 为 ___。\n"
               r"A. $2U_{2}-U_{1}$ B. $\frac{U_{2}^{2}}{U_{1}}$ "
               r"C. $\sqrt{\frac{U_{1}^{2}+U_{2}^{2}}{2}}$ "
               r"D. $\sqrt{2U_{2}^{2}-U_{1}^{2}}$" "\n"
               '<div><img src="imgs/crop.jpg" alt="Image" /></div>')
        parsed = _parse_option_row(row)
        self.assertEqual(list(parsed), list("ABCD"))
        self.assertEqual(parsed["C"], r"C. $\sqrt{\frac{U_{1}^{2}+U_{2}^{2}}{2}}$")

    def test_recovered_choice_replaces_picture_in_reading_order(self) -> None:
        def item(index: int, label: str, parent: str) -> dict:
            return {"self_ref": f"#/texts/{index}", "parent": {"$ref": parent},
                    "label": "text" if index == 0 else "list_item",
                    "orig": f"{label}. OCR", "text": f"{label}. OCR",
                    "prov": [{"page_no": 1, "bbox": {}, "charspan": [0, 6]}],
                    **({"enumerated": False, "marker": ""} if index else {})}

        data = {"texts": [item(0, "A", "#/groups/0"), item(1, "B", "#/groups/1"),
                          item(2, "D", "#/groups/2")],
                "groups": [{"children": [{"$ref": "#/texts/0"}, {"$ref": "#/groups/1"},
                                         {"$ref": "#/groups/2"}]},
                           {"children": [{"$ref": "#/texts/1"}]},
                           {"children": [{"$ref": "#/texts/2"}]}],
                "body": {"children": [{"$ref": "#/groups/0"}, {"$ref": "#/pictures/0"}]},
                "pictures": [{"self_ref": "#/pictures/0", "parent": {"$ref": "#/body"},
                              "prov": [{"page_no": 1, "bbox": {}, "charspan": [0, 0]}]}]}
        parsed = {letter: f"{letter}. $U_{number}$" for number, letter in enumerate("ABCD", 1)}
        self.assertTrue(_apply_option_row(data, {"peers": [0, 1, 2], "picture": 0}, parsed))
        self.assertEqual(data["groups"][0]["children"],
                         [{"$ref": f"#/texts/{index}"} for index in (0, 1, 3, 2)])
        self.assertEqual(data["body"]["children"], [{"$ref": "#/groups/0"}])


if __name__ == "__main__":
    unittest.main()
