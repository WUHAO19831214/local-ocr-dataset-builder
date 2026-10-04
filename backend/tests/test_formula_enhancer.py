from __future__ import annotations

import unittest
import subprocess
import sys
from unittest.mock import patch

from backend.app.formula_enhancer import _accept_candidate, _apply_option_row, _clean_candidate, _parse_option_row, _read_paddle_output, _should_revisit


class FormulaEnhancerTests(unittest.TestCase):
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
