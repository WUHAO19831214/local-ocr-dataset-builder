from __future__ import annotations

import unittest

from backend.app.formula_enhancer import _accept_candidate, _apply_option_row, _clean_candidate, _parse_option_row


class FormulaEnhancerTests(unittest.TestCase):
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
