from __future__ import annotations

import unittest

from backend.app.formula_enhancer import _accept_candidate, _clean_candidate


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


if __name__ == "__main__":
    unittest.main()
