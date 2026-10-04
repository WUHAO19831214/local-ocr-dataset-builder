from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from backend.app.word_exporter import export_editable_docx


@unittest.skipUnless(shutil.which("pandoc"), "Pandoc is required for native Word math")
class WordMathExportTests(unittest.TestCase):
    def test_existing_display_equations_and_inline_fraction_stay_native(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            markdown = root / "sample.md"
            output = root / "sample.docx"
            markdown.write_text(
                "电阻 $\\frac{U}{I}$。\n\n$$\nW=mgR\n$$\n\n"
                "$$\nmgh+mgR=\\frac{1}{2}mv_0^2\n$$\n\n"
                "$$\nv_{0}=\\frac{3}{2}\\sqrt{2gR}\n$$\n", encoding="utf-8",
            )
            export_editable_docx(markdown, output)
            with ZipFile(output) as archive:
                document = archive.read("word/document.xml").decode("utf-8")
        self.assertGreaterEqual(document.count("<m:oMath>"), 4)
        self.assertNotIn("\\frac", document)
        self.assertNotIn("\\sqrt", document)
        self.assertNotIn("W=mgR", document)


if __name__ == "__main__":
    unittest.main()
