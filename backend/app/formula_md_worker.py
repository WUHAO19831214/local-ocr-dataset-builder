"""Re-export patched Docling JSON with its original document structure."""

from __future__ import annotations

import re
import sys
import html
from pathlib import Path

from docling_core.types.doc import DoclingDocument, ImageRefMode


def main() -> None:
    json_path, md_path = map(Path, sys.argv[1:3])
    document = DoclingDocument.model_validate_json(json_path.read_text(encoding="utf-8"))
    markdown = document.export_to_markdown(image_mode=ImageRefMode.REFERENCED)

    def rewrite(match: re.Match[str]) -> str:
        filename = Path(match.group(2)).name
        if (md_path.parent / "images" / filename).is_file():
            return f"![{match.group(1)}](images/{filename})"
        return match.group(0)

    markdown = re.sub(r"!\[([^]]*)\]\(([^)]+)\)", rewrite, markdown)

    # Docling escapes Markdown punctuation even inside LaTeX spans. Pandoc
    # needs the literal underscore and comparison operators to create OMML.
    def restore_math(match: re.Match[str]) -> str:
        value = html.unescape(match.group(0))
        value = value.replace(r"\_", "_")
        delimiter = "$$" if value.startswith("$$") else "$"
        return f"{delimiter}{value[len(delimiter):-len(delimiter)].strip()}{delimiter}"

    markdown = re.sub(r"\$\$[^$]*\$\$|\$[^$\n]*\$", restore_math, markdown)
    markdown = re.sub(
        r"\n*\$\$([^$]*?)\$\$\n*",
        lambda match: f"\n\n$$\n{match.group(1).strip()}\n$$\n\n",
        markdown,
    )
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip() + "\n"
    md_path.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
