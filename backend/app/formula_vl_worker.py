"""Run inside the existing PaddleOCR-VL environment; one model load per document."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from paddleocr import PaddleOCRVL


def main() -> int:
    manifest_path, result_path = map(Path, sys.argv[1:3])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pipeline = PaddleOCRVL(
        pipeline_version="v1.6",
        device="cpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )
    results: dict[str, str] = {}
    crop_cache: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="paddle-formula-output-") as tmp:
        for position, region in enumerate(manifest, start=1):
            index = region["index"]
            try:
                if region["crop"] not in crop_cache:
                    for result in pipeline.predict(input=region["crop"]):
                        result.save_to_markdown(save_path=tmp)
                    result_md = Path(tmp) / f"{Path(region['crop']).stem}.md"
                    crop_cache[region["crop"]] = result_md.read_text(encoding="utf-8") if result_md.is_file() else ""
                results[str(index)] = crop_cache[region["crop"]]
                print(f"REGION {position}/{len(manifest)} · P{region['page']} · #{index}", flush=True)
            except Exception as exc:
                results[str(index)] = ""
                print(f"ERROR P{region['page']} · #{index}: {exc}", flush=True)
            # Keep completed predictions even if a later region fails.
            result_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
