"""Retry only unresolved formula crops with the locally cached CodeFormulaV2 model."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.models.stages.code_formula.code_formula_model import (
    CodeFormulaModel,
    CodeFormulaModelOptions,
)


def main() -> int:
    manifest_path, result_path = map(Path, sys.argv[1:3])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"MODEL_LOADING · CodeFormulaV2 · {len(manifest)} regions", flush=True)
    model = CodeFormulaModel(
        enabled=True,
        artifacts_path=None,
        options=CodeFormulaModelOptions(do_code_enrichment=False, do_formula_enrichment=True),
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
    )
    print("MODEL_READY · CodeFormulaV2", flush=True)
    results: dict[str, str] = {}
    for position, region in enumerate(manifest, start=1):
        index = str(region["index"])
        print(f"REGION_START {position}/{len(manifest)} · P{region['page']} · #{index}", flush=True)
        try:
            with Image.open(region["code_crop"]) as source:
                image = source.convert("RGB")
            item = SimpleNamespace(label="formula", text="")
            result = next(iter(model(None, [SimpleNamespace(item=item, image=image)])))
            results[index] = result.text
            print(f"REGION {position}/{len(manifest)} · P{region['page']} · #{index}", flush=True)
        except Exception as exc:
            results[index] = ""
            print(f"ERROR P{region['page']} · #{index}: {exc}", flush=True)
        result_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
