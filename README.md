# Local OCR Dataset Builder

独立 macOS 桌面 App，用本机已安装的 `docling + ocrmac` 流程把 PDF 转成 Local Textbook Workbench 可加载的数据集目录。

输出结构：

```text
output_name/
├── output_name.md
├── output_name.json
├── images/
├── output_name_原版式.docx
└── output_name_可编辑.docx
```

## 依赖

OCR 运行依赖现有环境，只读取和调用，不修改：

```text
/Users/wuhao/my-pdf-tool/my-pdf-tool/venv/bin/docling
```

默认同时输出两个 Word 文件。`原版式`每页采用原 PDF 页面图像，适合保留图片、公式和排版，但页面内容不能逐字编辑。`可编辑`使用 OCR Markdown 生成文本、图片和公式；若系统可找到 Pandoc，LaTeX 公式会转换为 Word 数学对象。OCR 误识别不会因导出 Word 自动修复。

### 高精度公式复核

处理模式选“高精度公式复核”时，先运行 Docling 公式增强，再用本机已有的 PaddleOCR-VL 1.6 对疑似行内公式、公式块及含符号的短选项按原 PDF 位置裁图重识别。只有包含可复现数学结构、且中文主体变化不过大的结果会写回 Markdown/JSON，并用于生成可编辑 Word。完整原始 OCR 保存在 `output_name_原始OCR.md/json`；每处原文、新候选、是否采用和原 PDF 裁图保存在 `output_name_公式复核.json` 与 `formula_regions/`。由于模型可能误认上下标及字符，仍应逐项对照裁图检查。

此模式依赖现有 `/Users/wuhao/LocalProjects/Codex/macbook-air-m2/paddleocr-vl-benchmark/.venv` 和其中缓存的 PaddleOCR-VL 1.6 模型。CPU 上会逐区域运行，整份理科试卷可能需要较长时间。普通与 Docling 公式模式无需这项依赖。

## 开发模式

后端：

```bash
python3 -m venv .desktop-venv
source .desktop-venv/bin/activate
pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8765 --reload
```

前端：

```bash
cd frontend
npm install --cache ./.npm-cache
npm run dev
```

桌面壳开发运行：

```bash
source .desktop-venv/bin/activate
pip install -r desktop/requirements.txt
npm --prefix frontend run build
python desktop_app.py
```

## 打包

```bash
chmod +x scripts/build_macos_app.sh
./scripts/build_macos_app.sh
```

打包产物：

```text
dist/Local OCR Dataset Builder.app
```

## API

- `GET /api/health`
- `POST /api/jobs/start`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/logs`
- `POST /api/system/open-path`

## 默认测试参数

```text
PDF: /Users/wuhao/my-pdf-tool/my-pdf-tool/dajiaderiyufudaoshu12_p1_2.pdf
输出根目录: /Users/wuhao/my-pdf-tool/datasets
输出名称: test_builder_p1_2
语言: ja-JP,zh-Hans
```
