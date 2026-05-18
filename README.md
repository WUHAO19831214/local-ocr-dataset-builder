# Local OCR Dataset Builder

独立 macOS 桌面 App，用本机已安装的 `docling + ocrmac` 流程把 PDF 转成 Local Textbook Workbench 可加载的数据集目录。

输出结构：

```text
output_name/
├── output_name.md
├── output_name.json
└── images/
```

## 依赖

OCR 运行依赖现有环境，只读取和调用，不修改：

```text
/Users/wuhao/my-pdf-tool/my-pdf-tool/venv/bin/docling
```

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

