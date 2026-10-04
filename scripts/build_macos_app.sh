#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "==> Building frontend"
cd "$ROOT_DIR/frontend"
npm install --cache ./.npm-cache
npm run build

echo "==> Preparing desktop Python environment"
cd "$ROOT_DIR"
python3 -m venv .desktop-venv
source .desktop-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt -r desktop/requirements.txt

echo "==> Packaging macOS app"
pyinstaller \
  --noconfirm \
  --clean \
  --windowed \
  --name "Local OCR Dataset Builder" \
  --collect-submodules pymupdf \
  --collect-data docx \
  --add-data "frontend/dist:frontend/dist" \
  --add-data "backend/app/formula_vl_worker.py:backend/app" \
  --add-data "backend/app/formula_code_worker.py:backend/app" \
  --add-data "backend/app/formula_md_worker.py:backend/app" \
  desktop_app.py

APP_PATH="$ROOT_DIR/dist/Local OCR Dataset Builder.app"

echo "==> Clearing extended attributes and signing app"
xattr -cr "$APP_PATH"
codesign --force --deep --sign - "$APP_PATH"

echo "==> Done"
echo "$APP_PATH"
