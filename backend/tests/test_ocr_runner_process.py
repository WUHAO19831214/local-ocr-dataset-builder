from __future__ import annotations

import tempfile
import unittest
import subprocess
import sys
import pymupdf
from pathlib import Path
from unittest.mock import patch

from backend.app.ocr_runner import OcrRunnerError, _docling_timeout_seconds, _read_docling_output, _run_docling
from backend.app.process_registry import register_process, stop_active_processes, unregister_process


class OcrRunnerProcessTests(unittest.TestCase):
    def test_large_basic_ocr_gets_time_per_page(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "large.pdf"
            pdf = pymupdf.open()
            for _ in range(10):
                pdf.new_page()
            pdf.save(path)
            pdf.close()
            self.assertEqual(_docling_timeout_seconds(path, "formula_vl"), 900)
            self.assertEqual(_docling_timeout_seconds(path, "formula"), 600)

    def test_app_shutdown_stops_tracked_process_group(self) -> None:
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                   start_new_session=True)
        register_process(process)
        try:
            stop_active_processes()
            self.assertNotEqual(process.wait(timeout=5), 0)
        finally:
            unregister_process(process)
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)

    def test_silent_docling_is_stopped_at_hard_timeout(self) -> None:
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                   stdout=subprocess.PIPE, text=True, start_new_session=True)
        try:
            with patch("backend.app.ocr_runner.DOCLING_TIMEOUT_SECONDS", 0.2):
                with self.assertRaises(OcrRunnerError):
                    _read_docling_output(process, lambda _: None)
            self.assertIsNotNone(process.poll())
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
            if process.stdout is not None:
                process.stdout.close()

    def test_high_precision_runs_one_conversion_without_slow_docling_formula_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            process = unittest.mock.MagicMock()
            process.stdout = iter([])
            process.wait.return_value = 0
            with patch("backend.app.ocr_runner.subprocess.Popen", return_value=process) as launch:
                _run_docling(Path(temporary) / "input.pdf", Path(temporary) / "output",
                             ("md", "json"), "zh-Hans", "formula_vl", True, lambda _: None)
            command = launch.call_args.args[0]
            self.assertEqual([command[index + 1] for index, value in enumerate(command[:-1])
                              if value == "--to"], ["md", "json"])
            self.assertNotIn("--enrich-formula", command)
            self.assertIn("--force-ocr", command)
            self.assertTrue(launch.call_args.kwargs["start_new_session"])

    def test_docling_formula_mode_has_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            process = unittest.mock.MagicMock()
            process.stdout = iter([])
            process.wait.return_value = 0
            with patch("backend.app.ocr_runner.subprocess.Popen", return_value=process) as launch:
                _run_docling(Path(temporary) / "input.pdf", Path(temporary) / "output",
                             ("md", "json"), "zh-Hans", "formula", False, lambda _: None)
            command = launch.call_args.args[0]
            self.assertIn("--enrich-formula", command)
            self.assertEqual(command[command.index("--document-timeout") + 1], "600")


if __name__ == "__main__":
    unittest.main()
