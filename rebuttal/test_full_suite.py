"""Check all-row eligibility, interrupted append recovery, and safe model cache."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from full_common import valid_cell, read_rows
from full_suite import stage, fully_complete
from publish_full import verify_section, fetch_table


class FullSuiteContract(unittest.TestCase):
    def row(self, **changes):
        row = dict(valid=True, full_dataset=True, total_requests=500,
                   successful_requests=500, failed_requests=0, output_length=512,
                   completion_tokens_min=512, completion_tokens_max=512,
                   total_completion_tokens=256000, wall_time_seconds=100,
                   output_tokens_per_second=2560.0)
        row.update(changes)
        return row

    def test_samples_failures_and_wrong_metric_cannot_be_full_results(self):
        self.assertTrue(valid_cell(self.row(), 500))
        for change in [dict(full_dataset=False), dict(total_requests=32),
                       dict(successful_requests=499, failed_requests=1),
                       dict(completion_tokens_min=511), dict(output_tokens_per_second=3000.0)]:
            self.assertFalse(valid_cell(self.row(**change), 500), change)

    def test_torn_last_append_preserves_completed_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            path.write_bytes(b'{"valid":true}\n{"valid":')
            self.assertEqual(read_rows(path), [{"valid": True}])
            self.assertEqual(path.read_bytes(), b'{"valid":true}\n')
            self.assertEqual(path.with_suffix(".jsonl.interrupted-tail").read_bytes(), b'{"valid":')

    def test_cache_detects_and_repairs_corrupted_copy_but_protects_unowned_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "config.json").write_text('{"hidden_size":2560}')
            (source / "model.safetensors").write_bytes(b"trained-checkpoint-bytes")
            destination = root / "cache"
            expected = stage(source, destination)
            (destination / "model.safetensors").write_bytes(b"corrupt")
            repaired = stage(source, destination)
            self.assertEqual(expected["files"], repaired["files"])
            (destination / "model.safetensors").unlink()
            self.assertEqual(expected["files"], stage(source, destination)["files"])
            unowned = root / "unowned"
            unowned.mkdir()
            with self.assertRaisesRegex(RuntimeError, "unowned"):
                stage(source, unowned)
            self.assertTrue(unowned.exists())

    def test_document_verification_checks_all_cells_not_only_marker(self):
        xml = '<fragment><p>digest</p><table><tr><th><p>dataset</p></th><th><p>C1</p></th></tr><tr><td><p>AIME25</p></td><td><p>123.456</p></td></tr></table></fragment>'
        verify_section(xml, "digest", [["dataset", "C1"], ["AIME25", "123.456"]])
        with self.assertRaisesRegex(RuntimeError, "differs"):
            verify_section(xml, "digest", [["dataset", "C1"], ["AIME25", "999.000"]])

    def test_missing_document_heading_accepts_empty_search_response(self):
        for content in ["", "  ", None, "<fragment/>"]:
            with patch("publish_full.lark", return_value={"data": {"document": {"content": content}}}):
                self.assertIsNone(fetch_table("FULLTABLE qwen3-4b/dflash/fixed512"))


if __name__ == "__main__":
    unittest.main()
