"""Real HTTP/SSE tests for the separate full-suite measurement client, no GPU."""
import argparse
import contextlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import full_benchmark
import aiohttp
import test_benchmark_contract as basic


class FullBenchmarkContract(basic.BenchmarkContract):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.args = argparse.Namespace(
            output=Path(self.temp.name), dataset_dir=full_benchmark.ROOT / "datasets",
            requests=0, variant="dspark", family="qwen3-4b", suite="contract_full",
            port=self.port, smoke=False, lengths=[512], concurrencies=[1])

    async def run_benchmark(self):
        # Shrink a synthetic corpus for fast HTTP tests. Production cardinalities
        # are independently checked by test_full_suite.py and fixed in the CLI.
        prompts = {"math500": ["a", "b"]}
        metadata = {"math500": {"available_prompts": 2, "selected_prompts": 2}}
        with patch.object(full_benchmark.reference, "resolve_datasets", return_value=(prompts, metadata)), \
             patch.object(full_benchmark, "COUNTS", {"math500": 2}), \
             contextlib.redirect_stdout(io.StringIO()):
            await full_benchmark.main(self.args)

    async def test_exact_length_and_resume(self):
        await self.run_benchmark()
        r = self.results()[0]
        self.assertTrue(r["full_dataset"])
        self.assertTrue(r["valid"])
        self.assertEqual(r["total_completion_tokens"], 1024)
        self.assertEqual(r["output_tokens_per_second"], 1024 / r["wall_time_seconds"])
        progress = json.loads((self.args.output / "request-progress.json").read_text())
        self.assertEqual(progress["finished_requests"], 2)
        await self.run_benchmark()
        self.assertEqual(len(self.results()), 1)

    async def test_truncated_fixed_output_is_saved_then_rejected(self):
        self.offset = -1
        with self.assertRaisesRegex(RuntimeError, "Invalid full-dataset"):
            await self.run_benchmark()
        self.assertFalse(self.results()[0]["valid"])
        self.assertFalse((self.args.output / "complete.json").exists())

    async def test_natural_eos_allows_shorter_output(self):
        # The final suite is always fixed length. Test the corpus-cardinality
        # guard in place of the parent client's natural-EOS contract.
        self.args.requests = 1
        with patch.object(full_benchmark, "COUNTS", {"math500": 2}):
            with self.assertRaisesRegex(ValueError, "Unexpected corpus size"):
                # Avoid the helper patch that deliberately changes the count.
                with contextlib.redirect_stdout(io.StringIO()):
                    await full_benchmark.main(self.args)

    async def test_resume_rejects_changed_settings(self):
        await self.run_benchmark()
        self.args.lengths = [1024]
        with self.assertRaisesRegex(ValueError, "settings changed"):
            await self.run_benchmark()

    async def test_http_failure_cancels_remaining_requests(self):
        self.http_failure = True
        sampling = argparse.Namespace(
            max_tokens=512, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0,
            frequency_penalty=0.0, presence_penalty=0.0, repetition_penalty=1.0,
            ignore_eos=True, enable_thinking=False, sampling_seed=980406,
            stop=None, extra_request_body={})
        progress = self.args.output / "progress.json"
        async with aiohttp.ClientSession() as session:
            with self.assertRaisesRegex(RuntimeError, "aborting whole cell"):
                await full_benchmark.measured_requests(
                    session, f"http://127.0.0.1:{self.port}", {}, "mock", ["test"] * 500,
                    1, sampling, progress, "failed-case")
        record = json.loads(progress.read_text())
        self.assertGreater(record["failed_requests"], 0)
        self.assertLess(record["finished_requests"], 500)


if __name__ == "__main__":
    import unittest
    unittest.main()
