"""Exercise measurement validation with a real local SSE HTTP server, no GPU."""
import argparse
import asyncio
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from aiohttp import web
import benchmark


class BenchmarkContract(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.offset = 0
        self.bodies = []
        async def completion(request):
            if getattr(self, "http_failure", False):
                return web.Response(status=503, text="mock service unavailable")
            body = await request.json()
            self.bodies.append(body)
            n = body["max_completion_tokens"] + self.offset
            events = [{"choices": [{"delta": {"content": "hello"}}]},
                      {"choices": [{"delta": {"content": " world"}}], "usage": {"completion_tokens": n}}]
            return web.Response(text="".join("data: " + json.dumps(e) + "\n\n" for e in events) + "data: [DONE]\n\n",
                                content_type="text/event-stream")
        app = web.Application()
        app.router.add_get("/v1/models", lambda _: web.json_response({"data": [{"id": "mock"}]}))
        app.router.add_get("/get_server_info", lambda _: web.json_response({"mock": True}))
        app.router.add_post("/flush_cache", lambda _: web.Response(text="ok"))
        app.router.add_post("/v1/chat/completions", completion)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        self.temp = tempfile.TemporaryDirectory()
        self.args = argparse.Namespace(output=Path(self.temp.name), datasets=["math500"], requests=2,
                                      variant="dspark", phase="contract", port=self.port,
                                      temperature=1.0, lengths=[512], natural_max_tokens=512, concurrencies=[1])

    async def asyncTearDown(self):
        await self.runner.cleanup()
        self.temp.cleanup()

    async def run_benchmark(self):
        with contextlib.redirect_stdout(io.StringIO()):
            await benchmark.main(self.args)

    def results(self):
        return [json.loads(x) for x in (self.args.output / "results.jsonl").read_text().splitlines()]

    async def test_exact_length_and_resume(self):
        await self.run_benchmark()
        r = self.results()[0]
        self.assertTrue(r["valid"])
        self.assertEqual(r["total_completion_tokens"], 1024)
        self.assertEqual(r["output_tokens_per_second"], 1024 / r["wall_time_seconds"])
        self.assertTrue(all(b["ignore_eos"] for b in self.bodies))
        await self.run_benchmark()
        self.assertEqual(len(self.results()), 1)

    async def test_truncated_fixed_output_is_saved_then_rejected(self):
        self.offset = -1
        with self.assertRaisesRegex(RuntimeError, "Invalid benchmark"):
            await self.run_benchmark()
        self.assertFalse(self.results()[0]["valid"])
        self.assertEqual(self.results()[0]["length_error_indices"], [0, 1])

    async def test_natural_eos_allows_shorter_output(self):
        self.args.lengths = [0]
        self.offset = -1
        await self.run_benchmark()
        self.assertTrue(self.results()[0]["valid"])
        self.assertFalse(self.bodies[-1]["ignore_eos"])

    async def test_resume_rejects_changed_settings(self):
        await self.run_benchmark()
        self.args.temperature = 0.0
        with self.assertRaisesRegex(ValueError, "settings changed"):
            await self.run_benchmark()


if __name__ == "__main__":
    unittest.main()
