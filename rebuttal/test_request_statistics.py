import unittest
from request_statistics import summarize_requests


class RequestStatisticsContract(unittest.TestCase):
    def test_truncation_uses_real_finish_reason_and_successful_denominator(self):
        # A natural stop exactly at the cap must not be called truncation.
        result=summarize_requests([
            {"success":True,"completion_tokens":512,"finish_reason":"length"},
            {"success":True,"completion_tokens":512,"finish_reason":"stop"},
            {"success":True,"completion_tokens":4,"finish_reason":"stop"},
            {"success":False,"completion_tokens":0,"finish_reason":"length"}])
        self.assertEqual(result["successful_requests"],3)
        self.assertEqual(result["length_ended_requests"],1)
        self.assertAlmostEqual(result["truncation_rate_percent"],100/3)
        self.assertEqual(result["actual_output_tokens_total"],1028)
        self.assertEqual(result["actual_output_tokens_p50"],512)
        self.assertEqual(result["actual_output_tokens_min"],4)

    def test_missing_reasons_and_zero_successes_are_not_reported_as_zero_truncation(self):
        self.assertIsNone(summarize_requests([])["truncation_rate_percent"])
        r=summarize_requests([{"success":True,"completion_tokens":512}])
        self.assertEqual(r["unknown_finish_reason_requests"],1)
        self.assertIsNone(r["truncation_rate_percent"])


if __name__ == "__main__":
    unittest.main()
