"""Summarize real successful-request token counts and server finish reasons."""
from collections import Counter


def summarize_requests(requests):
    successful = [r for r in requests if r.get("success") is True]
    lengths = sorted(int(r["completion_tokens"]) for r in successful)
    reasons = Counter(r.get("finish_reason") or "unknown" for r in successful)
    count = len(successful)

    def percentile(q):
        if not lengths:
            return None
        position = (len(lengths) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(lengths) - 1)
        return lengths[lower] + (lengths[upper] - lengths[lower]) * (position - lower)

    return {"successful_requests": count, "failed_requests": len(requests) - count,
            "actual_output_tokens_total": sum(lengths),
            "actual_output_tokens_mean": sum(lengths) / count if count else None,
            "actual_output_tokens_min": min(lengths) if lengths else None,
            "actual_output_tokens_max": max(lengths) if lengths else None,
            "actual_output_tokens_p50": percentile(.50), "actual_output_tokens_p95": percentile(.95),
            "finish_reason_counts": dict(reasons), "unknown_finish_reason_requests": reasons["unknown"],
            "length_ended_requests": reasons["length"],
            "truncation_rate_percent": reasons["length"] / count * 100 if count and not reasons["unknown"] else None,
            "definition": "successful requests with finish_reason=length / all successful requests *100",
            "percentile_method": "linear interpolation over sorted actual usage.completion_tokens"}
