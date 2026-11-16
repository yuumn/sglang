"""Verify requested order, complete coverage and version-sensitive publication."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from priority_plan import make_plan, make_dflash4l_plan, coordinates, expected_tokens, ack_digest, selected_units
from full_common import DATASETS, VARIANTS
from export_priority import export


class PriorityContract(unittest.TestCase):
    def test_dflash4l_task_is_exactly_six_full_datasets_five_concurrencies_fixed512(self):
        plan=make_dflash4l_plan()
        self.assertEqual(plan["total_cells"],30)
        self.assertEqual(expected_tokens(plan["units"]),3_916_800)
        points=coordinates(plan["units"][0])
        self.assertEqual(points,{f"qwen3-4b/dflash-4L/{d}/{c}/512" for d in DATASETS for c in [1,2,4,8,16]})
    def test_four_requested_stages_precede_remaining_full_matrix_without_duplicates(self):
        plan=make_plan()
        first=plan["units"][:8]
        self.assertEqual([(u["stage"],u["family"],u["variant"]) for u in first], [
            (1,"qwen3-4b","dspark"),(1,"qwen3-4b","dspark_latentspec"),
            (2,"qwen3-8b","dspark"),(2,"qwen3-8b","dspark_latentspec"),
            (3,"qwen3-4b","latentspec"),(3,"qwen3-4b","dflash"),
            (4,"qwen3-8b","latentspec"),(4,"qwen3-8b","dflash")])
        self.assertEqual(sum(u["expected_cells"] for u in first),200)
        self.assertEqual(expected_tokens(first),80_793_600)
        all_coordinates=[c for u in plan["units"] for c in coordinates(u)]
        expected={f"{f}/{v}/{d}/{c}/{n}" for f in ["qwen3-4b","qwen3-8b"] for v in VARIANTS
                  for d in DATASETS for c in [1,2,4,8,16] for n in [512,2048,4096]}
        self.assertEqual(len(all_coordinates),1080)
        self.assertEqual(set(all_coordinates),expected)
        self.assertEqual(len(selected_units(plan,{"stop_after_stage":4})),8)

    def test_new_scope_contains_only_stages_one_and_two_even_without_control_file(self):
        plan=make_plan(max_stage=2)
        self.assertEqual(len(plan["units"]),4)
        self.assertEqual(plan["total_cells"],120)
        self.assertEqual(expected_tokens(plan["units"]),70_553_600)
        self.assertTrue(all(u["stage"] in (1,2) for u in selected_units(plan,{"stop_after_stage":None})))

    def test_publication_ack_changes_when_a_result_version_changes(self):
        row=dict(family="qwen3-4b",variant="dspark",dataset="math500",concurrency=1,output_length=512,
                 output_tokens_per_second=1000,created_at="attempt1")
        before=ack_digest([row]);row["created_at"]="attempt2"
        self.assertNotEqual(before,ack_digest([row]))

    def test_export_does_not_promote_partial_results_to_complete_units(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            (out/"execution-control.json").write_text(json.dumps({"stop_after_stage":4}))
            result=export(out)
            self.assertEqual(result["total_cells"],200)
            self.assertEqual(result["rows"],[])
            self.assertTrue(all(not u["complete"] and u["sha256"] is None for u in result["units"]))

    def test_export_requires_matching_unit_settings_and_actual_graph256(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            unit=make_plan()["units"][0]
            measured=out/"measurements"/unit["unit_id"]
            measured.mkdir(parents=True)
            config={"datasets":["math500","aime25"],"lengths":[512,2048,4096]}
            (measured/"benchmark-config.json").write_text(json.dumps(config))
            fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
            row=dict(family="qwen3-4b",variant="dspark",dataset="math500",concurrency=1,output_length=512,
                valid=True,full_dataset=True,total_requests=500,successful_requests=500,failed_requests=0,
                completion_tokens_min=512,completion_tokens_max=512,total_completion_tokens=256000,
                wall_time_seconds=100,output_tokens_per_second=2560.0,decode_cuda_graph_max_bs=256,
                configuration_sha256=fingerprint)
            path=measured/"results.jsonl"
            path.write_text(json.dumps(row)+"\n")
            self.assertEqual(len(export(out)["rows"]),1)
            for changes in [{"decode_cuda_graph_max_bs":32},{"configuration_sha256":"different"},{"dataset":"mbpp"}]:
                path.write_text(json.dumps(dict(row,**changes))+"\n")
                self.assertEqual(export(out)["rows"],[])


if __name__ == "__main__":
    unittest.main()
