import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from request_statistics import summarize_requests
from publish_request_statistics import matrix,table_xml,table_values,write_csv


class StatisticsPublicationContract(unittest.TestCase):
    def example(self):
        requests=[dict(success=True,completion_tokens=512,finish_reason="length",response_id="a"),
                  dict(success=True,completion_tokens=512,finish_reason="stop",response_id="b")]
        row=dict(case_id="case",created_at="time",family="qwen3-4b",variant="dspark",dataset="math500",
                 concurrency=1,output_length=512,total_requests=2,output_tokens_per_second=123.4567,
                 output_statistics=summarize_requests(requests))
        return requests,row

    def test_statistics_table_roundtrip_preserves_actual_lengths_and_reason_based_rate(self):
        _,row=self.example();values=matrix([row])
        self.assertEqual(values[1][3],"2")
        self.assertEqual(values[1][-2:],["1","50.00%"])
        self.assertEqual(table_values(ET.fromstring(table_xml(values))),values)

    def test_request_csv_is_checked_against_summary_and_contains_real_reasons(self):
        requests,row=self.example()
        payload=json.dumps(dict(case_id="case",created_at="time",requests=requests))+"\n"
        with tempfile.TemporaryDirectory() as folder:
            args=argparse.Namespace(phase="phase",output=Path(folder))
            with patch("publish_request_statistics.subprocess.run",return_value=subprocess.CompletedProcess([],0,payload,"")):
                path,count,digest=write_csv(args,{"unit_id":"u01"},[row])
                data=list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig"))))
                self.assertEqual(count,2)
                self.assertEqual([r["finish_reason"] for r in data],["length","stop"])
                self.assertEqual([r["actual_output_tokens"] for r in data],["512","512"])
                row["output_statistics"]["length_ended_requests"]=2
                with self.assertRaisesRegex(RuntimeError,"differ"):
                    write_csv(args,{"unit_id":"u01"},[row])


if __name__ == "__main__":
    unittest.main()
